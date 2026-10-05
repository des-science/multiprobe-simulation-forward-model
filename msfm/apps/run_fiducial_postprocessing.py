# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created March 2024
Author: Arne Thomsen

Transform the full sky weak lensing signal and intrinsic alignment maps into multiple survey footprint cut-outs and
store them in .tfrecord files. The parallelization is done over the .tfrecord files, every jobarray element corresponds
to one.

For the fiducial, the main loop runs over the different permutations (simulation runs).

The forward model is functionally equivalent to the one in run_grid_postprocessing.py (multiplicative shear bias,
shape-noise model, per-bin or power-law galaxy biasing), evaluated at the fiducial parameters and their
finite-difference perturbations (for example for Fisher forecasts). The astrophysical amplitudes are read from the
config instead of being sampled from the Latin hypercube, and the per-example m-bias draw is shared between the
fiducial and its perturbations to keep the finite differences consistent.

Meant for
 - Euler (CPU nodes, local scratch)
 - esub jobarrays
 - Read the CosmoGrid directly from the SAN
 - CosmoGridV1.1
"""

import numpy as np
import tensorflow as tf
import os
import argparse
import warnings
import time
import yaml
import h5py
import pickle

from msfm.utils import (
    logger,
    imports,
    filenames,
    input_output,
    files,
    lensing,
    clustering,
    cosmogrid,
    postprocessing,
    tfrecords,
    power_spectra,
    scales,
    parameters,
    configuration,
)

hp = imports.import_healpy()

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


def resources(args):
    args = setup(args)

    if args.cluster == "perlmutter":
        # Same billing logic as run_grid_postprocessing.resources: NERSC bills max(cores/128,
        # mem/512GB) on the shared QOS, so the memory sets a floor on the charge and cores under it
        # are effectively free. With the derivatives every permutation loops over all
        # (2 * n_params + 1) cosmology dirs, peaking at ~17.2 GB -> total = 4 * 5120 = 20 GB ->
        # billed ~3.9% of a node, vs 12.5% at the old 16 cores. (A later packed run measured only
        # ~11.5 GB average-at-peak per task, so the 20 GB request is comfortably conservative.)
        resources = {
            "main_time": 2 if not args.no_derivatives else 1,
            "main_n_cores": 4,
            "main_memory": 5120,
            "main_scratch": 0,
            "merge_time": 2,
            "merge_n_cores": 32,
            "merge_memory": 1952,
            "merge_scratch": 0,
        }
    elif args.cluster == "euler":
        resources = {"main_time": 4, "main_memory": 4096, "main_n_cores": 8, "merge_memory": 4096, "merge_n_cores": 16}

        if args.from_san:
            # in MB. One projected_probes_maps_v11dmb.h5 should be around 1 GB
            resources["main_scratch"] = 4000
        else:
            resources["main_scratch"] = 0

    return resources


def setup(args):
    description = "Postprocess the CosmoGrid projections into forward-modeled survey footprints in .tfrecord files"
    parser = argparse.ArgumentParser(description=description, add_help=True)

    parser.add_argument(
        "--n_files",
        type=int,
        default=2500,
        help="number of .tfrecord files to produce, this should be equal to the number of tasks in esub",
    )
    parser.add_argument(
        "--dir_in",
        type=str,
        required=True,
        help="input root dir of the full sky CosmoGrid projections",
    )
    parser.add_argument(
        "--dir_out",
        type=str,
        required=True,
        help="output root dir of the forward-modeled survey footprints",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="configuration .yaml file",
    )
    parser.add_argument(
        "--cosmogrid_version",
        type=str,
        default="1.1",
        choices=["1.1", "1"],
        help="version of the input CosmoGrid",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default="perlmutter",
        choices=("perlmutter", "euler"),
        help="the cluster to execute on",
    )
    parser.add_argument(
        "--no_derivatives",
        action="store_true",
        help="do not compute the derivatives",
    )
    parser.add_argument(
        "--file_suffix",
        type=str,
        default="",
        help="Optional suffix to be appended to the end of the filename, for example to distinguish different runs",
    )
    parser.add_argument(
        "--max_sleep",
        type=int,
        default=120,
        help="set the maximal amount of time to sleep before copying to avoid clashes",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=str,
        default="info",
        choices=("critical", "error", "warning", "info", "debug"),
        help="logging level",
    )
    parser.add_argument("--debug", action="store_true", help="activate debug mode")

    args, _ = parser.parse_known_args(args)

    # print arguments
    logger.set_all_loggers_level(args.verbosity)
    for key, value in vars(args).items():
        LOGGER.info(f"{key} = {value}")

    # paths
    args.config = os.path.abspath(args.config)

    args.from_san = "/home/ipa/refreg" in args.dir_in
    if args.from_san:
        LOGGER.warning("Reading the CosmoGrid from the SAN")

    args.to_san = "/home/ipa/refreg" in args.dir_out
    if args.to_san:
        LOGGER.warning("Writing the .tfrecords to the SAN")
    elif not os.path.isdir(args.dir_out):
        input_output.robust_makedirs(args.dir_out)

    # compute
    try:
        LOGGER.info(f"Running on {len(os.sched_getaffinity(0))} cores")
    except AttributeError:
        pass

    return args


def main(indices, args):
    args = setup(args)

    LOGGER.timer.start("main")
    LOGGER.info(f"Got index set of size {len(indices)}")

    # I/O delay
    if args.debug:
        args.max_sleep = 0
        LOGGER.warning("debug mode")
    sleep_sec = np.random.uniform(0, args.max_sleep) if args.max_sleep > 0 else 0
    LOGGER.info(f"Waiting for {sleep_sec:.2f}s to prevent overloading IO")
    time.sleep(sleep_sec)

    # configuration
    conf = files.load_config(args.config)
    # shape-noise model; a "prior" source-clustering bias is sampled from the Latin hypercube on the grid and has no
    # fiducial counterpart
    _, sn_bias, _, _ = files.get_shape_noise(conf)
    assert sn_bias != "prior", "The prior source-clustering bias is not implemented for the fiducial"
    if not args.to_san:
        with open(os.path.join(args.dir_out, "config.yaml"), "w") as f:
            yaml.dump(conf, f)

    # modeling
    configuration.print_and_check_modeling_in_config(conf)

    baryonified = conf["analysis"]["modelling"]["baryonified"]

    # the fiducial serialization (tfrecords.parse_forward_fiducial) always contains both probes
    assert conf["analysis"]["modelling"]["lensing"]["store"], "The fiducial always stores both probes"
    assert conf["analysis"]["modelling"]["clustering"]["store"], "The fiducial always stores both probes"
    assert not conf["analysis"]["modelling"]["store_cross_maps"], "Cross maps are not implemented for the fiducial"

    extended_nla = conf["analysis"]["modelling"]["lensing"]["extended_nla"]

    # B-mode information-loss study: additionally carry the metacal B-mode convergence through the lensing forward
    # model and store the extra cross-spectra that involve a B channel (see run_tfrecords_alm_to_cl_bmode)
    keep_b_mode = conf["analysis"]["modelling"]["lensing"].get("b_mode_cls", False)
    if keep_b_mode:
        LOGGER.warning("b_mode_cls is on: also computing the metacal B-mode Cls block (cl_bmode_*)")

    quadratic_biasing = conf["analysis"]["modelling"]["clustering"]["quadratic_biasing"]
    stochasticity = conf["analysis"]["modelling"]["clustering"]["stochasticity"]
    assert not quadratic_biasing, "The quadratic biasing has not been implemented yet"
    assert not stochasticity, "The stochasticity has not been implemented yet"

    # directories
    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    meta_info_file = os.path.join(repo_dir, conf["files"]["meta_info"])

    cosmo_params_info = cosmogrid.get_cosmo_params_info(meta_info_file, "fiducial")
    cosmo_dirs = [cosmo_dir.decode("utf-8") for cosmo_dir in cosmo_params_info["path_par"]]
    if baryonified:
        cosmo_dirs = [cosmo_dir for cosmo_dir in cosmo_dirs]
        LOGGER.info(f"Using the baryonified inputs, then there's {len(cosmo_dirs) - 1} fiducial perturbations")
    else:
        cosmo_dirs = [cosmo_dir for cosmo_dir in cosmo_dirs if "bary" not in cosmo_dir]
        LOGGER.info(f"Using the dark matter only inputs, then there's {len(cosmo_dirs) - 1} fiducial perturbations")
    cosmo_dirs_in = [os.path.join(args.dir_in, "fiducial", cosmo_dir) for cosmo_dir in cosmo_dirs]

    # CosmoGrid
    n_patches = conf["analysis"]["n_patches"]
    n_perms_per_cosmo = conf["analysis"]["fiducial"]["n_perms_per_cosmo"]
    n_noise_per_signal = conf["analysis"]["fiducial"]["n_noise_per_signal"]
    n_examples_per_cosmo = n_patches * n_perms_per_cosmo * n_noise_per_signal
    LOGGER.info(
        f"For the fiducial cosmology, there's {n_examples_per_cosmo} in total: "
        f"{n_patches} patches times {n_perms_per_cosmo} permutations times {n_noise_per_signal} noise realizations"
    )

    # .tfrecords
    if n_perms_per_cosmo % args.n_files == 0:
        n_perms_per_file = n_perms_per_cosmo // args.n_files
        n_examples_per_file = n_examples_per_cosmo // args.n_files
    else:
        raise ValueError(
            f"The total number of examples per cosmology {n_perms_per_cosmo}"
            f" has to be evenly divisible by the number of files {args.n_files}"
        )
    LOGGER.info(f"There are {n_perms_per_cosmo} fiducial permutations and {n_examples_per_file} examples per file")

    # CosmoGrid perturbations in cosmological parameters
    cosmo_pert_labels = [label.split("cosmo_")[1].replace("/", "") for label in cosmo_dirs_in]
    LOGGER.info(f"There's {len(cosmo_pert_labels)} cosmological labels = {cosmo_pert_labels}")

    # separate label lists for astrophysics perturbations
    ia_params = list(conf["analysis"]["params"]["ia"]["nla"])
    if extended_nla:
        ia_params += conf["analysis"]["params"]["ia"]["tatt"]
    ia_pert_labels = parameters.get_fiducial_perturbation_labels(ia_params)[1:]
    LOGGER.info(f"There's {len(ia_pert_labels)} intrinsic alignment labels = {ia_pert_labels}")

    bg_params = list(conf["analysis"]["params"]["bg"]["linear"])
    if quadratic_biasing:
        bg_params += conf["analysis"]["params"]["bg"]["quadratic"]
    bg_pert_labels = parameters.get_fiducial_perturbation_labels(bg_params)[1:]
    LOGGER.info(f"There's {len(bg_pert_labels)} linear galaxy clustering labels = {bg_pert_labels}")

    # analysis files
    pixel_file = files.load_pixel_file(conf)
    data_vec_len = len(pixel_file[0])
    noise_file = files.load_noise_file(conf)

    # transforms
    lensing_transform = _get_lensing_transform(conf, pixel_file)
    clustering_transform = _get_clustering_transform(conf, pixel_file)
    m_bias_dist = lensing.get_m_bias_distribution(conf)

    LOGGER.warning(f"Starting the main loop trough indices {indices}")

    # index corresponds to a .tfrecord file ###########################################################################
    for index in indices:
        LOGGER.warning(f"Starting index {index}")
        LOGGER.timer.start("index")

        if args.to_san:
            LOGGER.info("Writing the .tfrecord to local scratch to be later copied to the SAN")
            san_dir_out = args.dir_out
            args.dir_out = os.environ["TMPDIR"]

        if args.debug:
            args.dir_out = os.path.join(args.dir_out, "debug")
            os.makedirs(args.dir_out, exist_ok=True)

        tfr_file = filenames.get_filename_tfrecords(
            args.dir_out,
            tag=conf["survey"]["name"] + args.file_suffix,
            index=index,
            simset="fiducial",
            with_bary=baryonified,
        )
        LOGGER.info(f"Index {index} is writing to {tfr_file}")

        # used to index permutations
        i_perm_start = index * n_perms_per_file
        i_perm_end = (index + 1) * n_perms_per_file

        n_done = 0
        with tf.io.TFRecordWriter(tfr_file) as file_writer:
            # loop over the example realizations
            for i_perm in LOGGER.progressbar(
                range(i_perm_start, i_perm_end),
                at_level="info",
                desc="Looping through the permutations\n",
                total=i_perm_end - i_perm_start,
            ):
                LOGGER.info(f"i_perm = {i_perm} in range({i_perm_start},{i_perm_end})")

                state_file = os.path.join(args.dir_out, f"program_state{i_perm:04}.pkl")
                if args.debug and os.path.exists(state_file):
                    with open(state_file, "rb") as f:
                        state = pickle.load(f)
                        kg_perts = state["kg_perts"]
                        ia_perts = state["ia_perts"]
                        dg_perts = state["dg_perts"]
                        bg_perts = state["bg_perts"]
                        all_sn_samples = state["all_sn_samples"]
                        all_pn_samples = state["all_pn_samples"]
                        cl_perts = state["cl_perts"]
                        cl_ia_perts = state["cl_ia_perts"]
                        cl_bg_perts = state["cl_bg_perts"]
                        cl_bmode_perts = state["cl_bmode_perts"]
                        cl_bmode_ia_perts = state["cl_bmode_ia_perts"]
                        cl_bmode_bg_perts = state["cl_bmode_bg_perts"]
                        all_i_example = state["all_i_example"]
                    LOGGER.warning(f"Debug mode, reading the state from {state_file}")
                else:
                    # annoyingly, we need to keep track of the patches like this
                    n_metacal_bins = len(conf["survey"]["metacal"]["z_bins"])
                    n_maglim_bins = len(conf["survey"]["maglim"]["z_bins"])
                    n_cosmo_perts = len(cosmo_pert_labels)
                    n_ia_perts = len(ia_pert_labels)
                    n_bg_perts = len(bg_pert_labels)

                    # maps
                    kg_perts = np.zeros((n_patches, n_cosmo_perts, data_vec_len, n_metacal_bins), dtype=np.float32)
                    dg_perts = np.zeros((n_patches, n_cosmo_perts, data_vec_len, n_maglim_bins), dtype=np.float32)
                    ia_perts = np.zeros((n_patches, n_ia_perts, data_vec_len, n_metacal_bins), dtype=np.float32)
                    bg_perts = np.zeros((n_patches, n_bg_perts, data_vec_len, n_maglim_bins), dtype=np.float32)

                    all_sn_samples = np.zeros(
                        (n_patches, n_noise_per_signal, data_vec_len, n_metacal_bins), dtype=np.float32
                    )
                    all_pn_samples = np.zeros(
                        (n_patches, n_noise_per_signal, data_vec_len, n_maglim_bins), dtype=np.float32
                    )
                    all_i_example = np.zeros((n_patches,), dtype=np.int32)

                    # delta-NLA cross-term map: only ever nonzero at the true-fiducial cosmo dir (see
                    # postprocessing.postprocess_fiducial_permutations), captured there and reused for every
                    # cosmological perturbation, like tomo_m_bias and the shape noise
                    all_ds = [None] * n_patches
                    all_ds_b = [None] * n_patches

                    # power spectra
                    n_bins = n_metacal_bins + n_maglim_bins
                    n_cross_bins = n_bins * (n_bins + 1) // 2
                    n_ell = 3 * conf["analysis"]["n_side"]

                    all_alm_sn = []
                    all_alm_pn = []

                    cl_perts = np.zeros(
                        (n_patches, n_cosmo_perts, n_noise_per_signal, n_ell, n_cross_bins), dtype=np.float32
                    )
                    cl_ia_perts = np.zeros(
                        (n_patches, n_ia_perts, n_noise_per_signal, n_ell, n_cross_bins), dtype=np.float32
                    )
                    cl_bg_perts = np.zeros(
                        (n_patches, n_bg_perts, n_noise_per_signal, n_ell, n_cross_bins), dtype=np.float32
                    )

                    # B-mode Cls block: the 12-channel [E, maglim, B] stack minus the standard 8-channel subset.
                    # Only metacal (spin-2) has a B-mode, so there are n_metacal_bins extra channels.
                    n_bmode_channels = n_bins + n_metacal_bins
                    n_bmode_cross = (n_bmode_channels * (n_bmode_channels + 1) // 2) - n_cross_bins
                    all_alm_sn_b = []
                    if keep_b_mode:
                        cl_bmode_perts = np.zeros(
                            (n_patches, n_cosmo_perts, n_noise_per_signal, n_ell, n_bmode_cross), dtype=np.float32
                        )
                        cl_bmode_ia_perts = np.zeros(
                            (n_patches, n_ia_perts, n_noise_per_signal, n_ell, n_bmode_cross), dtype=np.float32
                        )
                        cl_bmode_bg_perts = np.zeros(
                            (n_patches, n_bg_perts, n_noise_per_signal, n_ell, n_bmode_cross), dtype=np.float32
                        )
                    else:
                        cl_bmode_perts = cl_bmode_ia_perts = cl_bmode_bg_perts = None

                    # multiplicative shear bias: like on the grid, one draw per example is baked into the
                    # .tfrecords (see run_grid_postprocessing.py), but the draw is shared by the fiducial and all
                    # of its perturbations so that the finite differences stay consistent
                    tomo_m_bias = m_bias_dist.sample(n_patches).numpy().astype(np.float32)

                    if args.no_derivatives:
                        LOGGER.warning("Not computing the derivatives")
                        cosmo_dirs_in = [cosmo_dirs_in[0]]

                    # loop over the perturbations in the right order, there's 2 * n_cosmos + 1 iterations
                    for i_cosmo, cosmo_dir_in in LOGGER.progressbar(
                        enumerate(cosmo_dirs_in),
                        at_level="info",
                        desc="\nLooping through the perturbations\n",
                        total=len(cosmo_dirs_in),
                    ):
                        LOGGER.info(f"cosmo_dir_in = {cosmo_dir_in}")
                        is_fiducial = "cosmo_fiducial" in cosmo_dir_in

                        # this is what was previously stored in the separate .h5 files
                        data_vec_container = postprocessing.postprocess_fiducial_permutations(
                            args, conf, cosmo_dir_in, i_perm, pixel_file, noise_file
                        )

                        for i_patch in range(n_patches):
                            i_signal = i_perm * n_patches + i_patch
                            all_i_example[i_patch] = i_signal

                            # shape (n_pix, n_z_bins)
                            kg_in = data_vec_container["kg"][i_patch]
                            ia_in = data_vec_container["ia"][i_patch]
                            dg_in = data_vec_container["dg"][i_patch]
                            if quadratic_biasing:
                                dg2_in = data_vec_container["dg2"][i_patch]
                            else:
                                dg2_in = None

                            # parallel metacal B-mode convergence inputs (None disables the B channel downstream)
                            if keep_b_mode:
                                kg_b_in = data_vec_container["kg_b"][i_patch]
                                ia_b_in = data_vec_container["ia_b"][i_patch]
                            else:
                                kg_b_in = ia_b_in = None

                            # delta-NLA cross-term input; only populated by postprocess_fiducial_permutations at the
                            # true-fiducial cosmo dir, captured into all_ds below and reused for other cosmo dirs
                            if extended_nla and is_fiducial:
                                ds_in = data_vec_container["ds"][i_patch]
                                ds_b_in = data_vec_container["ds_b"][i_patch] if keep_b_mode else None
                            else:
                                ds_in = all_ds[i_patch]
                                ds_b_in = all_ds_b[i_patch]

                            # astrophysics perturbations are calculated with respect to the fiducial cosmo params
                            if is_fiducial:
                                # shape (n_noise_per_signal, n_pix, n_z_bins) load the shape noise realization
                                sn_samples_in = data_vec_container["sn"][i_patch]
                                sn_b_samples_in = data_vec_container["sn_b"][i_patch] if keep_b_mode else None

                                # add the signal and ia maps and smooth everything
                                kg, sn_samples, alm_kg, alm_sn, alm_kg_b, alm_sn_b = lensing_transform(
                                    kg_in,
                                    ia_in,
                                    ia_label="fiducial",
                                    m_bias=tomo_m_bias[i_patch],
                                    is_true_fiducial=True,
                                    sn_samples=sn_samples_in,
                                    np_seed=i_signal,
                                    kg_b=kg_b_in,
                                    ia_b=ia_b_in,
                                    sn_b_samples=sn_b_samples_in,
                                    ds=ds_in,
                                    ds_b=ds_b_in,
                                )
                                all_ds[i_patch] = ds_in
                                all_ds_b[i_patch] = ds_b_in

                                # convert dg to galaxy number and draw the poisson noise realization
                                dg, pn_samples, alm_dg, alm_pn = clustering_transform(
                                    dg_in, dg2_in, bg_label="fiducial", is_true_fiducial=True, np_seed=i_signal
                                )

                                all_sn_samples[i_patch] = sn_samples
                                all_pn_samples[i_patch] = pn_samples

                                all_alm_sn.append(alm_sn)
                                all_alm_pn.append(alm_pn)
                                all_alm_sn_b.append(alm_sn_b)

                                # intrinsic alignment perturbations
                                for i_ia, ia_pert_label in enumerate(ia_pert_labels):
                                    ia_perts[i_patch, i_ia], alm_ia, alm_ia_b = lensing_transform(
                                        kg_in,
                                        ia_in,
                                        ia_label=ia_pert_label,
                                        m_bias=tomo_m_bias[i_patch],
                                        np_seed=i_signal,
                                        kg_b=kg_b_in,
                                        ia_b=ia_b_in,
                                        ds=ds_in,
                                        ds_b=ds_b_in,
                                    )
                                    cl_ia_perts[i_patch, i_ia] = power_spectra.run_tfrecords_alm_to_cl(
                                        alm_ia, alm_sn, alm_dg, alm_pn
                                    )
                                    if keep_b_mode:
                                        cl_bmode_ia_perts[i_patch, i_ia] = power_spectra.run_tfrecords_alm_to_cl_bmode(
                                            alm_ia,
                                            alm_sn,
                                            alm_dg,
                                            alm_pn,
                                            alm_ia_b,
                                            alm_sn_b,
                                            cl_reference=cl_ia_perts[i_patch, i_ia],
                                        )

                                # galaxy clustering perturbations
                                for i_bg, bg_pert_label in enumerate(bg_pert_labels):
                                    bg_perts[i_patch, i_bg], alm_bg = clustering_transform(
                                        dg_in, dg2_in, bg_label=bg_pert_label, np_seed=i_signal
                                    )
                                    cl_bg_perts[i_patch, i_bg] = power_spectra.run_tfrecords_alm_to_cl(
                                        alm_kg, alm_sn, alm_bg, alm_pn
                                    )
                                    if keep_b_mode:
                                        cl_bmode_bg_perts[i_patch, i_bg] = power_spectra.run_tfrecords_alm_to_cl_bmode(
                                            alm_kg,
                                            alm_sn,
                                            alm_bg,
                                            alm_pn,
                                            alm_kg_b,
                                            alm_sn_b,
                                            cl_reference=cl_bg_perts[i_patch, i_bg],
                                        )

                            # cosmological perturbations
                            else:
                                kg, alm_kg, alm_kg_b = lensing_transform(
                                    kg_in,
                                    ia_in,
                                    ia_label="fiducial",
                                    m_bias=tomo_m_bias[i_patch],
                                    np_seed=i_signal,
                                    kg_b=kg_b_in,
                                    ia_b=ia_b_in,
                                    ds=ds_in,
                                    ds_b=ds_b_in,
                                )
                                dg, alm_dg = clustering_transform(dg_in, dg2_in, bg_label="fiducial", np_seed=i_signal)

                            kg_perts[i_patch, i_cosmo] = kg
                            dg_perts[i_patch, i_cosmo] = dg
                            cl_perts[i_patch, i_cosmo] = power_spectra.run_tfrecords_alm_to_cl(
                                alm_kg, all_alm_sn[i_patch], alm_dg, all_alm_pn[i_patch]
                            )
                            if keep_b_mode:
                                cl_bmode_perts[i_patch, i_cosmo] = power_spectra.run_tfrecords_alm_to_cl_bmode(
                                    alm_kg,
                                    all_alm_sn[i_patch],
                                    alm_dg,
                                    all_alm_pn[i_patch],
                                    alm_kg_b,
                                    all_alm_sn_b[i_patch],
                                    cl_reference=cl_perts[i_patch, i_cosmo],
                                )

                    if args.debug:
                        state = {
                            "kg_perts": kg_perts,
                            "ia_perts": ia_perts,
                            "dg_perts": dg_perts,
                            "bg_perts": bg_perts,
                            "all_sn_samples": all_sn_samples,
                            "all_pn_samples": all_pn_samples,
                            "cl_perts": cl_perts,
                            "cl_ia_perts": cl_ia_perts,
                            "cl_bg_perts": cl_bg_perts,
                            "cl_bmode_perts": cl_bmode_perts,
                            "cl_bmode_ia_perts": cl_bmode_ia_perts,
                            "cl_bmode_bg_perts": cl_bmode_bg_perts,
                            "all_i_example": all_i_example,
                        }
                        with open(state_file, "wb") as f:
                            LOGGER.warning(f"Debug mode, writing the state to {state_file}")
                            pickle.dump(state, f)

                # the .tfrecord entries are individual examples
                LOGGER.info(f"Writing the {n_patches} patches to the .tfrecord")
                for i_patch in range(n_patches):
                    serialized = _serialize_and_verify(
                        n_noise_per_signal,
                        # labels
                        cosmo_pert_labels,
                        ia_pert_labels,
                        bg_pert_labels,
                        # arrays
                        kg_perts[i_patch],
                        ia_perts[i_patch],
                        dg_perts[i_patch],
                        bg_perts[i_patch],
                        all_sn_samples[i_patch],
                        all_pn_samples[i_patch],
                        cl_perts[i_patch],
                        cl_ia_perts[i_patch],
                        cl_bg_perts[i_patch],
                        all_i_example[i_patch],
                        cl_bmode_perts=cl_bmode_perts[i_patch] if keep_b_mode else None,
                        cl_bmode_ia_perts=cl_bmode_ia_perts[i_patch] if keep_b_mode else None,
                        cl_bmode_bg_perts=cl_bmode_bg_perts[i_patch] if keep_b_mode else None,
                    )

                    file_writer.write(serialized)

                n_done += 1

        if args.to_san:
            postprocessing._rsync_tfrecord_to_san(conf, tfr_file, san_dir_out)

        LOGGER.info(f"Done with index {index} after {LOGGER.timer.elapsed('index')}")
        yield index


def _data_vector_smoothing(dv, l_min, l_max, theta_fwhm, np_seed, conf, pixel_file, mask):
    # Gaussian Random Field
    if conf["analysis"]["modelling"]["degrade_to_grf"]:
        dv, alm = scales.data_vector_to_grf_data_vector(
            np_seed,
            dv,
            data_vec_pix=pixel_file[0],
            n_side=conf["analysis"]["n_side"],
            l_min=l_min,
            l_max=l_max,
            theta_fwhm=theta_fwhm,
            arcmin=True,
            mask=mask,
            conf=conf,
            hard_cut=conf["analysis"]["scale_cuts"]["hard_cut"],
        )

    # standard smoothing with a Gaussian kernel
    else:
        dv, alm = scales.data_vector_to_smoothed_data_vector(
            dv,
            data_vec_pix=pixel_file[0],
            n_side=conf["analysis"]["n_side"],
            l_min=l_min,
            l_max=l_max,
            theta_fwhm=theta_fwhm,
            arcmin=True,
            mask=mask,
            conf=conf,
            hard_cut=conf["analysis"]["scale_cuts"]["hard_cut"],
        )

    return dv, alm


def _get_lensing_transform(conf, pixel_file):
    extended_nla = conf["analysis"]["modelling"]["lensing"]["extended_nla"]
    tomo_Aia_perts_dict = parameters.get_tomo_amplitude_perturbations_dict("Aia", conf)
    metacal_mask = files.get_tomo_dv_masks(conf)["metacal"]

    if extended_nla:
        bta_fid = conf["analysis"]["fiducial"]["bta"]
        delta_bta = conf["analysis"]["fiducial"]["perturbations"]["bta"]

        # bta value per IA perturbation label; fiducial unless the label perturbs bta itself
        bta_perts_dict = {label: bta_fid for label in tomo_Aia_perts_dict}
        bta_perts_dict["delta_bta_m"] = bta_fid - delta_bta
        bta_perts_dict["delta_bta_p"] = bta_fid + delta_bta

        # Aia/n_Aia stay at their fiducial tomo-amplitude for the new bta perturbation labels
        tomo_Aia_perts_dict["delta_bta_m"] = tomo_Aia_perts_dict["fiducial"]
        tomo_Aia_perts_dict["delta_bta_p"] = tomo_Aia_perts_dict["fiducial"]

        if conf["analysis"]["modelling"]["lensing"].get("b_mode_cls", False):
            LOGGER.warning(
                "b_mode_cls + extended_nla together is untested (no grid-side precedent); proceeding anyway"
            )
    else:
        bta_perts_dict = None

    def lensing_smoothing(kg, np_seed):
        kg, alm = _data_vector_smoothing(
            kg,
            conf["analysis"]["scale_cuts"]["lensing"]["l_min"],
            conf["analysis"]["scale_cuts"]["lensing"]["l_max"],
            conf["analysis"]["scale_cuts"]["lensing"]["theta_fwhm"],
            np_seed,
            conf,
            pixel_file,
            metacal_mask,
        )

        return kg, alm

    def _lensing_channel(kg, ia, ia_label, m_bias, ds=None):
        """The IA combination + m-bias + mask applied identically to an E- or B-mode convergence data vector."""
        # important not to use +=, since then the array is transformed in place
        if extended_nla:
            # first term of TATT (cross term), see run_grid_postprocessing.py; ds already contains the ia map
            kg = kg + tomo_Aia_perts_dict[ia_label] * (ia + bta_perts_dict[ia_label] * ds)
        else:
            kg = kg + tomo_Aia_perts_dict[ia_label] * ia

        # multiplicative shear bias like on the grid; drawn once per example in main so that the fiducial and the
        # perturbations share the same value (the noise is not m-biased)
        kg = (1.0 + m_bias) * kg

        kg = metacal_mask * kg

        return kg

    def lensing_transform(
        kg,
        ia,
        ia_label,
        m_bias,
        is_true_fiducial=False,
        sn_samples=None,
        np_seed=None,
        # B-mode convergence channel (B-mode Cls study); when kg_b is None the B outputs are returned as None
        kg_b=None,
        ia_b=None,
        sn_b_samples=None,
        # delta-NLA cross-term maps (extended_nla only); ds_b parallels kg_b/ia_b for the B-mode study
        ds=None,
        ds_b=None,
    ):
        assert bool(not is_true_fiducial) != bool(sn_samples is not None)
        keep_b_mode = kg_b is not None
        if keep_b_mode:
            assert (ia_b is not None) and (bool(not is_true_fiducial) != bool(sn_b_samples is not None))

        kg = _lensing_channel(kg, ia, ia_label, m_bias, ds=ds)
        if keep_b_mode:
            # the B-mode convergence goes through the exact same IA combination + m-bias + mask (by linearity)
            kg_b = _lensing_channel(kg_b, ia_b, ia_label, m_bias, ds=ds_b)

        # only smooth the shape noise and return the alms for the fiducial, not the perturbations
        if is_true_fiducial:
            assert sn_samples is not None, "sn has to be provided if is_true_fiducial is True"

            smooth_sn_samples, alm_sn_samples = [], []
            alm_sn_b_samples = [] if keep_b_mode else None
            for i, sn in enumerate(sn_samples):
                sn = metacal_mask * sn
                sn, alm_sn = lensing_smoothing(sn, np_seed + i)

                smooth_sn_samples.append(sn)
                alm_sn_samples.append(alm_sn)

                if keep_b_mode:
                    sn_b = metacal_mask * sn_b_samples[i]
                    _, alm_sn_b = lensing_smoothing(sn_b, np_seed + i)
                    alm_sn_b_samples.append(alm_sn_b)

            sn_samples = np.stack(smooth_sn_samples, axis=0)
            alm_sn_samples = np.stack(alm_sn_samples, axis=0)
            if keep_b_mode:
                alm_sn_b_samples = np.stack(alm_sn_b_samples, axis=0)

            # noiseless
            kg, alm_kg = lensing_smoothing(kg, np_seed)
            alm_kg_b = lensing_smoothing(kg_b, np_seed)[1] if keep_b_mode else None

            return kg, sn_samples, alm_kg, alm_sn_samples, alm_kg_b, alm_sn_b_samples

        else:
            kg, alm_kg = lensing_smoothing(kg, np_seed)
            alm_kg_b = lensing_smoothing(kg_b, np_seed)[1] if keep_b_mode else None

            return kg, alm_kg, alm_kg_b

    return lensing_transform


def _get_clustering_transform(conf, pixel_file):
    n_side = conf["analysis"]["n_side"]
    n_noise_per_signal = conf["analysis"]["fiducial"]["n_noise_per_signal"]
    quadratic_biasing = conf["analysis"]["modelling"]["clustering"]["quadratic_biasing"]

    maglim_mask = files.get_tomo_dv_masks(conf)["maglim"]
    tomo_n_gal_maglim = np.array(conf["survey"]["maglim"]["n_gal"]) * hp.nside2pixarea(n_side, degrees=True)

    # redshift dependence of the bias
    power_law_biasing = conf["analysis"]["modelling"]["clustering"]["power_law_biasing"]
    per_bin_biasing = conf["analysis"]["modelling"]["clustering"]["per_bin_biasing"]
    if power_law_biasing:
        tomo_bg_perts_dict = parameters.get_tomo_amplitude_perturbations_dict("bg", conf)
    elif per_bin_biasing:
        tomo_bg_perts_dict = parameters.get_per_bin_bias_perturbations_dict(conf)
    else:
        raise ValueError("Unsupported configuration of clustering bias")
    if quadratic_biasing:
        tomo_bg2_perts_dict = parameters.get_tomo_amplitude_perturbations_dict("bg2", conf)

    # survey systematics
    if conf["analysis"]["modelling"]["clustering"]["maglim_survey_systematics_map"]:
        tomo_maglim_sys_dv = files.get_clustering_systematics(conf, pixel_type="data_vector")
    else:
        tomo_maglim_sys_dv = None

    def clustering_smoothing(dg, np_seed):
        dg, alm = _data_vector_smoothing(
            dg,
            conf["analysis"]["scale_cuts"]["clustering"]["l_min"],
            conf["analysis"]["scale_cuts"]["clustering"]["l_max"],
            conf["analysis"]["scale_cuts"]["clustering"]["theta_fwhm"],
            np_seed,
            conf,
            pixel_file,
            maglim_mask,
        )

        return dg, alm

    def clustering_counts(dg, bg_tomo, dg2=None, bg2_tomo=None):
        """To focus on the function arguments that are actually varying within clustering_transform"""

        galaxy_counts = clustering.galaxy_density_to_count(
            tomo_n_gal_maglim,
            # linear
            dg,
            bg_tomo,
            # quadratic
            dg2,
            bg2_tomo,
            # rest
            systematics_map=tomo_maglim_sys_dv,
            mask=maglim_mask,
        )

        return galaxy_counts

    def clustering_transform(dg, dg2, bg_label, is_true_fiducial=False, np_seed=None):
        if quadratic_biasing:
            assert dg2 is not None, "dg2 has to be provided if quadratic_biasing is True"

            if bg_label == "fiducial":
                dg = clustering_counts(dg, tomo_bg_perts_dict["fiducial"], dg2, tomo_bg2_perts_dict["fiducial"])
            elif "bg_" in bg_label:
                dg = clustering_counts(dg, tomo_bg_perts_dict[bg_label], dg2, tomo_bg2_perts_dict["fiducial"])
            elif "bg2_" in bg_label:
                dg = clustering_counts(dg, tomo_bg_perts_dict["fiducial"], dg2, tomo_bg2_perts_dict[bg_label])
            else:
                raise ValueError(f"Inconsistent bias label {bg_label}")
        else:
            assert dg2 is None, "dg2 has to be None if quadratic_biasing is False"
            dg = clustering_counts(dg, tomo_bg_perts_dict[bg_label])

        dg = maglim_mask * dg

        # only draw the Poisson noise and return the alms for the fiducial, not the perturbations
        if is_true_fiducial:
            pn_samples = clustering.galaxy_count_to_noise(dg, n_noise_per_signal, np_seed=np_seed)

            smooth_pn_samples, alm_pn_samples = [], []
            for i, pn in enumerate(pn_samples):
                pn = maglim_mask * pn
                pn, alm_pn = clustering_smoothing(pn, np_seed + i)

                smooth_pn_samples.append(pn)
                alm_pn_samples.append(alm_pn)

            pn_samples = np.stack(smooth_pn_samples, axis=0)
            alm_pn_samples = np.stack(alm_pn_samples, axis=0)

            # noiseless
            dg, alm_dg = clustering_smoothing(dg, np_seed)

            return dg, pn_samples, alm_dg, alm_pn_samples

        else:
            dg, alm_dg = clustering_smoothing(dg, np_seed)

            return dg, alm_dg

    return clustering_transform


def _serialize_and_verify(
    n_noise_per_signal,
    # labels
    cosmo_pert_labels,
    ia_pert_labels,
    bg_pert_labels,
    # arrays
    kg_perts,
    ia_perts,
    dg_perts,
    bg_perts,
    sn_samples,
    pn_samples,
    cl_perts,
    cl_ia_perts,
    cl_bg_perts,
    i_signal,
    cl_bmode_perts=None,
    cl_bmode_ia_perts=None,
    cl_bmode_bg_perts=None,
):
    keep_b_mode = cl_bmode_perts is not None

    # serialize
    serialized = tfrecords.parse_forward_fiducial(
        cosmo_pert_labels,
        kg_perts,
        dg_perts,
        # lensing
        ia_pert_labels,
        ia_perts,
        sn_samples,
        # clustering
        bg_pert_labels,
        bg_perts,
        pn_samples,
        # power spectra
        cl_perts,
        cl_ia_perts,
        cl_bg_perts,
        i_signal,
        # B-mode power spectra (None -> not serialized)
        cl_bmode_perts=cl_bmode_perts,
        cl_bmode_ia_perts=cl_bmode_ia_perts,
        cl_bmode_bg_perts=cl_bmode_bg_perts,
    ).SerializeToString()

    # verify
    inv_tfr = tfrecords.parse_inverse_fiducial(
        serialized,
        cosmo_pert_labels + ia_pert_labels + bg_pert_labels,
        range(n_noise_per_signal),
        with_bmode=keep_b_mode,
    )

    # maps
    inv_kg_perts = tf.stack([inv_tfr[f"kg_{pert_label}"] for pert_label in cosmo_pert_labels], axis=0)
    inv_ia_perts = tf.stack([inv_tfr[f"kg_{pert_label}"] for pert_label in ia_pert_labels], axis=0)
    inv_dg_perts = tf.stack([inv_tfr[f"dg_{pert_label}"] for pert_label in cosmo_pert_labels], axis=0)
    inv_bg_perts = tf.stack([inv_tfr[f"dg_{pert_label}"] for pert_label in bg_pert_labels], axis=0)

    assert np.allclose(inv_kg_perts, kg_perts)
    assert np.allclose(inv_ia_perts, ia_perts)
    assert np.allclose(inv_dg_perts, dg_perts)
    assert np.allclose(inv_bg_perts, bg_perts)
    for i_noise in range(n_noise_per_signal):
        assert np.allclose(inv_tfr[f"sn_{i_noise}"], sn_samples[i_noise])
        assert np.allclose(inv_tfr[f"pn_{i_noise}"], pn_samples[i_noise])
    assert np.allclose(inv_tfr["i_signal"], i_signal)

    # power spectra
    inv_cl_perts = tf.stack([inv_tfr[f"cl_{pert_label}"] for pert_label in cosmo_pert_labels], axis=0)
    inv_cl_ia_perts = tf.stack([inv_tfr[f"cl_{pert_label}"] for pert_label in ia_pert_labels], axis=0)
    inv_cl_bg_perts = tf.stack([inv_tfr[f"cl_{pert_label}"] for pert_label in bg_pert_labels], axis=0)

    assert np.allclose(inv_cl_perts, cl_perts)
    assert np.allclose(inv_cl_ia_perts, cl_ia_perts)
    assert np.allclose(inv_cl_bg_perts, cl_bg_perts)

    # B-mode power spectra (parallel field, plain reshape, no cross-bin gather)
    if keep_b_mode:
        inv_cl_bmode_perts = tf.stack([inv_tfr[f"cl_bmode_{pert_label}"] for pert_label in cosmo_pert_labels], axis=0)
        inv_cl_bmode_ia_perts = tf.stack([inv_tfr[f"cl_bmode_{pert_label}"] for pert_label in ia_pert_labels], axis=0)
        inv_cl_bmode_bg_perts = tf.stack([inv_tfr[f"cl_bmode_{pert_label}"] for pert_label in bg_pert_labels], axis=0)
        assert np.allclose(inv_cl_bmode_perts, cl_bmode_perts)
        assert np.allclose(inv_cl_bmode_ia_perts, cl_bmode_ia_perts)
        assert np.allclose(inv_cl_bmode_bg_perts, cl_bmode_bg_perts)

    LOGGER.debug("Decoded the map part of the .tfrecord successfully")

    # legacy power spectra
    inv_cls = tfrecords.parse_inverse_fiducial_cls(serialized)
    assert np.allclose(inv_cls["cls"], cl_perts[0])
    assert np.allclose(inv_cls["i_signal"], i_signal)

    LOGGER.debug("Decoded the cls part of the .tfrecord successfully")

    return serialized


def merge(indices, args):
    args = setup(args)
    conf = files.load_config(args.config)

    # for proper bookkeeping
    n_patches = conf["analysis"]["n_patches"]
    n_perms_per_cosmo = conf["analysis"]["fiducial"]["n_perms_per_cosmo"]
    n_examples = n_patches * n_perms_per_cosmo

    tfr_pattern = filenames.get_filename_tfrecords(
        args.dir_out,
        tag=conf["survey"]["name"] + args.file_suffix,
        index=None,
        simset="fiducial",
        with_bary=conf["analysis"]["modelling"]["baryonified"],
        return_pattern=True,
    )

    keep_b_mode = conf["analysis"]["modelling"]["lensing"].get("b_mode_cls", False)

    cls_dset = tf.data.Dataset.list_files(tfr_pattern)
    cls_dset = cls_dset.interleave(tf.data.TFRecordDataset, cycle_length=16, block_length=1)
    # the default arguments for parse_inverse_fiducial_cls are fine since we're not in graph mode
    cls_dset = cls_dset.map(lambda x: tfrecords.parse_inverse_fiducial_cls(x, with_bmode=keep_b_mode))
    if args.debug:
        cls_dset = cls_dset.take(10)

    cls = []
    cls_bmode = [] if keep_b_mode else None
    i_examples = []
    for example in LOGGER.progressbar(
        cls_dset, total=n_examples, desc="Looping through the .tfrecords", at_level="info"
    ):
        cls.append(example["cls"].numpy())
        if keep_b_mode:
            cls_bmode.append(example["cls_bmode"].numpy())
        i_examples.append(int(example["i_signal"]))

    # noise realizations
    n_noise = example["cls"].numpy().shape[0]
    i_noise = np.arange(n_noise)
    i_noise = np.tile(i_noise, n_examples)

    # concatenate the different simulation runs and noise realizations along the same axis
    # cls.shape[0] = n_examples * n_noise
    cls = np.concatenate(cls, axis=0)
    if keep_b_mode:
        cls_bmode = np.concatenate(cls_bmode, axis=0)

    # i_examples.shape[0] = n_examples
    i_examples = np.array(i_examples)
    i_examples = np.repeat(i_examples, n_noise, axis=0)

    # sort by example index
    i_sort = np.argsort(i_examples)
    cls = cls[i_sort, ...]
    if keep_b_mode:
        cls_bmode = cls_bmode[i_sort, ...]
    i_examples = i_examples[i_sort]
    i_noise = i_noise[i_sort]

    # perform the binning (all examples at the same time)
    binned_cls, bin_edges = power_spectra.bin_according_to_config(cls, conf)

    if keep_b_mode:
        # the B-block has 42 columns, which is not triangular, so bin_according_to_config (with_cross=True) would trip
        # its triangular assert. Bin each column independently on the identical fixed ell grid instead.
        n_bmode_cross = cls_bmode.shape[-1]
        binned_cls_bmode, bin_edges_bmode = power_spectra.smooth_and_bin_cls(
            cls_bmode,
            with_cross=False,
            l_mins_smoothing=[None] * n_bmode_cross,
            l_maxs_smoothing=[None] * n_bmode_cross,
            fixed_binning=True,
            n_bins=conf["analysis"]["power_spectra"]["n_bins"],
            l_min_binning=conf["analysis"]["power_spectra"]["l_min"],
            l_max_binning=conf["analysis"]["power_spectra"]["l_max"],
        )

    # separate folder on the same level as tfrecords
    if args.debug:
        out_dir = os.path.join(args.dir_out, "../../cls/debug")
    else:
        out_dir = os.path.join(args.dir_out, "../../cls")
    os.makedirs(out_dir, exist_ok=True)

    LOGGER.info(f"Saving the results in {out_dir}")
    with h5py.File(os.path.join(out_dir, "fiducial_cls.h5"), "w") as f:
        f.create_dataset("cls/raw", data=cls)
        f.create_dataset("cls/binned", data=binned_cls)
        f.create_dataset("cls/bin_edges", data=bin_edges)
        if keep_b_mode:
            f.create_dataset("cls/bmode_raw", data=cls_bmode)
            f.create_dataset("cls/bmode_binned", data=binned_cls_bmode)
            f.create_dataset("cls/bmode_bin_edges", data=bin_edges_bmode)
        f.create_dataset("i_signal", data=i_examples)
        f.create_dataset("i_noise", data=i_noise)

    LOGGER.info("Done with merging of the fiducial power spectra")
