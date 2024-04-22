# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created March 2024
Author: Arne Thomsen

Transform the full sky weak lensing signal and intrinsic alignment maps into multiple survey footprint cut-outs and 
store them in .tfrecord files. The parallelization is done over the .tfrecord files, every jobarray element corresponds
to one.

For the grid, the main loop runs over the cosmologies.

Meant for 
 - Euler (CPU nodes, local scratch) 
 - esub jobarrays
 - Read the CosmoGrid directly from the SAN
 - CosmoGridV1.1
"""

import numpy as np
import tensorflow as tf
import os, argparse, warnings, time, yaml, h5py, copy_guardian

from sobol_seq import i4_sobol

from msfm.utils import (
    logger,
    imports,
    filenames,
    input_output,
    files,
    lensing,
    clustering,
    cosmogrid,
    preprocessing,
    tfrecords,
    power_spectra,
    scales,
    redshift,
    parameters,
)

hp = imports.import_healpy(parallel=True)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


def resources(args):
    args = setup(args)

    # TODO, check if this runs at scale
    resources = {"main_memory": 4096, "main_time": 4, "main_n_cores": 4, "merge_memory": 4096, "merge_n_cores": 16}

    if args.from_san:
        # in MB. One projected_probes_maps_v11dmb.h5 should be around 1 GB
        resources["main_scratch"] = 4096
    else:
        resources["main_scratch"] = 0

    return resources


def setup(args):
    description = "Preprocess the CosmoGrid projections into forward-modeled survey footprints in .tfrecord files"
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
    if not args.to_san:
        with open(os.path.join(args.dir_out, "config.yaml"), "w") as f:
            yaml.dump(conf, f)

    # directories
    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    meta_info_file = os.path.join(repo_dir, conf["files"]["meta_info"])

    cosmo_params_info = cosmogrid.get_cosmo_params_info(meta_info_file, "grid")
    cosmo_dirs = [cosmo_dir.decode("utf-8") for cosmo_dir in cosmo_params_info["path_par"]]
    cosmo_dirs_in = [os.path.join(args.dir_in, "grid", cosmo_dir) for cosmo_dir in cosmo_dirs]

    # CosmoGrid
    n_patches = conf["analysis"]["n_patches"]
    n_cosmos = conf["analysis"]["grid"]["n_cosmos"]
    n_perms_per_cosmo = conf["analysis"]["grid"]["n_perms_per_cosmo"]
    n_noise_per_example = conf["analysis"]["grid"]["n_noise_per_example"]
    n_examples_per_cosmo = n_patches * n_perms_per_cosmo * n_noise_per_example
    LOGGER.info(
        f"For every cosmology, theres {n_examples_per_cosmo} examples: "
        f"{n_patches} patches times {n_perms_per_cosmo} permutations times {n_noise_per_example} noise realizations"
    )

    # .tfrecords
    if n_cosmos % args.n_files == 0:
        n_cosmos_per_file = n_cosmos // args.n_files
        n_examples_per_file = n_examples_per_cosmo * n_cosmos_per_file
        LOGGER.info(f"The number of files implies {n_cosmos_per_file} cosmological parameters per .tfrecord file")
    else:
        raise ValueError(
            f"The total number of cosmologies {n_cosmos} has to be evenly divisible by the number of files {args.n_files}"
        )
    LOGGER.info(
        f"In total, there are n_examples_per_cosmo * n_cosmos_per_file = {n_examples_per_cosmo} * {n_cosmos_per_file}"
        f" = {n_examples_per_file} examples per file"
    )

    # analysis files
    pixel_file = files.load_pixel_file(conf)
    noise_file = files.load_noise_file(conf)

    # transforms
    lensing_transform = _get_lensing_transform(conf, pixel_file)
    clustering_transform = _get_clustering_transform(conf, pixel_file)

    # index corresponds to a .tfrecord file ###########################################################################
    for index in indices:
        LOGGER.timer.start("index")

        if args.to_san:
            LOGGER.info("Writing the .tfrecord to local scratch to be later copied to the SAN")
            san_dir_out = args.dir_out
            args.dir_out = os.environ["TMPDIR"]

        tfr_file = filenames.get_filename_tfrecords(
            args.dir_out,
            tag=conf["survey"]["name"] + args.file_suffix,
            index=index,
            simset="grid",
            with_bary=conf["analysis"]["modelling"]["baryonified"],
        )
        LOGGER.info(f"Index {index} is writing to {tfr_file}")

        # index for the cosmological parameters
        i_cosmo_start = index * n_cosmos_per_file
        i_cosmo_end = (index + 1) * n_cosmos_per_file
        LOGGER.info(f"And includes {cosmo_dirs[i_cosmo_start : i_cosmo_end]}")

        with tf.io.TFRecordWriter(tfr_file) as file_writer:
            # loop over the cosmological parameters
            for i_cosmo, cosmo_dir_in in LOGGER.progressbar(
                zip(range(i_cosmo_start, i_cosmo_end), cosmo_dirs_in[i_cosmo_start:i_cosmo_end]),
                at_level="debug",
                desc="Looping through cosmologies\n",
                total=i_cosmo_end - i_cosmo_start,
            ):
                LOGGER.debug(f"Taking inputs from {cosmo_dir_in}")

                # cut out the survey footprints, generate the shape noise, perform mode removal, ...
                data_vec_container = preprocessing.preprocess_grid_permutations(
                    args, conf, cosmo_dir_in, pixel_file, noise_file
                )

                # (n_examples_per_cosmo, n_pix, n_z_bins)
                kg_examples = data_vec_container["kg"]
                ia_examples = data_vec_container["ia"]
                dg_examples = data_vec_container["dg"]
                if conf["analysis"]["modelling"]["quadratic_biasing"]:
                    dg2_examples = data_vec_container["dg2"]
                else:
                    dg2_examples = [None] * n_examples_per_cosmo
                # (n_examples_per_cosmo, n_noise_per_examplen_pix, n_z_bins)
                sn_examples = data_vec_container["sn"]

                i_sobol, cosmo, Aia, n_Aia, bg, n_bg, bg2, n_bg2 = _extend_sobol_squence(
                    conf, cosmo_params_info, i_cosmo
                )

                # redshift evolution, only calculate the integrals once here
                current_lensing_transform = lambda kg, ia, sn_samples, np_seed: lensing_transform(
                    kg, ia, sn_samples, Aia, n_Aia, np_seed
                )
                current_clustering_transform = lambda dg, dg2, np_seed: clustering_transform(
                    dg, bg, n_bg, dg2, bg2, n_bg2, np_seed
                )

                # loop over the n_examples_per_cosmo
                for i_example, (kg, ia, sn_samples, dg, dg2) in LOGGER.progressbar(
                    enumerate(zip(kg_examples, ia_examples, sn_examples, dg_examples, dg2_examples)),
                    at_level="info",
                    desc="Looping through the per cosmology examples",
                    total=n_examples_per_cosmo // n_noise_per_example,
                ):
                    if args.debug and i_example > n_patches:
                        LOGGER.warning(f"Debug mode, only processing the first {n_patches} examples")
                        break

                    # maps
                    kg, sn_samples, alm_kg, alm_sn_samples = current_lensing_transform(
                        kg, ia, sn_samples, np_seed=i_sobol + i_example
                    )
                    dg, pn_samples, alm_dg, alm_pn_samples = current_clustering_transform(
                        dg, dg2, np_seed=i_sobol + i_example
                    )

                    # power spectra
                    cls = power_spectra.run_tfrecords_alm_to_cl(alm_kg, alm_sn_samples, alm_dg, alm_pn_samples)

                    serialized = tfrecords.parse_forward_grid(
                        kg, sn_samples, dg, pn_samples, cls, cosmo, i_sobol, i_example
                    ).SerializeToString()

                    _verify_tfrecord(
                        serialized, n_noise_per_example, kg, sn_samples, dg, pn_samples, cosmo, i_sobol, i_example, cls
                    )

                    file_writer.write(serialized)

        if args.to_san:
            _rsync_tfrecord_to_san(conf, tfr_file, san_dir_out)

        LOGGER.info(f"Done with index {index} after {LOGGER.timer.elapsed('index')}")
        yield index


def _data_vector_smoothing(dv, l_min, theta_fwhm, np_seed, conf, pixel_file):
    # Gaussian Random Field
    if conf["analysis"]["modelling"]["degrade_to_grf"]:
        dv, alm = scales.data_vector_to_grf_data_vector(
            np_seed,
            dv,
            data_vec_pix=pixel_file[0],
            n_side=conf["analysis"]["n_side"],
            l_min=l_min,
            theta_fwhm=theta_fwhm,
            arcmin=True,
        )
    # standard smoothing with a Gaussian kernel
    else:
        dv, alm = scales.data_vector_to_smoothed_data_vector(
            dv,
            data_vec_pix=pixel_file[0],
            n_side=conf["analysis"]["n_side"],
            l_min=l_min,
            theta_fwhm=theta_fwhm,
            arcmin=True,
        )

    return dv, alm


def _get_lensing_transform(conf, pixel_file):
    z0 = conf["analysis"]["modelling"]["z0"]
    tomo_z_metacal, tomo_nz_metacal = files.load_redshift_distributions("metacal", conf)
    m_bias_dist = lensing.get_m_bias_distribution(conf)
    metacal_mask = files.get_tomo_dv_masks(conf)["metacal"]

    def lensing_smoothing(kg, np_seed):
        kg, alm = _data_vector_smoothing(
            kg,
            conf["analysis"]["scale_cuts"]["lensing"]["l_min"],
            conf["analysis"]["scale_cuts"]["lensing"]["theta_fwhm"],
            np_seed,
            conf,
            pixel_file,
        )

        return kg, alm

    def lensing_transform(kg, ia, sn_samples, Aia, n_Aia, np_seed=None):
        # intrinsic alignment
        tomo_Aia = redshift.get_tomo_amplitudes(Aia, n_Aia, tomo_z_metacal, tomo_nz_metacal, z0)
        LOGGER.debug(f"Per z bin Aia = {tomo_Aia}")

        kg = kg + tomo_Aia * ia

        # fixing this in the .tfrecords simplifies reproducibility
        m_bias = m_bias_dist.sample()
        kg *= 1.0 + m_bias

        kg *= metacal_mask
        kg, alm_kg = lensing_smoothing(kg, np_seed)

        smooth_sn_samples, alm_sn_samples = [], []
        for shape_noise in sn_samples:
            shape_noise *= metacal_mask

            smooth_sn, alm_sn = lensing_smoothing(shape_noise, np_seed)

            smooth_sn_samples.append(smooth_sn)
            alm_sn_samples.append(alm_sn)

        sn_samples = np.stack(smooth_sn_samples, axis=0)
        alm_sn_samples = np.stack(alm_sn_samples, axis=0)

        return kg, sn_samples, alm_kg, alm_sn_samples

    return lensing_transform


def _get_clustering_transform(conf, pixel_file):
    n_side = conf["analysis"]["n_side"]
    n_noise_per_example = conf["analysis"]["grid"]["n_noise_per_example"]
    z0 = conf["analysis"]["modelling"]["z0"]

    # modeling
    quadratic_biasing = conf["analysis"]["modelling"]["quadratic_biasing"]
    stochasticity = conf["analysis"]["modelling"]["galaxy_stochasticity"]

    maglim_mask = files.get_tomo_dv_masks(conf)["maglim"]
    tomo_z_maglim, tomo_nz_maglim = files.load_redshift_distributions("maglim", conf)
    tomo_n_gal_maglim = np.array(conf["survey"]["maglim"]["n_gal"]) * hp.nside2pixarea(n_side, degrees=True)

    # survey systematics
    if conf["analysis"]["modelling"]["maglim_survey_systematics_map"]:
        tomo_maglim_sys_dv = files.get_clustering_systematics(conf, pixel_type="data_vector")
    else:
        tomo_maglim_sys_dv = None

    def clustering_smoothing(dg, np_seed):
        dg, alm = _data_vector_smoothing(
            dg,
            conf["analysis"]["scale_cuts"]["clustering"]["l_min"],
            conf["analysis"]["scale_cuts"]["clustering"]["theta_fwhm"],
            np_seed,
            conf,
            pixel_file,
        )

        return dg, alm

    def clustering_transform(
        # linear
        dg,
        bg,
        n_bg,
        # quadratic
        dg2=None,
        bg2=None,
        n_bg2=None,
        # noise
        np_seed=None,
    ):
        assert (not quadratic_biasing and (bg2 is None) and (n_bg2 is None)) or (
            quadratic_biasing and (bg2 is not None) and (n_bg2 is not None)
        ), f"The galaxy biasing setup must be consistent"

        # the linear galaxy bias is needed in both cases
        tomo_bg = redshift.get_tomo_amplitudes(bg, n_bg, tomo_z_maglim, tomo_nz_maglim, z0)
        LOGGER.debug(f"Per z bin linear bg = {tomo_bg}")

        if quadratic_biasing:
            tomo_bg2 = redshift.get_tomo_amplitudes(bg2, n_bg2, tomo_z_maglim, tomo_nz_maglim, z0)
            LOGGER.debug(f"Per z bin quadratic bg2 = {tomo_bg2}")

            dg = clustering.galaxy_density_to_count(
                tomo_n_gal_maglim,
                # linear
                dg,
                tomo_bg,
                # quadratic
                dg2,
                tomo_bg2,
                # misc
                conf=conf,
                stochasticity=stochasticity,
                data_vec_pix=pixel_file[0],
                systematics_map=tomo_maglim_sys_dv,
                mask=maglim_mask,
                np_seed=np_seed + 1,
            )
        else:
            dg = clustering.galaxy_density_to_count(
                tomo_n_gal_maglim,
                # linear
                dg,
                tomo_bg,
                # misc
                conf=conf,
                stochasticity=stochasticity,
                data_vec_pix=pixel_file[0],
                systematics_map=tomo_maglim_sys_dv,
                mask=maglim_mask,
                np_seed=np_seed + 1,
            )

        # draw noise, mask, smooth
        pn_samples = clustering.galaxy_count_to_noise(dg, n_noise_per_example, np_seed=np_seed + 2)

        smooth_pn_samples, alm_pn_samples = [], []
        for pn in pn_samples:
            pn *= maglim_mask

            smooth_pn, alm_smooth_pn = clustering_smoothing(pn, np_seed)

            smooth_pn_samples.append(smooth_pn)
            alm_pn_samples.append(alm_smooth_pn)

        pn_samples = np.stack(smooth_pn_samples, axis=0)
        alm_pn_samples = np.stack(alm_pn_samples, axis=0)

        # noiseless
        dg, alm_dg = clustering_smoothing(dg, np_seed)

        # shapes (n_pix, n_z_maglim), (n_noise_per_example, n_pix, n_z_maglim)
        # (n_noise_per_example, )
        return dg, pn_samples, alm_dg, alm_pn_samples

    return clustering_transform


def _extend_sobol_squence(conf, cosmo_params_info, i_cosmo):
    with_bary = conf["analysis"]["modelling"]["baryonified"]
    quadratic_biasing = conf["analysis"]["modelling"]["quadratic_biasing"]

    all_params = parameters.get_parameters(conf=conf)

    cosmogrid_params = conf["analysis"]["params"]["cosmo"].copy()
    if with_bary:
        cosmogrid_params += conf["analysis"]["params"]["bary"]

    cosmo = [cosmo_params_info[cosmo_param][i_cosmo] for cosmo_param in cosmogrid_params]
    cosmo = np.array(cosmo, dtype=np.float32)

    sobol_priors = parameters.get_prior_intervals(all_params, conf=conf)
    # extend the Sobol sequence by astrophysical parameters
    i_sobol = cosmo_params_info["sobol_index"][i_cosmo]
    sobol_point, _ = i4_sobol(sobol_priors.shape[0], i_sobol)
    sobol_params = sobol_point * np.squeeze(np.diff(sobol_priors)) + sobol_priors[:, 0]
    sobol_params = sobol_params.astype(np.float32)

    # add these to the label, the parameters are ordered as in sobol_priors
    Aia = sobol_params[6 + 2 * with_bary]
    n_Aia = sobol_params[7 + 2 * with_bary]
    bg = sobol_params[8 + 2 * with_bary]
    n_bg = sobol_params[9 + 2 * with_bary]
    cosmo = np.concatenate((cosmo, np.array([Aia, n_Aia, bg, n_bg])))
    if quadratic_biasing:
        bg2 = sobol_params[10 + 2 * with_bary]
        n_bg2 = sobol_params[11 + 2 * with_bary]
        cosmo = np.concatenate((cosmo, np.array([bg2, n_bg2])))
    else:
        bg2 = None
        n_bg2 = None

    # verify that the Sobol sequences (stored and newly generated) are identical for the cosmo params
    assert np.allclose(sobol_params[0], cosmo[0], rtol=1e-3, atol=1e-5)  # Om
    assert np.allclose(sobol_params[1], cosmo[1], rtol=1e-3, atol=1e-5)  # s8
    assert np.allclose(sobol_params[2], cosmo[2], rtol=1e-3, atol=1e-3)  # Ob
    assert np.allclose(sobol_params[3], cosmo[3], rtol=1e-3, atol=1e-5)  # H0
    assert np.allclose(sobol_params[4], cosmo[4], rtol=1e-3, atol=1e-5)  # ns
    assert np.allclose(sobol_params[5], cosmo[5], rtol=1e-3, atol=1e-5)  # w0
    if with_bary:
        assert np.allclose(sobol_params[6], np.log10(cosmo[6]), rtol=1e-3, atol=1e-5)  # bary_Mc
        assert np.allclose(sobol_params[7], cosmo[7], rtol=1e-3, atol=1e-5)  # bary_nu
    LOGGER.debug("The parameters derived from the sobol sequence are identical to the stored ones")

    return i_sobol, cosmo, Aia, n_Aia, bg, n_bg, bg2, n_bg2


def _verify_tfrecord(serialized, n_noise_per_example, kg, sn_samples, dg, pn_samples, cosmo, i_sobol, i_example, cls):
    inv_maps = tfrecords.parse_inverse_grid(serialized, range(n_noise_per_example))

    for i_noise in range(n_noise_per_example):
        assert np.allclose(inv_maps[f"kg_{i_noise}"], kg + sn_samples[i_noise])
        assert np.allclose(inv_maps[f"dg_{i_noise}"], dg + pn_samples[i_noise])
    assert np.allclose(inv_maps["cosmo"], cosmo)
    assert np.allclose(inv_maps["i_sobol"], i_sobol)
    assert np.allclose(inv_maps["i_example"], i_example)
    LOGGER.debug("Decoded the map part of the .tfrecord successfully")

    inv_cls = tfrecords.parse_inverse_grid_cls(serialized)

    assert np.allclose(inv_cls["cls"], cls)
    assert np.allclose(inv_cls["cosmo"], cosmo)
    assert np.allclose(inv_cls["i_sobol"], i_sobol)
    assert np.allclose(inv_cls["i_example"], i_example)
    LOGGER.debug("Decoded the cls part of the .tfrecord successfully")


def _rsync_tfrecord_to_san(conf, tfr_file, san_dir_out):
    # like in run_fiducial_preprocessing.py
    LOGGER.info("Copying the .tfrecord to the SAN")
    LOGGER.timer.start("copy_to_san")

    san_conf = conf["dirs"]["connections"]["san"]
    with copy_guardian.BoundedSemaphore(san_conf["max_connections"], timeout=san_conf["timeout"]):
        connection = copy_guardian.Connection(
            host=san_conf["host"],
            user=san_conf["user"],
            private_key=san_conf["private_key"],
            port=san_conf["port"],
        )
        connection.rsync_to(tfr_file, os.path.join(san_dir_out, os.path.basename(tfr_file)))

    LOGGER.info(f"Done copying {tfr_file} to the SAN after {LOGGER.timer.elapsed('copy_to_san')}")


def merge(indices, args):
    args = setup(args)
    conf = files.load_config(args.config)

    n_cosmos = conf["analysis"]["grid"]["n_cosmos"]
    n_patches = conf["analysis"]["n_patches"]
    n_perms_per_cosmo = conf["analysis"]["grid"]["n_perms_per_cosmo"]
    n_noise_per_example = conf["analysis"]["grid"]["n_noise_per_example"]
    n_signal_per_cosmo = n_patches * n_perms_per_cosmo

    tfr_pattern = filenames.get_filename_tfrecords(
        args.dir_out, tag=conf["survey"]["name"] + args.file_suffix, index=None, simset="grid", return_pattern=True
    )

    cls_dset = tf.data.Dataset.list_files(tfr_pattern)
    # flat_map to not mix cosmologies
    cls_dset = cls_dset.flat_map(tf.data.TFRecordDataset)
    # the default arguments for parse_inverse_fiducial_cls are fine since we're not in graph mode
    cls_dset = cls_dset.map(tfrecords.parse_inverse_grid_cls, num_parallel_calls=tf.data.AUTOTUNE)
    # every batch is a single cosmology
    cls_dset = cls_dset.batch(n_signal_per_cosmo)

    cls = []
    binned_cls = []
    bin_edges = []
    cosmos = []
    i_sobols = []
    i_examples = []
    i_noises = []
    for example in LOGGER.progressbar(
        cls_dset, total=n_cosmos, desc="Looping through the different cosmologies in the .tfrecords", at_level="info"
    ):
        cl = example["cls"].numpy()
        cosmo = example["cosmo"].numpy()
        i_sobol = example["i_sobol"].numpy()
        i_example = example["i_example"].numpy()

        # concatenate the noise realizations along the same axis as the examples
        cl = np.concatenate([cl[:, i, ...] for i in range(cl.shape[1])], axis=0)

        # perform the binning (all examples of a single cosmology at once)
        binned_cl, bin_edge = power_spectra.bin_cls(
            cl,
            l_mins=conf["analysis"]["scale_cuts"]["lensing"]["l_min"]
            + conf["analysis"]["scale_cuts"]["clustering"]["l_min"],
            l_maxs=conf["analysis"]["scale_cuts"]["lensing"]["l_max"]
            + conf["analysis"]["scale_cuts"]["clustering"]["l_max"],
            n_bins=conf["analysis"]["power_spectra"]["n_bins"],
            with_cross=True,
        )

        # tiling has the same form as the above concatenation
        cosmo = np.tile(cosmo, (n_noise_per_example, 1))
        i_sobol = np.tile(i_sobol, n_noise_per_example)
        i_example = np.tile(i_example, n_noise_per_example)

        # noise is treated separately because it's along a separate dimension in the .tfrecords. This here is preserves
        # the order imposed above in power_spectrum = ...
        i_noise = np.arange(n_noise_per_example)
        i_noise = np.repeat(i_noise, n_signal_per_cosmo)

        cls.append(cl)
        binned_cls.append(binned_cl)
        bin_edges.append(bin_edge)
        cosmos.append(cosmo)
        i_sobols.append(i_sobol)
        i_examples.append(i_example)
        i_noises.append(i_noise)

    # results
    cls = np.stack(cls, axis=0)
    binned_cls = np.stack(binned_cls, axis=0)
    bin_edges = np.stack(bin_edges, axis=0)
    cosmos = np.stack(cosmos, axis=0)
    i_sobols = np.array(i_sobols)
    i_examples = np.array(i_examples)
    i_noises = np.array(i_noises)

    # separate folder on the same level as tfrecords
    if args.debug:
        out_dir = args.dir_out
    else:
        out_dir = os.path.join(args.dir_out, "../../cls")
    os.makedirs(out_dir, exist_ok=True)

    LOGGER.info(f"Saving the results in {out_dir}")
    with h5py.File(os.path.join(out_dir, "grid_cls.h5"), "w") as f:
        f.create_dataset("cls/raw", data=cls)
        f.create_dataset("cls/binned", data=binned_cls)
        f.create_dataset("cls/bin_edges", data=bin_edges)
        f.create_dataset("cosmo", data=cosmos)
        f.create_dataset("i_sobol", data=i_sobols)
        f.create_dataset("i_example", data=i_examples)
        f.create_dataset("i_noise", data=i_noises)

    LOGGER.info(f"Done with merging of the grid power spectra")
