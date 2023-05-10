# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created December 2022
Author: Arne Thomsen

Convert the .h5 files containing the lensing and clustering maps to scrambled .tfrecord files suitable for training
with the delta loss at the fiducial cosmology and its perturbations

Adapted from https://github.com/tomaszkacprzak/CosmoPointNet/blob/main/CosmoPointNet/apps/run_build_tfrecords.py 
by Tomasz Kacprzak
and 
https://cosmo-gitlab.phys.ethz.ch/jafluri/arne_handover/-/blob/main/map_projection/tfr_generation/fiducial.py
https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/kids1000_analysis/input_pipeline.py
https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/kids1000_analysis/data.py
by Janis Fluri
"""

import numpy as np
import tensorflow as tf
import healpy as hp
import os, argparse, warnings, h5py

from numpy.random import default_rng

from msfm.utils import logger, input_output, cosmogrid, tfrecords, analysis, filenames, parameters, clustering, scales

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


def resources(args):
    return dict(main_memory=8192, main_time=4, main_scratch=0, main_n_cores=1)


def setup(args):
    description = "Make weak lensing and intrinsic alignment datavectors from full sky maps"
    parser = argparse.ArgumentParser(description=description, add_help=True)

    parser.add_argument(
        "-v",
        "--verbosity",
        type=str,
        default="info",
        choices=("critical", "error", "warning", "info", "debug"),
        help="logging level",
    )
    parser.add_argument("--with_bary", action="store_true", help="include baryons")
    parser.add_argument("--n_files", type=int, default=100, help="number of .tfrecord files to produce")
    parser.add_argument(
        "--dir_in",
        type=str,
        default=b"/global/cfs/cdirs/des/cosmogrid/DESY3/fiducial",
        help="input root dir of the simulations",
    )
    parser.add_argument(
        "--dir_out",
        type=str,
        default="/pscratch/sd/a/athomsen/DESY3/fiducial",
        help="output root dir of the .tfrecords",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="configuration yaml file",
    )
    parser.add_argument("--include_maglim_systematics", action="store_true", help="Whether to apply the")
    parser.add_argument("--np_seed", type=int, default=7, help="random seed to shuffle the patches")
    parser.add_argument("--debug", action="store_true", help="activate debug mode")

    args, _ = parser.parse_known_args(args)

    if not os.path.isdir(args.dir_out):
        input_output.robust_makedirs(args.dir_out)

    args.config = os.path.abspath(args.config)

    logger.set_all_loggers_level(args.verbosity)

    return args


def main(indices, args):
    args = setup(args)

    LOGGER.timer.start("main")
    LOGGER.info(f"Got index set of size {len(indices)}")
    LOGGER.info(f"Running on {len(os.sched_getaffinity(0))} cores")

    conf = analysis.load_config(args.config)

    # setup up directories
    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    meta_info_file = os.path.join(repo_dir, conf["files"]["meta_info"])
    cosmo_params_info = cosmogrid.get_cosmo_params_info(meta_info_file, "fiducial")

    # constants
    n_patches = conf["analysis"]["n_patches"]
    n_perms_per_cosmo = conf["analysis"]["fiducial"]["n_perms_per_cosmo"]
    n_examples_per_cosmo = n_patches * n_perms_per_cosmo
    data_vec_pix, _, _, _ = analysis.load_pixel_file()

    # intrinsic alignment
    tomo_Aia_perts_dict = parameters.get_tomo_amplitude_perturbations_dict("Aia", conf)

    # linear galaxy bias
    tomo_n_gal_maglim = tf.constant(conf["survey"]["maglim"]["n_gal"]) * hp.nside2pixarea(
        conf["analysis"]["n_side"], degrees=True
    )
    tomo_bg_perts_dict = parameters.get_tomo_amplitude_perturbations_dict("bg", conf)

    def transform_clustering(dg, label, is_fiducial=False, noise_fac=None):
        galaxy_counts = clustering.galaxy_density_to_number(
            dg,
            tomo_n_gal_maglim,
            tomo_bg_perts_dict[label],
            conf=conf,
            include_systematics=args.include_maglim_systematics,
        )

        # ic("after galaxy_density_to_number", galaxy_counts.dtype)

        # only draw the noise for the fiducial, not the perturbations
        if is_fiducial:
            galaxy_counts, noise_fac = clustering.galaxy_number_add_noise(
                galaxy_counts, noise_fac=None, return_noise_fac=True
            )
        else:
            galaxy_counts = clustering.galaxy_number_add_noise(
                galaxy_counts, noise_fac=noise_fac, return_noise_fac=False
            )

        # ic("after galaxy_number_add_noise", galaxy_counts.dtype)

        galaxy_counts = scales.data_vector_to_smoothed_data_vector(
            galaxy_counts,
            l_min=conf["analysis"]["scale_cuts"]["clustering"]["l_min"],
            l_max=conf["analysis"]["scale_cuts"]["clustering"]["l_max"],
            n_side=conf["analysis"]["n_side"],
            data_vec_pix=data_vec_pix,
        )

        # ic("after data_vector_to_smoothed_data_vector", galaxy_counts.dtype)

        if is_fiducial:
            # shape (n_pix, n_z_maglim)
            return galaxy_counts, noise_fac
        else:
            return galaxy_counts

    # tomo_l_min_maglim = conf["analysis"]["scale_cuts"]["maglim"]["l_min"]
    # tomo_l_max_maglim = conf["analysis"]["scale_cuts"]["maglim"]["l_max"]

    # set up the paths
    cosmo_dirs = [cosmo_dir.decode("utf-8") for cosmo_dir in cosmo_params_info["path_par"]]

    # remove baryon perturbations for the fiducial set
    if not args.with_bary:
        cosmo_dirs = [cosmo_dir for cosmo_dir in cosmo_dirs if not "bary" in cosmo_dir]

    cosmo_dirs_in = [os.path.join(args.dir_in, cosmo_dir) for cosmo_dir in cosmo_dirs]
    LOGGER.debug(cosmo_dirs_in)

    n_cosmos = len(cosmo_dirs_in)
    LOGGER.info(f"Got simulation set fiducial of size {n_cosmos} with base path {args.dir_in}")

    if n_examples_per_cosmo % args.n_files == 0:
        n_examples_per_file = n_examples_per_cosmo // args.n_files
    else:
        raise ValueError(
            f"The total number of examples per cosmology {n_examples_per_cosmo}"
            f" has to be divisible by the number of files {args.n_files}"
        )
    LOGGER.info(f"n_examples_per_file = {n_examples_per_file}")

    if args.debug:
        i_examples = np.arange(6)
        LOGGER.debug(f"Debug mode, using ordered indices {i_examples}")
    else:
        # shuffle the indices
        rng = default_rng(seed=args.np_seed)
        i_examples = rng.permutation(n_examples_per_cosmo)

    # CosmoGrid perturbations in cosmological parameters
    pert_labels = [label.split("cosmo_")[1].replace("/", "") for label in cosmo_dirs_in]
    LOGGER.info(f"There's {len(pert_labels)} cosmological labels = {pert_labels}")

    # separate label lists for astrophysics perturbations
    ia_pert_labels = parameters.get_fiducial_perturbation_labels(conf["analysis"]["params"]["ia"])[1:]
    LOGGER.info(f"There's {len(ia_pert_labels)} intrinsic alignment labels = {ia_pert_labels}")

    bg_pert_labels = parameters.get_fiducial_perturbation_labels(conf["analysis"]["params"]["bg"])[1:]
    LOGGER.info(f"There's {len(bg_pert_labels)} galaxy clustering labels = {bg_pert_labels}")

    # index corresponds to a .tfrecord file ###########################################################################
    for index in indices:
        LOGGER.timer.start("index")

        tfr_file = filenames.get_filename_tfrecords(
            args.dir_out, tag=conf["survey"]["name"], index=index, simset="fiducial"
        )
        LOGGER.info(f"Index {index} is writing to {tfr_file}")

        js = index * n_examples_per_file
        je = (index + 1) * n_examples_per_file

        n_done = 0
        with tf.io.TFRecordWriter(tfr_file) as file_writer:
            for j in LOGGER.progressbar(range(js, je), at_level="info", desc="Storing DES examples\n", total=je - js):

                if args.debug:
                    if n_done > 5:
                        LOGGER.warning("Debug mode, aborting after 5 subindices")
                        break

                # get the indices to the example
                i_example = i_examples[j]
                LOGGER.info(f"j = {j} in range({js},{je}): i_example = {i_example}")

                # loop over the perturbations in the right order
                kg_perts = []
                dg_perts = []
                ia_perts = []
                bg_perts = []
                # (2 * n_cosmos + 1,) iterations
                for cosmo_dir_in in cosmo_dirs_in:
                    file_cosmo = filenames.get_filename_data_vectors(cosmo_dir_in, with_bary=args.with_bary)
                    LOGGER.debug(f"Taking inputs from {cosmo_dir_in}")

                    (kg, ia, dg) = load_example(file_cosmo, i_example, ["kg", "ia", "dg"])

                    # add the fiducial intrinsic alignment
                    kg_perts.append(kg + tomo_Aia_perts_dict["fiducial"] * ia)

                    # # add the fiducial galaxy biasing, only draw the poisson noise here
                    # dg, noise_fac = transform_clustering(dg, "fiducial", is_fiducial=True)
                    # dg_perts.append(dg)

                    # dg = clustering.galaxy_density_to_number(
                    #     dg,
                    #     tomo_n_gal_maglim,
                    #     tomo_bg_perts_dict["fiducial"],
                    #     include_systematics=args.include_maglim_systematics,
                    #     conf=conf,
                    # )

                    # # smoothing, different scale per redshift bin
                    # for i_tomo, (l_min, l_max) in enumerate(zip(tomo_l_min_maglim, tomo_l_max_maglim)):
                    #     dg[..., i_tomo] = scales.map_to_smoothed_map(dg[..., i_tomo], l_min, l_max)

                    # dg_perts.apppend(dg)

                    # TODO galaxy bias and Poisson noise is only applied later
                    # dg_perts.append(dg)

                    # add the perturbations to the two intrinsic alignment parameters
                    if "cosmo_fiducial" in cosmo_dir_in:
                        # load the shape noise realization
                        (sn_realz,) = load_example(file_cosmo, i_example, ["sn"])

                        # draw the poisson noise realization
                        dg, poisson_noise_fac = transform_clustering(dg, "fiducial", is_fiducial=True)

                        # Aia perturbations
                        ia_perts.append(kg + tomo_Aia_perts_dict["delta_Aia_m"] * ia)
                        ia_perts.append(kg + tomo_Aia_perts_dict["delta_Aia_p"] * ia)

                        # n_Aia perturbations
                        ia_perts.append(kg + tomo_Aia_perts_dict["delta_n_Aia_m"] * ia)
                        ia_perts.append(kg + tomo_Aia_perts_dict["delta_n_Aia_p"] * ia)

                        # bg perturbations
                        bg_perts.append(transform_clustering(dg, "delta_bg_m", noise_fac=poisson_noise_fac))
                        bg_perts.append(transform_clustering(dg, "delta_bg_p", noise_fac=poisson_noise_fac))

                        # n_bg perturbations
                        bg_perts.append(transform_clustering(dg, "delta_n_bg_m", noise_fac=poisson_noise_fac))
                        bg_perts.append(transform_clustering(dg, "delta_n_bg_p", noise_fac=poisson_noise_fac))

                    else:
                        dg = transform_clustering(dg, "fiducial", noise_fac=poisson_noise_fac)

                    # here for consistent poisson noise
                    dg_perts.append(dg)

                    # ic(dg.dtype)
                    # ic(kg.dtype)

                # serialize the lists of tensors of shape (n_pix, n_z_bins)
                serialized = tfrecords.parse_forward_fiducial(
                    pert_labels,
                    kg_perts,
                    dg_perts,
                    # lensing
                    ia_pert_labels,
                    ia_perts,
                    sn_realz,
                    # clustering
                    bg_pert_labels,
                    bg_perts,
                    i_example,
                ).SerializeToString()

                # check correctness
                i_noise = 0
                inv_data_vectors, inv_index = tfrecords.parse_inverse_fiducial(
                    serialized, pert_labels + ia_pert_labels + bg_pert_labels, i_noise=i_noise
                )
                inv_kg_perts = tf.stack([inv_data_vectors[f"kg_{pert_label}"] for pert_label in pert_labels], axis=0)
                inv_ia_perts = tf.stack(
                    [inv_data_vectors[f"kg_{pert_label}"] for pert_label in ia_pert_labels], axis=0
                )
                inv_dg_perts = tf.stack([inv_data_vectors[f"dg_{pert_label}"] for pert_label in pert_labels], axis=0)
                inv_bg_perts = tf.stack(
                    [inv_data_vectors[f"dg_{pert_label}"] for pert_label in bg_pert_labels], axis=0
                )
                inv_sn = inv_data_vectors["sn"]

                assert np.allclose(inv_kg_perts, kg_perts)
                assert np.allclose(inv_ia_perts, ia_perts)
                assert np.allclose(inv_dg_perts, dg_perts)
                assert np.allclose(inv_bg_perts, bg_perts)
                assert np.allclose(inv_sn, sn_realz[i_noise])
                assert np.allclose(inv_index[0], i_example)

                LOGGER.debug("decoded successfully")

                file_writer.write(serialized)

                n_done += 1

        LOGGER.info(f"Done with index {index} after {LOGGER.timer.elapsed('index')}")
        yield index


def load_example(filename, i_example, map_labels):
    with h5py.File(filename, "r") as f:
        maps = []
        for map_label in map_labels:
            # shape (n_examples_per_cosmo, n_pix, n_z_bins) before the indexing
            maps.append(f[map_label][i_example, ...])

    return maps
