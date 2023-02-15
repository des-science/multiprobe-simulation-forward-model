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
import os, argparse, warnings, h5py

from numpy.random import default_rng
from icecream import ic

from msfm.utils import logger, input_output, cosmogrid, tfrecords
from msfm.utils.filenames import *

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
        "--repo_dir",
        type=str,
        default="/global/homes/a/athomsen/multiprobe-simulation-forward-model",
        help="root dir of the msfm repo to convert relative paths to absolute ones",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="configuration yaml file",
    )
    parser.add_argument("--np_seed", type=int, default=7, help="random seed to shuffle the patches")
    parser.add_argument("--debug", action="store_true", help="activate debug mode")

    args, _ = parser.parse_known_args(args)

    if not os.path.isdir(args.dir_out):
        input_output.robust_makedirs(args.dir_out)

    args.repo_dir = os.path.abspath(args.repo_dir)

    logger.set_all_loggers_level(args.verbosity)

    return args


def main(indices, args):
    args = setup(args)

    LOGGER.timer.start("main")
    LOGGER.info(f"Got index set of size {len(indices)}")
    LOGGER.info(f"Running on {len(os.sched_getaffinity(0))} cores")

    conf_file = os.path.join(args.repo_dir, args.config)
    conf = input_output.read_yaml(conf_file)
    LOGGER.info(f"Loaded configuration file")

    meta_info_file = os.path.join(args.repo_dir, conf["files"]["meta_info"])
    cosmo_params_info = cosmogrid.get_cosmo_params_info(meta_info_file, "fiducial")

    # constants
    n_patches = conf["analysis"]["n_patches"]
    n_perms_per_cosmo = conf["analysis"]["fiducial"]["n_perms_per_cosmo"]
    n_examples_per_cosmo = n_patches * n_perms_per_cosmo
    delta_Aia = conf["analysis"]["fiducial"]["perturbations"]["Aia"]

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

    # shuffle the indices
    rng = default_rng(seed=args.np_seed)
    i_examples = rng.permutation(n_examples_per_cosmo)

    LOGGER.info(f"n_examples_per_file = {n_examples_per_file}")

    pert_labels = [label.split("cosmo_")[1].replace("/", "") for label in cosmo_dirs_in]
    # manually add intrinsic alignment after the fiducial
    pert_labels = [pert_labels[0]] + ["delta_Aia_m", "delta_Aia_p"] + pert_labels[1:]
    LOGGER.info(f"{len(pert_labels)} labels = {pert_labels}")

    # index corresponds to a .tfrecord file ###########################################################################
    for index in indices:
        LOGGER.timer.start("index")

        tfr_file = get_filename_tfrecords(args.dir_out, tag=conf["survey"]["name"], index=index, simset="fiducial")
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
                for cosmo_dir_in in cosmo_dirs_in:
                    file_cosmo = get_filename_data_vectors(cosmo_dir_in, with_bary=args.with_bary)
                    LOGGER.debug(f"Taking inputs from {cosmo_dir_in}")

                    kg = load_kg(file_cosmo, i_example)
                    kg_perts.append(kg)

                    if "cosmo_fiducial" in cosmo_dir_in:
                        ia, sn_realz = load_ia_and_sn(file_cosmo, i_example)

                        LOGGER.debug(f"Adding intrinsic alignment as a perturbation to the fiducial")
                        kg_perts.append(kg - delta_Aia * ia)
                        kg_perts.append(kg + delta_Aia * ia)

                # shape (2 * n_cosmos + 1, n_pix, n_z_bins) for the delta loss
                kg_perts = np.stack(kg_perts, axis=0)
                LOGGER.debug(f"The tensor of kappa perturbations has shape {kg_perts.shape}")
                LOGGER.debug(f"The tensor of noise realizations has shape {sn_realz.shape}")

                serialized = tfrecords.parse_forward_fiducial(
                    kg_perts, pert_labels, sn_realz, i_example
                ).SerializeToString()

                # check correctness
                inv_data_vectors, inv_index = tfrecords.parse_inverse_fiducial(serialized, pert_labels, i_noise=0)
                inv_kg_perts = tf.stack([inv_data_vectors[f"kg_{pert_label}"] for pert_label in pert_labels], axis=0)
                inv_sn = inv_data_vectors["sn"]

                assert np.allclose(inv_kg_perts, kg_perts)
                assert np.allclose(inv_sn, sn_realz[0])
                assert np.allclose(inv_index, i_example)

                LOGGER.debug("decoded successfully")

                file_writer.write(serialized)

                n_done += 1

        LOGGER.info(f"Done with index {index} after {LOGGER.timer.elapsed('index')}")
        yield index


def load_kg(filename, ind_example):
    with h5py.File(filename, "r") as f:
        # shape (n_examples_per_cosmo, n_pix, n_z_bins) before the indexing
        kg = f["kg"][ind_example, ...]

    LOGGER.debug(f"Successfully loaded kg")
    return kg


def load_ia_and_sn(filename, ind_example):
    with h5py.File(filename, "r") as f:
        # shape (n_examples_per_cosmo, n_pix, n_z_bins) before the indexing
        ia = f["ia"][ind_example, ...]
        # shape (n_examples_per_cosmo, n_noise, n_pix, n_z_bins) before the indexing
        sn = f["sn"][ind_example, ...]

    LOGGER.debug(f"Successfully loaded ia and sn")
    return ia, sn
