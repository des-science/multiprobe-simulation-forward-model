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

from msfm.utils import logging, input_output, cosmogrid, tfrecords
from msfm.utils.filenames import *

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logging.get_logger(__file__)


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
    parser.add_argument("--with_bary", action="store_true", help="activate debug mode")
    parser.add_argument("--n_files", type=int, default=100, help="number of .tfrecord files to produce")
    parser.add_argument(
        "--dir_in",
        type=str,
        default=b"/global/cfs/cdirs/des/cosmogrid/DESY3/grid",
        help="input root dir of the simulations",
    )
    parser.add_argument(
        "--dir_out",
        type=str,
        default="/pscratch/sd/a/athomsen/DESY3/grid",
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

    logging.set_all_loggers_level(args.verbosity)

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
    params_info = cosmogrid.get_parameter_info(meta_info_file, "fiducial")

    # constants
    n_patches = conf["analysis"]["n_patches"]
    n_perms_per_param = conf["analysis"]["fiducial"]["n_perms_per_param"]
    n_examples_per_param = n_patches * n_perms_per_param

    # set up the paths
    meta_info_file = os.path.join(args.repo_dir, conf["files"]["meta_info"])
    params_info = cosmogrid.get_parameter_info(meta_info_file, "fiducial")
    params_dir = [param_dir.decode("utf-8") for param_dir in params_info["path_par"]]

    # remove baryon perturbations for the fiducial set
    if not args.with_bary:
        params_dir = [param_dir for param_dir in params_dir if not "bary" in param_dir]

    params_dir_in = [os.path.join(args.dir_in, param_dir) for param_dir in params_dir]
    LOGGER.debug(params_dir_in)

    n_params = len(params_dir_in)
    LOGGER.info(f"Got simulation set fiducial of size {n_params} with base path {args.dir_in}")

    if n_examples_per_param % args.n_files == 0:
        n_examples_per_file = n_examples_per_param // args.n_files
    else:
        raise ValueError(
            f"The total number of examples per parameter {n_examples_per_param} "
            f"has to be divisible by the number of files {args.n_files}"
        )

    # shuffle the indices
    rng = default_rng(seed=args.np_seed)
    i_examples = rng.permutation(n_examples_per_param)
    # i_examples = np.arange(800)

    LOGGER.debug(f"n_examples_per_file = {n_examples_per_file}")

    # index corresponds to a .tfrecord file ###########################################################################
    for index in indices:
        LOGGER.timer.start("index")

        file_tfrecord = get_filename_tfrecords(
            args.dir_out, tag=conf["survey"]["name"], index=index, simset="fiducial"
        )
        LOGGER.info(f"Index {index} is writing to {file_tfrecord}")

        js = index * n_examples_per_file
        je = (index + 1) * n_examples_per_file

        n_done = 0
        with tf.io.TFRecordWriter(file_tfrecord) as file_writer:

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
                for param_dir_in in params_dir_in:
                    file_param = get_filename_data_vectors(param_dir_in, with_bary=args.with_bary)

                    kg = load_kg(file_param, i_example)
                    kg_perts.append(kg)

                    if "cosmo_fiducial" in param_dir_in:
                        ia, sn_realz = load_ia_and_sn(file_param, i_example)

                # shape (2 * n_params + 1, n_pix, n_z_bins) for the delta loss
                kg_perts = np.stack(kg_perts, axis=0)

                serialized = tfrecords.parse_forward_fiducial(kg_perts, ia, sn_realz).SerializeToString()

                # check correctness
                inv_kg, inv_ia, inv_sn = tfrecords.parse_inverse_fiducial(serialized)

                assert np.allclose(inv_kg, kg_perts)
                assert np.allclose(inv_ia, ia)
                assert np.allclose(inv_sn, sn_realz)

                file_writer.write(serialized)

                n_done += 1

        LOGGER.info(f"Done with index {index} after {LOGGER.timer.elapsed('index')}")
        yield index


def load_kg(filename, ind_example):
    with h5py.File(filename, "r") as f:
        # shape (n_examples_per_param, n_pix, n_z_bins)
        kg = f["kg"][ind_example, ...]

    LOGGER.debug(f"Successfully loaded kg from file {filename}")
    return kg


def load_ia_and_sn(filename, ind_example):
    with h5py.File(filename, "r") as f:
        # shape (n_examples_per_param, n_pix, n_z_bins)
        ia = f["ia"][ind_example, ...]
        # shape (n_examples_per_param, n_noise, n_pix, n_z_bins)
        sn = f["sn"][ind_example, ...]

    LOGGER.debug(f"Successfully loaded ia and sn from file {filename}")
    return ia, sn
