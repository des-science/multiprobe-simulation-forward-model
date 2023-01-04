# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created December 2022
Author: Arne Thomsen

Convert the .h5 files containing the lensing and clustering maps to scrambled .tfrecord files suitable for training
(fiducial) and prediction (grid)

Adapted from https://github.com/tomaszkacprzak/CosmoPointNet/blob/main/CosmoPointNet/apps/run_build_tfrecords.py 
by Tomasz Kacprzak
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os, argparse, warnings, h5py

from numpy.random import default_rng
from numba import njit
from icecream import ic

from msfm.utils import logging, input_output, cosmogrid, tfrecords
from msfm.utils.filenames import *

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logging.get_logger(__file__)


def resources(args):
    return dict(main_memory=1000, main_time_per_index=4)


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
    parser.add_argument(
        "--simset", type=str, default="grid", choices=("grid", "fiducial"), help="set of simulations to use"
    )
    parser.add_argument(
        "--dir_in",
        type=str,
        default="/global/cfs/cdirs/des/cosmogrid/DESY3/grid",
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

    logging.set_all_loggers_level(args.verbosity)

    return args


def main(indices, args):
    args = setup(args)

    LOGGER.timer.start("main")

    conf_file = os.path.join(args.repo_dir, args.config)
    conf = input_output.read_yaml(conf_file)
    LOGGER.info(f"Loaded configuration file")

    meta_info_file = os.path.join(args.repo_dir, conf["files"]["meta_info"])
    params_info = cosmogrid.get_parameter_info(meta_info_file, args.simset)

    # constants
    n_patches = conf["analysis"]["n_patches"]
    n_perms_per_param = conf["analysis"][args.simset]["n_perms_per_param"]
    n_examples_per_param = n_patches * n_perms_per_param

    # set up the paths
    meta_info_file = os.path.join(args.repo_dir, conf["files"]["meta_info"])
    params_info = cosmogrid.get_parameter_info(meta_info_file, args.simset)
    params_dir = params_info["path_par"]

    # parameter level
    dirs_in = [os.path.join(args.dir_in, param_dir.decode("utf-8")) for param_dir in params_dir]

    n_params = len(dirs_in)
    LOGGER.info(f"Got simulation set {args.simset} of size {n_params} with base path {args.dir_in}")

    n_examples = n_params * n_examples_per_param

    n_files = len(indices)
    if n_examples % n_files == 0:
        n_examples_per_file = n_examples // n_files
    else:
        raise ValueError(
            f"The total number of examples {n_params} * {n_examples_per_param} = {n_examples} "
            f"has to be divisible by the number of files {n_files}"
        )

    # shuffle the indices
    ind_shape = (n_params, n_examples_per_param)
    rng = default_rng(seed=args.np_seed)
    ind_shuffle = rng.permutation(n_examples)
    LOGGER.info(f"Iterating over {n_examples} indices of shape {ind_shape}")
    assert n_examples == np.prod(ind_shape)

    LOGGER.debug(f"n_files = {n_files}")
    LOGGER.debug(f"n_examples_per_file = {n_examples_per_file}")

    # index corresponds to a .tfrecord file ###########################################################################
    for index in indices:
        file_tfrecord = get_filename_tfrecords(
            args.dir_out, tag=conf["survey"]["name"], index=index, simset=args.simset
        )
        LOGGER.info(f"Index {index} is writing to {file_tfrecord}")

        js = index * n_examples_per_file
        je = (index + 1) * n_examples_per_file

        n_done = 0
        with tf.io.TFRecordWriter(file_tfrecord) as file_writer:

            for j in LOGGER.progressbar(range(js, je), at_level="info", desc="Storing DES patches"):
                if args.debug:
                    if n_done > 5:
                        LOGGER.warning("Debug mode, aborting after 5 subindices")
                        break

                # get the indices to parameter and patch
                ind_param, ind_example = np.unravel_index(ind_shuffle[j], ind_shape)
                LOGGER.debug(f"j = {j} in range({js},{je}): ind_param={ind_param} ind_example={ind_example}")

                # ind_param = 0
                # ind_example = 0

                # maps
                dir_param = dirs_in[ind_param]
                file_param = get_filename_data_vectors(dir_param)
                kg, ia, sn = load_datavectors(file_param, ind_example)

                # labels
                cosmo = np.array([params_info[ind_param][p] for p in conf["analysis"]["cosmo"]], dtype=np.float32)
                sobol = params_info[ind_param]["sobol_index"]
                LOGGER.debug(f"cosmo = {cosmo}")
                LOGGER.debug(f"sobol seed = {sobol}")

                serialized = tfrecords.parse_forward_maps(kg, ia, sn, cosmo, sobol).SerializeToString()

                # check correctness
                inv_kg, inv_ia, inv_sn, inv_cosmo, inv_sobol = tfrecords.parse_inverse_maps(serialized)
                assert np.allclose(inv_kg, kg)
                assert np.allclose(inv_ia, ia)
                assert np.allclose(inv_sn, sn)
                assert np.allclose(inv_cosmo, cosmo)
                assert np.allclose(inv_sobol, sobol)

                file_writer.write(serialized)

                n_done += 1

        yield index


def load_datavectors(filename, ind_example):
    with h5py.File(filename, "r") as f:
        # shape (n_examples_per_param, n_pix, n_z_bins)
        kg = f["kg"][ind_example, ...]
        ia = f["ia"][ind_example, ...]
        sn = f["sn"][ind_example, ...]

    LOGGER.debug(f"Successfully loaded file {filename}")
    return kg, ia, sn


# This main only exists for testing purposes when not using esub
if __name__ == "__main__":
    args = [
        "--simset=grid",
        "--dir_in=/Users/arne/data/CosmoGrid_example/DES/grid",
        "--dir_out=/Users/arne/data/CosmoGrid_example/DES/grid",
        "--repo_dir=/Users/arne/git/multiprobe-simulation-forward-model",
        "--config=configs/config.yaml",
        "--debug",
        "--verbosity=debug",
    ]

    indices = [0, 1, 2, 3]
    # indices = [0]
    for _ in main(indices, args):
        pass

    LOGGER.info(f"Done with main after {LOGGER.timer.elapsed('main')}")
