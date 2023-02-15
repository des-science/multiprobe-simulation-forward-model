# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created December 2022
Author: Arne Thomsen

Convert the .h5 files containing the lensing and clustering maps to ordered .tfrecord files suitable for evaluation
of summary statistics.

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
    description = "Make .tfrecord files from the survey footprints collected in the .h5 files."
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

    args.repo_dir = os.path.abspath(args.repo_dir)

    logger.set_all_loggers_level(args.verbosity)

    return args


def main(indices, args):
    args = setup(args)

    LOGGER.timer.start("main")
    LOGGER.info(f"Got index set of size {len(indices)}")
    # LOGGER.info(f"Running on {len(os.sched_getaffinity(0))} cores")

    conf_file = os.path.join(args.repo_dir, args.config)
    conf = input_output.read_yaml(conf_file)
    LOGGER.info(f"Loaded configuration file")

    meta_info_file = os.path.join(args.repo_dir, conf["files"]["meta_info"])
    params_info = cosmogrid.get_parameter_info(meta_info_file, "grid")
    LOGGER.info(f"Loaded meta information")

    # constants
    target_cosmo_params = conf["analysis"]["cosmo"]
    n_cosmo_params = len(target_cosmo_params)

    # CosmoGrid
    n_patches = conf["analysis"]["n_patches"]
    n_perms_per_param = conf["analysis"]["grid"]["n_perms_per_param"]
    n_examples_per_param = n_patches * n_perms_per_param

    # set up the paths
    meta_info_file = os.path.join(args.repo_dir, conf["files"]["meta_info"])
    params_dir = [param_dir.decode("utf-8") for param_dir in params_info["path_par"]]

    params_dir_in = [os.path.join(args.dir_in, param_dir) for param_dir in params_dir]
    n_params = len(params_dir_in)
    LOGGER.info(f"Got simulation set grid of size {n_params} with base path {args.dir_in}")

    # configure file structure
    if n_params % args.n_files == 0:
        n_params_per_file = n_params // args.n_files
        LOGGER.info(f"The number of files implies {n_params_per_file} cosmological parameters per file")
    else:
        raise ValueError(
            f"The total number of parameters {n_params} has to be divisible by the number of files {args.n_files}"
        )
    n_examples_per_file = n_examples_per_param * n_params_per_file

    LOGGER.info(
        f"In total, there are n_examples_per_param * n_params_per_file = {n_examples_per_param} * {n_params_per_file}"
        f" = {n_examples_per_file} examples per file"
    )

    # index corresponds to a .tfrecord file ###########################################################################
    for index in indices:
        LOGGER.timer.start("index")

        tfr_file = get_filename_tfrecords(args.dir_out, tag=conf["survey"]["name"], index=index, simset="grid")
        LOGGER.info(f"Index {index} is writing to {tfr_file}")

        # index for the cosmological parameters
        js = index * n_params_per_file
        je = (index + 1) * n_params_per_file
        LOGGER.info(f"And includes {params_dir[js : je]}")

        n_done = 0
        with tf.io.TFRecordWriter(tfr_file) as file_writer:
            # loop over the cosmological parameters
            for i_param, param_dir_in in LOGGER.progressbar(
                zip(range(js, je), params_dir_in[js:je]),
                at_level="info",
                desc="Looping through cosmological parameters",
                total=je - js,
            ):
                if args.debug and n_done > 5:
                    LOGGER.warning("Debug mode, aborting after 5 subindices")
                    break

                LOGGER.debug(f"Taking inputs from {param_dir_in}")
                file_param = get_filename_data_vectors(param_dir_in, with_bary=args.with_bary)
                kg_examples, ia_examples, sn_examples = load_data_vectors(file_param)

                # select the relevant cosmological parameters
                cosmo = [params_info[cosmo_param][i_param] for cosmo_param in conf["analysis"]["cosmo"]]
                cosmo = np.array(cosmo, dtype=np.float32)

                i_sobol = params_info["sobol_index"][i_param]

                # loop over the n_examples_per_param
                for kg, ia, sn_realz in LOGGER.progressbar(
                    zip(kg_examples, ia_examples, sn_examples),
                    at_level="debug",
                    desc="Looping through the examples",
                    total=n_examples_per_param,
                ):
                    serialized = tfrecords.parse_forward_grid(kg, ia, sn_realz, cosmo, i_sobol).SerializeToString()

                    # check correctness
                    i_noise = 0
                    inv_kg, inv_ia, inv_sn, inv_cosmo, inv_i_sobol = tfrecords.parse_inverse_grid(serialized, i_noise)

                    assert np.allclose(inv_kg, kg)
                    assert np.allclose(inv_ia, ia)
                    assert np.allclose(inv_sn, sn_realz[i_noise])
                    assert np.allclose(inv_cosmo, cosmo)
                    assert np.allclose(inv_i_sobol, i_sobol)

                    LOGGER.debug("decoded successfully")

                    file_writer.write(serialized)

                    n_done += 1

        LOGGER.info(f"Done with index {index} after {LOGGER.timer.elapsed('index')}")
        yield index


def load_data_vectors(
    filename,
):
    with h5py.File(filename, "r") as f:
        # shape (n_examples_per_param, n_pix, n_z_bins) before the indexing
        kg = f["kg"][:]
        ia = f["ia"][:]
        sn_realz = f["sn"][:]

    LOGGER.debug(f"Successfully loaded the data vectors")
    return kg, ia, sn_realz
