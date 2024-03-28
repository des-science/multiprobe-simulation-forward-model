# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created March 2024
Author: Arne Thomsen

Transform the full sky weak lensing signal and intrinsic alignment maps into multiple survey footprint cut-outs and 
store them in .tfrecord files. The parallelization is done over the .tfrecord files, every jobarray element corresponds
to one.

Meant for 
 - Euler (CPU nodes, local scratch) 
 - esub jobarrays
 - Read the CosmoGrid directly from the SAN
 - CosmoGridV1.1
"""

import numpy as np
import tensorflow as tf
import os, argparse, warnings, time, yaml

from msfm.utils import (
    logger,
    imports,
    filenames,
    input_output,
    files,
    maps,
    lensing,
    clustering,
    cosmogrid,
    preprocessing,
)

hp = imports.import_healpy(parallel=True)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


def resources(args):
    args = setup(args)

    # TODO
    resources = {
        "main_memory": 2048,
        "main_time": 4,
        "main_n_cores": 4,
    }

    if args.from_san:
        # in MB. One projected_probes_maps_v11dmb.h5 should be around 1 GB
        resources["main_scratch"] = 2000
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
        "--cosmogrid_version", type=str, default="1.1", choices=["1.1", "1"], help="version of the input CosmoGrid"
    )

    parser.add_argument("--debug", action="store_true", help="activate debug mode")
    parser.add_argument("--with_bary", action="store_true", help="whether to include the baryonification in the input")
    parser.add_argument(
        "--from_san",
        action="store_true",
        help="copy the CosmoGrid files from the SAN instead of accessing them locally",
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

    args, _ = parser.parse_known_args(args)

    # print arguments
    logger.set_all_loggers_level(args.verbosity)
    for key, value in vars(args).items():
        LOGGER.info(f"{key} = {value}")

    if not os.path.isdir(args.dir_out):
        input_output.robust_makedirs(args.dir_out)

    args.config = os.path.abspath(args.config)

    return args


def main(indices, args):
    args = setup(args)

    LOGGER.timer.start("main")
    LOGGER.info(f"Got index set of size {len(indices)}")
    try:
        LOGGER.info(f"Running on {len(os.sched_getaffinity(0))} cores")
    except AttributeError:
        pass

    # I/O delay
    if args.debug:
        args.max_sleep = 0
        LOGGER.warning("debug mode")
    sleep_sec = np.random.uniform(0, args.max_sleep) if args.max_sleep > 0 else 0
    LOGGER.info(f"Waiting for {sleep_sec:.2f}s to prevent overloading IO")
    time.sleep(sleep_sec)

    # configuration
    conf = files.load_config(args.config)
    with open(os.path.join(args.dir_out, "config.yaml"), "w") as f:
        yaml.dump(conf, f)

    # directories TODO compactify this
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
        f"{n_perms_per_cosmo} permutations and {n_noise_per_example} noise realizations"
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

    # modeling settings
    degrade_to_grf = conf["analysis"]["modelling"]["degrade_to_grf"]
    if degrade_to_grf:
        LOGGER.warning(f"Degrading the weak lensing maps to Gaussian Random Fields")

    # index corresponds to a .tfrecord file ###########################################################################
    for index in indices:
        LOGGER.timer.start("index")

        tfr_file = filenames.get_filename_tfrecords(
            args.dir_out, tag=conf["survey"]["name"] + args.file_suffix, index=index, simset="grid"
        )
        LOGGER.info(f"Index {index} is writing to {tfr_file}")

        # index for the cosmological parameters
        js = index * n_cosmos_per_file
        je = (index + 1) * n_cosmos_per_file
        LOGGER.info(f"And includes {cosmo_dirs[js : je]}")

        n_done = 0
        with tf.io.TFRecordWriter(tfr_file) as file_writer:
            # loop over the cosmological parameters
            for i_cosmo, cosmo_dir_in in LOGGER.progressbar(
                zip(range(js, je), cosmo_dirs_in[js:je]),
                at_level="info",
                desc="Looping through cosmologies",
                total=je - js,
            ):
                LOGGER.debug(f"Taking inputs from {cosmo_dir_in}")

                # shape (n_examples_per_cosmo, n_pix, n_z_bins) for signal
                # or    (n_examples_per_cosmo, n_noise_per_examplen_pix, n_z_bins) for noise
                kg_examples, ia_examples, sn_examples, dg_examples = preprocessing.preprocess_permutations(
                    args, conf, "grid", cosmo_dir_in, pixel_file, noise_file
                )

                breakpoint()
