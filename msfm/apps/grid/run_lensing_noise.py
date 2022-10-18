# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created October 2022
Author: Arne Thomsen

Generate shape noise realizations for the weak lensing forward model
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os, argparse, warnings, h5py, time

from numba import njit

from msfm.apps.grid.run_lensing_datavectors import get_data_vec
from msfm.utils import logging, input_output, shear
from msfm.utils.filenames import *

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logging.get_logger(__file__)

# set the environmental variable OMP_NUM_THREADS to the number of logical processors for healpy parallelixation
try:
    n_cpus = len(os.sched_getaffinity(0))
except AttributeError:
    LOGGER.debug(f"os.sched_getaffinity is not available on this system, use os.cpu_count() instead")
    n_cpus = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(n_cpus)

import healpy as hp


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
        "--grid_dir_in",
        type=str,
        default="/global/cfs/cdirs/des/cosmogrid/DESY3/grid",
        help="input root dir of the simulation grid",
    )
    parser.add_argument(
        "--grid_dir_out",
        type=str,
        default="/pscratch/sd/a/athomsen/DESY3/grid",
        help="output root dir of the simulation grid",
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
    parser.add_argument("--debug", action="store_true", help="debug/test mode")
    parser.add_argument("--max_sleep", type=int, default=120, help="sleep before copying to avoid clashes")

    args, _ = parser.parse_known_args(args)

    logging.set_all_loggers_level(args.verbosity)

    return args


def main(indices, args):
    args = setup(args)

    LOGGER.timer.start("main")

    if args.debug:
        args.max_sleep = 0
        LOGGER.warning("!!! debug/test mode !!!")

    sleep_sec = np.random.uniform(0, args.max_sleep) if args.max_sleep > 0 else 0
    LOGGER.info(f"Waiting for {sleep_sec:.2f}s to prevent overloading IO")
    time.sleep(sleep_sec)

    conf_file = os.path.join(args.repo_dir, args.config)
    conf = input_output.read_yaml(conf_file)
    LOGGER.info(f"Loaded configuration file")

    # constants
    n_side = conf["analysis"]["n_side"]
    n_pix = conf["analysis"]["n_pix"]
    n_patches = conf["analysis"]["n_patches"]
    n_perms_per_param = conf["analysis"]["n_perms_per_param"]
    z_bins = conf["survey"]["metacal"]["z_bins"]
    n_z_bins = len(z_bins)

    lmax = 3 * n_side - 1
    l = hp.Alm.getlm(lmax)[0]
    l[l == 1] = 0

    kappa2gamma_fac, gamma2kappa_fac = shear.get_kaiser_squires_factors(l)
    l_mask_fac = shear.get_l_mask(l)

    # TODO correct metacal bias
    tomo_bias = [0.7, 1.2, 1.8, 3]

    pixel_file = os.path.join(args.repo_dir, conf["files"]["pixels"])
    with h5py.File(pixel_file, "r") as f:
        # pixel indices of padded data vector
        data_vec_pix = f["metacal/map_cut_outs/data_vec_ids"][:]
        data_vec_len = len(data_vec_pix)

        # pixel indices of the non padded patches (non tomographic)
        non_tomo_patches_pix = f["metacal/masks/RING/non_tomo"][:]
        non_tomo_patches_len = len(non_tomo_patches_pix)

        # to correct the shear for patch cut outs that have been mirrored
        gamma2_signs = f["metacal/map_cut_outs/patches/gamma_2_sign"][:]

        tomo_patches_pix = []
        tomo_corresponding_pix = []
        for z_bin in conf["survey"]["metacal"]["z_bins"]:
            # shape (4, pix_in_bin)
            patches_pix = f[f"metacal/map_cut_outs/patches/RING/{z_bin}"][:]
            # shape (pix_in_bin,)
            corresponding_pix = f[f"metacal/map_cut_outs/RING_ids_to_data_vec/{z_bin}"][:]

            tomo_patches_pix.append(patches_pix)
            tomo_corresponding_pix.append(corresponding_pix)
    LOGGER.info(f"Loaded pixel file")

    noise_file = os.path.join(args.repo_dir, conf["files"]["noise"])
    with h5py.File(noise_file, "r") as f:
        tomo_gamma_cat = []
        tomo_n_bar = []
        for z_bin in conf["survey"]["metacal"]["z_bins"]:
            # shape (n_gal, 3) with e1, e2, w
            gamma_cat = f[f"{z_bin}/cat"][:]
            n_bar = f[f"{z_bin}/n_bar"][()]

            tomo_gamma_cat.append(gamma_cat)
            tomo_n_bar.append(n_bar)
    LOGGER.info(f"Loaded noise file")

    # grid directories
    grid_params_file = os.path.join(args.repo_dir, conf["files"]["grid_params"])
    grid_params = np.load(grid_params_file)

    # permutation level
    grid_dirs_in = [
        os.path.join(args.grid_dir_in, grid_param, f"perm_{i:04d}")
        for grid_param in grid_params
        for i in range(n_perms_per_param)
    ]

    # parameter level
    grid_dirs_out = [
        os.path.join(args.grid_dir_out, grid_param) for grid_param in grid_params for _ in range(n_perms_per_param)
    ]

    n_grid = len(grid_dirs_in)
    LOGGER.info(f"Got grid of size {n_grid} with base path {args.grid_dir_in}")

    # other directories
    datapath = os.path.join(args.repo_dir, conf["files"]["healpy_data"])

    # index corresponds to simulation permutation on the grid
    for index in indices:
        grid_dir_in = grid_dirs_in[index]  # permutation level
        grid_dir_out = grid_dirs_out[index]  # parameter level
        perm_id = index % n_perms_per_param
        LOGGER.info(f"Index {index} takes input from {grid_dir_in}")

        if not os.path.isdir(grid_dir_out):
            input_output.robust_makedirs(grid_dir_out)

        full_maps_file = get_filename_full_maps(grid_dir_in)

        # output containers
        data_noise = np.zeros((n_patches, data_vec_len, n_z_bins), dtype=np.float32)
        if args.debug:
            data_counts = np.zeros((n_patches, data_vec_len, n_z_bins), dtype=np.float32)
            LOGGER.debug(f"Created output container for Metacalibration galaxy counts maps")

        for i_z, z_bin in enumerate(z_bins):
            # only consider this tomographic bin
            patches_pix = tomo_patches_pix[i_z]
            corresponding_pix = tomo_corresponding_pix[i_z]
            n_bar = tomo_n_bar[i_z]
            bias = tomo_bias[i_z]
            gamma_cat = tomo_gamma_cat[i_z]
            n_gals_cat = gamma_cat.shape[0]

            # base (rotated) footprint
            base_patch_pix = patches_pix[0]

            map_dir = f"dg/{z_bin}"
            with h5py.File(full_maps_file, "r") as f:
                delta_full = f[map_dir][:]
            LOGGER.info(f"Loaded {map_dir} from {full_maps_file}")

            # normalize to number density contrast
            delta_full = (delta_full - np.mean(delta_full)) / np.mean(delta_full)

            # number of galaxies per pixel
            counts_full = n_bar * (1 + bias * delta_full)
            counts_full = np.where(0 < counts_full, counts_full, 0)
            counts_full = np.random.poisson(counts_full)

            for i_patch, patch_pix in enumerate(patches_pix):
                LOGGER.info(f"Starting with patch index {i_patch}")

                counts_patch = counts_full[patch_pix]
                n_gals_patch = np.sum(counts_patch)

                # indices to sum over all of the galaxies in the individual pixels
                seg_ids = []
                for id, n_gals in enumerate(counts_patch):
                    seg_ids.extend(n_gals * [id])

                # inputs to the tf.function have to be tensors
                counts_patch = tf.constant(counts_patch, dtype=tf.int32)
                seg_ids = tf.constant(seg_ids, dtype=tf.int32)

                # randomize
                phase = tf.random.uniform(shape=(n_gals_cat,), minval=0, maxval=2 * np.pi)
                gamma_abs = tf.math.abs(gamma_cat[:, 0] + 1j * gamma_cat[:, 1])
                gamma1 = tf.math.cos(phase) * gamma_abs
                gamma2 = tf.math.sin(phase) * gamma_abs

                # joint samples for e1, e2 and w
                emp_dist = tfp.distributions.Empirical(
                    samples=tf.stack([gamma1, gamma2, gamma_cat[:, 2]], axis=1), event_ndims=1
                )

                samples = emp_dist.sample(sample_shape=n_gals_patch)

                LOGGER.timer.start("noise_gen")
                gamma1, gamma2 = tf_noise_gen(samples, seg_ids)
                LOGGER.debug(f"Noise generation successfull after {LOGGER.timer.elapsed('noise_gen')}")

                gamma1_patch = np.zeros(n_pix, dtype=np.float32)
                gamma1_patch[base_patch_pix] = gamma1

                gamma2_patch = np.zeros(n_pix, dtype=np.float32)
                gamma2_patch[base_patch_pix] = gamma2

                # mode removal
                LOGGER.timer.start("mode_removal")
                _, gamma_alm_E, gamma_alm_B = hp.map2alm(
                    [np.zeros_like(gamma1_patch), gamma1_patch, gamma2_patch],
                    use_pixel_weights=True,
                    datapath=datapath,
                )
                kappa_alm = gamma_alm_E * gamma2kappa_fac

                kappa_alm *= l_mask_fac

                kappa_patch = hp.alm2map(kappa_alm, nside=n_side)

                LOGGER.debug(f"Mode removal successfull after {LOGGER.timer.elapsed('mode_removal')}")

                # cut out padded data vector
                kappa_dv = get_data_vec(kappa_patch, data_vec_len, corresponding_pix, base_patch_pix)

                data_noise[i_patch, :, i_z] = kappa_dv

        # save the results
        data_vec_file = get_filename_data_vectors(grid_dir_out)
        with h5py.File(data_vec_file, "a") as f:
            try:
                # create dataset for every parameter level directory, collecting the permutation levels
                f.create_dataset(name="sn", shape=(n_perms_per_param * n_patches, data_vec_len, n_z_bins))
            except ValueError:
                LOGGER.info(f"dataset sn already exists in {data_vec_file}")

            f["sn"][n_patches * perm_id : n_patches * (perm_id + 1)] = data_noise
        LOGGER.info(f"Stored noise in {data_vec_file}")

        LOGGER.info(f"Done with index {index} after {LOGGER.timer.elapsed('main')}")
        yield index



# the input tensors have variable length because of the varying number of galaxies in the count map
@tf.function(
    input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32), tf.TensorSpec(shape=(None,), dtype=tf.int32)]
)
def tf_noise_gen(samples, seg_ids):
    LOGGER.warning(
        f"Tracing the noise generator tf.function for samples.shape = {samples.shape} and seg_ids.shape = {seg_ids.shape}"
    )

    # shape (total_gals, 3)
    e_samples = samples[:, :2]
    w_samples = tf.expand_dims(samples[:, 2], axis=1)

    # apply weights
    samples = tf.concat([e_samples * w_samples, w_samples], axis=1)

    sum_per_pix = tf.math.segment_sum(samples, seg_ids)

    # normalize with weights
    e_per_pix = sum_per_pix[:, :2] / tf.expand_dims(sum_per_pix[:, 2], axis=1)

    return e_per_pix[:, 0], e_per_pix[:, 1]

# This main only exists for testing purposes when not using esub
if __name__ == "__main__":
    args = [
        "--grid_dir_in=/Users/arne/data/CosmoGrid_example/DES/grid",
        "--grid_dir_out=/Users/arne/data/CosmoGrid_example/DES/grid",
        "--repo_dir=/Users/arne/git/multiprobe-simulation-forward-model",
        "--config=configs/config.yaml",
        "--max_sleep=0",
        "--debug",
        "--verbosity=debug",
    ]

    indices = [0, 1, 2, 3]
    # indices = [0]
    for _ in main(indices, args):
        pass
