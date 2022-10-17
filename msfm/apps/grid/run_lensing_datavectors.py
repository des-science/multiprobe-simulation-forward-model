# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created on 23.9.2022
Author: Arne Thomsen

This file contains functions to transform the full sky weak lensing signal and intrinsic alignment maps from UFalcon 
into multiple survey footprint cut-outs

- Made to be run with esub
"""

import numpy as np
import os, argparse, warnings, h5py, time

from numba import njit

from msfm.utils import logging, input_output
from msfm.utils.filenames import *

# set the environmental variable OMP_NUM_THREADS to the number of logical processors
# os.environ["OMP_NUM_THREADS"] =
import healpy as hp

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

    lmax = 3 * n_side - 1
    l = hp.Alm.getlm(lmax)[0]
    l[l == 1] = 0

    # from eq. (11) in https://academic.oup.com/mnras/article/505/3/4626/6287258
    kappa2gamma_fac = np.where(
        np.logical_and(l != 1, l != 0),
        -np.sqrt(((l + 2.0) * (l - 1)) / ((l + 1) * l)),
        0,
    )
    gamma2kappa_fac = np.where(
        np.logical_and(l != 1, l != 0),
        1 / kappa2gamma_fac,
        0,
    )
    l_mask_fac = np.where(np.logical_and(l != 1, l != 0), 1.0, 0.0)

    pixel_file = os.path.join(args.repo_dir, conf["files"]["pixels"])
    with h5py.File(pixel_file) as f:
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
        data_vectors = {}  # NEST ordering and padding
        data_patches = {}  # RING ordering and no padding
        for map_type in conf["survey"]["map_types"]["lensing"]:
            z_bins = conf["survey"]["metacal"]["z_bins"]
            n_z_bins = len(z_bins)

            # TODO do every patch multiple times
            data_vectors[map_type] = np.zeros((n_patches, data_vec_len, n_z_bins), dtype=np.float32)
            data_patches[map_type] = np.zeros((n_patches, non_tomo_patches_len, n_z_bins), dtype=np.float32)

            for i_z, z_bin in enumerate(z_bins):
                # only consider this tomographic bin
                patches_pix = tomo_patches_pix[i_z]
                corresponding_pix = tomo_corresponding_pix[i_z]

                # base (rotated) footprint
                base_patch_pix = patches_pix[0]

                map_dir = f"{map_type}/{z_bin}"
                with h5py.File(full_maps_file, "r") as f:
                    kappa_full = f[map_dir][:]
                LOGGER.info(f"Loaded {map_dir} from {full_maps_file}")

                # kappa -> gamma (full sky)
                kappa_alm = hp.map2alm(kappa_full, lmax=lmax, use_pixel_weights=True, datapath=datapath)
                gamma_alm = kappa_alm * kappa2gamma_fac
                _, gamma1_full, gamma2_full = hp.alm2map(
                    [np.zeros_like(gamma_alm), gamma_alm, np.zeros_like(gamma_alm)], nside=n_side
                )

                for i_patch, patch_pix in enumerate(patches_pix):
                    LOGGER.info(f"Starting with patch index {i_patch}")

                    # The 90° rots do NOT change the shear, however, the mirroring does,
                    # therefore we have to swap sign of gamma2 for the last 2 patches!
                    gamma2_sign = gamma2_signs[i_patch]
                    LOGGER.debug(f"Using gamma2 sign {gamma2_sign}")

                    # TODO each patch is done multiple times

                    # masking TODO use Janis' memory efficient numba function?
                    gamma1_patch = np.zeros(n_pix, dtype=np.float32)
                    gamma1_patch[base_patch_pix] = gamma1_full[patch_pix]

                    gamma2_patch = np.zeros(n_pix, dtype=np.float32)
                    gamma2_patch[base_patch_pix] = gamma2_full[patch_pix]

                    # fix the sign
                    gamma2_patch *= gamma2_sign

                    # mode removal
                    _, gamma_alm_E, gamma_alm_B = hp.map2alm(
                        [np.zeros_like(gamma1_patch), gamma1_patch, gamma2_patch],
                        use_pixel_weights=True,
                        datapath=datapath,
                    )
                    kappa_alm = gamma_alm_E * gamma2kappa_fac
                    LOGGER.debug(f"Mode removal successfull")

                    # band limiting
                    kappa_alm *= l_mask_fac

                    kappa_patch = hp.alm2map(kappa_alm, nside=n_side)

                    # cut out padded data vector
                    kappa_dv = get_data_vec(kappa_patch, data_vec_len, corresponding_pix, base_patch_pix)

                    data_vectors[map_type][i_patch, :, i_z] = kappa_dv
                    data_patches[map_type][i_patch, :, i_z] = kappa_patch[non_tomo_patches_pix]

        # save the results
        data_vec_file = get_filename_data_vectors(grid_dir_out)
        save_output_container(
            conf,
            "datavectors",
            data_vec_file,
            data_vectors,
            perm_id,
            n_perms_per_param,
            n_patches,
            data_vec_len,
            n_z_bins,
        )
        LOGGER.info(f"Stored datavectors in {data_vec_file}")

        patches_file = get_filename_data_patches(grid_dir_out)
        save_output_container(
            conf,
            "patches",
            patches_file,
            data_patches,
            perm_id,
            n_perms_per_param,
            n_patches,
            non_tomo_patches_len,
            n_z_bins,
        )
        LOGGER.info(f"Stored patches in {patches_file}")

        LOGGER.info(f"Done with index {index}")
        yield index


@njit
def get_data_vec(m, data_vec_len, corresponding_pix, cutout_pix):
    """
    This function makes cutouts from full sky maps to a nice data vector that can be fed into a DeepSphere network
    :param m: The map one should make a cutout from
    :param data_vec_len: length of the full data vec (including padding)
    :param corresponding_pix: pixel inside the data vec that should be populated (excludes padding)
    :param cutout_pix: pixel that should be cut out from the map (excludes padding)
    :return: the data vec
    """
    data_vec = np.zeros(data_vec_len)
    n_pix = corresponding_pix.shape[0]

    assert corresponding_pix.shape[0] == cutout_pix.shape[0]

    # assign
    for i in range(n_pix):
        data_vec[corresponding_pix[i]] = m[cutout_pix[i]]

    return data_vec


def save_output_container(
    conf, label, filename, output_container, perm_id, n_perms_per_param, n_patches, output_len, n_z_bins
):
    with h5py.File(filename, "a") as f:
        for map_type in conf["survey"]["map_types"]["lensing"]:
            try:
                # create dataset for every parameter level directory, collecting the permutation levels
                f.create_dataset(name=map_type, shape=(n_perms_per_param * n_patches, output_len, n_z_bins))
            except ValueError:
                LOGGER.info(f"dataset {map_type} already exists in {filename}")

            f[map_type][n_patches * perm_id : n_patches * (perm_id + 1)] = output_container[map_type]
    LOGGER.info(f"Stored {label} in {filename}")


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
    for _ in main(indices, args):
        pass
