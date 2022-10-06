# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created on 23.9.2022
Author: Arne Thomsen

This file contains functions to transform the full sky weak lensing signal and intrinsic alignment maps from UFalcon 
into multiple survey footprint cut-outs

- Made to be run with esub
- Paths for Perlmutter/Cori
"""

import numpy as np
import healpy as hp
import os, argparse, warnings, h5py

from numba import njit

from desy3_analysis.utils import logging, io
from desy3_analysis.utils.filenames import *

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
    parser.add_argument("--test", action="store_true", help="test mode")
    parser.add_argument(
        "--grid_dir", type=str, default="/global/cfs/cdirs/des/cosmogrid/DESY3/grid", help="input root dir"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/global/homes/a/athomsen/cosmogrid_desy3/configs/config.yaml",
        help="configuration yaml file",
    )

    args, _ = parser.parse_known_args(args)

    logging.set_all_loggers_level(args.verbosity)

    return args


def main(indices, args):
    # setup
    args = setup(args)
    conf = io.read_yaml(args.config)

    # constants
    nside = conf["constants"]["nside"]
    npix = hp.nside2npix(npix)

    lmax = 3 * nside - 1
    l = hp.Alm.getlm(lmax)[0]
    l[l == 1] = 0

    # TODO add mask
    patch_pix = None

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

    # grid directories
    grid_perms = np.load(conf["files"]["grid_perms"])
    grid_dirs = [os.path.join(args.grid_dir, grid_perm) for grid_perm in grid_perms]
    n_grid = len(grid_dirs)
    LOGGER.info(f"got grid of size {n_grid} with base path {args.grid_dir}")

    # index corresponds to simulation permutation on the grid
    for index in indices:
        grid_dir = grid_dirs[index]
        LOGGER.info(f"working in {grid_dir}")

        full_maps_file = get_filename_full_maps(grid_dir)

        for map_type in conf["map_types"]["lensing"]:
            for z_bin in conf["z_bins"]["metacal"]:

                map_dir = f"{map_type}/{z_bin}"
                with h5py.File(full_maps_file, "r") as f:
                    kappa_full = f[map_dir][:]
                LOGGER.info(f"loaded {map_dir} from {full_maps_file}")

                # kappa -> gamma (full sky)
                kappa_alm = hp.map2alm(kappa_full, lmax=lmax, use_pixel_weights=True)
                gamma_alm = kappa_alm * kappa2gamma_fac
                _, gamma1_full, gamma2_full = hp.alm2map([np.zeros_like(gamma_alm), gamma_alm, np.zeros_like(gamma_alm)], nside=nside)

                # masking
                gamma1_patch = np.zeros(npix, dtype=np.float32)
                gamma1_patch = gamma1_full[patch_pix]

                gamma2_patch = np.zeros(npix, dtype=np.float32)
                gamma2_patch = gamma2_full[patch_pix]

                # mode removal
                _, gamma_alm_E, gamma_alm_B = hp.map2alm([np.zeros_like(gamma1_patch), gamma1_patch, gamma2_patch], use_pixel_weights=True)
                kappa_alm = gamma_alm_E * gamma2kappa_fac

                # band limiting
                kappa_alm *= l_mask_fac

                kappa_patch = hp.alm2map(kappa_alm, nside=nside)

                # cut out padded data vector

        yield index


@njit()
def get_data_vec(m, data_vec_len, corresponding_pix, cutout_pix):
    """
    This function makes cutouts from full sky maps to a nice data vector that can be fed into a DeepSphere network
    :param m: The map one should make a cutout from
    :param data_vec_len: length of the full data vec
    :param corresponding_pix: pixel inside the data vec that should be populated
    :param cutout_pix: pixel that should be cut out from the map
    :return: the data vec
    """

    data_vec = np.zeros(data_vec_len)
    n_pix = corresponding_pix.shape[0]

    # assign
    for i in range(n_pix):
        data_vec[corresponding_pix[i]] = m[cutout_pix[i]]

    return data_vec


