# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created February 2023
Author: Arne Thomsen

Functions to handle the configuration and read in the survey files on the data vector pixels, masks and noise
"""

import os, h5py, warnings
import numpy as np

from icecream import ic

from msfm.utils import logger, input_output

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


def load_config(conf=None):
    """Loads or passes through a config

    Args:
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). Defaults to None.

    Raises:
        ValueError: When an invalid conf is passed

    Returns:
        dict: A configuration dictionary
    """
    # load the default config within this repo
    if conf is None:
        file_dir = os.path.dirname(__file__)
        repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
        conf = os.path.join(repo_dir, "configs/config.yaml")
        conf = input_output.read_yaml(conf)

    # load a config specified by a path
    elif isinstance(conf, str):
        conf = input_output.read_yaml(conf)

    # pass through an existing config
    elif isinstance(conf, dict):
        pass

    else:
        raise ValueError(f"conf {conf} must be None, a str specifying the path to the .yaml file, or the read dict")

    LOGGER.info(f"Loaded the config")
    return conf


def load_pixel_file(conf=None):
    """Loads the .h5 file that contains the pixel indices associated with the survey like the different patches. That
    file is generated in notebooks/survey_file_gen/pixel_file.ipynb. If the conf and repo_dir arguments are not passed,
    the default within the directory where this file resides is used.

    Args:
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). The relative paths are stored here. Defaults to
            None.

    Returns:
        data_vec_pix: data vector pixels including padding in NEST ordering (non-tomographic)
        non_tomo_patches_pix: non padded patches in RING ordering (non-tomographic)
        gamma2_signs: signs for gamma2 that come from mirroring the survey patch
        tomo_patches_pix: tomographic patch indices in RING ordering to cut out from the full sky maps
        tomo_corresponding_pix: needed to convert the pixels in RING ordering to NEST
    """
    conf = load_config(conf)

    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    pixel_file = os.path.join(repo_dir, conf["files"]["pixels"])

    with h5py.File(pixel_file, "r") as f:
        # pixel indices of padded data vector
        data_vec_pix = f["metacal/map_cut_outs/data_vec_ids"][:]

        # pixel indices of the non padded patches (non tomographic)
        patches_pix = f["metacal/masks/RING/non_tomo"][:]

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
    LOGGER.info(f"Loaded the pixel file")

    return data_vec_pix, patches_pix, gamma2_signs, tomo_patches_pix, tomo_corresponding_pix


def get_tomo_masks(conf=None):
    """Masks the data vectors for the different tomographic bins.

    Args:
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). Defaults to None.

    Returns:
        np.ndarray: Mask array of shape (n_pix, n_z_bins) that is zero for the padding and one for the data
    """
    data_vec_pix, _, _, _, tomo_corresponding_pix = load_pixel_file(conf)

    masks = []
    # loop over the tomographic bins
    for pix in tomo_corresponding_pix:
        mask = np.zeros(len(data_vec_pix), dtype=np.int32)
        # loop over individual pixels
        for p in pix:
            mask[p] = 1
        masks.append(mask)

    return np.array(masks).T


def load_noise_file(conf=None):
    """Loads the .h5 file that contains the noise information of the survey. That
    file is generated in notebooks/survey_file_gen/noise_file.ipynb

    Args:
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). The relative paths are stored here. Defaults to
            None.

    Returns:
        tomo_gamma_cat: list for the tomographic bins containing all of the gamma values for the galaxies in the survey
        tomo_n_bar: tomographic list of the mean number of galaxies per pixel
    """
    conf = load_config(conf)

    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    noise_file = os.path.join(repo_dir, conf["files"]["noise"])

    with h5py.File(noise_file, "r") as f:
        tomo_gamma_cat = []
        tomo_n_bar = []
        for z_bin in conf["survey"]["metacal"]["z_bins"]:
            # shape (n_gal, 3) with e1, e2, w
            gamma_cat = f[f"{z_bin}/cat"][:]
            n_bar = f[f"{z_bin}/n_bar"][()]

            tomo_gamma_cat.append(gamma_cat)
            tomo_n_bar.append(n_bar)
    LOGGER.info(f"Loaded the noise file")

    return tomo_gamma_cat, tomo_n_bar
