# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created February 2023
Author: Arne Thomsen

Functions to handle the configuration and read in the survey files on the data vector pixels, masks and noise
"""

import os, h5py, warnings
import numpy as np

from msfm.utils import logger, input_output, filenames

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
    file is generated in notebooks/survey_file_gen/pixel_file.ipynb. If the conf argument is not passed, the default
    within the directory where this file resides is used.

    Args:
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). The relative paths are stored here. Defaults to
            None.

    Returns:
        data_vec_pix: data vector pixels including padding in NEST ordering (non-tomographic)
        metacal_tomo_patches_pix: Tomographic patch indices in RING ordering to cut out from the full sky maps
        metacal_tomo_corresponding_pix: Needed to convert the pixels in RING ordering to NEST inside the datavector
        gamma2_signs: Signs for gamma2 that come from mirroring the survey patch
        maglim_patches_pix: Patch indices in RING ordering to cut out from the full sky maps (non-tomographic)
        maglim_corresponding_pix: Needed to convert the pixels in RING ordering to NEST inside the datavector
    """
    conf = load_config(conf)

    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    pixel_file = os.path.join(repo_dir, conf["files"]["pixels"])

    with h5py.File(pixel_file, "r") as f:
        print(pixel_file)
        # pixel indices of padded data vector
        data_vec_pix = f["data_vec"][:]

        # Metacal sample: weak lensing
        metacal_tomo_patches_pix = []
        metacal_tomo_corresponding_pix = []
        for z_bin in conf["survey"]["metacal"]["z_bins"]:
            # shape (4, pix_in_bin)
            patches_pix_dict = f[f"metacal/patches/{z_bin}"][:]
            # shape (pix_in_bin,)
            corresponding_pix_dict = f[f"metacal/patch_to_data_vec/{z_bin}"][:]

            metacal_tomo_patches_pix.append(patches_pix_dict)
            metacal_tomo_corresponding_pix.append(corresponding_pix_dict)

        # to correct the shear for patch cut outs that have been mirrored
        gamma2_signs = f["metacal/gamma_2_sign"][:]

        # Maglim sample: galaxy clustering
        maglim_patches_pix = f["maglim/patches"][:]
        maglim_corresponding_pix = f["maglim/patch_to_data_vec"][:]

    LOGGER.info(f"Loaded the pixel file")

    # package into dictionaries
    patches_pix_dict = {}
    patches_pix_dict["metacal"] = metacal_tomo_patches_pix
    patches_pix_dict["maglim"] = maglim_patches_pix

    corresponding_pix_dict = {}
    corresponding_pix_dict["metacal"] = metacal_tomo_corresponding_pix
    corresponding_pix_dict["maglim"] = maglim_corresponding_pix

    return data_vec_pix, patches_pix_dict, corresponding_pix_dict, gamma2_signs


def get_tomo_masks(conf=None):
    """Masks the data vectors for the different tomographic bins.

    Args:
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). Defaults to None.

    Returns:
        np.ndarray: Mask array of shape (n_pix, n_z_bins) that is zero for the padding and one for the data
    """
    data_vec_pix, _, metacal_tomo_conversion_pix, _, _, _ = load_pixel_file(conf)

    masks = []
    # loop over the tomographic bins
    for pix in metacal_tomo_conversion_pix:
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


def load_redshift_distributions(galaxy_sample, conf=None):
    """Load the redshift distributions from disk to memory.

    Args:
        galaxy_sample (str): Either "metacal" or "maglim".
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). The relative paths are stored here. Defaults to
            None.

    Returns:
        list: Per redshift bin z an nz values of the distribution.
    """
    conf = load_config(conf)

    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    redshift_dir = os.path.join(repo_dir, conf["dirs"]["redshift_distributions"])

    n_z_bins = len(conf["survey"][galaxy_sample]["z_bins"])

    tomo_z = []
    tomo_nz = []
    for i_tomo in range(1, n_z_bins + 1):
        z_dist_file = filenames.get_filename_z_distribution(redshift_dir, galaxy_sample, i_tomo)
        z_dist = np.loadtxt(z_dist_file)

        tomo_z.append(z_dist[:, 0])
        tomo_nz.append(z_dist[:, 1])

    return tomo_z, tomo_nz
