"""
Created on May 2023
Author: Arne Thomsen

Contains functions that are specific to galaxy clustering. Since the healpy alm transform only takes either a single
or three maps (polarized case), these functions are not vectorized accross the example dimension.
"""

import numpy as np
import healpy as hp
import tensorflow as tf
import logging

from msfm.utils import files

hp_LOGGER = logging.getLogger("healpy")
hp_LOGGER.disabled = True


def galaxy_density_to_number(dg, n_gal, bg, conf=None, include_systematics=False, sys_pixel_type="data_vector"):
    """Transform a galaxy density to a galaxy number map, according to the constants defined in the config file.

    Args:
        dg (Union[np.ndarray, tf.Tensor]): Galaxy density contrast map or datavector. Optionally per tomographic bin.
        n_gal (np.ndarray): Average number of galaxies per pixel (optionally per tomographic bin).
        bg (np.ndarray): Effective linear galaxy biasing parameter (optionally per tomographic bin).
        with_systematics (list): Whether to multiply with the maglim systematics map. These are in datavector format.
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). The relative paths are stored here. Defaults to
            None.
        sys_pixel_type (str, optional): Either "map" or "data_vector", determines whether the systematics map is
            returned as a full sky healpy map or in data vector format.


    Raises:
        ValueError: If something apart from a numpy array or tensorflow tensor is passed.

    Returns:
        ng: Galaxy number count map.
    """
    tomo_sys_dv = files.get_clustering_systematics(conf, pixel_type=sys_pixel_type)

    if isinstance(dg, np.ndarray):
        ng = n_gal * (1 + bg * dg)
        ng = np.where(0 < ng, ng, 0)

    elif isinstance(dg, tf.Tensor):
        ng = n_gal * (1 + bg * dg)
        ng = tf.where(0 < ng, ng, 0)

    else:
        raise ValueError(f"Unsupported type {type(dg)} for dg")

    if include_systematics:
        # mask zeros, this is expecially important for the padded data vectors
        ng[tomo_sys_dv != 0.0] /= tomo_sys_dv[tomo_sys_dv != 0.0]

    return ng


def galaxy_number_add_noise(ng, noise_fac=None, return_noise_fac=False):
    """Draw Poisson noise according to the galaxy number map, or multiply with a Poisson noise factor (drawn at the
    fiducial, applied to the perturbations).

    Args:
        dg (Union[np.ndarray, tf.Tensor]): Galaxy number count map or datavector. Optionally per tomographic bin.
        noise_fac (Union[np.ndarray, tf.Tensor, optional): Same shape as dg. This is only used for the fiducial
            perturbations, that should have the same noise characteristics as the fiducial Poisson noise sample.
            Defaults to None, then fresh noise is drawn.
        return_noise_fac (bool, optional): Whether to return the Poisson noise factor. Defaults to False.

    Raises:
        ValueError: If something apart from a numpy array or tensorflow tensor is passed.

    Returns:
        ng: Noisy galaxy number count map.
    """
    if isinstance(ng, np.ndarray):
        # draw noise
        if noise_fac is None:
            noisy_dg = np.random.poisson(ng).astype(np.float32)

            if return_noise_fac:
                noise_fac = noisy_dg / ng

        # apply previous noise realization
        else:
            noisy_dg = ng * noise_fac

    elif isinstance(ng, tf.Tensor):
        raise NotImplementedError

    if return_noise_fac:
        return noisy_dg, noise_fac
    else:
        return noisy_dg
