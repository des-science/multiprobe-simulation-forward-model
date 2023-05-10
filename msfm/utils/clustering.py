"""
Created on May 2023
Author: Arne Thomsen

Contains functions that are specific to galaxy clustering.
"""

import numpy as np
import healpy as hp
import tensorflow as tf
import logging

from msfm.utils import analysis

hp_LOGGER = logging.getLogger("healpy")
hp_LOGGER.disabled = True

def galaxy_density_to_number(dg, n_gal, bg, conf=None, include_systematics=False):
    """Transform a galaxy density to a galaxy number map, according to the constants defined in the config file.

    Args:
        dg (Union[np.ndarray, tf.Tensor]): Galaxy density contrast map or datavector.
        n_gal (np.ndarray): Average number of galaxies per pixel (optionally per tomographic bin).
        bg (np.ndarray): Effective linear galaxy biasing parameter (optionally per tomographic bin).
        with_systematics (list): Whether to multiply with the maglim systematics map. These are in datavector format.
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). The relative paths are stored here. Defaults to
            None.

    Raises:
        ValueError: If something apart from a numpy array or tensorflow tensor is passed.

    Returns:
        dg: Galaxy number count map.
    """
    tomo_sys_map = analysis.get_clustering_systematics(conf)

    if isinstance(dg, np.ndarray):
        dg = n_gal * (1 + bg * dg)
        dg = np.where(0 < dg, dg, 0)

        if include_systematics:
            dg *= np.stack(tomo_sys_map, axis=-1)

    elif isinstance(dg, tf.Tensor):
        dg = n_gal * (1 + bg * dg)
        dg = tf.where(0 < dg, dg, 0)

        if include_systematics:
            dg *= tf.stack(tomo_sys_map, axis=-1)

    else:
        raise ValueError(f"Unsupported type {type(dg)} for dg")

    return dg


def galaxy_number_add_noise(dg, noise_fac=None, return_noise_fac=False):
    if isinstance(dg, np.ndarray):
        # draw noise
        if noise_fac is None:
            noisy_dg = np.random.poisson(dg).astype(np.float32)

            if return_noise_fac:
                noise_fac = noisy_dg / dg

        # apply previous noise realization
        else:
            noisy_dg = dg * noise_fac

    elif isinstance(dg, tf.Tensor):
        raise NotImplementedError
    
    if return_noise_fac:
        return noisy_dg, noise_fac
    else:
        return noisy_dg


# def galaxy_density_to_number(dg, n_gal, bg, conf=None, include_systematics=False, draw_noise=True, noise_factor=None):
#     """Transform a galaxy density to a galaxy number map, according to the constants defined in the config file.

#     Args:
#         dg (Union[np.ndarray, tf.Tensor]): Galaxy density contrast map or datavector.
#         n_gal (np.ndarray): Average number of galaxies per pixel (optionally per tomographic bin).
#         bg (np.ndarray): Effective linear galaxy biasing parameter (optionally per tomographic bin).
#         with_systematics (list): Whether to multiply with the maglim systematics map. These are in datavector format.
#         conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
#             passed through) or None (the default config is loaded). The relative paths are stored here. Defaults to
#             None.

#     Raises:
#         ValueError: If something apart from a numpy array or tensorflow tensor is passed.

#     Returns:
#         dg: Galaxy number count map.
#     """
#     tomo_sys_map = analysis.get_clustering_systematics(conf)

#     if isinstance(dg, np.ndarray):
#         dg = n_gal * (1 + bg * dg)
#         dg = np.where(0 < dg, dg, 0)

#         if draw_noise and noise_factor is None:
#             dg = np.random.poisson(dg)
#         elif not draw_noise and isinstance(noise_factor, np.ndarray):
#             dg *= noise_factor
#         else:
#             raise ValueError

#         if include_systematics:
#             dg *= np.stack(tomo_sys_map, axis=-1)

#     elif isinstance(dg, tf.Tensor):
#         dg = n_gal * (1 + bg * dg)
#         dg = tf.where(0 < dg, dg, 0)

#         if draw_noise:
#             dg = tf.random.poisson(shape=[], lam=dg)

#         if include_systematics:
#             dg *= tf.stack(tomo_sys_map, axis=-1)

#     else:
#         raise ValueError(f"Unsupported type {type(dg)} for dg")

#     return dg


# def galaxy_density_to_number(dg, tomo_n_gal, tomo_bg, include_systematics=False, apply_normalization=False, conf=None):
#     """Transform a galaxy density to a galaxy number map, according to the constants defined in the config file.

#     Args:
#         dg (Union[np.ndarray, tf.Tensor]): _description_
#         tomo_n_gal (np.ndarray): Average number of galaxies per pixel and tomographic bin.
#         tomo_bg (np.ndarray): Effective linear galaxy biasing parameter per tomographic bin.
#         with_systematics (list): Whether to multiply with the maglim systematics map.
#         conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
#             passed through) or None (the default config is loaded). The relative paths are stored here. Defaults to
#             None.

#     Raises:
#         ValueError: If something apart from a numpy array or tensorflow tensor is passed.

#     Returns:
#         dg: Galaxy number count map.
#     """
#     conf = analysis.load_config(conf)
#     n_side = conf["analysis"]["n_side"]
#     tomo_n_gal_metacal = np.array(conf["survey"]["maglim"]["n_gal"]) * hp.nside2pixarea(n_side, degrees=True)
#     tomo_sys_map = analysis.get_clustering_systematics(conf)

#     if isinstance(dg, np.ndarray):
#         dg = tomo_n_gal * (1 + tomo_bg * dg)
#         dg = np.where(0 < dg, dg, 0)
#         dg = np.random.poisson(dg)

#         if include_systematics:
#             dg *= np.stack(tomo_sys_map, axis=-1)

#     elif isinstance(dg, tf.Tensor):
#         dg = tomo_n_gal * (1 + tomo_bg * dg)
#         dg = tf.where(0 < dg, dg, 0)
#         dg = tf.random.poisson(shape=[], lam=dg)

#         if include_systematics:
#             dg *= tf.stack(tomo_sys_map, axis=-1)

#     else:
#         raise ValueError(f"Unsupported type {type(dg)} for dg")

#     return dg
