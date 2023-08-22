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

    ng = n_gal * (1 + bg * dg)

    # transform like in DeepLSS Appendix E and https://github.com/tomaszkacprzak/deep_lss/blob/3c145cf8fe04c4e5f952dca984c5ce7e163b8753/deep_lss/lss_astrophysics_model_batch.py#L609
    if isinstance(dg, np.ndarray):
        ng_clip = np.clip(ng, a_min=0, a_max=None)
        ng = ng_clip * np.sum(ng) / np.sum(ng_clip)

    elif isinstance(dg, tf.Tensor):
        ng_clip = tf.clip_by_value(ng, clip_value_min=0, clip_value_max=1e5)
        ng = ng_clip * tf.reduce_sum(ng) / tf.reduce_sum(ng_clip)

    else:
        raise ValueError(f"Unsupported type {type(dg)} for dg")

    if include_systematics:
        # mask zeros, this is expecially important for the padded data vectors
        ng[tomo_sys_dv != 0.0] /= tomo_sys_dv[tomo_sys_dv != 0.0]

    return ng


def galaxy_number_sample_noise(ng, n_noise):
    """
    TODO
    Draw Poisson noise according to the galaxy number map, or multiply with a Poisson noise factor (drawn at the
    fiducial, applied to the perturbations).

    Args:
        dg (Union[np.ndarray, tf.Tensor]): Galaxy number count map or datavector. Optionally per tomographic bin.

    Raises:
        ValueError: If something apart from a numpy array or tensorflow tensor is passed.

    Returns:
        ng: Noisy galaxy number count map.
    """
    if isinstance(ng, np.ndarray):
        # draw noise, poisson realizations along axis
        noisy_ngs = np.random.poisson(np.repeat(ng[np.newaxis, :], n_noise, axis=0)).astype(np.float32)

        # shape (n_noise, n_pix)
        poisson_noise = noisy_ngs - ng

    elif isinstance(ng, tf.Tensor):
        raise NotImplementedError

    return poisson_noise
