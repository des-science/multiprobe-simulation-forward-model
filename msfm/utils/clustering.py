"""
Created on May 2023
Author: Arne Thomsen

Contains functions that are specific to galaxy clustering. Since the healpy alm transform only takes either a single
or three maps (polarized case), these functions are not vectorized accross the example dimension.
"""

import numpy as np
import tensorflow as tf

from msfm.utils import files


def galaxy_density_to_count(
    dg, ng_bar, bg, bg2=None, conf=None, include_systematics=False, sys_pixel_type="data_vector"
):
    """Transform a galaxy density to a galaxy count map, according to the constants defined in the config file.
    Negative values are clipped and the maps tranformed to conserve the total number of galaxies like in DeepLSS.

    Args:
        dg (Union[np.ndarray, tf.Tensor]): Galaxy density contrast map or datavector. Optionally per tomographic bin
            in the last array dimension.
        ng_bar (np.ndarray): Average number of galaxies per pixel (optionally per tomographic bin).
        bg (np.ndarray): Effective linear galaxy biasing parameter (optionally per tomographic bin).
        bg2 (np.ndarray, optional): Effective quadratic galaxy biasing parameter (optionally per tomographic bin).
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). The relative paths are stored here. Defaults to
            None.
        include_systematics (bool): Whether to multiply with the maglim systematics map. Defaults to False.
        sys_pixel_type (str, optional): Either "map" or "data_vector", determines whether the systematics map is
            returned as a full sky healpy map or in data vector format.

    Raises:
        ValueError: If something apart from a numpy array or tensorflow tensor is passed.

    Returns:
        ng: Galaxy number count map.
    """
    tomo_sys_dv = files.get_clustering_systematics(conf, pixel_type=sys_pixel_type)

    # linear bias
    if bg2 is None:
        ng = ng_bar * (1 + bg * dg)

    # quadratic bias
    else:
        ng = ng_bar * (1 + bg * dg + bg2 * dg**2)

    # transform like in DeepLSS Appendix E and https://github.com/tomaszkacprzak/deep_lss/blob/3c145cf8fe04c4e5f952dca984c5ce7e163b8753/deep_lss/lss_astrophysics_model_batch.py#L609
    # this ensures that all of the values are positive, while the total number of galaxies is conserved
    if isinstance(dg, np.ndarray):
        ng_clip = np.clip(ng, a_min=0, a_max=None, dtype=np.float32)
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


def galaxy_count_to_noise(ng, n_noise, np_seed=None):
    """
    Draw Poisson noise according to the given map of galaxy counts.

    Args:
        ng (Union[np.ndarray, tf.Tensor]): Galaxy number count map or datavector. Optionally per tomographic bin.
        n_noise (int): Number of noise realizations to draw.
        np_seed (int, optional): Seed for the numpy random number generator. Defaults to None.

    Raises:
        ValueError: If something apart from a numpy array or tensorflow tensor is passed.

    Returns:
        poisson_noise: Pure (e.g. the input galaxy count map has been subtracted) Poisson noise consistent with the
            input.
    """

    if isinstance(ng, np.ndarray):
        rng = np.random.default_rng(np_seed)

        # draw noise, poisson realizations along axis
        noisy_ngs = rng.poisson(np.repeat(ng[np.newaxis, :], n_noise, axis=0), size=None).astype(np.float32)

        # shape (n_noise, n_pix) is broadcast along the first axis
        poisson_noise = noisy_ngs - ng

    elif isinstance(ng, tf.Tensor):
        raise NotImplementedError

    return poisson_noise
