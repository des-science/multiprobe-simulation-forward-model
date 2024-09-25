"""
Created on May 2023
Author: Arne Thomsen

Contains functions that are specific to galaxy clustering. Since the healpy alm transform only takes either a single
or three maps (polarized case), these functions are not vectorized accross the example dimension.
"""

import numpy as np

# import tensorflow as tf
import os

from msfm.utils import files, imports

hp = imports.import_healpy()


def galaxy_density_to_count(
    ng_bar,
    # linear
    dg,
    bg,
    # quadratic
    dg2=None,
    bg2=None,
    # modeling
    conf=None,
    systematics_map=None,
    stochasticity=None,
    # format
    nest=True,
    data_vec_pix=None,
    mask=None,
    np_seed=None,
):
    """Transform a galaxy density to a galaxy count map, according to the constants defined in the config file.
    Negative values are clipped and the maps tranformed to conserve the total number of galaxies like in DeepLSS.

    Args:
        ng_bar (np.ndarray): Average number of galaxies per pixel (optionally per tomographic bin).
        dg (Union[np.ndarray, tf.Tensor]): Galaxy density contrast map or datavector. Optionally per tomographic bin
            in the last array dimension.
        bg (np.ndarray): Effective linear galaxy biasing parameter (optionally per tomographic bin).
        dg2 (np.ndarray, optional): Squared galaxy density contrast map (optionally per tomographic bin).
        bg2 (np.ndarray, optional): Effective quadratic galaxy biasing parameter (optionally per tomographic bin).
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). The relative paths are stored here. Defaults to
            None.
        systematics_map (bool): Whether to multiply with the maglim systematics map. Defaults to False.
        stochasticity (float, optional): Raises a NotImplementedError if not None. Defaults to None.


    Raises:
        ValueError: If something apart from a numpy array or tensorflow tensor is passed.

    Returns:
        ng: Galaxy number count map.
    """

    # decorrelate the galaxy density contrast from the galaxy number
    if stochasticity is not None:
        raise NotImplementedError("The current implementation of stochasticity is known to be wrong, don't use it")

        assert isinstance(stochasticity, float), f"stochasticity must be a float, got {type(stochasticity)}"
        assert 0 < stochasticity < 1, f"stochasticity must be between 0 and 1, got {stochasticity}"
        assert isinstance(dg, np.ndarray), f"dg must be a numpy array, got {type(dg)}"
        assert nest, f"The healpy maps must be in nest ordering to add the stochasticity, got ring instead"
        assert data_vec_pix is not None, f"data_vec_pix must be passed if stochasticity is not None"

        # healpy path
        conf = files.load_config(conf)
        file_dir = os.path.dirname(__file__)
        repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
        hp_datapath = os.path.join(repo_dir, conf["files"]["healpy_data"])

        # convert to full sky map
        n_side = conf["analysis"]["n_side"]
        n_pix = conf["analysis"]["n_pix"]

        rng = np.random.default_rng(np_seed)

        # tomographic bins
        n_z = dg.shape[1]
        for i_z in range(n_z):
            dg_full = np.zeros(n_pix)
            dg_full[data_vec_pix] = dg[:, i_z]

            dg_full = hp.reorder(dg_full, n2r=True)
            alm = hp.map2alm(dg_full, pol=False, use_pixel_weights=True, datapath=hp_datapath)

            # draw random phases
            random_phases = stochasticity * rng.uniform(-np.pi, np.pi, alm.shape[0])
            scrambled_alm = np.exp(1j * random_phases) * alm

            dg_full = hp.alm2map(scrambled_alm, nside=n_side, pol=False)
            dg_full = hp.reorder(dg_full, r2n=True)

            dg[:, i_z] = dg_full[data_vec_pix]

    # linear bias
    if (bg2 is None) and (dg2 is None):
        ng = ng_bar * (1 + bg * dg)

    # quadratic bias
    elif (bg2 is not None) and (dg2 is not None):
        ng = ng_bar * (1 + bg * dg + bg2 * dg2)

    else:
        raise ValueError("Both or none of dg2 and bg2 must be passed")

    # transform like in DeepLSS Appendix E and https://github.com/tomaszkacprzak/deep_lss/blob/3c145cf8fe04c4e5f952dca984c5ce7e163b8753/deep_lss/lss_astrophysics_model_batch.py#L609
    # this ensures that all of the values are positive, while the total number of galaxies is conserved
    if isinstance(dg, np.ndarray):
        ng_clip = np.clip(ng, a_min=0, a_max=None, dtype=np.float32)
        ng = ng_clip * np.sum(ng) / np.sum(ng_clip)
    elif isinstance(dg, tf.Tensor):
        import tensorflow as tf

        ng_clip = tf.clip_by_value(ng, clip_value_min=0, clip_value_max=1e5)
        ng = ng_clip * tf.reduce_sum(ng) / tf.reduce_sum(ng_clip)
    else:
        raise ValueError(f"Unsupported type {type(dg)} for dg")

    if systematics_map is not None:
        # mask zeros, this is expecially important for the padded data vectors
        ng[systematics_map != 0.0] /= systematics_map[systematics_map != 0.0]

    # mask the footprint within the padded data vector
    if mask is not None:
        ng *= mask

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

    # elif isinstance(ng, tf.Tensor):
    #     raise NotImplementedError

    return poisson_noise
