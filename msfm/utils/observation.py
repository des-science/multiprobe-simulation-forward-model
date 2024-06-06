# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created May 2024
Author: Arne Thomsen

Utilities to forward model (mock) observations to be consistent with the CosmoGrid maps.
"""

import os
import numpy as np
from msfm.utils import files, logger, lensing, imports, scales, maps
from typing import Union

hp = imports.import_healpy(parallel=True)

LOGGER = logger.get_logger(__file__)


def forward_model_observation_map(
    wl_gamma: np.ndarray = None,
    gc_count: np.ndarray = None,
    conf: Union[str, dict] = None,
    apply_norm: bool = True,
    with_padding: bool = True,
    nest: bool = True,
):
    """Take a (mock) observation and apply the same transformations to it as within the CosmoGrid pipeline, such that
    everything (masking, mode removal, normalization, ...) is consistent.

    Args:
        wl_gamma (np.ndarray, optional): The weak lensing shear map of shape (n_pix,n_z_metacal,2), where the first
            axis corresponds to a full sky map at the correct n_side and the last axis contains the gamma_1 and gamma_2
            components. Note that the input footprint has to be rotated to the correct position on the sky. Defaults
            to None.
        gc_count (np.ndarray, optional): The galaxy clustering galaxy number count map of shape (n_pix,n_z_maglim),
            like for wl_gamma. Defaults to None.
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). Defaults to None.
        apply_norm (bool, optional): Whether to rescale the maps to approximate unit range. Defaults to True.
        with_padding (bool, optional): Whether to include the padding of the data vectors (the healpy DeepSphere \
            networks) need this. Defaults to True.
        nest (bool, optional): Whether the full sky input maps wl_gamma and gc_count are in nested (or ring if false)
            ordering. Defaults to True.
    """
    assert wl_gamma is not None or gc_count is not None, "Either wl_gamma or gc_count must be provided."

    conf = files.load_config(conf)

    n_side = conf["analysis"]["n_side"]
    n_pix = conf["analysis"]["n_pix"]
    n_z_metacal = len(conf["survey"]["metacal"]["z_bins"])
    n_z_maglim = len(conf["survey"]["maglim"]["z_bins"])
    data_vec_pix, _, _, _ = files.load_pixel_file(conf)

    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    hp_datapath = os.path.join(repo_dir, conf["files"]["healpy_data"])

    if wl_gamma is not None:
        assert wl_gamma.shape == (
            n_pix,
            n_z_metacal,
            2,
        ), f"Expected shape {(n_pix, n_z_metacal, 2)}, got {wl_gamma.shape}"

        if nest:
            wl_gamma[..., 0] = maps.tomographic_reorder(wl_gamma[..., 0], n2r=True)
            wl_gamma[..., 1] = maps.tomographic_reorder(wl_gamma[..., 1], n2r=True)
        _, gamma2kappa_fac, _ = lensing.get_kaiser_squires_factors(l_max=3 * n_side - 1)

        wl_kappa = np.zeros((n_pix, n_z_metacal), dtype=np.float32)
        for i in range(n_z_metacal):
            wl_kappa[:, i] = lensing.mode_removal(
                wl_gamma[:, i, 0], wl_gamma[:, i, 1], gamma2kappa_fac, n_side, hp_datapath=hp_datapath
            )

        wl_kappa = maps.tomographic_reorder(wl_kappa, r2n=True)
        wl_kappa = wl_kappa[data_vec_pix]

        if apply_norm:
            wl_kappa = wl_kappa / conf["analysis"]["normalization"]["lensing"]

        wl_kappa, _ = scales.data_vector_to_smoothed_data_vector(
            wl_kappa,
            data_vec_pix=data_vec_pix,
            n_side=n_side,
            l_min=conf["analysis"]["scale_cuts"]["lensing"]["l_min"],
            theta_fwhm=conf["analysis"]["scale_cuts"]["lensing"]["theta_fwhm"],
            arcmin=True,
        )

    if gc_count is not None:
        assert gc_count.shape == (
            n_pix,
            n_z_maglim,
        ), f"Expected shape {(n_pix, n_z_maglim)}, got {gc_count.shape}"

        if apply_norm:
            pass

        if not nest:
            gc_count = maps.tomographic_reorder(gc_count, r2n=True)

        gc_count = gc_count[data_vec_pix]

        gc_count, _ = scales.data_vector_to_smoothed_data_vector(
            gc_count,
            data_vec_pix=data_vec_pix,
            n_side=n_side,
            l_min=conf["analysis"]["scale_cuts"]["lensing"]["l_min"],
            theta_fwhm=conf["analysis"]["scale_cuts"]["lensing"]["theta_fwhm"],
            arcmin=True,
        )

    if wl_gamma is not None and gc_count is not None:
        observation = np.concatenate([wl_kappa, gc_count], axis=-1)
    elif wl_gamma is not None:
        observation = wl_kappa
    else:
        observation = gc_count

    if not with_padding:
        masks_dict = files.get_tomo_dv_masks(conf)
        masks_metacal = masks_dict["metacal"]
        masks_maglim = masks_dict["maglim"]
        # only keep indices that are in all (per tomographic bin and galaxy sample) masks
        mask_total = np.prod(np.concatenate([masks_metacal, masks_maglim], axis=-1), axis=-1)
        mask_total = mask_total.astype(bool)
        footprint_pix = data_vec_pix[mask_total]

        observation_full_sky = np.zeros((n_pix, observation.shape[-1]), dtype=observation.dtype)
        observation_full_sky[data_vec_pix] = observation
        observation = observation[footprint_pix]

    return observation
