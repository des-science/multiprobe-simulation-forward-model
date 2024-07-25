# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created May 2024
Author: Arne Thomsen

Utilities to forward model (mock) observations to be consistent with the CosmoGrid maps.
"""

import os
import numpy as np
from msfm.utils import files, logger, lensing, imports, scales, maps, power_spectra
from typing import Union

hp = imports.import_healpy()

LOGGER = logger.get_logger(__file__)


def forward_model_observation_map(
    wl_gamma_map: np.ndarray = None,
    gc_count_map: np.ndarray = None,
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
    assert wl_gamma_map is not None or gc_count_map is not None, "Either wl_gamma or gc_count must be provided."

    conf = files.load_config(conf)

    n_side = conf["analysis"]["n_side"]
    n_pix = conf["analysis"]["n_pix"]
    n_z_metacal = len(conf["survey"]["metacal"]["z_bins"])
    n_z_maglim = len(conf["survey"]["maglim"]["z_bins"])

    data_vec_pix, patches_pix_dict, corresponding_pix_dict, _ = files.load_pixel_file(conf)
    data_vec_len = len(data_vec_pix)
    masks_dict = files.get_tomo_dv_masks(conf)
    masks_metacal = masks_dict["metacal"]
    masks_maglim = masks_dict["maglim"]

    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    hp_datapath = os.path.join(repo_dir, conf["files"]["healpy_data"])

    if wl_gamma_map is not None:
        assert wl_gamma_map.shape == (
            n_pix,
            n_z_metacal,
            2,
        ), f"Expected shape {(n_pix, n_z_metacal, 2)}, got {wl_gamma_map.shape}"

        # the input to the mode removal must always be in RING ordering
        if nest:
            wl_gamma_map[..., 0] = maps.tomographic_reorder(wl_gamma_map[..., 0], n2r=True)
            wl_gamma_map[..., 1] = maps.tomographic_reorder(wl_gamma_map[..., 1], n2r=True)
        _, gamma2kappa_fac, _ = lensing.get_kaiser_squires_factors(l_max=3 * n_side - 1)

        wl_kappa_dv = np.zeros((data_vec_len, n_z_metacal), dtype=np.float32)
        for i in range(n_z_metacal):
            # full sky (but only partially occupied)
            wl_kappa_map = lensing.mode_removal(
                wl_gamma_map[:, i, 0], wl_gamma_map[:, i, 1], gamma2kappa_fac, n_side, hp_datapath=hp_datapath
            )

            # full sky (but only footprint occupied) -> padded data vector
            wl_kappa_dv[:, i] = maps.map_to_data_vec(
                hp_map=wl_kappa_map,
                data_vec_len=data_vec_len,
                corresponding_pix=corresponding_pix_dict["metacal"][i],
                cutout_pix=patches_pix_dict["metacal"][i][0],
                remove_mean=True,
            )

        if apply_norm:
            wl_kappa_dv = wl_kappa_dv / conf["analysis"]["normalization"]["lensing"]

        wl_kappa_dv *= masks_metacal
        wl_kappa_dv, wl_alms = scales.data_vector_to_smoothed_data_vector(
            wl_kappa_dv,
            data_vec_pix=data_vec_pix,
            n_side=n_side,
            l_min=conf["analysis"]["scale_cuts"]["lensing"]["l_min"],
            theta_fwhm=conf["analysis"]["scale_cuts"]["lensing"]["theta_fwhm"],
            arcmin=True,
        )
        wl_kappa_dv *= masks_metacal

    if gc_count_map is not None:
        assert gc_count_map.shape == (
            n_pix,
            n_z_maglim,
        ), f"Expected shape {(n_pix, n_z_maglim)}, got {gc_count_map.shape}"

        # the input to map_to_data_vec must always be in RING ordering
        if nest:
            gc_count_map = maps.tomographic_reorder(gc_count_map, n2r=True)

        gc_count_dv = np.zeros((data_vec_len, n_z_maglim), dtype=np.float32)
        for i in range(n_z_maglim):
            # full sky (but only footprint occupied) -> padded data vector
            gc_count_dv[:, i] = maps.map_to_data_vec(
                hp_map=gc_count_map[:, i],
                data_vec_len=data_vec_len,
                corresponding_pix=corresponding_pix_dict["maglim"],
                cutout_pix=patches_pix_dict["maglim"][0],
            )

        if apply_norm:
            pass

        gc_count_dv *= masks_maglim
        gc_count_dv, gc_alms = scales.data_vector_to_smoothed_data_vector(
            gc_count_dv,
            data_vec_pix=data_vec_pix,
            n_side=n_side,
            l_min=conf["analysis"]["scale_cuts"]["clustering"]["l_min"],
            theta_fwhm=conf["analysis"]["scale_cuts"]["clustering"]["theta_fwhm"],
            arcmin=True,
        )
        gc_count_dv *= masks_maglim

    if wl_gamma_map is not None and gc_count_map is not None:
        observation = np.concatenate([wl_kappa_dv, gc_count_dv], axis=-1)
        observation_cls = power_spectra.get_cls(np.concatenate([wl_alms, gc_alms], axis=-1))
    elif wl_gamma_map is not None:
        observation = wl_kappa_dv
        observation_cls = power_spectra.get_cls(wl_alms)
    elif gc_count_map is not None:
        observation = gc_count_dv
        observation_cls = power_spectra.get_cls(gc_alms)

    # go from padded datavector to non-padded patch
    if not with_padding:
        # only keep indices that are in all (per tomographic bin and galaxy sample) masks
        mask_total = np.prod(np.concatenate([masks_metacal, masks_maglim], axis=-1), axis=-1)
        mask_total = mask_total.astype(bool)
        footprint_pix = data_vec_pix[mask_total]

        observation_full_sky = np.zeros((n_pix, observation.shape[-1]), dtype=observation.dtype)
        observation_full_sky[data_vec_pix] = observation
        observation = observation_full_sky[footprint_pix]

    return observation, observation_cls


def forward_model_cosmogrid():
    pass
