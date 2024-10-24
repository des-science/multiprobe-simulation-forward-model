# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created May 2024
Author: Arne Thomsen

Utilities to forward model (mock) observations to be consistent with the CosmoGrid maps.
"""

import os, h5py
import numpy as np
from msfm.utils import (
    files,
    logger,
    lensing,
    imports,
    scales,
    maps,
    power_spectra,
    filenames,
    redshift,
    clustering,
    lensing,
)
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
            l_max=conf["analysis"]["scale_cuts"]["lensing"]["l_max"],
            theta_fwhm=conf["analysis"]["scale_cuts"]["lensing"]["theta_fwhm"],
            arcmin=True,
            mask=masks_metacal,
            hard_cut=conf["analysis"]["scale_cuts"]["hard_cut"],
            conf=conf,
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
                corresponding_pix=corresponding_pix_dict["maglim"][i],
                cutout_pix=patches_pix_dict["maglim"][i][0],
            )

        if apply_norm:
            pass

        gc_count_dv *= masks_maglim
        gc_count_dv, gc_alms = scales.data_vector_to_smoothed_data_vector(
            gc_count_dv,
            data_vec_pix=data_vec_pix,
            n_side=n_side,
            l_min=conf["analysis"]["scale_cuts"]["clustering"]["l_min"],
            l_max=conf["analysis"]["scale_cuts"]["clustering"]["l_max"],
            theta_fwhm=conf["analysis"]["scale_cuts"]["clustering"]["theta_fwhm"],
            arcmin=True,
            mask=masks_maglim,
            hard_cut=conf["analysis"]["scale_cuts"]["hard_cut"],
            conf=conf,
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
    else:
        raise ValueError("At least one of wl_gamma or gc_count must be provided.")

    if with_padding:
        return observation, observation_cls, data_vec_pix
    else:
        # only keep indices that are in all (per tomographic bin and galaxy sample) masks
        mask_total = np.prod(np.concatenate([masks_metacal, masks_maglim], axis=-1), axis=-1)
        mask_total = mask_total.astype(bool)
        footprint_pix = data_vec_pix[mask_total]

        observation_full_sky = np.zeros((n_pix, observation.shape[-1]), dtype=observation.dtype)
        observation_full_sky[data_vec_pix] = observation
        observation = observation_full_sky[footprint_pix]

        return observation, observation_cls, footprint_pix


def forward_model_cosmogrid(
    map_dir,
    conf=None,
    noisy=False,
    # lensing
    with_lensing=True,
    tomo_Aia=None,
    bta=None,
    # clustering
    with_clustering=True,
    tomo_bg=None,
    tomo_qbg=None,
):
    """Take a full-sky CosmoGrid maps as they are projected with UFalcon and transform them into fiducial probe maps
    that are in the same format as the (synthetic) observations like a gamma map for weak lensing, no smoothing, etc.
    The steps here are the same what is implemented in the run_grid/fiducial_postprocessing.py files. The later steps
    of that pipeline are implemented in forward_model_observation_map.
    This function is for example useful for the benchmark simulations.

    Args:
        map_dir (str): The directory where the full-sky CosmoGrid map is stored in the .h5 format.
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). Defaults to None.
        with_lensing (bool, optional): Whether to include the weak lensing map. Defaults to True.
        with_clustering (bool, optional): Whether to include the galaxy clustering map. Defaults to True.
        noisy (bool, optional): Whether to generate shape and poisson noise or return noiseless maps. Defaults to
            False.

    Returns:
        (np.ndarray, np.ndarray): Weak lensing and galaxy clustering full-sky maps of shape (n_pix, n_z, 2) and
            (n_pix, n_z) respectively.
    """

    conf = files.load_config(conf)

    # constants
    n_side = conf["analysis"]["n_side"]
    n_pix = conf["analysis"]["n_pix"]
    data_vec_pix, patches_pix_dict, _, _ = files.load_pixel_file(conf)
    z0 = conf["analysis"]["modelling"]["z0"]

    map_file = filenames.get_filename_full_maps(map_dir, with_bary=conf["analysis"]["modelling"]["baryonified"])
    with h5py.File(map_file, "r") as f:
        if with_lensing:
            maglim_mask = files.get_tomo_dv_masks(conf)["maglim"]
            kappa2gamma_fac, _, _ = lensing.get_kaiser_squires_factors(3 * n_side - 1)
            metacal_bins = conf["survey"]["metacal"]["z_bins"]

            file_dir = os.path.dirname(__file__)
            repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
            hp_datapath = os.path.join(repo_dir, conf["files"]["healpy_data"])

            kg = []
            ia = []
            ds = []
            dg = []
            for z_bin in metacal_bins:
                kg.append(hp.ud_grade(f[f"map/kg/{z_bin}"], n_side))
                ia.append(hp.ud_grade(f[f"map/ia/{z_bin}"], n_side))
                ia.append(hp.ud_grade(f[f"map/ds/{z_bin}"], n_side))
                if noisy:
                    dg.append(hp.ud_grade(f[f"map/dg/{z_bin}"], n_side))
            kg = np.stack(kg, axis=-1)
            ia = np.stack(ia, axis=-1)
            ds = np.stack(ds, axis=-1)
            if noisy:
                dg = np.stack(dg, axis=-1)

            # create the noiseless fiducial map
            extended_nla = conf["analysis"]["modelling"]["lensing"]["extended_nla"] if bta is None else True

            if tomo_Aia is None:
                Aia = conf["analysis"]["fiducial"]["Aia"]
                n_Aia = conf["analysis"]["fiducial"]["n_Aia"]
                tomo_z_metacal, tomo_nz_metacal = files.load_redshift_distributions("metacal", conf)
                tomo_Aia = redshift.get_tomo_amplitudes(Aia, n_Aia, tomo_z_metacal, tomo_nz_metacal, z0)
                LOGGER.info(f"Using Aia={Aia} from the config")
            else:
                LOGGER.info(f"Using Aia={tomo_Aia} from the function call")

            if bta is None and extended_nla:
                bta = conf["analysis"]["fiducial"]["bta"]
                LOGGER.info(f"Using bta={bta} from the config")
            else:
                LOGGER.info(f"Using bta={bta} from the function call")

            if extended_nla:
                wl_kappa_map = kg + tomo_Aia * (ia + bta * ds)
                LOGGER.info("Using delta-NLA")
            else:
                wl_kappa_map = kg + tomo_Aia * ia
                LOGGER.info("Using standard NLA")

            gamma1 = []
            gamma2 = []
            for i in range(wl_kappa_map.shape[-1]):
                patch_pix = patches_pix_dict["metacal"][i][0]

                # kappa -> gamma (full sky)
                kappa_alm = hp.map2alm(
                    wl_kappa_map[:, i],
                    use_pixel_weights=True,
                    datapath=hp_datapath,
                )

                gamma_alm = kappa_alm * kappa2gamma_fac
                _, gamma1_full, gamma2_full = hp.alm2map(
                    [np.zeros_like(gamma_alm), gamma_alm, np.zeros_like(gamma_alm)], nside=n_side
                )

                if noisy:
                    import tensorflow as tf
                    import tensorflow_probability as tfp

                    tomo_bias = conf["survey"]["metacal"]["galaxy_bias"]
                    tomo_n_gal = np.array(conf["survey"]["metacal"]["n_gal"]) * hp.nside2pixarea(n_side, degrees=True)
                    tomo_gamma_cat, _ = files.load_noise_file(conf)

                    dg = (dg - np.mean(dg, axis=0)) / np.mean(dg, axis=0)
                    counts_map = clustering.galaxy_density_to_count(
                        tomo_n_gal, dg, tomo_bias, systematics_map=None
                    ).astype(int)

                    with tf.device("/CPU:0"):
                        counts = counts_map[patch_pix, i]

                        # create joint distribution, as this is faster than random indexing
                        gamma_abs = tf.math.abs(tomo_gamma_cat[i][:, 0] + 1j * tomo_gamma_cat[i][:, 1])
                        w = tomo_gamma_cat[i][:, 2]
                        cat_dist = tfp.distributions.Empirical(
                            samples=tf.stack([gamma_abs, w], axis=-1), event_ndims=1
                        )

                        gamma1_noise, gamma2_noise = lensing.noise_gen(counts, cat_dist, n_noise_per_example=1)
                        gamma1_noise = gamma1_noise[:, 0]
                        gamma2_noise = gamma2_noise[:, 0]
                else:
                    gamma1_noise = 0
                    gamma2_noise = 0

                gamma1_patch = np.zeros(n_pix, dtype=np.float32)
                gamma1_patch[patch_pix] = gamma1_full[patch_pix] + gamma1_noise

                gamma2_patch = np.zeros(n_pix, dtype=np.float32)
                gamma2_patch[patch_pix] = gamma2_full[patch_pix] + gamma2_noise

                gamma1.append(gamma1_patch)
                gamma2.append(gamma2_patch)

            gamma1 = np.stack(gamma1, axis=-1)
            gamma2 = np.stack(gamma2, axis=-1)

            wl_gamma_patch = np.stack([gamma1, gamma2], axis=-1)
        else:
            wl_gamma_patch = None

        if with_clustering:
            patch_pix = patches_pix_dict["maglim"][0]
            maglim_mask = files.get_tomo_dv_masks(conf)["maglim"]

            maglim_bins = conf["survey"]["maglim"]["z_bins"]
            tomo_n_gal_maglim = np.array(conf["survey"]["maglim"]["n_gal"]) * hp.nside2pixarea(n_side, degrees=True)

            dg = []
            for z_bin in maglim_bins:
                dg.append(hp.ud_grade(f[f"map/dg/{z_bin}"], n_side))
            dg = np.stack(dg, axis=-1)

            # cut out the footprint
            dg_patch = np.zeros_like(dg)
            dg_patch[patch_pix] = dg[patch_pix]

            # subtract and divide by mean (within the patch)
            dg_patch[patch_pix] = (dg_patch[patch_pix] - np.mean(dg_patch[patch_pix])) / np.mean(dg_patch[patch_pix])

            dg_patch = maps.tomographic_reorder(dg_patch, r2n=True)
            dg_dv = dg_patch[data_vec_pix]

            # density contrast to count
            if tomo_bg is None:
                if conf["analysis"]["modelling"]["clustering"]["power_law_biasing"]:
                    bg = conf["analysis"]["fiducial"]["bg"]
                    n_bg = conf["analysis"]["fiducial"]["n_bg"]
                    tomo_z_maglim, tomo_nz_maglim = files.load_redshift_distributions("maglim", conf)
                    tomo_bg = redshift.get_tomo_amplitudes(bg, n_bg, tomo_z_maglim, tomo_nz_maglim, z0)
                elif conf["analysis"]["modelling"]["clustering"]["per_bin_biasing"]:
                    tomo_bg = np.array(
                        [conf["analysis"]["fiducial"][f"bg{i}"] for i in range(1, len(maglim_bins) + 1)]
                    )
                LOGGER.info(f"Using bg={tomo_bg} from the config")
            else:
                LOGGER.info(f"Using bg={tomo_bg} from the function call")

            if tomo_qbg is None:
                if conf["analysis"]["modelling"]["clustering"]["quadratic_biasing"]:
                    if conf["analysis"]["modelling"]["clustering"]["power_law_biasing"]:
                        bg = conf["analysis"]["fiducial"]["qbg"]
                        n_bg = conf["analysis"]["fiducial"]["n_qbg"]
                        tomo_z_maglim, tomo_nz_maglim = files.load_redshift_distributions("maglim", conf)
                        tomo_qbg = redshift.get_tomo_amplitudes(bg, n_bg, tomo_z_maglim, tomo_nz_maglim, z0)
                    elif conf["analysis"]["modelling"]["clustering"]["per_bin_biasing"]:
                        tomo_qbg = np.array(
                            [conf["analysis"]["fiducial"][f"qbg{i}"] for i in range(1, len(maglim_bins) + 1)]
                        )
                    LOGGER.info(f"Using qbg={tomo_qbg} from the config")
                    qdg_dv = dg_dv**2 * np.sign(dg_dv)
                else:
                    tomo_qbg = None
                    qdg_dv = None
                    LOGGER.info("No quadratic biasing")
            else:
                LOGGER.info(f"Using qbg={tomo_qbg} from the function call")
                qdg_dv = dg_dv**2 * np.sign(dg_dv)

            gc_count_dv = clustering.galaxy_density_to_count(
                tomo_n_gal_maglim,
                dg_dv,
                tomo_bg,
                qdg_dv,
                tomo_qbg,
                mask=maglim_mask,
                nest=True,
            )

            if noisy:
                gc_count_dv += clustering.galaxy_count_to_noise(gc_count_dv, n_noise=1)[0]

            gc_count_patch = np.zeros((n_pix, gc_count_dv.shape[-1]))
            gc_count_patch[data_vec_pix] = gc_count_dv
            gc_count_patch = maps.tomographic_reorder(gc_count_patch, n2r=True)
        else:
            gc_count_patch = None

        return wl_gamma_patch, gc_count_patch
