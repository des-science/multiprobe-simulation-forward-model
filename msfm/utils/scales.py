"""
Created on May 2023
Author: Arne Thomsen

Tools to handle the scale cuts/Gaussian smoothing.
"""

import numpy as np
import os, logging

from msfm.utils import files, logger, imports

hp = imports.import_healpy()

LOGGER = logger.get_logger(__file__)


def rad_to_arcmin(theta):
    return theta / np.pi * (180 * 60)


def arcmin_to_rad(theta):
    return theta * np.pi / (60 * 180)


def ell_to_angle(ell, arcmin=False, method="naive"):
    # method like 6.2 of https://academic.oup.com/mnras/article/505/4/5714/6296446
    if method == "naive":
        theta = np.pi / ell

    elif method == "physical":
        theta = 1 / ell

    if arcmin:
        theta = rad_to_arcmin(theta)

    return theta


def angle_to_ell(theta, arcmin=False, method="naive"):
    # method like 6.2 of https://academic.oup.com/mnras/article/505/4/5714/6296446
    if arcmin:
        theta = arcmin_to_rad(theta)

    if method == "naive":
        ell = np.pi / theta

    elif method == "physical":
        ell = 1 / theta

    return ell


def alm_to_smoothed_map(alm, n_side, l_min, l_max=None, theta_max=None, nest=False):
    """Takes in alm coefficients and returns a map that has been smoothed according to l_min and l_max.

    Args:
        alm (np.array): Single spherical harmonics decomposition.
        n_side (int): Healpix nside of the output map.
        l_min (int): Largest scale.
        l_max (int): Smallest scale, specified as an ell.
        theta_max (float): Smallest scale, specified as an angle, which is used as the FWHM of a Gaussian.
        nest (bool, optional): Whether the (full sky) output map should be in NEST ordering.

    Returns:
        np.array: Healpy map of shape (n_pix,)
    """
    assert not (l_max is None and theta_max is None), "Either l_max or theta_max must be specified"
    assert l_max is None or theta_max is None, "Only one of l_max or theta_max can be specified"

    # alm are computed for the standard l_max = 3 * n_side - 1
    l = hp.Alm.getlm(3 * n_side - 1)[0]

    # remove large scales (hard cut)
    alm[l < l_min] = 0.0

    # remove small scales (Gaussian smoothing)
    if l_max is not None:
        theta_max = ell_to_angle(l_max)

    full_map = hp.alm2map(alm, nside=n_side, fwhm=theta_max)

    if nest:
        full_map = hp.reorder(full_map, r2n=True)

    return full_map


def map_to_smoothed_map(full_map, n_side, l_min, l_max, nest=False):
    """Takes in a (multiple) full sky healpy map(s) and returns a (multiple) map(s) that has (have) been smoothed
    according to l_min and l_max. The input can either be a single map, or a stack of multiple tomographic bins along
    axis = 1 = -1.

    Args:
        hp_map (np.array): Full sky healpy map(s) of the appropriate n_side and shape (n_pix,) or (n_pix, n_z_bins).
        l_min (Union[int, list]): Largest scale(s).
        l_max (Union[int, list]): Smallest scale(s).
        n_side (int): Healpix nside of the output map.
        nest (bool, optional): Whether the (full sky) input map is in NEST ordering.

    Returns:
        np.array: Smoothed (full sky) healpy map(s) of shape (n_pix,) or (n_pix, n_z_bins).
    """

    # healpy path
    conf = files.load_config()
    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    hp_datapath = os.path.join(repo_dir, conf["files"]["healpy_data"])

    # multiple tomographic bins along final axis
    if isinstance(l_min, list) and isinstance(l_max, list):
        assert full_map.ndim == 2
        assert len(l_min) == len(l_max) == full_map.shape[1]

        for i_tomo, (current_l_min, current_l_max) in enumerate(zip(l_min, l_max)):
            current_map = full_map[:, i_tomo]
            if nest:
                current_map = hp.reorder(current_map, n2r=True)

            alm = hp.map2alm(
                current_map,
                use_pixel_weights=True,
                datapath=hp_datapath,
            )

            full_map[:, i_tomo] = alm_to_smoothed_map(alm, n_side, current_l_min, current_l_max, nest=nest)

    # single map
    elif isinstance(l_min, int) and isinstance(l_max, int):
        assert full_map.ndim == 1

        if nest:
            full_map = hp.reorder(full_map, n2r=True)

        alm = hp.map2alm(
            full_map,
            use_pixel_weights=True,
            datapath=hp_datapath,
        )

        full_map = alm_to_smoothed_map(alm, n_side, l_min, l_max, nest=nest)

    else:
        raise ValueError(f"Unknown dtype for l_min or l_max")

    return full_map


def data_vector_to_smoothed_data_vector(data_vector, l_min, l_max, n_side, data_vec_pix):
    """Takes in a (multiple) padded data vector(s) and returns a (multiple) data vectors(s) that has (have) been
    smoothed according to l_min and l_max. The input can either be a single map, or a stack of multiple tomographic
    bins along axis = 1 = -1.

    Args:
        data_vector (np.array): Partial sky padded data vector(s) of shape (len(data_vec_pix),) or
            (len(data_vec_pix), n_z_bins).
        l_min (Union[int, list]): Largest scale(s).
        l_max (Union[int, list]): Smallest scale(s).
        n_side (int): Healpix nside of the output map.
        data_vec_pix (np.array): Indices of the (padded) data vector in NEST ordering.

    Returns:
        np.array: Smoothed data vector(s) of shape (len(data_vec_pix),) or (len(data_vec_pix), n_z_bins).
    """

    n_pix = hp.nside2npix(n_side)

    # multiple tomographic bins along final axis
    if isinstance(l_min, list) and isinstance(l_max, list):
        assert data_vector.ndim == 2
        assert len(l_min) == len(l_max) == data_vector.shape[1]

        full_map = np.zeros((n_pix, data_vector.shape[1]), dtype=np.float32)
        full_map[data_vec_pix] = data_vector

    # single map
    elif isinstance(l_min, int) and isinstance(l_max, int):
        assert data_vector.ndim == 1

        full_map = np.zeros(n_pix, dtype=np.float32)
        full_map[data_vec_pix] = data_vector

    else:
        raise ValueError(f"Unknown dtype for l_min or l_max")

    data_vector = map_to_smoothed_map(full_map, n_side, l_min, l_max, nest=True)[data_vec_pix]

    return data_vector


# Gaussian Random Fields ##############################################################################################


def data_vector_to_grf_data_vector(data_vector, n_side, l_min, l_max, data_vec_pix, np_seed):
    """Takes in a (multiple) padded data vector(s) and returns a (multiple) data vectors(s) that has (have) been
    smoothed according to l_min and l_max and transformed to a Gaussian Random Field. This destroys all non-Gaussian
    information and is meant for testing purposes only, to be compared with Cls. The input can either be a single map,
    or a stack of multiple tomographic bins along axis = 1 = -1.

    Args:
        data_vector (np.array): Partial sky padded data vector(s) of shape (len(data_vec_pix),) or
            (len(data_vec_pix), n_z_bins).
        n_side (int): Healpix nside of the output map.
        l_min (Union[int, list]): Largest scale(s).
        l_max (Union[int, list]): Smallest scale(s).
        data_vec_pix (np.array): Indices of the (padded) data vector in NEST ordering.
        np_seed (int): A numpy random seed used in the (intrinsically random) generation alm -> map. It's important
            that this is the same for all tomographic bins and maps that are added later on the GRF level (like signal
            and noise maps, both for weak lensing and galaxy clustering).

    Returns:
        np.array: Smoothed data vector(s) of shape (len(data_vec_pix),) or (len(data_vec_pix), n_z_bins).
    """

    n_pix = hp.nside2npix(n_side)

    # multiple tomographic bins along final axis
    if isinstance(l_min, list) and isinstance(l_max, list):
        assert data_vector.ndim == 2
        assert len(l_min) == len(l_max) == data_vector.shape[1]

        for i_tomo, (current_l_min, current_l_max) in enumerate(zip(l_min, l_max)):
            full_map = np.zeros((n_pix), dtype=np.float32)
            full_map[data_vec_pix] = data_vector[:, i_tomo]
            full_map = hp.reorder(full_map, n2r=True)

            cl = hp.anafast(full_map, alm=False, pol=False, use_pixel_weights=True)

            # remove large scales
            cl[np.arange(0, current_l_min)] = 0

            # remove small scales and make a Gaussian Random Field
            np.random.seed(np_seed)
            grf = hp.synfast(cl, nside=n_side, pol=False, fwhm=ell_to_angle(current_l_max)).astype(np.float32)
            grf = hp.reorder(grf, r2n=True)

            # padding is populated too
            data_vector[:, i_tomo] = grf[data_vec_pix]

    # single map
    elif isinstance(l_min, int) and isinstance(l_max, int):
        assert data_vector.ndim == 1

        full_map = np.zeros(n_pix, dtype=np.float32)
        full_map[data_vec_pix] = data_vector
        full_map = hp.reorder(full_map, n2r=True)

        cl = hp.anafast(full_map, alm=False, pol=False, use_pixel_weights=True)

        # remove large scales
        cl[np.arange(0, current_l_min)] = 0

        # remove small scales and make a Gaussian Random Field
        np.random.seed(np_seed)
        grf = hp.synfast(cl, nside=n_side, pol=False, fwhm=ell_to_angle(current_l_max)).astype(np.float32)
        grf = hp.reorder(grf, r2n=True)

        # padding is populated too
        data_vector = grf[data_vec_pix]

    else:
        raise ValueError(f"Unknown dtype for l_min or l_max")

    return data_vector


def alm_to_grf_map(alm, n_side, l_min, l_max, np_seed):
    """TODO this function has not been tested yet in conjunction with run_datavectors.py

    Take in an alm vector and return a full sky Gaussian Random Field in ring ordering. This is for testing purposes
    only and enables a comparison of the networks to power spectra. Note that it's important that the different
    tomographic bins are all generated according to the same np.random.seed, otherwise the tomographic information
    gets lost too.

    Args:
        alm (np.ndarray): Vector of complex alm coefficients. Only a single tomographic bin at a time is supported
            by this function, unlike some of the above in this file.
        n_side (int): Healpix nside of the output map.
        l_min (Union[int, list]): Largest scale(s).
        l_max (Union[int, list]): Smallest scale(s).
        np_seed (int): A numpy random seed used in the (intrinsically random) generation alm -> map.

    Returns:
        np.array: Smoothed full sky healpy map for a single tomographic bin of shape (n_pix,).
    """
    cl = hp.alm2cl(alm)

    # remove large scales
    cl[np.arange(0, l_min)] = 0

    # remove small scales and make a Gaussian Random Field
    np.random.seed(np_seed)
    grf = hp.synfast(cl, nside=n_side, pol=False, fwhm=ell_to_angle(l_max)).astype(np.float32)

    return grf
