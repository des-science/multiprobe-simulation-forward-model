"""
Created on May 2023
Author: Arne Thomsen

Tools to handle the scale cuts/Gaussian smoothing.
"""

import numpy as np
import os, logging

from msfm.utils import files, logger

LOGGER = logger.get_logger(__file__)

# set the environmental variable OMP_NUM_THREADS to the number of logical processors for healpy parallelixation
try:
    n_cpus = len(os.sched_getaffinity(0))
except AttributeError:
    LOGGER.debug(f"os.sched_getaffinity is not available on this system, use os.cpu_count() instead")
    n_cpus = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(n_cpus)
LOGGER.info(f"Setting up healpy to run on {n_cpus} CPUs")

import healpy as hp

hp_LOGGER = logging.getLogger("healpy")
hp_LOGGER.disabled = True


def alm_to_smoothed_map(alm, l_min, l_max, n_side, nest=False):
    """Takes in alm coefficients and returns a map that has been smoothed according to l_min and l_max.

    Args:
        alm (np.array): Single spherical harmonics decomposition.
        l_min (int): Largest scale.
        l_max (int): Smallest scale.
        n_side (int): Healpix nside of the output map.
        nest (bool, optional): Whether the (full sky) output map should be in NEST ordering.

    Returns:
        np.array: Healpy map of shape (n_pix,)
    """
    LOGGER.debug(f"Smoothing according to l_min = {l_min}, l_max = {l_max}")

    # alm are computed for the standard l_max = 3 * n_side - 1
    l = hp.Alm.getlm(3 * n_side - 1)[0]

    # remove large scales (hard cut)
    alm[l < l_min] = 0.0

    # remove small scales (Gaussian smoothing)
    full_map = hp.alm2map(alm, nside=n_side, fwhm=np.pi / l_max)

    if nest:
        full_map = hp.reorder(full_map, r2n=True)

    return full_map


def map_to_smoothed_map(full_map, l_min, l_max, n_side, nest=False):
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

            full_map[:, i_tomo] = alm_to_smoothed_map(alm, current_l_min, current_l_max, n_side, nest=nest)

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

        full_map = alm_to_smoothed_map(alm, l_min, l_max, n_side, nest=nest)

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
        nest (bool, optional): Whether the (full sky) input map is in NEST ordering.

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

    data_vector = map_to_smoothed_map(full_map, l_min, l_max, n_side, nest=True)[data_vec_pix]

    return data_vector


# Gaussian Random Fields ##############################################################################################


def data_vector_to_grf_data_vector(data_vector, l_min, l_max, n_side, data_vec_pix, np_seed=None):
    """Takes in a (multiple) padded data vector(s) and returns a (multiple) data vectors(s) that has (have) been
    smoothed according to l_min and l_max and transformed to a Gaussian Random Field. This destroys all non-Gaussian
    information and is meant for testing purposes only, to be compared with Cls. The input can either be a single map,
    or a stack of multiple tomographic bins along axis = 1 = -1.

    Args:
        data_vector (np.array): Partial sky padded data vector(s) of shape (len(data_vec_pix),) or
            (len(data_vec_pix), n_z_bins).
        l_min (Union[int, list]): Largest scale(s).
        l_max (Union[int, list]): Smallest scale(s).
        n_side (int): Healpix nside of the output map.
        nest (bool, optional): Whether the (full sky) input map is in NEST ordering.

    Returns:
        np.array: Smoothed data vector(s) of shape (len(data_vec_pix),) or (len(data_vec_pix), n_z_bins).
    """

    n_pix = hp.nside2npix(n_side)

    # multiple tomographic bins along final axis
    if isinstance(l_min, list) and isinstance(l_max, list):
        assert data_vector.ndim == 2
        assert len(l_min) == len(l_max) == data_vector.shape[1]

        # same seed for all tomographic bins
        if np_seed is None:
            np_seed = int(abs(data_vector[10000, 0] * 100 + 100))

        for i_tomo, (current_l_min, current_l_max) in enumerate(zip(l_min, l_max)):
            full_map = np.zeros((n_pix), dtype=np.float32)
            full_map[data_vec_pix] = data_vector[:, i_tomo]
            full_map = hp.reorder(full_map, n2r=True)

            cl = hp.anafast(full_map, alm=False, pol=False, use_pixel_weights=True)

            # remove large scales
            cl[np.arange(0, current_l_min)] = 0

            # remove small scales and make a Gaussian Random Field
            np.random.seed(np_seed)
            grf = hp.synfast(cl, nside=n_side, pol=False, fwhm=np.pi / current_l_max).astype(np.float32)
            grf = hp.reorder(grf, r2n=True)

            # padding is populated too
            data_vector[:, i_tomo] = grf[data_vec_pix]

    # single map
    elif isinstance(l_min, int) and isinstance(l_max, int):
        assert data_vector.ndim == 1
        if np_seed is None:
            np_seed = int(abs(data_vector[10000] * 100 + 100))

        full_map = np.zeros(n_pix, dtype=np.float32)
        full_map[data_vec_pix] = data_vector
        full_map = hp.reorder(full_map, n2r=True)

        cl = hp.anafast(full_map, alm=False, pol=False, use_pixel_weights=True)

        # remove large scales
        cl[np.arange(0, current_l_min)] = 0

        # remove small scales and make a Gaussian Random Field
        np.random.seed(np_seed)
        grf = hp.synfast(cl, nside=n_side, pol=False, fwhm=np.pi / current_l_max).astype(np.float32)
        grf = hp.reorder(grf, r2n=True)

        # padding is populated too
        data_vector = grf[data_vec_pix]

    else:
        raise ValueError(f"Unknown dtype for l_min or l_max")

    return data_vector


def alm_to_grf_map(alm, l_min, l_max, n_side, np_seed):
    """TODO this function has not been tested yet in conjunction with run_datavectors.py

    Take in an alm vector and return a full sky Gaussian Random Field in ring ordering. This is for testing purposes
    only and enables a comparison of the networks to power spectra. Note that it's important that the different
    tomographic bins are all generated according to the same np.random.seed, otherwise the tomographic information
    gets lost too.

    Args:
        alm (np.ndarray): Vector of complex alm coefficients. Only a single tomographic bin at a time is supported
            by this function, unlike some of the above in this file.
        l_min (Union[int, list]): Largest scale(s).
        l_max (Union[int, list]): Smallest scale(s).
        n_side (int): Healpix nside of the output map.
        np_seed (int): A numpy random seed used in the (intrinsically random) generation alm -> map.

    Returns:
        np.array: Smoothed full sky healpy map for a single tomographic bin of shape (n_pix,).
    """
    cl = hp.alm2cl(alm)

    # remove large scales
    cl[np.arange(0, l_min)] = 0

    # remove small scales and make a Gaussian Random Field
    np.random.seed(np_seed)
    grf = hp.synfast(cl, nside=n_side, pol=False, fwhm=np.pi / l_max).astype(np.float32)

    return grf
