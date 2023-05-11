"""
Created on May 2023
Author: Arne Thomsen

Tools to handle the scale cuts/Gaussian smoothing.
"""

import numpy as np
import os

# from msfm.utils import analysis, logger, maps
from msfm.utils import analysis, logger

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


def alm_to_smoothed_map(alm, l_min, l_max, n_side):
    """Takes in alm coefficients and returns a map that has been smoothed according to l_min and l_max.

    Args:
        alm (np.array): Single spherical harmonics decomposition.
        l_min (int): Largest scale.
        l_max (int): Smallest scale.
        n_side (int): Healpix nside of the output map.

    Returns:
        np.array: Healpy map of shape (n_pix,)
    """
    LOGGER.debug(f"Smoothing according to l_min = {l_min}, l_max = {l_max}")

    # alm are computed for the standard l_max = 3 * n_side - 1
    l = hp.Alm.getlm(3 * n_side - 1)[0]

    # remove large scales (hard cut)
    alm[l < l_min] = 0.0

    # remove small scales (Gaussian smoothing)
    hp_map = hp.alm2map(alm, nside=n_side, fwhm=np.pi / l_max)

    return hp_map


# def map_to_smoothed_map(full_map, l_min, l_max, n_side, nest=False):
def map_to_smoothed_map(full_map, l_min, l_max, nest=False):
    """Takes in a full sky healpy map and returns a map that has been smoothed according to l_min and l_max.

    Args:
        hp_map (np.array): Full sky healpy map of the appropriate n_side and shape (n_pix,) or (n_pix, n_z_bins).
        l_min (Union[int, list]): Largest scale (per tomographic bin).
        l_max (Union[int, list]): Smallest scale (per tomographic bin).
        n_side (int): Healpix nside of the output map.

    Returns:
        np.array: Healpy map of shape (n_pix,) or (n_pix, n_z_bins).
    """

    # healpy path
    conf = analysis.load_config()
    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    hp_datapath = os.path.join(repo_dir, conf["files"]["healpy_data"])

    # multiple tomographic bins
    if isinstance(l_min, list) and isinstance(l_max, list):
        assert full_map.ndim == 2
        assert len(l_min) == len(l_max) == full_map.shape[-1]

        for i_tomo, (current_l_min, current_l_max) in enumerate(zip(l_min, l_max)):
            # ic("pre")
            # ic(full_map[:, i_tomo].min())
            # ic(full_map[:, i_tomo].max())
            # ic(current_l_max)

            # alm = hp.map2alm(
            #     hp_map[:, i_tomo],
            #     use_pixel_weights=True,
            #     datapath=hp_datapath,
            # )

            # hp_map[:, i_tomo] = alm_to_smoothed_map(alm, current_l_min, current_l_max, n_side)

            full_map[:, i_tomo] = hp.smoothing(
                full_map[:, i_tomo],
                fwhm=np.pi / current_l_max,
                use_pixel_weights=True,
                datapath=hp_datapath,
                nest=nest,
            )

            # ic("post")
            # ic(full_map[:, i_tomo].min())
            # ic(full_map[:, i_tomo].max())

    # no tomographic axis
    else:
        assert full_map.ndim == 1

        # alm = hp.map2alm(
        #     hp_map,
        #     use_pixel_weights=True,
        #     datapath=hp_datapath,
        # )

        # hp_map = alm_to_smoothed_map(alm, l_min, l_max, n_side)
        full_map = alm_to_smoothed_map(
            full_map, fwhm=current_l_max, use_pixel_weights=True, datapath=hp_datapath, nest=nest
        )

    return full_map


# def data_vector_to_smoothed_data_vector(data_vector, l_min, l_max, n_side, conf=None, galaxy_sample="maglim"):
def data_vector_to_smoothed_data_vector(data_vector, l_min, l_max, n_side, data_vec_pix, nest=True):
    """Takes in a padded data vector (survey footprint) and returns a map that has been smoothed according to l_min
        and l_max.

    Args:
        data_vector (np.array): Partial sky data vector of shape (len(data_vec_pix),) or (len(data_vec_pix), n_z_bins).
        l_min (Union[int, list]): Largest scale (per tomographic bin).
        l_max (Union[int, list]): Smallest scale (per tomographic bin).
        n_side (int): Healpix nside of the output map.

    Returns:
        np.array: data vector of shape (len(data_vec_pix),) or (len(data_vec_pix), n_z_bins).
    """
    # data_vec_pix, patches_pix_dict, corresponding_pix_dict, _ = analysis.load_pixel_file(conf)

    # multiple tomographic bins
    if isinstance(l_min, list) and isinstance(l_max, list):
        assert data_vector.ndim == 2
        assert len(l_min) == len(l_max) == data_vector.shape[-1]

        full_map = np.zeros((hp.nside2npix(n_side), data_vector.shape[-1]), dtype=np.float32)
        full_map[data_vec_pix] = data_vector

    # no tomographic axis
    else:
        assert data_vector.ndim == 1

        full_map = np.zeros(hp.nside2npix(n_side), dtype=np.float32)
        full_map[data_vec_pix] = data_vector

    data_vector = map_to_smoothed_map(full_map, l_min, l_max, nest=nest)[data_vec_pix]

    # # RING ordering
    # smoothed_map = map_to_smoothed_map(full_map, l_min, l_max, n_side, nest=nest)

    # if galaxy_sample == "metacal":
    #     raise NotImplementedError

    # ic(smoothed_map.shape)
    # ic(corresponding_pix_dict[galaxy_sample].shape)
    # ic(patches_pix_dict[galaxy_sample][0].shape)

    # # NEST ordering
    # data_vector = maps.map_to_data_vec(
    #     smoothed_map,
    #     len(data_vec_pix),
    #     corresponding_pix_dict[galaxy_sample],
    #     patches_pix_dict[galaxy_sample][0],
    #     remove_mean=False,
    # )
    # data_vector = smoothed_map[data_vec_pix]

    return data_vector
