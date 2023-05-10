"""
Created on May 2023
Author: Arne Thomsen

Tools to handle the scale cuts/Gaussian smoothing.
"""

import numpy as np
import healpy as hp
import os

from msfm.utils import analysis


def alm_to_smoothed_map(alm, l_min, l_max, n_side):
    """Takes in alm coefficients and and returns a map that has been smoothed according to l_min and l_max.

    Args:
        alm (np.array): Spherical harmonics decomposition.
        l_min (int): Largest scale.
        l_max (int): Smallest scale.
        n_side (int): Healpix nside of the output map.

    Returns:
        np.array: Healpy map of shape (n_pix,)
    """

    # alm are computed for the standard l_max = 3 * n_side - 1
    l = hp.Alm.getlm(3 * n_side - 1)[0]

    # remove large scales (hard cut)
    alm[l < l_min] = 0.0

    # remove small scales (Gaussian smoothing)
    hp_map = hp.alm2map(alm, nside=n_side, fwhm=np.pi / l_max)

    return hp_map


# def map_to_smoothed_map(hp_map, l_min, l_max, n_side, data_vec_pix=None):
def map_to_smoothed_map(hp_map, l_min, l_max, n_side):
    """Takes in alm coefficients and and returns a map that has been smoothed according to l_min and l_max.

    Args:
        hp_map (np.array): Full sky healpy map of the appropriate n_side and shape (n_pix,), only a single tomographic
            bin.
        l_min (int): Largest scale.
        l_max (int): Smallest scale.
        n_side (int): Healpix nside of the output map.

    Returns:
        np.array: Healpy map of shape (n_pix,)
    """

    # healpy path
    conf = analysis.load_config()
    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    meta_info_file = os.path.join(repo_dir, conf["files"]["meta_info"])
    hp_datapath = os.path.join(repo_dir, conf["files"]["healpy_data"])

    # # handle data vectors (not full map)
    # if hp_map.shape[0] != hp.nside2npix(n_side):
    #     assert hp_map.
    #     temp_map = np.zeros((hp.nside2npix(n_side), hp_map.shape[1]))


    # multiple tomographic bins
    if isinstance(l_min, list) and isinstance(l_max, list):
        assert hp_map.ndim == 2
        assert len(l_min) == len(l_max) == hp_map.shape[-1]

        # # data vector
        # if hp_map.shape[0] != hp.nside2npix(n_side):
        #     temp
        #     hp_map = np.zeros(hp.nside2npix(n_side))
        #     hp_map[data_vec_pix] = current_map

        for i_tomo, (current_l_min, current_l_max) in enumerate(zip(l_min, l_max)):
            alm = hp.map2alm(
                hp_map[..., i_tomo],
                use_pixel_weights=True,
                datapath=hp_datapath,
            )

            hp_map[..., i_tomo] = alm_to_smoothed_map(alm, current_l_min, current_l_max, n_side)

    # no tomographic axis
    else:
        assert hp_map.ndim == 1

        # # data vector
        # if hp_map.shape[0] != hp.nside2npix(n_side):
        #     hp_map = np.zeros(hp.nside2npix(n_side))
        #     hp_map[data_vec_pix] = current_map

        alm = hp.map2alm(
            hp_map,
            use_pixel_weights=True,
            datapath=hp_datapath,
        )

        hp_map = alm_to_smoothed_map(alm, l_min, l_max, n_side)

    return hp_map

def data_vector_to_smoothed_data_vector(data_vector, l_min, l_max, n_side, data_vec_pix):
    """Takes in alm coefficients and and returns a map that has been smoothed according to l_min and l_max.

    Args:
        hp_map (np.array): Full sky healpy map of the appropriate n_side and shape (n_pix,), only a single tomographic
            bin.
        l_min (int): Largest scale.
        l_max (int): Smallest scale.
        n_side (int): Healpix nside of the output map.

    Returns:
        np.array: Healpy map of shape (n_pix,)
    """
    # multiple tomographic bins
    if isinstance(l_min, list) and isinstance(l_max, list):
        assert data_vector.ndim == 2
        assert len(l_min) == len(l_max) == data_vector.shape[-1]

        hp_map = np.zeros((hp.nside2npix(n_side), data_vector.shape[-1]), dtype=np.float32)
        hp_map[data_vec_pix] = data_vector

    # no tomographic axis
    else:
        assert data_vector.ndim == 1

        hp_map = np.zeros(hp.nside2npix(n_side), dtype=np.float32)
        hp_map[data_vec_pix] = data_vector

    data_vector = map_to_smoothed_map(hp_map, l_min, l_max, n_side)[data_vec_pix]

    return data_vector
