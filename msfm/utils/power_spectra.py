"""
Created in September 2023
Author: Arne Thomsen

Tools to handle the calculate power spectra consistent with the forward modeled maps. These are based off 
https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/scripts/human_summaries/eval_summaries
by Janis Fluri.
"""

import numpy as np
import scipy.stats

from msfm.utils import logger, imports

hp = imports.import_healpy()

LOGGER = logger.get_logger(__file__)


def get_alms(maps, nest=True, datapath=None):
    """Gets the alms from a list of maps containing different probes or tomographic bins.

    Args:
        maps (np.ndarray): Array of full sky healpy maps corresponding to the tomographic bins/probes. The shape is
            (n_channels, n_pix).
        nest (bool, optional): The ordering of the maps. Defaults to True.
        datapath (str, optional): Path to the healpy pixel_weights for the alm transform. Defaults to None.

    Returns:
        np.ndarray: Array of alms with shape (n_channels, n_alms)
    """
    alms = []
    for map in maps:

        if nest:
            map = hp.reorder(map, n2r=True)

        alms.append(hp.map2alm(map, use_pixel_weights=True, datapath=datapath))

    alms = np.stack(alms, axis=1)

    return alms


def get_cl_bins(l_min, l_max, n_bins):
    """Square root bins as Dominik, this helps with the more noisy smaller scales

    Args:
        l_min (int): Largest scale.
        l_max (int): Smallest scale.
        n_bins (int): Number of bins to average the Cls in.
    """
    return np.linspace(np.sqrt(l_min), np.sqrt(l_max), n_bins, endpoint=True) ** 2


def get_cls(alms, l_mins, l_maxs, n_bins, with_cross=True):
    """Calculates the auto- and cross-spectra from a list of alms

    Args:
        alms (np.ndarray): Array of shape (n_alms, n_z_bins) containing alms corresponding to the tomographic bins.
        l_mins (list): List of largest scales, same length as the number of tomographic bins.
        l_maxs (list): List of smallest scales, same length as the number of tomographic bins.
        n_bins (int): Number of bins to average the Cls in.
        cross (bool, optional): Whether to calculate the cross spectra or auto only. Defaults to True.

    Returns:
        np.ndarray: If cross=False: an array with the same length as alms containing the auto-spectra, otherwise an
            array with length n * (n + 1) / 2 containing all auto and cross spectra ordered as
            11, 12, 13, ..., 1n, 22, 23, ..., 2n, ..., nn, where n = len(alms)
    """

    assert alms.shape[1] == len(l_mins) == len(l_maxs)

    # get the number of alms
    n_alms = alms.shape[1]

    # get the cls
    cls = []
    for i in range(n_alms):
        for j in range(n_alms):
            if (i == j) or (i < j and with_cross):
                # always conservative for cross bins
                l_min = max(l_mins[i], l_mins[j])
                l_max = min(l_maxs[i], l_maxs[j])

                bins = get_cl_bins(l_min, l_max, n_bins)

                # square root of alms like Dominik
                cl = hp.alm2cl(alms1=np.sqrt(alms[:, i]), alms2=np.sqrt(alms[:, j]))
                binned_cl = scipy.stats.binned_statistic(np.arange(len(cl)), cl, statistic="mean", bins=bins)[0]

                cls.append(binned_cl)

    cls = np.stack(cls, axis=1)

    return cls


def get_cl_bin_indices(
    n_z_lensing=4, n_z_clustering=4, with_lensing=True, with_clustering=True, with_cross_z=True, with_cross_probe=True
):
    """Returns a list of indices corresponding to the auto and cross spectra of the selected probes and tomographic
    bins. Note that this assumes that the channels are assumed to be ordered as lensing first, followed by clustering.

    Args:
        n_z_lensing (int, optional): Number of tomographic bins for lensing. Defaults to 4, like for metacal.
        n_z_clustering (int, optional): Number of tomographic bins for clustering. Defaults to 4, like for reduced
            maglim.
        with_lensing (bool, optional): Whether to include include the weak lensing bins. Defaults to True.
        with_clustering (bool, optional): Whether to include include the galaxy clustering bins. Defaults to True.
        with_cross_z (bool, optional): Whether to include the tomographic cross bins. Defaults to True.
        with_cross_probe (bool, optional): Whether to include the cross probe bins. Defaults to True.

    Returns:
        list: List of indices corresponding to the auto and cross spectra of the selected probes and tomographic bins,
            that can be used for numpy fancy indexing.
    """
    assert not (
        with_cross_probe and (not with_lensing or not with_clustering)
    ), f"Cross probe correlations are only allowed if both lensing and clustering are considered."
    assert not (
        with_cross_probe and not with_cross_z
    ), f"Cross probe correlations are only allowed if cross z correlations are considered."

    # loop over all auto and cross spectra
    index = 0
    lensing_indices = []
    clustering_indices = []
    combined_indices = []
    names = []
    for i in range(n_z_lensing + n_z_clustering):
        for j in range(n_z_lensing + n_z_clustering):
            if i <= j:
                names.append(f"bin_{i}x{j}")

                # lensing only``
                if i < n_z_lensing and j < n_z_lensing:
                    if with_cross_z:
                        lensing_indices.append(index)
                    elif i == j:
                        lensing_indices.append(index)

                # clustering only
                elif i >= n_z_lensing and j >= n_z_lensing:
                    if with_cross_z:
                        clustering_indices.append(index)
                    elif i == j:
                        clustering_indices.append(index)

                # cross probe
                elif with_cross_probe and with_lensing and with_clustering:
                    combined_indices.append(index)

                index += 1

    total_indices = []

    if with_lensing:
        total_indices += lensing_indices
    if with_clustering:
        total_indices += clustering_indices
    if with_cross_probe:
        total_indices += combined_indices

    names = np.array(names)[total_indices]

    return sorted(total_indices), names
