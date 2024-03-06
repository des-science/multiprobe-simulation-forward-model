"""
Created in September 2023
Author: Arne Thomsen

Tools to handle the calculate power spectra consistent with the forward modeled maps. These are based off 
https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/scripts/human_summaries/eval_summaries
by Janis Fluri.
"""

import numpy as np
import scipy.stats

from msfm.utils import logger, imports, scales

hp = imports.import_healpy(parallel=True)

LOGGER = logger.get_logger(__file__)


def get_alms(maps, nest=True, datapath=None):
    """Gets the alms from a list of maps containing different probes or tomographic bins.

    Args:
        maps (np.ndarray): Array of full sky healpy maps corresponding to the tomographic bins/probes. The shape is
            (n_pix, n_z_bins).
        nest (bool, optional): The ordering of the maps. Defaults to True.
        datapath (str, optional): Path to the healpy pixel_weights for the alm transform. Defaults to None.

    Returns:
        np.ndarray: Array of alms with shape (n_channels, n_alms)
    """

    alms = []
    for i in range(maps.shape[1]):
        map = maps[:, i]

        if nest:
            map = hp.reorder(map, n2r=True)

        alms.append(hp.map2alm(map, use_pixel_weights=True, datapath=datapath))

    alms = np.stack(alms, axis=1)

    return alms


def get_cls(alms, with_cross=True):
    """Calculates the (non-binned) auto- and cross-spectra from an array of tomographic/multi-probe alms. Note that
    no binning is applied here.

    Args:
        alms (np.ndarray): Array of shape (n_alms, n_z_bins) containing alms corresponding to the tomographic bins.
        with_cross (bool, optional): Whether to calculate the cross spectra or auto only. Defaults to True.

    Returns:
        np.ndarray: If cross=False: an array with the same length as alms containing the auto-spectra, otherwise an
            array with length n * (n + 1) / 2 containing all auto and cross spectra ordered as
            11, 12, 13, ..., 1n, 22, 23, ..., 2n, ..., nn, where n = len(alms)
    """

    # get the number of alms
    n_alms = alms.shape[1]

    # get the cls
    cls = []
    for i in range(n_alms):
        for j in range(n_alms):
            if (i == j) or (i < j and with_cross):
                # NOTE there's no sqrt here
                cl = hp.alm2cl(alms1=alms[:, i], alms2=alms[:, j])

                cls.append(cl)

    cls = np.stack(cls, axis=1).astype(np.float32)

    return cls


def get_cl_bins(l_min, l_max, n_bins):
    """Square root spaced bins as Dominik, this helps with the more noisy smaller scales.

    Args:
        l_min (int): Largest scale.
        l_max (int): Smallest scale.
        n_bins (int): Number of bins to average the Cls in.
    """
    return np.linspace(np.sqrt(l_min), np.sqrt(l_max), n_bins, endpoint=True) ** 2


def bin_cls(cls, l_mins, l_maxs, n_bins, n_side=None, with_cross=True, per_cross_binning=True):
    """Take the raw Cls and bin them within a given range of scales. This is done for each cross bin separately,
    always taking the more conservative cut.

    Args:
        cls (np.ndarray): Array of shape (n_ell, n_z) or (n_examples, n_ell, n_z) containing the raw power
            spectra to be binned, where n_z is either the number of tomographic bins or tomographic cross bin
            combinations, consistent with the with_cross argument.
        l_mins (list): List of largest scales, same length as the number of tomographic bins.
        l_maxs (list): List of smallest scales, same length as the number of tomographic bins.
        n_bins (int): Number of bins to average the Cls in.
        with_cross (bool, optional): Whether to calculate the cross spectra or auto only. Defaults to True.
        per_cross_binning (bool, optional): Whether to use the same binning for all cross bins or different binning for
            each cross bin. Defaults to True.

    Returns:
        (np.ndarray, np.ndarray): binned_cls has shape (n_bins-1, n_z) or (n_examples, n_bins-1, n_z) and contains the
            mean Cls in each bin. bin_edges contains the binning details.
    """

    assert (
        cls.ndim == 2 or cls.ndim == 3 or cls.ndim == 4
    ), f"cls has shape {cls.shape}, which is not 2, 3 or 4 dimensional"
    assert len(l_mins) == len(l_maxs), f"l_mins and l_maxs have different lengths: {len(l_mins)} and {len(l_maxs)}"
    n_z = len(l_mins)

    if n_side is not None:
        l_mins = np.array(l_mins)
        l_maxs = np.array(l_maxs)
        l_mins = np.clip(l_mins, 0, 3 * n_side - 1)
        l_maxs = np.clip(l_maxs, 0, 3 * n_side - 1)

    if not per_cross_binning:
        LOGGER.warning(f"Using the same binning for all cross bins, this is currently hardcoded and only experimental")

    # minus indexing to be compatible with both the 2d and 3d case
    n_cross_z = cls.shape[-1]
    n_ell = cls.shape[-2]
    ell = np.arange(n_ell)

    if with_cross:
        assert n_cross_z == n_z * (n_z + 1) // 2
    else:
        assert n_cross_z == n_z

    # translate the two indices to a single one in a list
    cross_l_mins = []
    cross_l_maxs = []
    cross_bins = []
    for i in range(n_z):
        for j in range(n_z):
            if (i == j) or (i < j and with_cross):
                # always conservative for cross bins
                l_min = max(l_mins[i], l_mins[j])
                l_max = min(l_maxs[i], l_maxs[j])

                cross_l_mins.append(l_min)
                cross_l_maxs.append(l_max)

                if per_cross_binning:
                    # different binning for each cross bin
                    bins = get_cl_bins(l_min, l_max, n_bins)
                else:
                    # the same binning for all cross bins NOTE this is hardcoded for now
                    bins = get_cl_bins(30, 1500, n_bins)

                cross_bins.append(bins)

    assert len(cross_bins) == n_cross_z

    four_dim_input = cls.ndim == 4
    if four_dim_input:
        binned_cls_shape = cls.shape[:2] + (n_bins - 1, n_cross_z)
        cls = cls.reshape(-1, n_ell, n_cross_z)

    binned_cls = []
    bin_edges = []
    for i in range(n_cross_z):
        # select a single (cross) tomographic bin
        current_ell = ell
        current_cls = cls[..., i]
        current_bins = cross_bins[i]
        current_l_min = cross_l_mins[i]
        current_l_max = cross_l_maxs[i]

        # smooth in the same way as the maps
        current_cls = scales.cls_to_smoothed_cls(current_cls, l_min=current_l_min, l_max=current_l_max)

        # scipy.stats.binned_statistic includes values outside the bin range in the first/last bin, so those have to be
        # manually removed first
        if per_cross_binning:
            scale_cut = (current_l_min < ell) & (ell < current_l_max)
        else:
            scale_cut = (30 < ell) & (ell < 1500)
        current_cls = current_cls[..., scale_cut]
        current_ell = ell[scale_cut]

        binned = scipy.stats.binned_statistic(current_ell, current_cls, statistic="mean", bins=current_bins)

        binned_cls.append(binned[0])
        bin_edges.append(binned[1])

    # shape (n_bins-1, n_cross_z) or (n_examples, n_bins-1, n_cross_z)
    binned_cls = np.stack(binned_cls, axis=-1)
    if four_dim_input:
        binned_cls = binned_cls.reshape(binned_cls_shape)

    bin_edges = np.stack(bin_edges, axis=-1)
    if cls.ndim == 4:
        pass

    return binned_cls, bin_edges


def run_tfrecords_alm_to_cl(alm_kg, alm_sn_realz, alm_dg, alm_pn_realz):
    """To be used in run_grid_tfrecords.py and run_fiducial_tfrecords.py to compute the (noisy) Cls from the alms.
    Note that no binning is applied here, the Cls are returned in full.

    Args:
        alm_kg (np.ndarray): Shape (n_alms, n_z_bins) containing the lensing signal amls.
        alm_sn_realz (np.ndarray): Shape (n_noise, n_alms, n_z_bins) containing the shape noise amls.
        alm_dg (np.ndarray): Shape (n_alms, n_z_bins) containing the clustering signal amls.
        alm_pn_realz (np.ndarray): Shape (n_noise, n_alms, n_z_bins) containing the Poisson noise amls.

    Returns:
        np.ndarray: Shape (n_noise, n_cls, n_z_cross), where n_z_cross = n_z_bins * (n_z_bins + 1) / 2.
    """

    assert alm_sn_realz.shape[0] == alm_pn_realz.shape[0], f"alm_sn_realz and alm_pn_realz have different lengths"
    n_noise_per_example = alm_sn_realz.shape[0]

    cls = []
    for i_noise in range(n_noise_per_example):
        alms_lensing = alm_kg + alm_sn_realz[i_noise]
        alms_clustering = alm_dg + alm_pn_realz[i_noise]

        # concatenate redshift bins
        alms = np.concatenate((alms_lensing, alms_clustering), axis=1)

        cls.append(get_cls(alms, with_cross=True))

    # shape (n_noise_per_example, n_ell, n_cross_z)
    cls = np.stack(cls, axis=0)

    return cls
