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
        with_cross (bool, optional): Whether to calculate the cross spectra or auto only. Defaults to True.

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

                # NOTE there's no sqrt here
                cl = hp.alm2cl(alms1=alms[:, i], alms2=alms[:, j])
                binned_cl = scipy.stats.binned_statistic(np.arange(len(cl)), cl, statistic="mean", bins=bins)[0]

                cls.append(binned_cl)

    cls = np.stack(cls, axis=1).astype(np.float32)

    return cls


def run_tfrecords_alm_to_cl(conf, alm_kg, alm_sn_realz, alm_dg, alm_pn_realz):
    """To be used in run_grid_tfrecords.py and run_fiducial_tfrecords.py

    Args:
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). The relative paths are stored here. Defaults to
            None.
        alm_kg (np.ndarray): Shape (n_alms, n_z_bins) containing the lensing signal amls.
        alm_sn_realz (np.ndarray): Shape (n_noise, n_alms, n_z_bins) containing the shape noise amls.
        alm_dg (np.ndarray): Shape (n_alms, n_z_bins) containing the clustering signal amls.
        alm_pn_realz (np.ndarray): Shape (n_noise, n_alms, n_z_bins) containing the Poisson noise amls.

    Returns:
        np.ndarray: Shape (n_noise, n_cls, n_z_cross), where n_z_cross = n_z_bins * (n_z_bins + 1) / 2.
    """
    assert alm_sn_realz.shape[0] == alm_pn_realz.shape[0]

    n_noise_per_example = alm_sn_realz.shape[0]

    cls = []
    for i_noise in range(n_noise_per_example):
        alms_lensing = alm_kg + alm_sn_realz[i_noise]
        alms_clustering = alm_dg + alm_pn_realz[i_noise]

        alms = np.concatenate((alms_lensing, alms_clustering), axis=1)

        cls.append(
            get_cls(
                alms,
                l_mins=conf["analysis"]["scale_cuts"]["lensing"]["l_min"]
                + conf["analysis"]["scale_cuts"]["clustering"]["l_min"],
                l_maxs=conf["analysis"]["scale_cuts"]["lensing"]["l_max"]
                + conf["analysis"]["scale_cuts"]["clustering"]["l_max"],
                n_bins=conf["analysis"]["power_spectra"]["n_bins"],
            )
        )

    # shape (n_noise_per_example, n_cls, n_cross_bins)?
    cls = np.stack(cls, axis=0)

    return cls
