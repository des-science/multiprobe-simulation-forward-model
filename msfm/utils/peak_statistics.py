"""
Created in December 2023
Author: Arne Thomsen

Tools to handle the calculate peaks consistent with the forward modeled maps. These are based off 
https://github.com/des-science/y3-combined-peaks
By Virginia Ajani. The two classes from estats by Dominik Zürcher can be found in
https://cosmo-gitlab.phys.ethz.ch/cosmo_public/estats/-/blob/master/estats/map.py
https://cosmo-gitlab.phys.ethz.ch/cosmo_public/estats/-/blob/master/estats/summary.py
"""

import numpy as np
import os, h5py

from estats.map import map as estats_map
from estats.summary import summary as estats_summary

from msfm.utils import logger, imports

hp = imports.import_healpy()

LOGGER = logger.get_logger(__file__)


def get_peaks_bins(binning_file, n_z_bins, with_cross=True):
    """Loads the binning scheme from disk into memory.

    Args:
        binning_file (str): Absolute path to the file where the binning scheme is stored.
        n_z_bins (int): Number of tomographic bins.
        with_cross (bool, optional): Whether to calculate the cross spectra or auto only. Defaults to True.

    Returns:
        dict: Dictionary containing the (cross) tomographic binning information.
    """

    with h5py.File(binning_file, "r") as f:
        # the desired i,j indexing is easiest achieved with a dictionary
        edges = {}
        centers = {}
        fwhm = {}
        for i in range(n_z_bins):
            for j in range(n_z_bins):
                if (i == j) or (i < j and with_cross):
                    edges[f"{i}x{j}"] = f[f"edges/{i}x{j}"][:]
                    centers[f"{i}x{j}"] = f[f"centers/{i}x{j}"][:]
                    fwhm[f"{i}x{j}"] = f[f"theta_fwhm/{i}x{j}"][:]

    return edges, centers, fwhm


def get_peaks(
    maps,
    n_side=512,
    n_bins=15,
    theta_fwhm=[0.0],
    with_cross=True,
    binning_file=None,
    bins_edges=None,
    bins_centers=None,
    bins_fwhms=None,
):
    """Calculates the peaks statistic from tomographic maps, that can also include different probes.

    Args:
        maps (np.ndarray): Input maps of shape (n_pix, n_z_bins), where the tomographic bins may contain different
            probes.
        n_side (int, optional): HEALPix nside parameter. Defaults to 512.
        n_bins (int, optional): The number of bins used in the summary statistic. Defaults to 15.
        binning_dir (str): Absolute path to the directory where the binning scheme is stored/loaded.
        theta_fwhm (list, optional): A list or list of lits (tomographic case) of smoothing scales as the FWHM of a
            Gaussian in arcmin. Note that smoothing is done on the fly, so the input maps are ideally not presmoothed
            and all smoothing is done here. Defaults to [0.0], so no smoothing at all.
        with_cross (bool, optional): Whether to calculate the cross spectra or auto only. Defaults to True.
        binning_file (str, optional): Absolute path to the file where the binning scheme is stored. Defaults to None,
            then the binning scheme is not stored.
        bins_edges (dict, optional): Dictionary containing the bin edges for the cross peaks. Defaults to None.
        bins_centers (dict, optional): Dictionary containing the bin centers for the cross peaks. Defaults to None.
        bins_fwhms (dict, optional): Dictionary containing the smoothing scales for the cross peaks. Defaults to None.
    Returns:
        np.ndarray: The peaks statistic with shape (n_scales, n_bins, n_z_cross), which is usually flattened into one
            long summary vector.
    """

    # possibly for multiple probes
    n_z_bins = maps.shape[1]

    # whether list of lists
    per_tomo_bin_scales = isinstance(theta_fwhm, list) and all(isinstance(sublist, list) for sublist in theta_fwhm)
    if per_tomo_bin_scales:
        assert (
            len(theta_fwhm) == n_z_bins
        ), f"For per tomographic bin smoothing scales, the length of theta_fwhm must be {n_z_bins}."
        assert all(
            len(theta_fwhm[0]) == len(sublist) for sublist in theta_fwhm
        ), f"For per tomographic bin smoothing scales, all bins need the same number of scales."

    peaks = []
    for i in range(n_z_bins):
        for j in range(n_z_bins):
            if (i == j) or (i < j and with_cross):
                if per_tomo_bin_scales:
                    # always be conservative and take the maximum smoothing scale for cross bins
                    current_theta_fwhm = [max(theta) for theta in zip(theta_fwhm[i], theta_fwhm[j])]
                elif isinstance(theta_fwhm, list):
                    current_theta_fwhm = theta_fwhm
                else:
                    raise ValueError("Smoothing scales must be a list of lists or a list of floats.")

                # get the estat cross map object
                cross_map = estats_map(
                    polarizations="E",
                    kappa_E=[maps[:, i], maps[:, j]],
                    scales=current_theta_fwhm,
                    NSIDE=n_side,
                    verbosity=1,
                )

                # compute cross peaks
                cross_stats = cross_map.calc_summary_stats(
                    statistics=["CrossPeaks"], scales=current_theta_fwhm, trimming=False
                )

                # select 'E-modes'
                cross_peaks = cross_stats["E"]["CrossPeaks"]

                # downbin cross peaks, the value of 1000 is a default and hardcoded
                summary = estats_summary(
                    scales=current_theta_fwhm, CrossPeaks=1000, CrossPeaks_sliced_bins=n_bins, verbosity=1
                )
                summary.readin_stat_data(
                    cross_peaks, statistic="CrossPeaks", meta_list=["sims", n_bins], parameters=["type", "tomo"]
                )
                summary.generate_binning_scheme(statistics="CrossPeaks", bin=n_bins)

                # store the binning scheme to disk
                if binning_file is not None:
                    bin_edges, bin_centers = summary.get_binning_scheme(statistic="CrossPeaks", bin=n_bins)

                    with h5py.File(binning_file, "a") as f:
                        f.create_dataset(f"edges/{i}x{j}", data=bin_edges)
                        f.create_dataset(f"centers/{i}x{j}", data=bin_centers)
                        f.create_dataset(f"theta_fwhm/{i}x{j}", data=current_theta_fwhm)

                    LOGGER.info(
                        f"Saved binning scheme for {i}x{j} and theta_fwhm = {current_theta_fwhm} to {binning_file}"
                    )

                # from memory, to be used in conjunction with get_binning_scheme
                elif (bins_edges is not None) and (bins_centers is not None):
                    assert np.all(
                        current_theta_fwhm == bins_fwhms[f"{i}x{j}"]
                    ), f"Smoothing scales for cross bin {i}x{j} have to match."

                    bin_edges = bins_edges[f"{i}x{j}"]
                    bin_centers = bins_centers[f"{i}x{j}"]

                else:
                    raise ValueError(f"either binning_file or bins_centers and bins_edges must be provided.")

                # result
                summary.set_binning_scheme(bin_centers, bin_edges, statistic="CrossPeaks", bin=n_bins)
                summary.downbin_data(statistics=["CrossPeaks"])
                binned_peaks = summary.get_data("CrossPeaks").reshape(len(current_theta_fwhm), n_bins)
                peaks.append(binned_peaks)

    # the cross z bin channels come last
    peaks = np.stack(peaks, axis=-1)

    # shape (n_scales, n_bins, n_z_cross)
    return peaks
