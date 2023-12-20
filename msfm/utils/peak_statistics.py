"""
Created in December 2023
Author: Arne Thomsen

Tools to handle the calculate peaks consistent with the forward modeled maps. These are based off 
https://github.com/des-science/y3-combined-peaks
By Virginia Ajani.
"""

import numpy as np
import os

from estats.map import map as estats_map
from estats.summary import summary as estats_summary

from msfm.utils import logger, imports

hp = imports.import_healpy(parallel=True)

LOGGER = logger.get_logger(__file__)


def get_peaks(maps, binning_dir, n_side=512, n_bins=15, smoothing_scales=[0.0], with_cross=True, save_binning=False):
    """Calculates the peaks statistic from tomographic maps, that can also include different probes.

    Args:
        maps (np.ndarray): Input maps of shape (n_pix, n_z_bins), where the tomographic bins may contain different
            probes.
        binning_dir (str): Absolute path to the directory where the binning scheme is stored/loaded.
        n_side (int, optional): HEALPix nside parameter. Defaults to 512.
        n_bins (int, optional): The number of bins used in the summary statistic. Defaults to 15.
        smoothing_scales (list, optional): A list of smoothing scales as the FWHM of a Gaussian in arcmin. Note that
            smoothing is done on the fly, so the input maps are ideally not presmoothed and all smoothing is done
            here. Defaults to [0.0], so no smoothing at all.
        with_cross (bool, optional): Whether to calculate the cross spectra or auto only. Defaults to True.
        save_binning (bool, optional): Whether to save the binning scheme for this example. This is only meant as a
            preprocessing step. For any analysis, the same binning scheme should be used throughout all of the
            examples, so this should be False. Then, the binning information is loaded. Defaults to False.

    Returns:
        np.ndarray: The peaks statistic with shape (n_scales, n_bins, n_z_cross), which is usually flattened into one
            long summary vector.
    """
    n_z_bins = maps.shape[1]

    peaks = []
    for i in range(n_z_bins):
        for j in range(n_z_bins):
            if (i == j) or (i < j and with_cross):
                # get the estat cross map object
                cross_map = estats_map(
                    polarizations="E", kappa_E=[maps[:, i], maps[:, j]], scales=smoothing_scales, NSIDE=n_side
                )

                # compute cross peaks
                stats_cross_sims = cross_map.calc_summary_stats(
                    statistics=["CrossPeaks"], scales=smoothing_scales, trimming=False
                )

                # select 'E-modes'
                cross_peaks = stats_cross_sims["E"]["CrossPeaks"]

                # downbin cross peaks, the value of 1000 is a default and hardcoded
                binner = estats_summary(scales=smoothing_scales, CrossPeaks=1000, CrossPeaks_sliced_bins=n_bins)

                binner.readin_stat_data(
                    cross_peaks, statistic="CrossPeaks", meta_list=["sims", n_bins], parameters=["type", "tomo"]
                )

                # TODO is the location of this line alright?
                binner.generate_binning_scheme(statistics="CrossPeaks", bin=n_bins)

                centers_file = os.path.join(binning_dir, f"bin_centers_{i}x{j}.npy")
                edges_file = os.path.join(binning_dir, f"bin_edges_{i}x{j}.npy")

                if save_binning:
                    bin_edges, bin_centers = binner.get_binning_scheme(statistic="CrossPeaks", bin=n_bins)
                    np.save(centers_file, bin_centers)
                    np.save(edges_file, bin_edges)
                    LOGGER.info(f"Saved binning scheme for {i}x{j} to {centers_file} and {edges_file}")
                else:
                    bin_centers = np.load(centers_file)
                    bin_edges = np.load(edges_file)
                    binner.set_binning_scheme(bin_centers, bin_edges, statistic="CrossPeaks", bin=n_bins)

                binner.downbin_data(statistics=["CrossPeaks"])

                # result
                binned_peaks = binner.get_data("CrossPeaks").reshape(len(smoothing_scales), n_bins)
                peaks.append(binned_peaks)

    # the cross z bin channels come last
    peaks = np.stack(peaks, axis=-1)

    # shape (n_scales, n_bins, n_z_cross)
    return peaks
