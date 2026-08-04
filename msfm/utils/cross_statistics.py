# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created December 2023
Author: Arne Thomsen

To be used in conjunction with the results of power_spectra.py and peak_statistics.py
"""

import numpy as np


def get_cross_bin_indices(
    n_z_lensing=4,
    n_z_clustering=4,
    with_lensing=True,
    with_clustering=True,
    with_cross_z=True,
    with_cross_probe=None,
    lenses_before_sources=False,
):
    """Returns a list of indices corresponding to the auto and cross spectra of the selected probes and tomographic
    bins. Note that this assumes that the channels are assumed to be ordered as lensing first, followed by clustering.

    Args:
        n_z_lensing (int, optional): Number of tomographic bins for lensing. Defaults to 4, like for metacal.
        n_z_clustering (int, optional): Number of tomographic bins for clustering. Defaults to 4, like for reduced
            maglim.
        with_lensing (bool, optional): Whether to include include the weak lensing bins. Defaults to True.
        with_clustering (bool, optional): Whether to include include the galaxy clustering bins. Defaults to True.
        with_cross_z (bool or dict, optional): Whether to include the tomographic cross-z bins. A bool applies to both
            probes; a mapping ``{"lensing": bool, "clustering": bool}`` sets each probe independently (missing keys
            default to True). E.g. the DES Y3 3x2pt selection uses ``{"lensing": True, "clustering": False}`` (cosmic
            shear keeps its cross-z bins while galaxy clustering is auto-z only). Defaults to True.
        with_cross_probe (bool, optional): Whether to include the cross probe (galaxy-galaxy lensing) bins. Defaults to
            True.
        lenses_before_sources (bool, optional): When True, only include cross-probe (GGL) pairs where the clustering
            lens bin index is ≤ the lensing source bin index, i.e. the lenses are in front of the sources. For the
            DES Y3 metacal (source) / maglim (lens) tomographic bins this index rule is equivalent to selecting pairs
            with lens mean-z < source mean-z, and keeps 10 of the 16 pairs (lens l, source s), 1-indexed:
            (1,1) (1,2) (1,3) (1,4) (2,2) (2,3) (2,4) (3,3) (3,4) (4,4). Defaults to False.

    Returns:
        list: List of indices corresponding to the auto and cross spectra of the selected probes and tomographic bins,
            that can be used for numpy fancy indexing. The length of this list is n_z_bins * (n_z_bins + 1) / 2, where
            n_z_bins = n_z_lensing + n_z_clustering.
    """

    if with_cross_probe is None:
        with_cross_probe = with_lensing and with_clustering

    # with_cross_z may be a single bool (both probes) or a per-probe mapping.
    if isinstance(with_cross_z, dict):
        cz_lensing = bool(with_cross_z.get("lensing", True))
        cz_clustering = bool(with_cross_z.get("clustering", True))
    else:
        cz_lensing = cz_clustering = bool(with_cross_z)

    # loop over all auto and cross spectra
    index = 0
    lensing_indices = []
    clustering_indices = []
    cross_indices = []
    names = []
    for i in range(n_z_lensing + n_z_clustering):
        for j in range(n_z_lensing + n_z_clustering):
            if i <= j:
                names.append(f"bin_{i}x{j}")

                # lensing only
                if i < n_z_lensing and j < n_z_lensing:
                    if cz_lensing:
                        lensing_indices.append(index)
                    elif i == j:
                        lensing_indices.append(index)

                # clustering only
                elif i >= n_z_lensing and j >= n_z_lensing:
                    if cz_clustering:
                        clustering_indices.append(index)
                    elif i == j:
                        clustering_indices.append(index)

                # cross probe: i is the lensing source bin, j - n_z_lensing is the clustering lens bin
                elif with_cross_probe:
                    if lenses_before_sources and (j - n_z_lensing) > i:
                        pass  # lens bin is behind source bin — skip
                    else:
                        cross_indices.append(index)

                index += 1

    total_indices = []

    if with_lensing:
        total_indices += lensing_indices
    if with_clustering:
        total_indices += clustering_indices
    if with_cross_probe:
        total_indices += cross_indices

    total_indices = np.array(sorted(total_indices))
    names = np.array(names)[total_indices]

    return total_indices, names
