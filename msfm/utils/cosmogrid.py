# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created February 2023
Author: Arne Thomsen

Functions specific to the cosmogrid
"""

import h5py


def get_cosmo_params_info(meta_info_file, simset="grid"):
    """Returns directories on the level of cosmo_000001 and cosmo_delta_H0_p and so on

    Args:
        meta_info_file (str): path to the modified (in permutations_list.ipynb) CosmoGridV1 metainfo file
        simset (str, optional): Either "grid" or "fiducial". Defaults to 'grid'.

    Returns:
        ndarray: List containing the metainfo for all the unique cosmological parameters
    """
    with h5py.File(meta_info_file, "r") as f:
        params_info = f[f"parameters/{simset}"][:]

    return params_info
