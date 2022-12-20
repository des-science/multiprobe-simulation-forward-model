import numpy as np
import os, h5py

def get_parameters_list(meta_info_file, simset='grid'):
    """Returns directories on the level of cosmo_000001 and cosmo_delta_H0_p and so on

    Args:
        meta_info_file (str): path to the modified CosmoGridV1 metainfo file
        simset (str, optional): Either "grid" or "fiducial". Defaults to 'grid'.

    Returns:
        ndarray: List containing the directories of the unique cosmological parameters
    """
    with h5py.File(meta_info_file, "r") as f: 
        simulations = f[f"simulations/{simset}"][:]
        pars_list = simulations["path_par"]

    return np.unique(pars_list)
