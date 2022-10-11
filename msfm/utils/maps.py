import numpy as np
from numba import njit

from . import logging

LOGGER = logging.get_logger(__file__)

@njit(parallel=False)
def make_normallized_maps(gal_pix, e1, e2, w, n_pix):
    """
    Args:
        gal_pix (np.ndarray): shape (n_gal,), pixel ids of all of the galaxies
        e1 (np.ndarray): shape (n_gal,), ellipticities 1
        e2 (np.ndarray): shape (n_gal,), ellipticities 2
        w (np.ndarray): shape (n_gal,) Metacalibration weight
        n_pix (int): number of pixels in the maps

    Returns:
        e1_map, e2_map, abs_map, w_map, n_map: maps of size n_pix
    """
    assert gal_pix.shape == e1.shape == e2.shape == w.shape

    # maps for accumulation
    e1_map = np.zeros(n_pix)
    e2_map = np.zeros(n_pix)
    abs_map = np.zeros(n_pix)
    w_map = np.zeros(n_pix)
    n_map = np.zeros(n_pix)

    # loop over the whole catalog
    for pix_id, gamma1, gamma2, weight in zip(gal_pix, e1, e2, w):
        e1_map[pix_id] += gamma1*weight
        e2_map[pix_id] += gamma2*weight
        abs_map[pix_id] += weight*np.abs(gamma1 + gamma2*1j)
        w_map[pix_id] += weight
        n_map[pix_id] += 1

    # avoid division by zero
    w_mask = w_map != 0
    
    # normalize
    e1_map[w_mask] /= w_map[w_mask]
    e2_map[w_mask] /= w_map[w_mask]
    abs_map[w_mask] /= w_map[w_mask]

    return e1_map, e2_map, abs_map, w_map, n_map
