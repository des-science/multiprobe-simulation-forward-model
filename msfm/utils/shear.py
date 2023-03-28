"""
Created on October 2022
Author: Arne Thomsen

Tools to handle the scale cuts, kaiser-squires transformation and multiplicative and additive shear biases.
"""

import numpy as np
import healpy as hp
import tensorflow as tf
import tensorflow_probability as tfp

from msfm.utils import analysis, logger

LOGGER = logger.get_logger(__file__)


def get_kaiser_squires_factors(l):
    """Factors for a spherical Kaiser Squires transformation
    from eq. (11) in https://academic.oup.com/mnras/article/505/3/4626/6287258

    l = hp.Alm.getlm(lmax)[0]
    """
    kappa2gamma_fac = np.where(
        np.logical_and(l != 1, l != 0),
        -np.sqrt(((l + 2.0) * (l - 1)) / ((l + 1) * l)),
        0,
    )
    gamma2kappa_fac = np.where(
        np.logical_and(l != 1, l != 0),
        1 / kappa2gamma_fac,
        0,
    )
    return kappa2gamma_fac, gamma2kappa_fac


def get_l_mask(l):
    """Masks l equal to 0 and 1 (as these are problematic in the KS inversion)

    l = hp.Alm.getlm(lmax)[0]
    """
    return np.where(np.logical_and(l != 1, l != 0), 1.0, 0.0)


def get_m_bias_distribution(conf=None):
    conf = analysis.load_config(conf)

    m_bias_dist = tfp.distributions.MultivariateNormalDiag(
        loc=conf["analysis"]["shear_bias"]["multiplicative"]["mu"],
        scale_diag=conf["analysis"]["shear_bias"]["multiplicative"]["sigma"],
    )

    return m_bias_dist


def mode_removal(gamma1_patch, gamma2_patch, gamma2kappa_fac, l_mask_fac, n_side, hp_datapath=None):
    """Takes in survey patches of gamma maps and puts out survey patches of kappa maps that only contain E-modes

    Args:
        gamma1_patch (np.ndarray): Array of size n_pix, but only the survey patch is populated
        gamma2_patch (np.ndarray): Same
        gamma2kappa_fac (np.ndarray): Kaiser squires conversion factors
        l_mask_fac (np.ndarray): Mask l = 0,1
        n_side (int): Resolution of the map
        hp_datapath (str, optional): Path to a healpy pixel weights file. Defaults to None.

    Returns:
        np.ndarray: Array of size n_pix, but only the survey patch is populated
    """
    # gamma: map -> alm
    _, gamma_alm_E, gamma_alm_B = hp.map2alm(
        [np.zeros_like(gamma1_patch), gamma1_patch, gamma2_patch],
        use_pixel_weights=True,
        datapath=hp_datapath,
    )
    # gamma -> kappa
    kappa_alm = gamma_alm_E * gamma2kappa_fac
    kappa_alm *= l_mask_fac

    # kappa: alm -> map
    kappa_patch = hp.alm2map(kappa_alm, nside=n_side)

    return kappa_patch
