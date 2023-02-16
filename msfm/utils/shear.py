"""
Created on October 2022
Author: Arne Thomsen
"""

import numpy as np
import tensorflow_probability as tfp


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


def get_m_bias_distribution(conf):
    m_bias_dist = tfp.distributions.MultivariateNormalDiag(
        loc=conf["analysis"]["shear_bias"]["multiplicative"]["mu"],
        scale_diag=conf["analysis"]["shear_bias"]["multiplicative"]["sigma"],
    )

    return m_bias_dist
