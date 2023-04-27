# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

""" 
Created April 2023
Author: Arne Thomsen
"""

import numpy as np


def get_tomo_amplitudes(amplitude, exponent, tomo_z, tomo_nz, z0):
    """Parametrization of an effective per redshift bin evolution like in equastions (8) and (9) in DeepLSS
    https://arxiv.org/pdf/2203.09616.pdf. This is suitable both for the intrinsic alignment amplitude and the linear
    galaxy bias.

    Args:
        amplitude (float): Overall amplitude parameter.
        exponent (float): Parameter that determines the strenght of the redshift dependence.
        tomo_z (list): Per redshift bin z value of the distribution.
        tomo_nz (list): Per redshift bin n(z) value of the distribution.
        z0 (float): Pivot redshift.

    Returns:
        list: Per redshift bin amplitude.
    """
    tomo_amplitudes = []
    for z, nz in zip(tomo_z, tomo_nz):
        integrand = nz * ((1 + z) / (1 + z0)) ** exponent
        integral = np.sum(integrand) / np.sum(nz)

        tomo_amplitudes.append(amplitude * integral)

    return np.array(tomo_amplitudes).astype(np.float32)
