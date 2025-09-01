# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

""" 
Created April 2023
Author: Arne Thomsen
"""

import numpy as np

from msfm.utils import files


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


def get_tomo_amplitudes_vectorized(amplitude, exponent, tomo_z, tomo_nz, z0):
    """Parametrization of an effective per redshift bin evolution like in equastions (8) and (9) in DeepLSS
    https://arxiv.org/pdf/2203.09616.pdf. This is suitable both for the intrinsic alignment amplitude and the linear
    galaxy bias. Written by ChatGPT

    Args:
        amplitude (float): Overall amplitude parameter.
        exponent (float): Parameter that determines the strenght of the redshift dependence.
        tomo_z (list): Per redshift bin z value of the distribution.
        tomo_nz (list): Per redshift bin n(z) value of the distribution.
        z0 (float): Pivot redshift.

    Returns:
        list: Per redshift bin amplitude.
    """

    tomo_z_array = np.array(tomo_z)  # shape: (4, 159)
    tomo_nz_array = np.array(tomo_nz)  # shape: (4, 159)

    # Reshape tomo_z_array and tomo_nz_array to have an extra dimension for batch_size
    # This will allow broadcasting with amplitude and exponent
    # New shape will be (batch_size, 4, 159)
    tomo_z_broadcasted = tomo_z_array[np.newaxis, :, :]  # shape: (1, 4, 159)
    tomo_nz_broadcasted = tomo_nz_array[np.newaxis, :, :]  # shape: (1, 4, 159)

    # Expand amplitude and exponent to (batch_size, 1, 1) to broadcast correctly
    amplitude_broadcasted = amplitude[:, np.newaxis, np.newaxis]  # shape: (batch_size, 1, 1)
    exponent_broadcasted = exponent[:, np.newaxis, np.newaxis]  # shape: (batch_size, 1, 1)

    # Calculate the integrand using broadcasting over (batch_size, 4, 159)
    integrand = tomo_nz_broadcasted * ((1 + tomo_z_broadcasted) / (1 + z0)) ** exponent_broadcasted

    # Compute the integrals for each batch over the z dimension (axis=2)
    integrals = np.sum(integrand, axis=2) / np.sum(tomo_nz_broadcasted, axis=2)

    # Multiply by amplitude (broadcasted shape)
    tomo_amplitudes = amplitude_broadcasted[:, :, 0] * integrals  # shape: (batch_size, 4)

    # Return the result as a float32 array
    return tomo_amplitudes.astype(np.float32)


def get_tomo_amplitudes_according_to_config(conf, amplitude, exponent, sample="metacal"):
    tomo_z, tomo_nz = files.load_redshift_distributions(sample, conf)
    z0 = conf["analysis"]["modelling"]["z0"]

    return get_tomo_amplitudes_vectorized(amplitude, exponent, tomo_z, tomo_nz, z0)
