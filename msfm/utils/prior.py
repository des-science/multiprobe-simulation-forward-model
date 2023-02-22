# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created February 2023
Author: Arne Thomsen

Based off https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/kids1000_analysis/probability.py
by Janis Fluri
"""

import os
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import fsolve

from msfm.utils import parameters, analysis


def in_grid_prior(cosmos, conf=None):
    """Determines whether the elements of the given array of cosmological parameters are contained within the analysis
    prior. This is needed to build a vectorized log posterior.

    Args:
        cosmos (np.ndarray): A 2D array of cosmological parameters with shape (n_cosmos, n_params), where n_params
            has to be in the right ordering (as defined in the config) and n_theta corresponds to n_cosmos.
        conf (str, dict, optional): Config to use, can be either a string to the config.yaml file, the dictionary
            obtained by reading such a file or None, where the default config within the repo is used. Defaults to
            None.

    Raises:
        ValueError: If an incompatible type is passed to the conf argument

    Returns:
        in_prior: A 2D boolean array of the same shape as params that specifies whether the values in params are
        contained within the prior.
    """
    conf = analysis.load_config(conf)

    # make the params 2d
    cosmos = np.atleast_2d(cosmos)

    # get the number of params
    n_params = cosmos.shape[1]

    params = conf["analysis"]["params"][:n_params]

    priors = parameters.get_priors(params)
    # priors = np.array(conf["analysis"]["grid"]["prior"]["standard"])

    # check if we are in the prior
    in_prior = np.all(np.logical_and(priors[:, 0] <= cosmos, cosmos <= priors[:, 1]), axis=1)

    # hull of the border points
    hull = Delaunay(conf["analysis"]["grid"]["priors"]["Om_s8_border_points"])

    # check if we are in the hull
    in_prior[in_prior] = hull.find_simplex(cosmos[in_prior, :2]) >= 0

    # check if we are above the w0 threshold (same as get_min_w0 with margin = 0.01)
    #                                            Om                                   w0
    in_prior[in_prior] = 1.0 / (cosmos[in_prior, 0] - 1.0) + 0.01 <= cosmos[in_prior, 5]

    return in_prior[:, np.newaxis]


def log_posterior(cosmos, log_probs, conf=None):
    """Vectorized version of the log posterior to be used in the MCMC runs, for example with emcee.

    Args:
        cosmos (np.ndarray): A 2D array of cosmological parameterss with shape (n_cosmos, n_params), where n_params
            has to be in the right ordering (as defined in the config) and n_theta corresponds to n_cosmos.
        log_probs (np.ndarray): Log probabilities associated with the parameters. These for example come out of the
            Gaussian Process emulator.
        conf (str, dict, optional): Config to use, can be either a string to the config.yaml file, the dictionary
            obtained by reading such a file or None, where the default config within the repo is used. Defaults to
            None.

    Returns:
        np.ndarray: The log posterior values obtained by restricting the emulator's predictions to the prior range.
    """
    # make the params 2d
    cosmos = np.atleast_2d(cosmos)

    # - infinity if outside the pior range, given input probability otherwise
    return np.where(np.squeeze(in_grid_prior(cosmos, conf)), log_probs, -np.inf)


def get_min_w0(Om, margin=0.01):
    """Calculates the minimum possible w0 value given an Om value. The minimum w0 value is calculated with a formula
    from the concept creator and ensures that the "w0 phantom crossing" occurs after z = 0.

    Args:
        Om (float): Omega matter value
        margin (float, optional): Margin to add to the minimum value. Defaults to 0.01.

    Returns:
        float: The minimum w0
    """
    f = lambda w: 1.0 - ((Om - 1.0) / Om * (1.0 + w)) ** (1.0 / (3.0 * w))
    w0 = fsolve(f, -1.05)[0]
    return w0 + margin
