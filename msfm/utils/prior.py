# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created February 2023
Author: Arne Thomsen

Based off https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/kids1000_analysis/probability.py
by Janis Fluri
"""

import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import fsolve

from msfm.utils import files, parameters, logger

LOGGER = logger.get_logger(__file__)


def in_grid_prior(cosmos, conf=None, params=None):
    """Determines whether the elements of the given array of cosmological parameters are contained within the analysis
    prior. This is needed to build a vectorized log posterior.

    Args:
        cosmos (np.ndarray): A 2D array of cosmological parameters with shape (n_cosmos, n_params), where n_params
            has to be in the right ordering (as defined in the config) and n_theta corresponds to n_cosmos.
        conf (str, dict, optional): Config to use, can be either a string to the config.yaml file, the dictionary
            obtained by reading such a file or None, where the default config within the repo is used. Defaults to
            None.
        params (list, optional): List of strings containing "Om", "s8", "Ob", "H0", "ns" and "w0" in the same order as
            within the cosmos array.

    Raises:
        ValueError: If an incompatible type is passed to the conf argument

    Returns:
        in_prior: A 1D boolean array of the shape (n_cosmos,) that specifies whether the values in params are
        contained within the prior.
    """
    conf = files.load_config(conf)
    if params is None:
        params = (
            conf["analysis"]["params"]["cosmo"] + conf["analysis"]["params"]["ia"] + conf["analysis"]["params"]["bg"]
        )

    # make the params 2d
    cosmos = np.atleast_2d(cosmos)

    prior_intervals = parameters.get_prior_intervals(params)

    # check if we are in the prior intervals
    in_prior = np.all(np.logical_and(prior_intervals[:, 0] <= cosmos, cosmos <= prior_intervals[:, 1]), axis=1)

    # simplex in the Om - s8 plane
    try:
        i_Om = params.index("Om")
        i_s8 = params.index("s8")
    except ValueError:
        LOGGER.debug(f"The hull prior is only checked when Om and s8 are included as parameters")
    else:
        hull = Delaunay(conf["analysis"]["grid"]["priors"]["Om_s8_border_points"])

        # check if we are in the hull, shape (n_cosmos,)
        in_hull = hull.find_simplex(cosmos[:, [i_Om, i_s8]]) >= 0

        # what is False will stay false irrespective of the rhs
        in_prior[in_prior] = in_hull[in_prior]

    # w0 threshold
    try:
        i_Om = params.index("Om")
        i_w0 = params.index("w0")
    except ValueError:
        LOGGER.debug(f"The w0 threshold is only checked if Om and w0 are included as parameters")
    else:
        # check if we are above the w0 threshold (same as get_min_w0 with margin = 0.01)
        in_prior[in_prior] = 1.0 / (cosmos[in_prior, i_Om] - 1.0) + 0.01 <= cosmos[in_prior, i_w0]

    return in_prior


def log_posterior(cosmos, log_probs, conf=None, params=None):
    """Vectorized version of the log posterior to be used in the MCMC runs, for example with emcee.

    Args:
        cosmos (np.ndarray): A 2D array of cosmological parameterss with shape (n_cosmos, n_params), where n_params
            has to be in the right ordering (as defined in the config) and n_theta corresponds to n_cosmos.
        log_probs (np.ndarray): Log probabilities associated with the parameters of shape (n_cosmos, 1) or (n_cosmos,).
            These are output values of the Gaussian Process emulator for example.
        conf (str, dict, optional): Config to use, can be either a string to the config.yaml file, the dictionary
            obtained by reading such a file or None, where the default config within the repo is used. Defaults to
            None.

        Example usage:
            log_posterior = lambda X: prior.log_posterior(X, predictor(X), params=params, conf=conf)
            from emcee import EnsembleSampler
            sampler = EnsembleSampler(nwalkers, ndim, log_posterior, vectorize=True)

    Returns:
        np.ndarray: The log posterior values obtained by restricting the emulator's predictions to the prior range.
    """
    # - infinity if outside the pior range, given input probability otherwise
    return np.where(in_grid_prior(cosmos, conf, params), np.squeeze(log_probs), -np.inf)


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
