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

from msfm.utils.input_output import read_yaml


def grid_prior(cosmos, conf=None):
    """Calculates the CosmoGridV1 prior in the ordering specified in the config on a set of given params

    Args:
        cosmos (np.ndarray): A 2D array of cosmological parameterss with shape (n_cosmos, n_params), where n_params
            has to be in the right ordering and n_theta corresponds to n_cosmos.
        conf (str, dict, optional): Config to use, can be either a string to the config.yaml file, the dictionary
            obtained by reading such a file or None, where the default config within the repo is used. Defaults to
            None.

    Raises:
        ValueError: If an incompatible type is passed to the conf argument

    Returns:
        in_prior: A 2D boolean array of the same shape as params that specifies whether the values in params are
        contained within the prior.
    """
    if conf is None:
        conf = read_yaml(os.path.abspath("../../configs/config.yaml"))
    elif isinstance(conf, str):
        conf = read_yaml(conf)
    elif isinstance(conf, dict):
        pass
    else:
        raise ValueError(f"conf {conf} must be None, a str specifying the path to the .yaml file, or a dict.")

    # make the params 2d
    cosmos = np.atleast_2d(cosmos)

    # get the number of params
    n_params = cosmos.shape[1]

    prior = np.array(conf["analysis"]["grid"]["prior"]["standard"])

    # check if we are in the prior
    in_prior = np.all(np.logical_and(prior[:n_params, 0] <= cosmos, cosmos <= prior[:n_params, 1]), axis=1)

    # hull of the border points
    hull = Delaunay(conf["analysis"]["grid"]["prior"]["Om_s8_border_points"])

    # check if we are in the hull
    in_prior[in_prior] = hull.find_simplex(cosmos[in_prior, :2]) >= 0

    # check if we are above the w0 threshold (same as get_min_w0 with margin = 0.01)
    #                                            Om                                   w0
    in_prior[in_prior] = 1.0 / (cosmos[in_prior, 0] - 1.0) + 0.01 <= cosmos[in_prior, 5]

    return in_prior[:, np.newaxis]


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
