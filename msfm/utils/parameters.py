# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created February 2023
Author: Arne Thomsen

Functions to read in the parameter values stored in the config
"""

import numpy as np

from msfm.utils import analysis


def get_priors(params=None, conf=None):
    """Return the array of priors over the cosmological parameters
    (just the intervals without the additional restrictions)

    Args:
        params (list, optional): List of strings like ["Om", "s8", "H0", "Ob", "ns", "w0", "Aia"]. Defaults to None,
            then all parameters in config.yaml are used.
        conf (str, dict, optional): The config, either specified as a str pointing to a file or a dict. Defaults to
            None, then the standard config of this repo is used.

    Returns:
        np.ndarray: shape (n_params, 2) where [:,0] contains the lower and [:,1] the upper bounds for the parameters
            in the ordering specified by the params list.
    """
    conf = analysis.load_config(conf)

    if params is None:
        params = conf["analysis"]["params"]

    priors = np.array([conf["analysis"]["grid"]["priors"][param] for param in params])

    return priors


def get_fiducials(params=None, conf=None):
    """Return the array of fiducial values over the cosmological parameters

    Args:
        params (list, optional): List of strings like ["Om", "s8", "H0", "Ob", "ns", "w0", "Aia"]. Defaults to None,
            then all parameters in config.yaml are used.
        conf (str, dict, optional): The config, either specified as a str pointing to a file or a dict. Defaults to
            None, then the standard config of this repo is used.

    Returns:
        np.ndarray: shape (n_params,) in the ordering specified by the params list.
    """
    conf = analysis.load_config(conf)

    if params is None:
        params = conf["analysis"]["params"]

    fids = np.array([conf["analysis"]["fiducial"][param] for param in params])

    return fids


def get_fiducial_perturbations(params=None, conf=None):
    """Return the array of fiducial perturbations over the cosmological parameters

    Args:
        params (list, optional): List of strings like ["Om", "s8", "H0", "Ob", "ns", "w0", "Aia"]. Defaults to None,
            then all parameters in config.yaml are used.
        conf (str, dict, optional): The config, either specified as a str pointing to a file or a dict. Defaults to
            None, then the standard config of this repo is used.

    Returns:
        np.ndarray: shape (n_params,) in the ordering specified by the params list.
    """
    conf = analysis.load_config(conf)

    if params is None:
        params = conf["analysis"]["params"]

    perts = np.array([conf["analysis"]["fiducial"]["perturbations"][param] for param in params])

    return perts


def get_fiducial_perturbation_labels(params=None):
    """Returns a list of strings that denote the different perturbations around the fiducial cosmology.

    Args:
        param_labels (list): list of strings with the names of the cosmological parameters like for example
            ["Om", "s8", "H0", "Ob", "ns", "w0", "Aia"]

    Returns:
        pert_labels: list of strings denoting the fiducial perturbations. These are used in the .tfrecord files.
    """
    if params is None:
        conf = analysis.load_config()
        params = conf["analysis"]["params"]

    pert_labels = ["fiducial"]
    for param in params:
        pert_labels.append(f"delta_{param}_m")
        pert_labels.append(f"delta_{param}_p")

    return pert_labels
