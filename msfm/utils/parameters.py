# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created February 2023
Author: Arne Thomsen

Functions to read in the parameter values stored in the config
"""

import numpy as np

from msfm.utils import files, redshift

# Parameters stored in raw CosmoGrid units in the label tables / tfrecords (bary_Mc ~ 1e12-1e15)
# but expressed as log10 in the config priors/fiducials and everywhere downstream (summary
# training, inference). This tuple and raw_to_prior_units() below are the single source of truth
# for the raw->prior-units conversion; the forward model deliberately *stores* raw, so the
# conversion lives here as an offered utility rather than being baked into the write path.
LOG10_PARAMS = ("bary_Mc",)

# A raw Mc is > ~1e12 while a log10(Mc) is ~12, so this threshold cleanly separates the two and
# catches a double conversion (log10 applied to an already-log10'd table).
LOG10_RAW_MIN = 1e10


def raw_to_prior_units(cosmos, params, copy=True):
    """Convert label columns stored in raw CosmoGrid units to the config's prior units.

    Currently this is the raw-Mc -> log10(Mc) conversion for every parameter in LOG10_PARAMS
    that appears in ``params``. A sanity assert guards against converting an already-converted
    table (a log10(Mc) ~ 12 would otherwise be log10'd again into ~1.1).

    Args:
        cosmos (np.ndarray): (..., n_params) label array whose last axis is ordered like ``params``.
        params (sequence[str]): parameter name per column of the last axis.
        copy (bool): if True (default) operate on a copy; otherwise modify ``cosmos`` in place.

    Returns:
        np.ndarray: the converted array (a copy unless copy=False).
    """
    cosmos = np.asarray(cosmos)
    if copy:
        cosmos = cosmos.copy()
    params = list(params)
    for name in LOG10_PARAMS:
        if name in params:
            i = params.index(name)
            assert (
                np.min(cosmos[..., i]) > LOG10_RAW_MIN
            ), f"expected raw {name} labels (> {LOG10_RAW_MIN:g}), got log10 already?"
            cosmos[..., i] = np.log10(cosmos[..., i])
    return cosmos


def get_parameters(params=None, conf=None):
    """Return the list of cosmological parameters. This is meant to handle the default case when params is set to None.
    If params is not none, the value is simply passed through.

    Args:
        params (list, optional): List of strings like ["Om", "s8", "H0", "Ob", "ns", "w0", "Aia"]. Defaults to None,
            then all parameters according to config.yaml are used.
        conf (str, dict, optional): The config, either specified as a str pointing to a file or a dict. Defaults to
            None, then the standard config of this repo is used.

    Returns:
        np.ndarray: shape (n_params,) in the ordering specified by the params list.
    """
    if params is None:
        conf = files.load_config(conf)

        # sobol
        params = conf["analysis"]["params"]["cosmo"].copy()
        if conf["analysis"]["modelling"]["baryonified"]:
            params += conf["analysis"]["params"]["bary"]

        if conf["analysis"]["modelling"]["clustering"]["stochasticity"]:
            params += conf["analysis"]["params"]["stochasticity"]

        # latin hypercube
        params += conf["analysis"]["params"]["ia"]["nla"]
        if conf["analysis"]["modelling"]["lensing"]["extended_nla"]:
            params += conf["analysis"]["params"]["ia"]["tatt"]

        params += conf["analysis"]["params"]["bg"]["linear"]
        if conf["analysis"]["modelling"]["clustering"]["quadratic_biasing"]:
            params += conf["analysis"]["params"]["bg"]["quadratic"]
        try:
            _, sn_bias, _, _ = files.get_shape_noise(conf)
            if sn_bias == "prior":
                params += conf["analysis"]["params"]["sc"]
        except KeyError:
            pass

    return params


def get_prior_intervals(params=None, conf=None):
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
    conf = files.load_config(conf)
    params = get_parameters(params, conf)

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
    conf = files.load_config(conf)
    params = get_parameters(params, conf)

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
    conf = files.load_config(conf)
    params = get_parameters(params, conf)

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
    params = get_parameters(params)

    pert_labels = ["fiducial"]
    for param in params:
        pert_labels.append(f"delta_{param}_m")
        pert_labels.append(f"delta_{param}_p")

    return pert_labels


def get_per_bin_bias_perturbations_dict(conf=None):
    """Returns a dictionary containing the per redshift bin linear galaxy bias values for the fiducial and the
    perturbations, for the per_bin_biasing parametrization (params like ["bg1", "bg2", "bg3", "bg4"]). This is the
    per-bin analog of get_tomo_amplitude_perturbations_dict("bg", conf), which covers power_law_biasing: every
    perturbation label (like "delta_bg2_m") perturbs a single bin while the others stay at their fiducial values.

    Args:
        conf (str, dict, optional): The config, either specified as a str pointing to a file or a dict. Defaults to
            None, then the standard config of this repo is used.

    Returns:
        dict: Keys "fiducial" and "delta_{bgi}_{m,p}" (matching get_fiducial_perturbation_labels), values are the per
            bin amplitude arrays of shape (n_z_bins,).
    """
    conf = files.load_config(conf)

    bg_params = conf["analysis"]["params"]["bg"]["linear"]
    fiducial = np.array([conf["analysis"]["fiducial"][param] for param in bg_params], dtype=np.float32)

    tomo_bias_perturbations_dict = {"fiducial": fiducial}
    for i_bin, param in enumerate(bg_params):
        delta = conf["analysis"]["fiducial"]["perturbations"][param]

        for sign, label in zip((-1.0, 1.0), (f"delta_{param}_m", f"delta_{param}_p")):
            perturbed = fiducial.copy()
            perturbed[i_bin] += sign * delta
            tomo_bias_perturbations_dict[label] = perturbed

    return tomo_bias_perturbations_dict


def get_tomo_amplitude_perturbations_dict(param, conf=None):
    """Returns a dictionary containing the tomographic amplitudes calculated like in redshift.py for the different
    perturbations of the intrinsic alignment or galaxy biasing parameters.

    Args:
        param (str): Has to be either "Aia" or "bg"
        conf (str, dict, optional): The config, either specified as a str pointing to a file or a dict. Defaults to
            None, then the standard config of this repo is used.

    Raises:
        ValueError: When an unknown parameter string is passed.

    Returns:
        dict: Dictionary containing the per bin amplitude values for either Aia or bg and the perturbations.
    """
    conf = files.load_config(conf)

    # redshift
    if param == "Aia":
        sample = "metacal"
    elif param == "bg" or param == "bg2":
        sample = "maglim"
    else:
        raise ValueError(f"param {param} needs to be either 'Aia', 'bg' or 'bg2'")

    # fiducial values
    amplitude = conf["analysis"]["fiducial"][param]
    exponent = conf["analysis"]["fiducial"][f"n_{param}"]

    # perturbations
    delta_amplitude = conf["analysis"]["fiducial"]["perturbations"][param]
    delta_exponent = conf["analysis"]["fiducial"]["perturbations"][f"n_{param}"]

    tomo_amplitude_perturbations_dict = {
        "fiducial": redshift.get_tomo_amplitudes_according_to_config(conf, amplitude, exponent, sample),
        f"delta_{param}_m": redshift.get_tomo_amplitudes_according_to_config(
            conf, amplitude - delta_amplitude, exponent, sample
        ),
        f"delta_{param}_p": redshift.get_tomo_amplitudes_according_to_config(
            conf, amplitude + delta_amplitude, exponent, sample
        ),
        f"delta_n_{param}_m": redshift.get_tomo_amplitudes_according_to_config(
            conf, amplitude, exponent - delta_exponent, sample
        ),
        f"delta_n_{param}_p": redshift.get_tomo_amplitudes_according_to_config(
            conf, amplitude, exponent + delta_exponent, sample
        ),
    }

    return tomo_amplitude_perturbations_dict
