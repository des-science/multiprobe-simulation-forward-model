# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

""" 
Created February 2023
Author: Arne Thomsen

This file is based off 
https://github.com/tomaszkacprzak/CosmoPointNet/blob/main/CosmoPointNet/utils_tfrecords.py 
by Tomasz Kacprzak and
https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/kids1000_analysis/data.py
by Janis Fluri and see
https://www.tensorflow.org/tutorials/load_data/tfrecord
https://towardsdatascience.com/a-practical-guide-to-tfrecords-584536bc786c
"""

import warnings
import tensorflow as tf

from msfm.utils import logger

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


def parse_forward_grid(kg, sn_realz, dg, pn_realz, cosmo, i_sobol, i_example):
    """The grid cosmologies contain all of the maps and labels.

    Args:
        kg (np.ndarray): shape(n_pix, n_z_metacal), includes the sum of an original kg and ia map.
        sn_realz (np.ndarray): shape(n_noise, n_pix, n_z_metacal), shape noise consistent with the kg map.
        dg (np.ndarray): shape (n_pix, n_z_maglim), a map of galaxy counts (not just density contrast).
        pn_realz (np.ndarray): shape(n_noise, n_pix, n_z_maglim), poisson noise consistent with the dg map.
        cosmo (np.ndarray): shape(n_params) to be used as a label.
        i_sobol (int): Seed within the Sobol sequence.

    Returns:
        tf.train.Example: Example containing all of these tensors.
    """
    # LOGGER.warning(f"Tracing parse_forward_grid")

    # sn_realz has an additional shape noise axis
    assert kg.shape == sn_realz.shape[1:]
    # the data vector dimension matches (while n_z does not)
    assert kg.shape[0] == dg.shape[0]

    features = {
        # tensor shapes
        "n_pix": _int64_feature(kg.shape[0]),
        "n_z_metacal": _int64_feature(kg.shape[1]),
        "n_z_maglim": _int64_feature(dg.shape[1]),
        "n_params": _int64_feature(cosmo.shape[0]),
        # labels
        "cosmo": _bytes_feature(tf.io.serialize_tensor(cosmo)),
        "i_sobol": _int64_feature(i_sobol),
        "i_example": _int64_feature(i_example),
    }

    # lensing (metacal), shape noise realizations
    for i, sn in enumerate(sn_realz):
        features[f"kg_{i}"] = _bytes_feature(tf.io.serialize_tensor(kg + sn))

    # clustering (maglim), poisson noise realizations
    for i, pn in enumerate(pn_realz):
        features[f"dg_{i}"] = _bytes_feature(tf.io.serialize_tensor(dg + pn))

    # create an Example, wrapping the single features
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example


def parse_inverse_grid(
    serialized_example,
    n_noise=1,
    # shapes
    n_pix=None,
    n_z_metacal=None,
    n_z_maglim=None,
    n_params=None,
    # probes
    with_lensing=True,
    with_clustering=True,
):
    """Use the same structure as in in the forward pass above. Note that n_pix, n_z_bins and n_params have to be passed
    as function arguments to ensure that the function can be converted to a graph.

    Args:
        serialized_example (tf.train.Example.SerializeToString()): The stored data.
        n_noise (int, optional): Number of noise realizations to return, where the noise index always runs from 0 to
            n_noise - 1. Defaults to 1.
        n_pix (int, optional): Fixes the size of the tensors. Defaults to None.
        n_z_metacal (int, optional): Fixes the size of the tensors. Defaults to None.
        n_z_maglim (int, optional): Fixes the size of the tensors. Defaults to None.
        n_params (int, optional): Fixes the size of the tensors. Defaults to None.
        with_lensing (bool, optional): Whether to return the weak lensing maps. Defaults to True.
        with_clustering (bool, optional): Whether to return the galaxy clustering maps. Defaults to True.

    Returns:
        tf.tensors, int: Tensors containing the different fields, the cosmological parameters and indices i_sobol and
            i_example.
    """
    # LOGGER.warning(f"Tracing parse_inverse_grid")

    features = {
        # tensor shapes
        "n_pix": tf.io.FixedLenFeature([], tf.int64),
        "n_params": tf.io.FixedLenFeature([], tf.int64),
        # labels
        "cosmo": tf.io.FixedLenFeature([], tf.string),
        "i_sobol": tf.io.FixedLenFeature([], tf.int64),
        "i_example": tf.io.FixedLenFeature([], tf.int64),
    }

    if with_lensing:
        features["n_z_metacal"] = tf.io.FixedLenFeature([], tf.int64)
        for i in range(n_noise):
            features[f"kg_{i}"] = tf.io.FixedLenFeature([], tf.string)

    if with_clustering:
        features["n_z_maglim"] = tf.io.FixedLenFeature([], tf.int64)
        for i in range(n_noise):
            features[f"dg_{i}"] = tf.io.FixedLenFeature([], tf.string)

    serialized_data = tf.io.parse_single_example(serialized_example, features)

    # output container
    output_data = {}

    cosmo = tf.io.parse_tensor(serialized_data["cosmo"], out_type=tf.float32)
    if n_params is None:
        cosmo = tf.reshape(cosmo, shape=(serialized_data["n_params"],))
    else:
        cosmo = tf.ensure_shape(cosmo, shape=(n_params,))
    output_data["cosmo"] = cosmo

    for i in range(n_noise):
        if with_lensing:
            output_data = _parse_and_reshape_data_vector(
                output_data, serialized_data, f"kg_{i}", f"kg_{i}", n_pix, n_z_metacal, "n_z_metacal"
            )

        if with_clustering:
            output_data = _parse_and_reshape_data_vector(
                output_data, serialized_data, f"dg_{i}", f"dg_{i}", n_pix, n_z_maglim, "n_z_maglim"
            )

    # indices
    output_data["i_sobol"] = serialized_data["i_sobol"]
    output_data["i_example"] = serialized_data["i_example"]

    return output_data


def parse_forward_fiducial(
    cosmo_pert_labels,
    kg_perts,
    dg_perts,
    # lensing
    ia_pert_labels,
    ia_perts,
    sn_realz,
    # clustering
    bg_pert_labels,
    bg_perts,
    pn_realz,
    # label
    i_example,
):
    """The fiducials don't need a label and contain the perturbation for the delta loss with
    n_perts = 2 * n_params + 1

    Args:
        cosmo_pert_labels (list): Dictionary keys of length n_cosmo_perts and string elements. These are the
            cosmological parameters and common to both kg and dg.
        kg_perts (list): Kappa perturbations of length n_perts and elements of shape(n_pix, n_z_metacal).
        dg_perts (list): Delta perturbations of length n_perts and elements of shape(n_pix, n_z_maglim).
        ia_pert_labels (list): Dictionary keys for the intrinsic alignment perturbations, which only affect kg.
        ia_perts (list): Same length as ia_pert_labels, these are the perturbed kg tensors.
        sn_realz (np.ndarray): Shape noise realizations of shape(n_noise, n_pix, n_z_metacal).
        bg_pert_labels (list): Dictionary keys for the galaxy clustering perturbations, which only affect dg.
        bg_perts (list): Same length as bg_pert_labels, these are the perturbed dg tensors.
        pn_realz (np.ndarray): Poisson noise realizations of shape(n_noise, n_pix, n_z_maglim).
        i_example (int): example index (comes from simulation run and the patch), there are
            n_perms_per_cosmo * n_patches.

    Returns:
        tf.train.Example: Example containing all of these tensors.
    """
    # LOGGER.warning(f"Tracing parse_forward_fiducial")

    # the number of perturbations is the same
    assert len(kg_perts) == len(dg_perts) == len(cosmo_pert_labels)
    assert len(ia_pert_labels) == len(ia_perts)
    assert len(bg_pert_labels) == len(bg_perts)

    # the data vector dimension matches (while n_z does not)
    for kg_pert, dg_pert in zip(kg_perts, dg_perts):
        assert kg_pert.shape[0] == dg_pert.shape[0] == sn_realz.shape[1] == pn_realz.shape[1]

    assert len(sn_realz) == len(pn_realz), "the number of noise realizations has to be identical for sn and pn"

    # define the structure of a single example
    features = {
        # tensor shapes
        "n_pix": _int64_feature(kg_perts[0].shape[0]),
        "n_z_metacal": _int64_feature(kg_perts[0].shape[1]),
        "n_z_maglim": _int64_feature(dg_perts[0].shape[1]),
        # label
        "i_example": _int64_feature(i_example),
    }

    # cosmological perturbations (kappa and delta)
    for label, kg_pert, dg_pert in zip(cosmo_pert_labels, kg_perts, dg_perts):
        features[f"kg_{label}"] = _bytes_feature(tf.io.serialize_tensor(kg_pert))
        features[f"dg_{label}"] = _bytes_feature(tf.io.serialize_tensor(dg_pert))

    # intrinsic alignment perturbations (kappa)
    for label, ia_pert in zip(ia_pert_labels, ia_perts):
        features[f"kg_{label}"] = _bytes_feature(tf.io.serialize_tensor(ia_pert))

    # shape noise realizations
    for i, sn in enumerate(sn_realz):
        features[f"sn_{i}"] = _bytes_feature(tf.io.serialize_tensor(sn))

    # galaxy biasing (delta)
    for label, bg_pert in zip(bg_pert_labels, bg_perts):
        features[f"dg_{label}"] = _bytes_feature(tf.io.serialize_tensor(bg_pert))

    # poisson noise realizations
    for i, pn in enumerate(pn_realz):
        features[f"pn_{i}"] = _bytes_feature(tf.io.serialize_tensor(pn))

    # create an Example, wrapping the single features
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example


def parse_inverse_fiducial(
    serialized_example,
    pert_labels,
    n_noise=1,
    # shapes
    n_pix=None,
    n_z_metacal=None,
    n_z_maglim=None,
    # probes
    with_lensing=True,
    with_clustering=True,
):
    """Use the same structure as in in the forward pass above. Note that n_pix and n_z_bins have to be passed as
    arguments to ensure that the function can be converted to a graph.

    Args:
        serialized_example (tf.train.Example.SerializeToString()): The data loaded from the .tfrecord file.
        pert_labels (list): List of strings that contain the labels defining the keys. These include all parameters,
            so cosmological and astrophysics (intrinsic alignment and galaxy clustering).
        n_noise (int, optional): Number of noise realizations to return, where the noise index always runs from 0 to
            n_noise - 1. Defaults to 1.
        n_pix (int, optional): Fixes the size of the tensors. Defaults to None.
        n_z_metacal (int, optional): Fixes the size of the tensors. Defaults to None.
        n_z_maglim (int, optional): Fixes the size of the tensors. Defaults to None.
        with_lensing (bool, optional): Whether the weak lensing maps should be returned or not. Defaults to True.
        with_clustering (bool, optional): Whether the galaxy clustering maps should be returned or not. Defaults to
            True.

    Returns:
        dict, int: Dictionary of datavectors (fiducial, perturbations and shape noise) and the patch index (i_example).
    """
    # LOGGER.warning(f"Tracing parse_inverse_fiducial")

    features = {
        # tensor shapes, not recommended as reshaping with respect to them leads to a None shape in tf.function
        "n_pix": tf.io.FixedLenFeature([], tf.int64),
        "n_z_metacal": tf.io.FixedLenFeature([], tf.int64),
        "n_z_maglim": tf.io.FixedLenFeature([], tf.int64),
        # label
        "i_example": tf.io.FixedLenFeature([], tf.int64),
    }

    # all perturbation parameters
    for label in pert_labels:
        # kappa: cosmological + intrinsic alignment parameters
        if with_lensing and (not "bg" in label):
            features[f"kg_{label}"] = tf.io.FixedLenFeature([], tf.string)

        # delta: cosmological + galaxy clustering parameters
        if with_clustering and (not "Aia" in label):
            features[f"dg_{label}"] = tf.io.FixedLenFeature([], tf.string)

    # all desired noise realizations
    for i in range(n_noise):
        if with_lensing:
            # shape noise
            features[f"sn_{i}"] = tf.io.FixedLenFeature([], tf.string)

        if with_clustering:
            # poisson noise
            features[f"pn_{i}"] = tf.io.FixedLenFeature([], tf.string)

    serialized_data = tf.io.parse_single_example(serialized_example, features)

    # output container
    output_data = {}

    # all perturbation parameters
    for label in pert_labels:
        # kappa: cosmological + intrinsic alignment parameters
        if with_lensing and (not "bg" in label):
            output_data = _parse_and_reshape_data_vector(
                output_data, serialized_data, f"kg_{label}", f"kg_{label}", n_pix, n_z_metacal, "n_z_metacal"
            )

        # delta: cosmological + galaxy clustering parameters
        if with_clustering and (not "Aia" in label):
            output_data = _parse_and_reshape_data_vector(
                output_data, serialized_data, f"dg_{label}", f"dg_{label}", n_pix, n_z_maglim, "n_z_maglim"
            )

    # all desired noise realizations
    for i in range(n_noise):
        # shape noise
        if with_lensing:
            output_data = _parse_and_reshape_data_vector(
                output_data, serialized_data, f"sn_{i}", f"sn_{i}", n_pix, n_z_metacal, "n_z_metacal"
            )

        # poisson noise
        if with_clustering:
            output_data = _parse_and_reshape_data_vector(
                output_data, serialized_data, f"pn_{i}", f"pn_{i}", n_pix, n_z_maglim, "n_z_maglim"
            )

    # indices
    output_data["i_example"] = serialized_data["i_example"]

    return output_data


# helper functions ####################################################################################################


def _parse_and_reshape_data_vector(out_dict, serialized_example, key_in, key_out, n_pix, n_z_bins, n_z_bins_label):
    tensor = tf.io.parse_tensor(serialized_example[key_in], out_type=tf.float32)

    if (n_pix is None) or (n_z_bins is None):
        # reshape allows for None shapes within the graph, but is slower
        tensor = tf.reshape(tensor, shape=(serialized_example["n_pix"], serialized_example[n_z_bins_label]))
    else:
        # tf.ensure_shape fixes the shape inside the graph
        tensor = tf.ensure_shape(tensor, shape=(n_pix, n_z_bins))

    out_dict[key_out] = tensor

    return out_dict


# features ############################################################################################################

# https://www.tensorflow.org/tutorials/load_data/tfrecord#data_types_for_tftrainexample
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
