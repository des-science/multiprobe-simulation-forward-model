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
from icecream import ic

from msfm.utils import logger

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)

# def parse_forward_maps(kg, ia, sn, dg, cosmo, i_sobol):
def parse_forward_grid(kg, ia, sn_realz, cosmo, i_sobol):
    """The grid cosmologies contain all of the maps and labels

    Args:
        kg (np.ndarray): shape(n_pix, n_z_bins)
        ia (np.ndarray): shape(n_pix, n_z_bins)
        sn_realz (np.ndarray): shape(n_noise, n_pix, n_z_bins)
        cosmo (np.ndarray): shape(n_params)
        i_sobol (int): Seed within the Sobol sequence

    Returns:
        tf.train.Example: Example containing all of these tensors
    """
    # assert kg.shape == ia.shape == sn.shape == dg.shape
    assert kg.shape == ia.shape == sn_realz.shape[1:]

    features = {
        # tensor shapes
        "n_pix": _int64_feature(kg.shape[0]),
        "n_z_bins": _int64_feature(kg.shape[1]),
        "n_params": _int64_feature(cosmo.shape[0]),
        # lensing, metacal
        "kg": _bytes_feature(tf.io.serialize_tensor(kg)),
        "ia": _bytes_feature(tf.io.serialize_tensor(ia)),
        # clustering, maglim TODO
        # "dg": _bytes_feature(tf.io.serialize_tensor(dg)),
        # labels
        "cosmo": _bytes_feature(tf.io.serialize_tensor(cosmo)),
        "i_sobol": _int64_feature(i_sobol),
    }

    # shape noise realizations
    for i, sn in enumerate(sn_realz):
        features[f"sn_{i}"] = _bytes_feature(tf.io.serialize_tensor(sn))

    # create an Example, wrapping the single features
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example


def parse_inverse_grid(serialized_example, i_noise=0, n_pix=None, n_z_bins=None, n_params=None):
    """Use the same structure as in in the forward pass above. Note that n_pix, n_z_bins and n_params have to be passed 
    as function arguments to ensure that the function can be converted to a graph.

    Args:
        serialized_example (tf.train.Example.SerializeToString()): The data
        n_pix (int, optional): Fixes the size of the tensors. Defaults to None.
        n_z_bins (int, optional): Fixes the size of the tensors. Defaults to None.
        n_params (int, optional): Fixes the size of the tensors. Defaults to None.

    Returns:
        tf.tensors, int: Tensors containing the different fields, the cosmological parameters and an sobol index label
    """
    LOGGER.warning(f"Tracing parse_inverse_grid")

    features = {
        # tensor shapes
        "n_pix": tf.io.FixedLenFeature([], tf.int64),
        "n_z_bins": tf.io.FixedLenFeature([], tf.int64),
        "n_params": tf.io.FixedLenFeature([], tf.int64),
        # lensing, metacal
        "kg": tf.io.FixedLenFeature([], tf.string),
        "ia": tf.io.FixedLenFeature([], tf.string),
        "sn": tf.io.FixedLenFeature([], tf.string),
        # clustering, maglim TODO
        # "dg": tf.io.FixedLenFeature([], tf.string),
        # labels
        "cosmo": tf.io.FixedLenFeature([], tf.string),
        "i_sobol": tf.io.FixedLenFeature([], tf.int64),
    }

    data = tf.io.parse_single_example(serialized_example, features)

    kg = tf.io.parse_tensor(data["kg"], out_type=tf.float32)
    ia = tf.io.parse_tensor(data["ia"], out_type=tf.float32)
    sn = tf.io.parse_tensor(data[f"sn_{i_noise}"], out_type=tf.float32)

    # dg = tf.io.parse_tensor(content["dg"], out_type=tf.float32)
    cosmo = tf.io.parse_tensor(data["cosmo"], out_type=tf.float32)

    # defining the shapes like this works too, but is slower than when they are passed as function arguments
    if n_pix is None or n_z_bins is None or n_params is None:
        if n_pix is None:
            n_pix = data["n_pix"]
        if n_z_bins is None:
            n_z_bins = data["n_z_bins"]
        if n_params is None:
            n_params = data["n_params"]

        # only reshape even works with None shapes
        kg = tf.reshape(kg, shape=(n_pix, n_z_bins))
        ia = tf.reshape(ia, shape=(n_pix, n_z_bins))
        sn = tf.reshape(sn, shape=(n_pix, n_z_bins))
        # dg = tf.reshape(dg, shape=(content["n_pix"], content["n_z_bins"]))
        cosmo = tf.reshape(cosmo, shape=(n_params,))

    # tf.ensure_shape fixes the shape inside the graph
    kg = tf.ensure_shape(kg, shape=(n_pix, n_z_bins))
    ia = tf.ensure_shape(ia, shape=(n_pix, n_z_bins))
    sn = tf.ensure_shape(sn, shape=(n_pix, n_z_bins))
    # dg = tf.ensure_shape(dg, shape=(n_pix, n_z_bins))
    cosmo = tf.ensure_shape(cosmo, shape=(n_params,))

    i_sobol = data["i_sobol"]

    # return kg, ia, sn, dg, cosmo, i_sobol
    return kg, ia, sn, cosmo, i_sobol


def parse_forward_fiducial(kg_perts, pert_labels, sn_realz, index):
    """The fiducials don't need a label and contain the perturbation for the delta loss with
    n_perts = 2 * n_params + 1

    Args:
        kg_perts (np.ndarray): kappa perturbations of shape(n_perts, n_pix, n_z_bins).
        pert_labels (list): list of strings, defines the dictionary keys.
        sn_realz (np.ndarray): shape noise realizations of shape(n_noise, n_pix, n_z_bins).
        index: example index (comes from simulation run and the patch).

    Returns:
        tf.train.Example: Example containing all of these tensors.
    """
    assert kg_perts.shape[1:] == sn_realz.shape[1:]
    assert kg_perts.shape[0] == len(pert_labels)

    # define the structure of a single example
    features = {
        # tensor shapes
        "n_pix": _int64_feature(kg_perts.shape[1]),
        "n_z_bins": _int64_feature(kg_perts.shape[2]),
        # label
        "index": _int64_feature(index),
    }

    # kappa perturbations
    for label, kg_pert in zip(pert_labels, kg_perts):
        features[f"kg_{label}"] = _bytes_feature(tf.io.serialize_tensor(kg_pert))

    # shape noise realizations
    for i, sn in enumerate(sn_realz):
        features[f"sn_{i}"] = _bytes_feature(tf.io.serialize_tensor(sn))

    # TODO dg

    # create an Example, wrapping the single features
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example


def parse_inverse_fiducial(serialized_example, pert_labels, i_noise=0, n_pix=None, n_z_bins=None):
    """Use the same structure as in in the forward pass above. Note that n_pix and n_z_bins have to be passed as
    arguments to ensure that the function can be converted to a graph.


    Args:
        serialized_example (tf.train.Example.SerializeToString()): The data.
        pert_labels (list): List of strings that contain the labels defining the keys.
        i_noise (int, optional): Index to choose the noise realization to return. Defaults to 0.
        n_pix (int, optional): Fixes the size of the tensors. Defaults to None.
        n_z_bins (int, optional): Fixes the size of the tensors. Defaults to None.

    Returns:
        dict, int: Dictionary of datavectors (fiducial, perturbations and shape noise) and the patch index.
    """
    LOGGER.warning(f"Tracing parse_inverse_fiducial")

    features = {
        # tensor shapes, these aren't used because reshaping with respect to them leads to a None shape in tf.function
        "n_pix": tf.io.FixedLenFeature([], tf.int64),
        "n_z_bins": tf.io.FixedLenFeature([], tf.int64),
        # label
        "index": tf.io.FixedLenFeature([], tf.int64),
    }

    # kappa perturbations
    for label in pert_labels:
        features[f"kg_{label}"] = tf.io.FixedLenFeature([], tf.string)

    # single noise realization
    features[f"sn_{i_noise}"] = tf.io.FixedLenFeature([], tf.string)

    data = tf.io.parse_single_example(serialized_example, features)
    data_vectors = {}

    # parse the perturbations
    for label in pert_labels:
        key = f"kg_{label}"
        kg_pert = tf.io.parse_tensor(data[key], out_type=tf.float32)
        kg_pert = tf.ensure_shape(kg_pert, shape=(n_pix, n_z_bins))
        data_vectors[key] = kg_pert

    sn = tf.io.parse_tensor(data[f"sn_{i_noise}"], out_type=tf.float32)
    sn = tf.ensure_shape(sn, shape=(n_pix, n_z_bins))
    data_vectors[f"sn"] = sn

    # TODO dg

    index = data["index"]

    return data_vectors, index


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
