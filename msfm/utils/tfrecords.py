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


def parse_forward_grid(kg, sn_realz, dg, cosmo, i_sobol):
    """The grid cosmologies contain all of the maps and labels.

    Args:
        kg (np.ndarray): shape(n_pix, n_z_bins), includes the sum of an original kg and ia map.
        sn_realz (np.ndarray): shape(n_noise, n_pix, n_z_bins), consistent with the kg map.
        dg (np.ndarray): shape (n_pix, n_z_maglim), a map of galaxy counts (not just density contrast).
        cosmo (np.ndarray): shape(n_params) can be used as a label.
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
        # lensing, metacal
        "kg": _bytes_feature(tf.io.serialize_tensor(kg)),
        # clustering, maglim
        "dg": _bytes_feature(tf.io.serialize_tensor(dg)),
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


def parse_inverse_grid(serialized_example, i_noise=0, n_pix=None, n_z_metacal=None, n_z_maglim=None, n_params=None):
    """Use the same structure as in in the forward pass above. Note that n_pix, n_z_bins and n_params have to be passed
    as function arguments to ensure that the function can be converted to a graph.

    Args:
        serialized_example (tf.train.Example.SerializeToString()): The stored data.
        i_noise (int, optional): Noise index that determines which of the stored shape noise realization to use.
        n_pix (int, optional): Fixes the size of the tensors. Defaults to None.
        n_z_metacal (int, optional): Fixes the size of the tensors. Defaults to None.
        n_z_maglim (int, optional): Fixes the size of the tensors. Defaults to None.
        n_params (int, optional): Fixes the size of the tensors. Defaults to None.

    Returns:
        tf.tensors, int: Tensors containing the different fields, the cosmological parameters and an sobol index label
            (i_sobol, i_noise)
    """
    # LOGGER.warning(f"Tracing parse_inverse_grid")

    features = {
        # tensor shapes
        "n_pix": tf.io.FixedLenFeature([], tf.int64),
        "n_z_metacal": tf.io.FixedLenFeature([], tf.int64),
        "n_z_maglim": tf.io.FixedLenFeature([], tf.int64),
        "n_params": tf.io.FixedLenFeature([], tf.int64),
        # lensing, metacal
        "kg": tf.io.FixedLenFeature([], tf.string),
        f"sn_{i_noise}": tf.io.FixedLenFeature([], tf.string),
        # clustering, maglim
        "dg": tf.io.FixedLenFeature([], tf.string),
        # labels
        "cosmo": tf.io.FixedLenFeature([], tf.string),
        "i_sobol": tf.io.FixedLenFeature([], tf.int64),
    }

    data = tf.io.parse_single_example(serialized_example, features)

    kg = tf.io.parse_tensor(data["kg"], out_type=tf.float32)
    sn = tf.io.parse_tensor(data[f"sn_{i_noise}"], out_type=tf.float32)
    dg = tf.io.parse_tensor(data["dg"], out_type=tf.int16)
    cosmo = tf.io.parse_tensor(data["cosmo"], out_type=tf.float32)

    # defining the shapes like this works too, but is slower than when they are passed as function arguments
    if (n_pix is None) or (n_z_metacal is None) or (n_z_maglim is None) or (n_params is None):
        kg = tf.reshape(kg, shape=(data["n_pix"], data["n_z_metacal"]))
        sn = tf.reshape(sn, shape=(data["n_pix"], data["n_z_metacal"]))
        dg = tf.reshape(dg, shape=(data["n_pix"], data["n_z_maglim"]))
        cosmo = tf.reshape(cosmo, shape=(data["n_params"],))
    # tf.ensure_shape fixes the shape inside the graph
    else:
        kg = tf.ensure_shape(kg, shape=(n_pix, n_z_metacal))
        sn = tf.ensure_shape(sn, shape=(n_pix, n_z_metacal))
        dg = tf.ensure_shape(dg, shape=(n_pix, n_z_maglim))
        cosmo = tf.ensure_shape(cosmo, shape=(n_params,))

    index = (data["i_sobol"], i_noise)

    return kg, sn, dg, cosmo, index


def parse_forward_fiducial(pert_labels, kg_perts, dg_perts, ia_pert_labels, ia_perts, sn_realz, i_example):
    """The fiducials don't need a label and contain the perturbation for the delta loss with
    n_perts = 2 * n_params + 1

    Args:
        pert_labels (list): Dictionary keys of length n_perts and string elements. These are the cosmological
            parameters and common to both kg and dg.
        kg_perts (list): Kappa perturbations of length n_perts and elements of shape(n_pix, n_z_metacal).
        dg_perts (list): Delta perturbations of length n_perts and elements of shape(n_pix, n_z_maglim).
        ia_pert_labels (list): Dictionary keys for the intrinsic alignment perturbations, which only affect kg.
        ia_perts (list): Same length as ia_pert_labels, these are the perturbed kg tensors.
        sn_realz (np.ndarray): Shape noise realizations of shape(n_noise, n_pix, n_z_metacal).
        i_example (int): example index (comes from simulation run and the patch), there are
            n_perms_per_cosmo * n_patches.

    Returns:
        tf.train.Example: Example containing all of these tensors.
    """

    # the number of perturbations is the same
    assert len(kg_perts) == len(dg_perts) == len(pert_labels)

    # the data vector dimension matches (while n_z does not)
    for kg_pert, dg_pert in zip(kg_perts, dg_perts):
        assert kg_pert.shape[0] == dg_pert.shape[0] == sn_realz.shape[1]

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
    for label, kg_pert, dg_pert in zip(pert_labels, kg_perts, dg_perts):
        features[f"kg_{label}"] = _bytes_feature(tf.io.serialize_tensor(kg_pert))
        features[f"dg_{label}"] = _bytes_feature(tf.io.serialize_tensor(dg_pert))

    # intrinsic alignment perturbations (kappa)
    for label, ia_pert in zip(ia_pert_labels, ia_perts):
        features[f"kg_{label}"] = _bytes_feature(tf.io.serialize_tensor(ia_pert))

    # shape noise realizations
    for i, sn in enumerate(sn_realz):
        features[f"sn_{i}"] = _bytes_feature(tf.io.serialize_tensor(sn))

    # create an Example, wrapping the single features
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example


def parse_inverse_fiducial(serialized_example, pert_labels, i_noise=0, n_pix=None, n_z_metacal=None, n_z_maglim=None):
    """Use the same structure as in in the forward pass above. Note that n_pix and n_z_bins have to be passed as
    arguments to ensure that the function can be converted to a graph.


    Args:
        serialized_example (tf.train.Example.SerializeToString()): The data.
        pert_labels (list): List of strings that contain the labels defining the keys.
        i_noise (int, optional): Index to choose the noise realization to return. Defaults to 0.
        n_pix (int, optional): Fixes the size of the tensors. Defaults to None.
        n_z_metacal (int, optional): Fixes the size of the tensors. Defaults to None.
        n_z_maglim (int, optional): Fixes the size of the tensors. Defaults to None.

    Returns:
        dict, int: Dictionary of datavectors (fiducial, perturbations and shape noise) and the patch index, consisting
            of (i_example, i_noise).
    """
    LOGGER.warning(f"Tracing parse_inverse_fiducial")

    features = {
        # tensor shapes, not recommended as reshaping with respect to them leads to a None shape in tf.function
        "n_pix": tf.io.FixedLenFeature([], tf.int64),
        "n_z_metacal": tf.io.FixedLenFeature([], tf.int64),
        "n_z_maglim": tf.io.FixedLenFeature([], tf.int64),
        # label
        "i_example": tf.io.FixedLenFeature([], tf.int64),
    }

    # cosmological perturbations (kappa and delta)
    for label in pert_labels:
        features[f"kg_{label}"] = tf.io.FixedLenFeature([], tf.string)

        # no intrinsic alignment perturbations for delta
        if not "Aia" in label:
            features[f"dg_{label}"] = tf.io.FixedLenFeature([], tf.string)

    # single shape noise realization
    features[f"sn_{i_noise}"] = tf.io.FixedLenFeature([], tf.string)

    data = tf.io.parse_single_example(serialized_example, features)

    # output container
    data_vectors = {}

    # parse the cosmological perturbations
    for map_type, n_z_bins, n_z_bins_str in zip(
        ["kg", "dg"], [n_z_metacal, n_z_maglim], ["n_z_metacal", "n_z_maglim"]
    ):
        for label in pert_labels:
            # intrinsic alignment perturbations are only stored for kappa
            if not ((map_type == "dg") and ("Aia" in label)):
                key = f"{map_type}_{label}"
                pert = tf.io.parse_tensor(data[key], out_type=tf.float32)

                # reshape allows for None shapes within the graph, but is slower
                if (n_pix is None) or (n_z_metacal is None) or (n_z_maglim is None):
                    pert = tf.reshape(pert, shape=(data["n_pix"], data[n_z_bins_str]))
                # tf.ensure_shape fixes the shape inside the graph
                else:
                    pert = tf.ensure_shape(pert, shape=(n_pix, n_z_bins))

                data_vectors[key] = pert

    # parse the shape noise separately
    sn = tf.io.parse_tensor(data[f"sn_{i_noise}"], out_type=tf.float32)
    if (n_pix is None) or (n_z_metacal is None):
        sn = tf.reshape(sn, shape=(data["n_pix"], data["n_z_metacal"]))
    else:
        sn = tf.ensure_shape(sn, shape=(n_pix, n_z_metacal))
    data_vectors[f"sn"] = sn

    index = (data["i_example"], i_noise)

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
