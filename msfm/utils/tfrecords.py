""" adapted from https://github.com/tomaszkacprzak/CosmoPointNet/blob/main/CosmoPointNet/utils_tfrecords.py 
by Tomasz Kacprzak"""

import tensorflow as tf
from icecream import ic

# https://towardsdatascience.com/a-practical-guide-to-tfrecords-584536bc786c
# def parse_forward_maps(kg, ia, sn, dg, cosmo, sobol):
def parse_forward_grid(kg, ia, sn, cosmo, sobol):
    """The grid cosmologies contain all of the maps and labels

    Args:
        kg (np.ndarray): shape(n_pix, n_z_bins)
        ia (np.ndarray): shape(n_pix, n_z_bins)
        sn (np.ndarray): shape(n_pix, n_z_bins)
        cosmo (np.ndarray): shape(n_cosmo_params)
        sobol (int): Seed within the Sobol sequence

    Returns:
        tf.train.Example: Example containing all of these tensors
    """
    # assert kg.shape == ia.shape == sn.shape == dg.shape
    assert kg.shape == ia.shape == sn.shape

    data = {
        # tensor shapes
        "n_pix": _int64_feature(kg.shape[0]),
        "n_z_bins": _int64_feature(kg.shape[1]),
        "n_params": _int64_feature(cosmo.shape[0]),
        # lensing, metacal
        "kg": _bytes_feature(tf.io.serialize_tensor(kg)),
        "ia": _bytes_feature(tf.io.serialize_tensor(ia)),
        "sn": _bytes_feature(tf.io.serialize_tensor(sn)),
        # clustering, maglim TODO
        # "dg": _bytes_feature(tf.io.serialize_tensor(dg)),
        # labels
        "cosmo": _bytes_feature(tf.io.serialize_tensor(cosmo)),
        "sobol": _int64_feature(sobol),
    }

    # create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))
    return out


def parse_inverse_grid(element):
    """use the same structure as above"""

    data = {
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
        "sobol": tf.io.FixedLenFeature([], tf.int64),
    }

    content = tf.io.parse_single_example(element, data)

    kg = tf.io.parse_tensor(content["kg"], out_type=tf.float32)
    ia = tf.io.parse_tensor(content["ia"], out_type=tf.float32)
    sn = tf.io.parse_tensor(content["sn"], out_type=tf.float32)
    # dg = tf.io.parse_tensor(content["dg"], out_type=tf.float32)

    kg = tf.reshape(kg, shape=(content["n_pix"], content["n_z_bins"]))
    ia = tf.reshape(ia, shape=(content["n_pix"], content["n_z_bins"]))
    sn = tf.reshape(sn, shape=(content["n_pix"], content["n_z_bins"]))
    # dg = tf.reshape(dg, shape=(content["n_pix"], content["n_z_bins"]))

    cosmo = tf.io.parse_tensor(content["cosmo"], out_type=tf.float32)
    cosmo = tf.reshape(cosmo, shape=(content["n_params"],))

    sobol = content["sobol"]

    # return kg, ia, sn, dg, cosmo, sobol
    return kg, ia, sn, cosmo, sobol


def parse_forward_fiducial(kg_perts, pert_labels, sn_realz, index):
    """The fiducials don't need a label and contain the perturbation for the delta loss with
    n_perts = 2 * n_params + 1

    Args:
        kg_perts (np.ndarray): kappa perturbations of shape(n_perts, n_pix, n_z_bins)
        pert_labels (list): list of strings, defines the dictionary keys
        sn_realz (np.ndarray): shape noise realizations of shape(n_noise, n_pix, n_z_bins)
        index: example index (comes from simulation run and the patch)

    Returns:
        tf.train.Example: Example containing all of these tensors
    """
    assert kg_perts.shape[1:] == sn_realz.shape[1:]
    assert kg_perts.shape[0] == len(pert_labels)

    # define the structure of a single example
    data = {
        # tensor shapes
        "n_pix": _int64_feature(kg_perts.shape[1]),
        "n_z_bins": _int64_feature(kg_perts.shape[2]),
        # label
        "index": _int64_feature(index),
    }

    # kappa perturbations
    for label, kg_pert in zip(pert_labels, kg_perts):
        data[f"kg_{label}"] = _bytes_feature(tf.io.serialize_tensor(kg_pert))

    # noise realizations
    for i, sn in enumerate(sn_realz):
        data[f"sn_{i}"] = _bytes_feature(tf.io.serialize_tensor(sn))

    # TODO dg

    # create an Example, wrapping the single features
    example = tf.train.Example(features=tf.train.Features(feature=data))
    return example


# TODO make this i_noise some tf.variable like Janis suggests?
def parse_inverse_fiducial(example, pert_labels, i_noise=0):
    """use the same structure as above"""

    data = {
        # tensor shapes
        "n_pix": tf.io.FixedLenFeature([], tf.int64),
        "n_z_bins": tf.io.FixedLenFeature([], tf.int64),
        # label
        "index": tf.io.FixedLenFeature([], tf.int64),
    }

    # kappa perturbations
    for label in pert_labels:
        data[f"kg_{label}"] = tf.io.FixedLenFeature([], tf.string)

    # single noise realization
    data[f"sn_{i_noise}"] = tf.io.FixedLenFeature([], tf.string)

    content = tf.io.parse_single_example(example, data)

    # parse the features
    kg_perts = []
    for label in pert_labels:
        kg_pert = tf.io.parse_tensor(content[f"kg_{label}"], out_type=tf.float32)
        kg_pert = tf.reshape(kg_pert, shape=(content["n_pix"], content["n_z_bins"]))
        kg_perts.append(kg_pert)

    kg_perts = tf.stack(kg_perts, axis=0)

    sn = tf.io.parse_tensor(content[f"sn_{i_noise}"], out_type=tf.float32)

    # TODO dg

    index = content["index"]

    return kg_perts, sn, index


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
