""" adapted from https://github.com/tomaszkacprzak/CosmoPointNet/blob/main/CosmoPointNet/utils_tfrecords.py 
by Tomasz Kacprzak"""

import tensorflow as tf

# https://towardsdatascience.com/a-practical-guide-to-tfrecords-584536bc786c
# def parse_forward_maps(kg, ia, sn, dg, cosmo, sobol):
def parse_forward_maps(kg, ia, sn, cosmo, sobol):
    """define the dictionary -- the structure -- of a single example"""

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


def parse_inverse_maps(element):
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

def parse_forward_fiducial(kg, ia, sn):
    """ The fiducials don't need a label and contain the perturbation for the delta loss with
    n_perts = 2 * n_params + 1

    Args:
        kg_perts (np.ndarray): shape(n_perts, n_pix, n_z_bins)
        ia_perts (_type_): shape(n_perts, n_pix, n_z_bins)
        sn_perts (_type_): shape(n_perts, n_pix, n_z_bins)

    Returns:
        _type_: _description_
    """
    # assert kg.shape == ia.shape == sn.shape == dg.shape
    assert kg.shape == ia.shape == sn.shape

    # define the structure of a single example
    data = {
        # tensor shapes
        "n_perts": _int64_feature(kg.shape[0]),
        "n_pix": _int64_feature(kg.shape[1]),
        "n_z_bins": _int64_feature(kg.shape[2]),
        # lensing, metacal
        "kg": _bytes_feature(tf.io.serialize_tensor(kg)),
        "ia": _bytes_feature(tf.io.serialize_tensor(ia)),
        "sn": _bytes_feature(tf.io.serialize_tensor(sn)),
        # clustering, maglim TODO
        # "dg": _bytes_feature(tf.io.serialize_tensor(dg)),
    }

    # create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))
    return out

def parse_inverse_fiducial(element):
    """use the same structure as above"""

    data = {
        # tensor shapes
        "n_perts": tf.io.FixedLenFeature([], tf.int64),
        "n_pix": tf.io.FixedLenFeature([], tf.int64),
        "n_z_bins": tf.io.FixedLenFeature([], tf.int64),
        # lensing, metacal
        "kg": tf.io.FixedLenFeature([], tf.string),
        "ia": tf.io.FixedLenFeature([], tf.string),
        "sn": tf.io.FixedLenFeature([], tf.string),
        # clustering, maglim TODO
        # "dg": tf.io.FixedLenFeature([], tf.string),
    }

    content = tf.io.parse_single_example(element, data)

    kg = tf.io.parse_tensor(content["kg"], out_type=tf.float32)
    ia = tf.io.parse_tensor(content["ia"], out_type=tf.float32)
    sn = tf.io.parse_tensor(content["sn"], out_type=tf.float32)
    # dg = tf.io.parse_tensor(content["dg"], out_type=tf.float32)

    kg = tf.reshape(kg, shape=(content["n_perts"], content["n_pix"], content["n_z_bins"]))
    ia = tf.reshape(ia, shape=(content["n_perts"], content["n_pix"], content["n_z_bins"]))
    sn = tf.reshape(sn, shape=(content["n_perts"], content["n_pix"], content["n_z_bins"]))
    # dg = tf.reshape(dg, shape=(content["n_pix"], content["n_z_bins"]))

    # return kg, ia, sn, dg, cosmo, sobol
    return kg, ia, sn

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
