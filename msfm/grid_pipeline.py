# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created February 2023
Author: Arne Thomsen

This file is loosely based off 
https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/kids1000_analysis/input_pipeline.py
by Janis Fluri
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import warnings

from icecream import ic

from msfm.utils import analysis, logger, tfrecords, shear

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


def dset_add_bias(kg, sn, cosmo, index, m_bias_dist=None):
    """Adds a random multiplicative shear bias (the additive one is negligible)

    Args:
        kg (tf.tensor): shape(n_pix, n_z_bins) the sum of weak lensing signal and intrinsic alignment
        sn (tf.tensor): shape(n_pix, n_z_bins) one shape noise realization
        cosmo (tf.tensor): shape(n_params,) the cosmological parameter label
        index (tuple): A tuple of two integers (i_sobol, i_noise)
        m_bias_dist (tfp.distributions.Distribution, optional): TensorFlow probability distribution from which the bias
            is sampled. The samples have to be drawn within this function for randomness. Defaults to None, then the
            bias is equal to one.

    Returns:
        tuple: (kg, sn, cosmo, index) of the same shape as at the input
    """
    LOGGER.warning(f"Tracing dset_add_bias")

    if m_bias_dist is None:
        LOGGER.warning(f"No multiplicative shear bias is applied")

    elif isinstance(m_bias_dist, tfp.distributions.Distribution):
        m_bias = m_bias_dist.sample()
        # broadcast axis 0 of size n_pix
        kg *= 1 + m_bias

    return kg, sn, cosmo, index


def dset_add_noise(kg, sn, cosmo, index, noise_scale=1):
    """
    Args:
        kg (tf.tensor): shape(n_pix, n_z_bins) the sum of weak lensing signal and intrinsic alignment
        sn (tf.tensor): shape(n_pix, n_z_bins) one shape noise realization
        cosmo (tf.tensor): shape(n_params,) the cosmological parameter label
        index (tuple): A tuple of two integers (i_sobol, i_noise)
        m_bias_dist (tfp.distributions.Distribution, optional): TensorFlow probability distribution from which the bias
            is sampled. The samples have to be drawn within this function for randomness. Defaults to None, then the
            bias is equal to one.

    Returns:
        tuple: (kg, cosmo, index) but kg now also includes the shape noise component
    """
    LOGGER.warning(f"Tracing dset_add_noise")

    kg += noise_scale * sn

    return kg, cosmo, index


def get_grid_dset(
    conf: dict,
    repo_dir: str,
    tfr_pattern: str,
    batch_size: int,
    i_noise: int = 0,
    noise_scale: float = 1.0,
    n_readers: int = 8,
    n_prefetch: int = tf.data.AUTOTUNE,
    tf_seed: int = 31,
) -> tf.data.Dataset:
    """Builds the training dataset from the given file name pattern
    TODO add galaxy clustering maps

    Args:
        conf (dict): From configuration file in configs/config.yaml.
        repo_dir (str): Absolute path to the msfm repo.
        tfr_pattern (str): Glob pattern of the .fiducial tfrecord files.
        pert_labels (list): List of the perturbations to use in training, see the config for all possibilities.
        batch_size (int): Local batch size, will be multiplied with the number of deltas for the total batch size.
        i_noise (int): Index for the shape noise realizations. This has to be fixed and can't be a tf.Variable or
            other tensor (like randomly sampled).
        noise_scale (float): Factor by which to multiply the shape noise. This could also be a tf.Variable to change
            it according to a schedule during training
        n_readers (int, optional): Number of parallel readers, i.e. samples read out from different input files
            concurrently. This should be roughly less than a tenth of the number of files. Defaults to 8.
        n_prefetch (int, optional): Number of dataset elements to prefetch.
        tf_seed (int, optional): The global tensorflow seed to make the evaluation of this function deterministic.
            TODO check whether this is actually the case.

    Returns:
        tf.data.Dataset: A deterministic dataset that goes through the grid cosmologies in the order of the sobol seeds
    """
    LOGGER.info(f"Starting to generate the grid data set for i_noise = {i_noise}")

    # load the pixel file to get the size of the data vector
    data_vec_pix, _, _, _, _ = analysis.load_pixel_file(conf, repo_dir)
    n_pix = len(data_vec_pix)
    masks = tf.constant(analysis.get_tomo_masks(conf, repo_dir))
    n_z_bins = masks.shape[1]
    # n_params = len(conf["analysis"]["params"]["cosmo"]) + len(conf["analysis"]["params"]["astro"])
    n_params = len(conf["analysis"]["params"])

    # for determinism TODO double check whether this actually fixes everything
    tf.random.set_seed(tf_seed)

    # get the file names and dataset them
    dset = tf.data.Dataset.list_files(tfr_pattern, shuffle=False)

    # interleave, block_length is the number of files every reader reads
    dset = dset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=n_readers,
        block_length=1,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    dset_parse_inverse = lambda serialized_example: tfrecords.parse_inverse_grid(
        serialized_example, i_noise, n_pix, n_z_bins, n_params
    )
    # output signature (kg, sn, cosmo, index)
    dset = dset.map(dset_parse_inverse, num_parallel_calls=tf.data.AUTOTUNE)

    # add shear bias
    m_bias_dist = shear.get_m_bias_distribution(conf)
    dset = dset.map(lambda kg, sn, cosmo, index: dset_add_bias(kg, sn, cosmo, index, m_bias_dist))

    # add noise, output signature (kg, cosmo, index)
    dset = dset.map(lambda kg, sn, cosmo, index: dset_add_noise(kg, sn, cosmo, index, noise_scale))

    # batch
    dset = dset.batch(batch_size, drop_remainder=False)

    # mask to ensure the padding (relevant if there's additive shift somewhere). the masks tensor is broadcast
    dset = dset.map(lambda kg, cosmo, index: (tf.multiply(kg, masks), cosmo, index))

    dset = dset.prefetch(n_prefetch)

    LOGGER.info(
        f"Successfully generated the grid set with element_spec {dset.element_spec} for i_noise = {i_noise}"
    )
    return dset


def get_grid_multi_noise_dset(
    conf: dict,
    repo_dir: str,
    tfr_pattern: str,
    batch_size: int,
    n_noise: int = 1,
    noise_scale: float = 1,
    n_readers: int = 8,
    n_prefetch: int = tf.data.AUTOTUNE,
    tf_seed: int = 31,
) -> tf.data.Dataset:
    """A dataset made up of the above datasets, but for different i_noise (index of the shape noise realization). The
    sampling is uniform.

    Args:
        conf (dict): From configuration file in configs/config.yaml.
        repo_dir (str): Absolute path to the msfm repo.
        tfr_pattern (str): Glob pattern of the .fiducial tfrecord files.
        pert_labels (list): List of the perturbations to use in training, see the config for all possibilities.
        batch_size (int): Local batch size, will be multiplied with the number of deltas for the total batch size.
        n_noise (int): Number of noise indices to include.
        file_name_shuffle_buffer (int, optional): Defaults to 128.
        file_name_shuffle_seed (int, optional): Defaults to 17.
        examples_shuffle_buffer (int, optional): Defaults to 128.
        examples_shuffle_seed (int, optional): Defaults to 67.
        n_readers (int, optional): Number of parallel readers, i.e. samples read out from different input files
            concurrently. Defaults to 8.
        n_prefetch (int, optional): Number of dataset elements to prefetch.

    Returns:
        tf.data.Dataset: A dataset that returns samples with a given batchsize in the right ordering for the delta loss
            The label consists of (i_example, i_noise)
    """

    dset = tf.data.Dataset.sample_from_datasets(
        [
            get_grid_dset(
                conf, repo_dir, tfr_pattern, batch_size, i_noise, noise_scale, n_readers, 0, tf_seed + i_noise
            )
            for i_noise in range(n_noise)
        ]
    )

    dset = dset.prefetch(n_prefetch)
    LOGGER.info(
        f"Successfully generated the grid set with element_spec {dset.element_spec} for i_noise in [0, {n_noise}]"
    )
    return dset
