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

from msfm.utils import analysis, logger, tfrecords, shear, parameters

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


def dset_add_bias(data_vectors, index, pert_labels, m_bias_dist=None):
    """Adds a random multiplicative shear bias (the additive one is negligible)

    Args:
        data_vectors (dict): has keys "kg_{pert_label}" and "sn", which contain tensors of shape (n_pix, n_z_bins)
        index (tf.tensor): Integer that is just passed through
        pert_labels (list): The labels of the perturbations to loop through. These are needed explicitly for the
            function to be converted by autograph.
        m_bias_dist (tfp.distributions.Distribution, optional): TensorFlow probability distribution from which the bias
            is sampled. The samples have to be drawn within this function for randomness. Defaults to None, then the
            bias is equal to one.

    Returns:
        tuple: (data_vectors, index) of the same shape as at the input
    """
    LOGGER.warning(f"Tracing dset_add_bias")

    if m_bias_dist is None:
        LOGGER.warning(f"No multiplicative shear bias is applied")

    elif isinstance(m_bias_dist, tfp.distributions.Distribution):
        m_bias = m_bias_dist.sample()
        for label in pert_labels:
            # broadcast axis 0 of size n_pix
            data_vectors[f"kg_{label}"] *= 1 + m_bias

    return data_vectors, index


def dset_add_noise(data_vectors, index, pert_labels, noise_scale=1):
    """
    Args:
        data_vectors (dict): has keys "kg_{pert_label}" and "sn", which contain tensors of shape (n_pix, n_z_bins)
        index (tf.tensor): Integer that is just passed through
        pert_labels (list): The labels of the perturbations to loop through. These are needed explicitly for the
            function to be converted by autograph.

    Returns:
        tuple: (data_vectors, index), but data_vectors["sn"] is removed
    """
    LOGGER.warning(f"Tracing dset_add_noise")

    sn = data_vectors.pop("sn")
    for label in pert_labels:
        data_vectors[f"kg_{label}"] += noise_scale * sn

    return data_vectors, index


def dset_concat_perts(data_vectors, index, pert_labels):
    """Concatenates the batch and perturbations along axis 0

    Args:
        data_vectors (dict): has keys "kg_{pert_label}", which contain tensors of shape (n_pix, n_z_bins)
        index (tf.tensor): Integer that is just passed through
        pert_labels (list): The labels of the perturbations to loop through. These are needed explicitly for the
            function to be converted by autograph.

    Returns:
        (tf.tensor, tf.tensor): data_vectors is a tensor of shape (n_perts * batch_size, n_pix), where the first
            batch_size elements correspond to the fiducial value, the second to the first perturbation, the third to
            the second perturbation, etc.
    """
    LOGGER.warning(f"Tracing dset_concat_perts")

    data_vectors = tf.concat([data_vectors[f"kg_{pert_label}"] for pert_label in pert_labels], axis=0)

    return data_vectors, index


def get_fiducial_dset(
    tfr_pattern: str,
    local_batch_size: int,
    # configuration
    params: list = None,
    conf: dict = None,
    # shape noise settings
    i_noise: int = 0,
    noise_scale: float = 1.0,
    # performance
    n_readers: int = 8,
    n_prefetch: int = tf.data.AUTOTUNE,
    file_name_shuffle_buffer: int = 128,
    examples_shuffle_buffer: int = 128,
    # random seeds
    file_name_shuffle_seed: int = 17,
    examples_shuffle_seed: int = 67,
    is_eval: bool = False,
    eval_seed: int = 32,
    # distribution
    input_context: tf.distribute.InputContext = None,
) -> tf.data.Dataset:
    """Builds the training dataset from the given file name pattern
    TODO add galaxy clustering maps

    Args:
        tfr_pattern (str): Glob pattern of the .fiducial tfrecord files.
        batch_size (int): Local batch size, will be multiplied with the number of deltas for the total batch size.
        params (list): List of the cosmological parameters with respect to which the perturbations to be used in 
            training are returned, see the config for all possibilities. Defaults to None, then all are included
            according to the config.
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). Defaults to None.
        i_noise (int): Index for the shape noise realizations. This has to be fixed and can't be a tf.Variable or
            other tensor (like randomly sampled).
        noise_scale (float): Factor by which to multiply the shape noise. This could also be a tf.Variable to change
            it according to a schedule during training
        n_readers (int, optional): Number of parallel readers, i.e. samples read out from different input files
            concurrently. This should be roughly less than a tenth of the number of files. Defaults to 8.
        n_prefetch (int, optional): Number of dataset elements to prefetch.
        file_name_shuffle_buffer (int, optional): Defaults to 128.
        examples_shuffle_buffer (int, optional): Defaults to 128.
        file_name_shuffle_seed (int, optional): Defaults to 17.
        examples_shuffle_seed (int, optional): Defaults to 67.
        is_eval (bool, optional): If this is True, then the dataset won't be shuffled repeatedly, such that one can go
            through it deterministically exactly once. Defaults to False.
        eval_seed (int, optional): Fixed seed for evaluation. Defaults to 32.
        input_context (tf.distribute.InputContext, optional): For distributed training, this is passed to the
            dataset_fn like in https://www.tensorflow.org/tutorials/distribute/input#tfdistributestrategydistribute_datasets_from_function
            Then, the dataset is sharded. Defaults to None for a non distributed dataset.

            Example usage:
                def dataset_fn(input_context):
                    dset = fiducial_pipeline.get_fiducial_dset(
                        tfr_pattern,
                        params,
                        batch_size,
                        input_context=input_context,
                    )

    Returns:
        tf.data.Dataset: A dataset that returns samples with a given batchsize in the right ordering for the delta loss
            The label consists of (i_example, i_noise)
    """
    LOGGER.info(f"Starting to generate the fiducial training set for i_noise = {i_noise}")

    # load the pixel file to get the size of the data vector
    data_vec_pix, _, _, _, _ = analysis.load_pixel_file(conf)
    n_pix = len(data_vec_pix)
    masks = tf.constant(analysis.get_tomo_masks(conf), dtype=tf.float32)
    n_z_bins = masks.shape[1]
    pert_labels = parameters.get_fiducial_perturbation_labels(params)

    if is_eval:
        tf.random.set_seed(eval_seed)

    # get the file names
    dset = tf.data.Dataset.list_files(tfr_pattern, shuffle=False)

    # shard for distributed training
    if input_context is not None:
        # NOTE Taken from https://www.tensorflow.org/tutorials/distribute/input#usage_2. This is black magic since
        # print(input_context.num_input_pipelines) yields 1, so I don't know how the sharding happens, but it does, see
        # distributed_sharding.ipynb
        dset = dset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
        LOGGER.info(f"Sharding the dataset according to the input_context")

    # repeat and shuffle
    if not is_eval:
        dset = dset.repeat()
        dset = dset.shuffle(file_name_shuffle_buffer, seed=file_name_shuffle_seed)

    # interleave, block_length is the number of files every reader reads
    if is_eval:
        dset = dset.interleave(tf.data.TFRecordDataset, cycle_length=n_readers, block_length=1)
    else:
        dset = dset.interleave(
            tf.data.TFRecordDataset,
            cycle_length=n_readers,
            block_length=1,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )

    dset_parse_inverse = lambda serialized_example: tfrecords.parse_inverse_fiducial(
        serialized_example, pert_labels, i_noise, n_pix, n_z_bins
    )
    # output signature (data_vectors, (i_example, i_noise))
    dset = dset.map(dset_parse_inverse, num_parallel_calls=tf.data.AUTOTUNE)

    # add shear bias
    m_bias_dist = shear.get_m_bias_distribution(conf)
    dset = dset.map(lambda data_vectors, index: dset_add_bias(data_vectors, index, pert_labels, m_bias_dist))

    # add noise
    dset = dset.map(lambda data_vectors, index: dset_add_noise(data_vectors, index, pert_labels, noise_scale))

    # shuffle and batch
    if not is_eval:
        dset = dset.shuffle(examples_shuffle_buffer, seed=examples_shuffle_seed)
    dset = dset.batch(local_batch_size, drop_remainder=True)
    LOGGER.info(f"Batching into {local_batch_size} elements locally")

    # concatenate the perturbations into the batch dimension like in
    # https://github.com/des-science/y3-deep-lss/blob/main/deep_lss/utils/delta_loss.py#L167
    dset = dset.map(lambda data_vectors, index: dset_concat_perts(data_vectors, index, pert_labels))

    # mask to ensure the padding (relevant if there's additive shift somewhere). the masks tensor is broadcast
    dset = dset.map(lambda data_vectors, index: (tf.multiply(data_vectors, masks), index))

    dset = dset.prefetch(n_prefetch)

    LOGGER.info(
        f"Successfully generated the fiducial training set with element_spec {dset.element_spec} for"
        f" i_noise = {i_noise}"
    )
    return dset


def get_fiducial_multi_noise_dset(
    tfr_pattern: str,
    local_batch_size: int,
    # configuration
    params: list = None,
    conf: dict = None,
    # shape noise settings
    n_noise: int = 1,
    noise_scale: float = 1.0,
    # performance
    n_readers: int = 8,
    n_prefetch: int = tf.data.AUTOTUNE,
    file_name_shuffle_buffer: int = 128,
    examples_shuffle_buffer: int = 128,
    # random seeds
    file_name_shuffle_seed: int = 17,
    examples_shuffle_seed: int = 67,
    # distribution
    input_context: tf.distribute.InputContext = None,
) -> tf.data.Dataset:
    """A dataset made up of the above datasets, but for different i_noise (index of the shape noise realization). The
    sampling is uniform.

    Args:
        tfr_pattern (str): Glob pattern of the .fiducial tfrecord files.
        batch_size (int): Local batch size, will be multiplied with the number of deltas for the total batch size.
        params (list): List of the cosmological parameters with respect to which the perturbations to be used in 
            training are returned, see the config for all possibilities. Defaults to None, then all are included
            according to the config.        
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). Defaults to None.
        n_batches (int, optional): dset.take(n_batches) elements are taken from the dataset. This is done within this
            function because this operation can not be performed on a distributed dataset.
        n_noise (int): Number of noise indices to include.
        noise_scale (float): Factor by which to multiply the shape noise. This could also be a tf.Variable to change
            it according to a schedule during training
        n_readers (int, optional): Number of parallel readers, i.e. samples read out from different input files
            concurrently. Defaults to 8.
        n_prefetch (int, optional): Number of dataset elements to prefetch.
        file_name_shuffle_buffer (int, optional): Defaults to 128.
        examples_shuffle_buffer (int, optional): Defaults to 128.
        file_name_shuffle_seed (int, optional): Defaults to 17.
        examples_shuffle_seed (int, optional): Defaults to 67.

    Returns:
        tf.data.Dataset: A dataset that returns samples with a given batchsize in the right ordering for the delta loss
            The label consists of (i_example, i_noise)
    """

    dset = tf.data.Dataset.sample_from_datasets(
        [
            get_fiducial_dset(
                tfr_pattern,
                local_batch_size,
                params,
                conf,
                i_noise,
                noise_scale,
                n_readers,
                n_prefetch,
                file_name_shuffle_buffer // n_noise,
                examples_shuffle_buffer // n_noise,
                file_name_shuffle_seed + i_noise,
                examples_shuffle_seed + i_noise,
                input_context,
            )
            for i_noise in range(n_noise)
        ]
    )

    dset = dset.prefetch(n_prefetch)
    LOGGER.info(
        f"Successfully generated the fiducial training set with element_spec {dset.element_spec} for i_noise in"
        f" [0, {n_noise}]"
    )
    return dset
