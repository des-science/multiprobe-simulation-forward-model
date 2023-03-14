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

from msfm.utils import analysis, logger, tfrecords, shear

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


def dset_augmentations(
    kg: tf.constant,
    sn: tf.constant,
    cosmo: tf.constant,
    index: tuple,
    m_bias_dist: tfp.distributions.Distribution = None,
    noise_scale: float = 1.0,
    masks: tf.constant = None,
):
    """Applies random augmentations and general pre-processing to the maps. This includes in order:
        - Adds random multiplicative shear bias (the additive one is negligible) to the kappa maps
        - Adds the chosen shape noise realization to the kappa maps
        - Masks the resulting data vector (this is only required if an addition like an additive shear bias is applied)

    Args:
        kg (tf.tensor): shape(n_pix, n_z_bins) the sum of weak lensing signal and intrinsic alignment
        sn (tf.tensor): shape(n_pix, n_z_bins) one shape noise realization
        cosmo (tf.tensor): shape(n_params,) the cosmological parameter label
        index (tuple): A tuple of two integers (i_sobol, i_noise)
        m_bias_dist (tfp.distributions.Distribution, optional): TensorFlow probability distribution from which the bias
            is sampled. The samples have to be drawn within this function for randomness. Defaults to None, then the
            bias is always equal to one.
        noise_scale (float): Multiplicative factor applied to the shape noise before it is added. Defaults to 1.0,
            for a value of None, no shape noise is added to the kappa maps.
        masks (tf.tensor): Tensor Of shape (n_pix, n_z_bins) only containing zeros and ones.

    Returns:
        tuple: (data_vectors, index), where data_vectors is a tensor of shape (n_perts * batch_size, n_pix). The first
            batch_size elements correspond to the fiducial value, the second to the first perturbation, the third to
            the second perturbation, etc. (for compatibility with the delta loss). Furthermore, data_vectors["sn"] is
            removed.
    """

    LOGGER.warning(f"Tracing dset_augmentations")

    # shear bias
    if m_bias_dist is not None:
        # shape (n_z_bins,)
        m_bias = m_bias_dist.sample()
        # broadcast axis 0 of size n_pix
        kg *= 1.0 + m_bias
    else:
        LOGGER.warning(f"No multiplicative shear bias is applied")

    # shape noise
    if noise_scale is not None:
        kg += noise_scale * sn
    else:
        LOGGER.warning(f"No shape noise is added")

    # masking
    if masks is not None:
        kg *= masks
    else:
        LOGGER.warning(f"No masking is applied")

    return kg, cosmo, index


def get_grid_dset(
    tfr_pattern: str,
    local_batch_size: int,
    # configuration
    n_params: int = None,
    conf: dict = None,
    # shape noise settings
    i_noise: int = 0,
    noise_scale: float = 1.0,
    # performance
    n_readers: int = 8,
    n_prefetch: int = tf.data.AUTOTUNE,
    # random seeds
    tf_seed: int = 31,
    # distribution
    input_context: tf.distribute.InputContext = None,
) -> tf.data.Dataset:
    """Builds the training dataset from the given file name pattern
    TODO add galaxy clustering maps

    Args:
        tfr_pattern (str): Glob pattern of the .fiducial tfrecord files.
        batch_size (int): Local batch size, will be multiplied with the number of deltas for the total batch size.
        n_params (list): Number of the cosmological parameters stored in the .tfrecords, this is typically all of them.
            The value is used to reshape the stored tensors, and for nothing else.
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). Defaults to None.
        i_noise (int): Index for the shape noise realizations. This has to be fixed and can't be a tf.Variable or
            other tensor (like randomly sampled).
        noise_scale (float): Factor by which to multiply the shape noise. This could also be a tf.Variable to change
            it according to a schedule during training
        n_readers (int, optional): Number of parallel readers, i.e. samples read out from different input files
            concurrently. This should be roughly less than a tenth of the number of files. Defaults to 8.
        n_prefetch (int, optional): Number of dataset elements to prefetch.
        tf_seed (int, optional): The global tensorflow seed to make the evaluation of this function deterministic.
            TODO check whether this is actually the case.
        input_context (tf.distribute.InputContext, optional): For distributed training, this is passed to the
            dataset_fn like in https://www.tensorflow.org/tutorials/distribute/input#tfdistributestrategydistribute_datasets_from_function
            Then, the dataset is sharded. Defaults to None for a non distributed dataset.

            Example usage:
                def dataset_fn(input_context):
                    dset = fiducial_pipeline.get_fiducial_dset(
                        tfr_pattern,
                        batch_size,
                        input_context=input_context,
                    )

    Returns:
        tf.data.Dataset: A deterministic dataset that goes through the grid cosmologies in the order of the sobol seeds
    """
    LOGGER.info(f"Starting to generate the grid data set for i_noise = {i_noise}")

    # load the pixel file to get the size of the data vector
    data_vec_pix, _, _, _, _ = analysis.load_pixel_file(conf)
    n_pix = len(data_vec_pix)
    masks = tf.constant(analysis.get_tomo_masks(conf), dtype=tf.float32)
    n_z_bins = masks.shape[1]
    if n_params is None:
        conf = analysis.load_config(conf)
        params = conf["analysis"]["params"]
        n_params = len(params)
    m_bias_dist = shear.get_m_bias_distribution(conf)

    # for determinism TODO double check whether this actually fixes everything
    tf.random.set_seed(tf_seed)

    # get the file names and dataset them
    dset = tf.data.Dataset.list_files(tfr_pattern, shuffle=False)

    # shard for distributed evaluation
    if input_context is not None:
        # NOTE Taken from https://www.tensorflow.org/tutorials/distribute/input#usage_2. This is black magic since
        # input_context.num_input_pipelines = 1, so I don't know how the sharding happens, but it does, see
        # distributed_sharding.ipynb
        dset = dset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
        LOGGER.info(f"Sharding the dataset according to the input_context")

    # interleave, block_length is the number of files every reader reads
    dset = dset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=n_readers,
        block_length=1,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    # parse, output signature (kg, sn, cosmo, index)
    dset = dset.map(
        lambda serialized_example: tfrecords.parse_inverse_grid(
            serialized_example, i_noise, n_pix, n_z_bins, n_params
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # batch (first, for vectorization)
    dset = dset.batch(local_batch_size, drop_remainder=False)
    LOGGER.info(f"Batching into {local_batch_size} elements locally")

    # augmentations (all in one function, to make parallelization easier)
    dset = dset.map(
        lambda kg, sn, cosmo, index: dset_augmentations(
            kg,
            sn,
            cosmo,
            index,
            m_bias_dist,
            noise_scale,
            masks,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dset = dset.prefetch(n_prefetch)

    LOGGER.info(f"Successfully generated the grid set with element_spec {dset.element_spec} for i_noise = {i_noise}")
    return dset


def get_grid_multi_noise_dset(
    tfr_pattern: str,
    local_batch_size: int,
    # configuration
    conf: dict = None,
    # shape noise settings
    n_noise: int = 0,
    noise_scale: float = 1.0,
    # performance
    n_readers: int = 8,
    n_prefetch: int = tf.data.AUTOTUNE,
    # random seeds
    tf_seed: int = 31,
    # distribution
    input_context: tf.distribute.InputContext = None,
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
                tfr_pattern,
                local_batch_size,
                conf,
                i_noise,
                noise_scale,
                n_readers,
                0,
                tf_seed + i_noise,
                input_context,
            )
            for i_noise in range(n_noise)
        ]
    )

    dset = dset.prefetch(n_prefetch)
    LOGGER.info(
        f"Successfully generated the grid set with element_spec {dset.element_spec} for i_noise in [0, {n_noise}]"
    )
    return dset
