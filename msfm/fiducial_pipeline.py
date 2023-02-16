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

from msfm.utils import logger, tfrecords, survey, shear

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
    conf: dict,
    repo_dir: str,
    tfr_pattern: str,
    pert_labels: list,
    batch_size: int,
    i_noise: int = 0,
    noise_scale: float = 1.0,
    file_name_shuffle_buffer: int = 128,
    file_name_shuffle_seed: int = 17,
    examples_shuffle_buffer: int = 128,
    examples_shuffle_seed: int = 67,
    n_readers: int = 8,
    n_prefetch: int = tf.data.AUTOTUNE,
    is_eval: bool = False,
    eval_seed: int = 32,
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
        file_name_shuffle_buffer (int, optional): Defaults to 128.
        file_name_shuffle_seed (int, optional): Defaults to 17.
        examples_shuffle_buffer (int, optional): Defaults to 128.
        examples_shuffle_seed (int, optional): Defaults to 67.
        n_readers (int, optional): Number of parallel readers, i.e. samples read out from different input files
            concurrently. This should be roughly less than a tenth of the number of files. Defaults to 8.
        n_prefetch (int, optional): Number of dataset elements to prefetch.
        is_eval (bool, optional): If this is True, then the dataset won't be shuffled repeatedly, such that one can go
            through it deterministically exactly once. Defaults to False.
        eval_seed (int, optional): Fixed seed for evaluation. Defaults to 32.

    Returns:
        tf.data.Dataset: A dataset that returns samples with a given batchsize in the right ordering for the delta loss
            The label consists of (i_example, i_noise)
    """
    LOGGER.info(f"Starting to generate the fiducial training set for i_noise = {i_noise}")

    # load the pixel file to get the size of the data vector
    data_vec_pix, _, _, _, _ = survey.load_pixel_file(conf, repo_dir)
    n_pix = len(data_vec_pix)
    masks = tf.constant(survey.get_tomo_masks(conf, repo_dir))
    n_z_bins = masks.shape[1]

    if is_eval:
        tf.random.set_seed(eval_seed)

    # get the file names, shuffle and dataset them
    dset = tf.data.Dataset.list_files(tfr_pattern, shuffle=False)
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
    dset = dset.batch(batch_size, drop_remainder=True)

    # concatenate the perturbations into the batch dimension like in
    # https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/kids1000_analysis/losses.py#L122
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
    conf: dict,
    repo_dir: str,
    tfr_pattern: str,
    pert_labels: list,
    batch_size: int,
    n_noise: int = 1,
    noise_scale: float = 1,
    file_name_shuffle_buffer: int = 128,
    file_name_shuffle_seed: int = 17,
    examples_shuffle_buffer: int = 128,
    examples_shuffle_seed: int = 67,
    n_readers: int = 8,
    n_prefetch: int = tf.data.AUTOTUNE,
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
            get_fiducial_dset(
                conf,
                repo_dir,
                tfr_pattern,
                pert_labels,
                batch_size,
                i_noise,
                noise_scale,
                file_name_shuffle_buffer // n_noise,
                file_name_shuffle_seed + i_noise,
                examples_shuffle_buffer // n_noise,
                examples_shuffle_seed + i_noise,
                n_readers,
                n_prefetch=0,
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
