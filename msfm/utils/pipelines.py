# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created February 2023
Author: Arne Thomsen

This file is based off 
https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/kids1000_analysis/input_pipeline.py
by Janis Fluri
"""

import numpy as np
import tensorflow as tf
import warnings

from icecream import ic

from msfm.utils import logging, tfrecords, survey
from msfm.utils.filenames import *

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logging.get_logger(__file__)


def dset_remove_mean(data_vectors, index, pert_labels, corr_fact=1):
    """
    Args:
        data_vectors (dict): has keys "kg_{pert_label}" and "sn", which contain tensors of shape (n_pix, n_z_bins)
        pert_labels (list): The labels of the perturbations to loop through. These are needed explicitly for the
            function to be converted by autograph.
        index (tf.tensor): Integer that is just passed through
        corr_fact (int, optional): Correction factor because of the padding. Defaults to 1.

    Returns:
        tuple: (data_vectors, index) of the same shape as at the input
    """
    for label in pert_labels:
        # take the mean over the axis of size n_pix, weight has the dimension of the last axis (tomo) and is broadcast
        data_vectors[f"kg_{label}"] -= tf.reduce_mean(data_vectors[f"kg_{label}"], axis=0, keepdims=True) * corr_fact

    data_vectors["sn"] -= tf.reduce_mean(data_vectors["sn"], axis=0, keepdims=True) * corr_fact

    return data_vectors, index


def dset_add_noise(data_vectors, index, pert_labels, noise_scale=1):
    """
    Args:
        data_vectors (dict): has keys "kg_{pert_label}" and "sn", which contain tensors of shape (n_pix, n_z_bins)
        pert_labels (list): The labels of the perturbations to loop through. These are needed explicitly for the
            function to be converted by autograph.
        index (tf.tensor): Integer that is just passed through

    Returns:
        tuple: (data_vectors, index), but data_vectors["sn"] is removed
    """
    sn = data_vectors.pop("sn")
    for label in pert_labels:
        data_vectors[f"kg_{label}"] += sn

    return data_vectors, index


# TODO
def dset_add_biases(data_vectors, index):
    pass


def dset_concat_perts(data_vectors, index, pert_labels):
    data_vectors = tf.concat([data_vectors[f"kg_{pert_label}"] for pert_label in pert_labels], axis=0)
    return data_vectors, index


# TODO make this compatible with multi node training
def get_train_dset(
    conf: dict,
    repo_dir: str,
    tfr_pattern: str,
    pert_labels: list,
    i_noise: int,
    batch_size: int,
    noise_scale: float = 1,
    is_eval: bool = False,
    eval_seed: int = 32,
    file_name_shuffle_buffer: int = 128,
    file_name_shuffle_seed: int = 17,
    examples_shuffle_buffer: int = 128,
    examples_shuffle_seed: int = 67,
    n_readers: int = 8,
    n_prefetch: int = tf.data.AUTOTUNE,
) -> tf.data.Dataset:
    """Builds the training dataset from the given file name pattern
    TODO add galaxy clustering maps

    Args:
        conf (dict): From configuration file in configs/config.yaml.
        repo_dir (str): Absolute path to the msfm repo.
        tfr_pattern (str): Glob pattern of the .fiducial tfrecord files.
        pert_labels (list): List of the perturbations to use in training, see the config for all possibilities.
        i_noise (int): TODO make this a tf.variable? Index for the shape noise realizations.
        batch_size (int): Local batch size, will be multiplied with the number of deltas for the delta.
        is_eval (bool, optional): If this is True, then the dataset won't be shuffled repeated such that one can go
            through it deterministically exactly once. Defaults to False.
        eval_seed (int, optional): Fixed seed for evaluation. Defaults to 32.
        file_name_shuffle_buffer (int, optional): Defaults to 128.
        file_name_shuffle_seed (int, optional): Defaults to 17.
        examples_shuffle_buffer (int, optional): Defaults to 128.
        examples_shuffle_seed (int, optional): Defaults to 67.
        n_readers (int, optional): Number of parallel readers, i.e. samples read out from different input files
            concurrently. Defaults to 8.
        n_prefetch (int, optional): Number of dataset elements to prefetch.

    Returns:
        tf.data.Dataset: A dataset that returns samples with a given batchsize in the right ordering for the delta loss
    """
    # load the pixel file to calculate the non padded mean
    data_vec_pix, _, _, tomo_patches_pix, _ = survey.load_pixel_file(conf, repo_dir)
    n_pix = len(data_vec_pix)
    tomo_n_patch_pix = [len(patches_pix[0]) for patches_pix in tomo_patches_pix]
    mean_corr_fac = n_pix / np.array(tomo_n_patch_pix)
    n_z_bins = len(mean_corr_fac)

    if is_eval:
        tf.random.set_seed(eval_seed)

    # get the file names, shuffle and dataset them
    dset = tf.data.Dataset.list_files(tfr_pattern, shuffle=False)
    # TODO shard the dataset?
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

    # TODO could randomly sample the noise realization
    # k = 2
    # i_noise = tf.squeeze(tf.random.categorical(tf.ones((1,k))/k, 1))
    # tf.print(i_noise)

    # decode the dset into a dict of data vectors (shape (n_pix, n_z_bins)) for the perturbations and noise, and index
    dset_parse_inverse = lambda serialized_example: tfrecords.parse_inverse_fiducial(
        serialized_example, pert_labels, i_noise, n_pix, n_z_bins
    )
    dset = dset.map(dset_parse_inverse, num_parallel_calls=tf.data.AUTOTUNE)

    # remove the mean, TODO set num_parallel_calls?
    dset = dset.map(lambda data_vectors, index: dset_remove_mean(data_vectors, index, pert_labels, mean_corr_fac))

    # TODO add biases like https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/kids1000_analysis/input_pipeline.py#L203

    # add noise, TODO set num_parallel_calls?
    dset = dset.map(lambda data_vectors, index: dset_add_noise(data_vectors, index, pert_labels, noise_scale))

    # TODO mask like here? https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/kids1000_analysis/input_pipeline.py#L259

    # shuffle and batch
    if not is_eval:
        dset = dset.shuffle(examples_shuffle_buffer, seed=examples_shuffle_seed)
    dset = dset.batch(batch_size, drop_remainder=True)

    # concatenate the perturbations into the batch dimension like in
    # https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/kids1000_analysis/losses.py#L122
    dset = dset.map(lambda data_vectors, index: dset_concat_perts(data_vectors, index, pert_labels))

    dset = dset.prefetch(n_prefetch)

    return dset
