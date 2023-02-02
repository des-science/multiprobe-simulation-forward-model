# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created February 2023
Author: Arne Thomsen

This file is inspired by 
https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/kids1000_analysis/input_pipeline.py

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


def dset_remove_mean(kg_perts, sn, index, corr_fact=1):
    # take the mean over the axis of size n_pix, weight has the dimension of the last axis (tomo) and is broadcast
    kg_perts -= tf.reduce_mean(kg_perts, axis=1, keepdims=True) * corr_fact
    sn -= tf.reduce_mean(sn, axis=0, keepdims=True) * corr_fact

    return kg_perts, sn, index


# TODO
def dset_add_biases(example):
    pass


# TODO make this compatible with multi node training
def get_train_dset(
    conf: dict,
    repo_dir: str,
    tfr_pattern: str,
    pert_labels: list,
    i_noise: int,
    batch_size: int,
    is_eval: bool = False,
    eval_seed: int = 32,
    file_name_shuffle_buffer: int = 128,
    file_name_shuffle_seed: int = 17,
    examples_shuffle_buffer: int = 128,
    examples_shuffle_seed: int = 67,
    n_readers: int = 8,
    n_prefetch: int = 3,
) -> tf.data.Dataset:
    """Builds the training dataset from the given file name pattern

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
    data_vec_len = len(data_vec_pix)
    tomo_patch_len = [len(patches_pix) for patches_pix in tomo_patches_pix]
    mean_correction = data_vec_len / np.array(tomo_patch_len)

    if is_eval:
        tf.random.set_seed(eval_seed)

    # get the file names, shuffle and dataset them
    # TODO shard the dataset?
    dset = tf.data.Dataset.list_files(tfr_pattern, shuffle=False)
    if not is_eval:
        dset = dset.repeat()
        dset = dset.shuffle(file_name_shuffle_buffer, seed=file_name_shuffle_seed)

    # interleave
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

    # decode the dset, shapes kg_perts = (n_perts, n_pix, n_z_bins) and sn = (n_pix, n_z_bins)
    dset_parse_inverse = lambda example: tfrecords.parse_inverse_fiducial(example, pert_labels, i_noise)
    dset = dset.map(dset_parse_inverse, num_parallel_calls=tf.data.AUTOTUNE)

    # shuffle and batch
    if not is_eval:
        dset = dset.shuffle(examples_shuffle_buffer, seed=examples_shuffle_seed)
    dset = dset.batch(batch_size)

    # remove the mean
    dset = dset.map(lambda kg_perts, sn, index: dset_remove_mean(kg_perts, sn, index, mean_correction))

    # TODO add biases like https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/kids1000_analysis/input_pipeline.py#L203

    # add the noise by expanding it along the dimension of the perturbations
    dset_add_noise = lambda kg_perts, sn, index: (kg_perts + tf.expand_dims(sn, axis=1), index)
    dset = dset.map(dset_add_noise)

    # TODO mask like here? https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/kids1000_analysis/input_pipeline.py#L259

    # concatenate the perturbations into the batch dimension like in
    # https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/kids1000_analysis/losses.py#L122 
    dset_concat_perts = lambda kg_perts, index: (tf.concat(tf.unstack(kg_perts, axis=0), axis=0), index)
    dset = dset.map(dset_concat_perts)

    dset = dset.prefetch(n_prefetch)

    return dset
