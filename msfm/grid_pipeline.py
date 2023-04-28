# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created February 2023
Author: Arne Thomsen

This file is loosely based off 
https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/kids1000_analysis/input_pipeline.py
by Janis Fluri
"""

import tensorflow as tf
import warnings

from msfm.utils import analysis, logger, tfrecords

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


def dset_augmentations(
    kg: tf.constant,
    sn: tf.constant,
    dg: tf.constant,
    cosmo: tf.constant,
    index: tuple,
    noise_scale: float = 1.0,
    masks_metacal: tf.constant = None,
    mask_maglim: tf.constant = None,
):
    """Applies random augmentations and general pre-processing to the maps. This includes in order:
        lensing
        - DEPRECATED (This was moved to run_grid_tfrecords.py): Adds random multiplicative shear bias (the additive
            one is negligible) to the kappa maps
        - Adds the chosen shape noise realization to the kappa maps
        - Masks the resulting data vector (this is only required if an addition like an additive shear bias is applied)

        clustering
        - Masking only

    Args:
        kg (tf.tensor): shape(n_pix, n_z_metacal) the sum of weak lensing signal and intrinsic alignment
        sn (tf.tensor): shape(n_pix, n_z_metacal) one shape noise realization
        dg (tf.tensor): shape(n_pix, n_z_maglim) the galaxy number map
        cosmo (tf.tensor): shape(n_params,) the cosmological parameter label
        index (tuple): A tuple of two integers (i_sobol, i_noise)
        noise_scale (float): Multiplicative factor applied to the shape noise before it is added. Defaults to 1.0,
            for a value of None, no shape noise is added to the kappa maps.
        masks_metacal (tf.tensor): Tensor Of shape (n_pix, n_z_metacal) only containing zeros and ones.
        mask_maglim (tf.tensor): Tensor Of shape (n_pix, 1) only containing zeros and ones. There's only one mask for
            all tomographic bins.

    Returns:
        tuple: (data_vectors, cosmo, index), where data_vectors is a tensor of shape
        (batch_size, n_pix, n_z_metacal + n_z_maglim), cosmo is a label distributed on the Sobol sequence and index is
        a tuple containing (i_sobol, i_noise).
    """

    LOGGER.warning(f"Tracing dset_augmentations")

    """ lensing """

    # shape noise
    if noise_scale is not None:
        kg += noise_scale * sn
    else:
        LOGGER.warning(f"No shape noise is added")

    # masking
    if masks_metacal is not None:
        kg *= masks_metacal
    else:
        LOGGER.warning(f"No masking is applied to metacal")

    """ clustering """

    dg = tf.cast(dg, tf.float32)

    # masking
    if mask_maglim is not None:
        dg *= mask_maglim
    else:
        LOGGER.warning(f"No masking is applied to maglim")

    # concatenate the tomography axis
    data_vectors = tf.concat([kg, dg], axis=-1)

    return data_vectors, cosmo, index


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
    # distribution
    input_context: tf.distribute.InputContext = None,
) -> tf.data.Dataset:
    """Builds the training dataset from the given file name pattern

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
        tf.data.Dataset: A deterministic dataset that goes through the grid cosmologies in the order of the sobol
            seeds. The output is a tuple like (data_vectors, cosmo, index), where data_vectors is a tensor of shape
        (batch_size, n_pix, n_z_metacal + n_z_maglim), cosmo is a label distributed on the Sobol sequence and index is
        a tuple containing (i_sobol, i_noise).
    """
    LOGGER.info(f"Starting to generate the grid data set for i_noise = {i_noise}")

    # constants
    conf = analysis.load_config(conf)
    n_z_metacal = len(conf["survey"]["metacal"]["z_bins"])
    n_z_maglim = len(conf["survey"]["maglim"]["z_bins"])

    if n_params is None:
        params = (
            conf["analysis"]["params"]["cosmo"] + conf["analysis"]["params"]["ia"] + conf["analysis"]["params"]["bg"]
        )
        n_params = len(params)

    # load the pixel file to get the size of the data vector
    data_vec_pix, _, _, _ = analysis.load_pixel_file(conf)
    n_pix = len(data_vec_pix)

    # masking
    masks_dict = analysis.get_tomo_masks(conf)
    masks_metacal = tf.constant(masks_dict["metacal"], dtype=tf.float32)
    mask_maglim = tf.constant(masks_dict["maglim"], dtype=tf.float32)

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

    # parse, output signature (kg, sn, dg, cosmo, index)
    dset = dset.map(
        lambda serialized_example: tfrecords.parse_inverse_grid(
            serialized_example, i_noise, n_pix, n_z_metacal, n_z_maglim, n_params
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # batch (first, for vectorization)
    dset = dset.batch(local_batch_size, drop_remainder=False)
    LOGGER.info(f"Batching into {local_batch_size} elements locally")

    # augmentations (all in one function, to make parallelization faster)
    dset = dset.map(
        lambda kg, sn, dg, cosmo, index: dset_augmentations(
            kg,
            sn,
            dg,
            cosmo,
            index,
            noise_scale,
            masks_metacal,
            mask_maglim,
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
