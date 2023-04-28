# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created February 2023
Author: Arne Thomsen

This file is loosely based off 
https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/kids1000_analysis/input_pipeline.py
by Janis Fluri
"""

import healpy as hp  # this import could be avoided easily
import tensorflow as tf
import tensorflow_probability as tfp
import warnings

from msfm.utils import analysis, logger, tfrecords, shear, parameters, redshift

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


def dset_augmentations(
    data_vectors: dict,
    index: tuple,
    pert_labels: list,
    # kg
    m_bias_dist: tfp.distributions.Distribution = None,
    noise_scale: float = 1.0,
    masks_metacal: tf.constant = None,
    # dg
    tomo_n_gal_maglim: tf.constant = None,
    tomo_bg_perts_dict: dict = None,
    mask_maglim: tf.constant = None,
):
    """Applies random augmentations and general pre-processing to the kappa maps. This includes in order:
        - Adds random multiplicative shear bias (the additive one is negligible) to the kappa maps
        - Adds the chosen shape noise realization to the kappa maps
        - Masks the resulting data vector (this is only required if an addition like an additive shear bias is applied)
        - Concatenates the batch and perturbations along axis 0 (for compatibility with the delta loss)

    Args:
        data_vectors (dict): Has keys "kg_{pert_label}" and "sn", which contain tensors of shape
            (batch_size, n_pix, n_z_bins).
        index (tf.tensor): Tuple of integers (i_example, i_noise) that is only passed through.
        pert_labels (list): The labels of the perturbations to loop through. These are needed explicitly for the
            function to be converted by autograph. This list should include all parameters, so cosmo, ia and bg ones.
        m_bias_dist (tfp.distributions.Distribution, optional): TensorFlow probability distribution from which the bias
            is sampled. The samples have to be drawn within this function for randomness. Defaults to None, then the
            bias is always equal to one.
        noise_scale (float): Multiplicative factor applied to the shape noise before it is added. Defaults to 1.0,
            for a value of None, no shape noise is added to the kappa maps.
        masks_metacal (tf.tensor): Tensor of shape (n_pix, n_z_metacal) only containing zeros and ones.
        tomo_n_gal_maglim (tf.tensor): Average number of galaxies per pixel for the given nside of shape (n_z_maglim,).
        tomo_bg_perts_dict (dict): Dict of arrays of shape (n_z_maglim,) which contains one galaxy biasing amplitude
            per tomographic bin. The dictionary keys are the perturbation labels.
        masks_maglim (tf.tensor): Tensor of shape (n_pix, 1) only containing zeros and ones. The mask is identical for
            all tomographic bins.

    Returns:
        tuple: (data_vectors, index), where data_vectors is a tensor of shape (n_perts * batch_size, n_pix). The first
            batch_size elements correspond to the fiducial value, the second to the first perturbation, the third to
            the second perturbation, etc. (for compatibility with the delta loss). Furthermore, data_vectors["sn"] is
            removed.
    """

    LOGGER.warning(f"Tracing dset_augmentations")

    # shape noise
    sn = data_vectors.pop("sn")

    augmented_kg = []
    augmented_dg = []
    for label in pert_labels:
        # intrinsic alignemnt perturbations
        if "Aia" in label:
            kg_data_vector = data_vectors[f"kg_{label}"]
            # doesn't affect the clustering map
            dg_data_vector = data_vectors[f"dg_fiducial"]

            tomo_bg = tomo_bg_perts_dict["fiducial"]

        # galaxy bias perturbations
        elif "bg" in label:
            # doesn't affect the convergence map
            kg_data_vector = data_vectors[f"kg_fiducial"]
            # the perturbation comes in the per bin bias parameter
            dg_data_vector = data_vectors[f"dg_fiducial"]

            tomo_bg = tomo_bg_perts_dict[label]

        # cosmology perturbations
        else:
            kg_data_vector = data_vectors[f"kg_{label}"]
            dg_data_vector = data_vectors[f"dg_{label}"]

            tomo_bg = tomo_bg_perts_dict["fiducial"]

        """ lensing """

        if m_bias_dist is not None:
            # shear bias, only sample at the fiducial (which always comes first)
            if "fiducial" in label:
                # shape (n_z_bins,)
                m_bias = m_bias_dist.sample()

            # broadcast axis 0 of size n_pix
            kg_data_vector *= 1.0 + m_bias
        else:
            LOGGER.warning(f"No multiplicative shear bias is applied")

        # shape noise
        if noise_scale is not None:
            kg_data_vector += noise_scale * sn
        else:
            LOGGER.warning(f"No shape noise is added")

        # masking
        if masks_metacal is not None:
            kg_data_vector *= masks_metacal
        else:
            LOGGER.warning(f"No masking is applied to metacal")

        """ clustering """

        # go from galaxy density contrast to galaxy number map
        dg_data_vector = tomo_n_gal_maglim * (1 + tomo_bg * dg_data_vector)
        dg_data_vector = tf.where(0 < dg_data_vector, dg_data_vector, 0)

        # Poisson noise, only sample at the fiducial (which always comes first)
        if "fiducial" in label:
            dg_noisy_fiducial = tf.random.poisson(shape=[], lam=dg_data_vector)
            dg_fiducial_noise = dg_noisy_fiducial - dg_data_vector

            dg_data_vector = dg_noisy_fiducial
        else:
            # apply the "fiducial Poisson noise" to the perturbations
            dg_data_vector += dg_fiducial_noise

        # masking
        if mask_maglim is not None:
            dg_data_vector *= mask_maglim
        else:
            LOGGER.warning(f"No masking is applied to maglim")

        # results
        augmented_kg.append(kg_data_vector)
        augmented_dg.append(dg_data_vector)

    # concatenate the perturbation axis
    kg_data_vectors = tf.concat(augmented_kg, axis=0)
    dg_data_vectors = tf.concat(augmented_dg, axis=0)

    # concatenate the tomography axis
    data_vectors = tf.concat([kg_data_vectors, dg_data_vectors], axis=-1)

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
    is_cached: bool = False,
    n_readers: int = 8,
    n_prefetch: int = None,
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

    Args:
        tfr_pattern (str): Glob pattern of the .fiducial tfrecord files.
        local_batch_size (int): Local batch size, will be multiplied with the number of deltas for the total batch size.
        params (list): List of the cosmological parameters with respect to which the perturbations to be used in
            training are returned, see the config for all possibilities. Defaults to None, then all are included
            according to the config.
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). Defaults to None.
        i_noise (int): Index for the shape noise realizations. This has to be fixed and can't be a tf.Variable or
            other tensor (like randomly sampled).
        noise_scale (float): Factor by which to multiply the shape noise. This could also be a tf.Variable to change
            it according to a schedule during training. Set to None to not include any shape noise. Defaults to 1.0.
        is_cached (bool): Whether to cache on the level on the deserialized tensors. This is only feasible if all of
            the fiducial .tfrecords fit into RAM. Defaults to False.
        n_readers (int, optional): Number of parallel readers, i.e. samples read out from different input files
            concurrently. This should be roughly less than a tenth of the number of files. Defaults to 8.
        n_prefetch (int, optional): Number of dataset elements to prefetch. Defaults to None, then tf.data.AUTOTUNE
            is used.
        file_name_shuffle_buffer (int, optional): Defaults to 128.
        examples_shuffle_buffer (int, optional): Defaults to 128.
        file_name_shuffle_seed (int, optional): Defaults to 17.
        examples_shuffle_seed (int, optional): Defaults to 67.
        is_eval (bool, optional): If this is True, then the dataset won't be shuffled repeatedly, such that one can go
            through it deterministically exactly once. Defaults to False.
        eval_seed (int, optional): Fixed seed for evaluation. Defaults to 32.
        input_context (tf.distribute.InputContext, optional): For distributed training, this is passed to the
            dataset_fn like in
            https://www.tensorflow.org/tutorials/distribute/input#tfdistributestrategydistribute_datasets_from_function
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
            The index label consists of (i_example, i_noise)
    """
    LOGGER.info(f"Starting to generate the fiducial training set for i_noise = {i_noise}")
    conf = analysis.load_config(conf)

    # constants
    n_z_metacal = len(conf["survey"]["metacal"]["z_bins"])
    n_z_maglim = len(conf["survey"]["maglim"]["z_bins"])

    pert_labels = parameters.get_fiducial_perturbation_labels(params)
    cosmo_kg_pert_labels = [pert_label for pert_label in pert_labels if not "bg" in pert_label]
    m_bias_dist = shear.get_m_bias_distribution(conf)

    # galaxy biasing
    tomo_n_gal_maglim = tf.constant(conf["survey"]["maglim"]["n_gal"]) * hp.nside2pixarea(
        conf["analysis"]["n_side"], degrees=True
    )
    tomo_bg_perts_dict = parameters.get_tomo_amplitude_perturbations_dict("bg", conf)

    # load the pixel file to get the size of the data vector
    data_vec_pix, _, _, _ = analysis.load_pixel_file(conf)
    n_pix = len(data_vec_pix)

    # masking
    masks_dict = analysis.get_tomo_masks(conf)
    masks_metacal = tf.constant(masks_dict["metacal"], dtype=tf.float32)
    mask_maglim = tf.constant(masks_dict["maglim"], dtype=tf.float32)

    if is_eval:
        LOGGER.warning(f"Evaluation mode is activated, the random seed is fixed and the dataset is not repeated")
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

    # repeat and shuffle the files
    if not is_eval and not is_cached:
        LOGGER.info(f"Shuffling file names")
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

    # parse, output signature (data_vectors, (i_example, i_noise))
    dset = dset.map(
        lambda serialized_example: tfrecords.parse_inverse_fiducial(
            serialized_example, cosmo_kg_pert_labels, i_noise, n_pix, n_z_metacal, n_z_maglim
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if is_cached:
        LOGGER.warning(f"Caching the dataset")
        dset = dset.cache()
        dset = dset.repeat()

    # shuffle the tensors
    if not is_eval and examples_shuffle_buffer is not None:
        dset = dset.shuffle(examples_shuffle_buffer, seed=examples_shuffle_seed)

    # batch (first, for vectorization)
    dset = dset.batch(local_batch_size, drop_remainder=True)
    LOGGER.info(f"Batching into {local_batch_size} elements locally")

    # augmentations (all in one function, to make parallelization faster)
    dset = dset.map(
        lambda data_vectors, index: dset_augmentations(
            data_vectors,
            index,
            pert_labels,
            # kg
            m_bias_dist,
            noise_scale,
            masks_metacal,
            # dg
            tomo_n_gal_maglim,
            tomo_bg_perts_dict,
            mask_maglim,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # prefetch
    if n_prefetch is None:
        n_prefetch = tf.data.AUTOTUNE
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
    is_cached: bool = False,
    n_readers: int = 8,
    n_prefetch: int = None,
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

        tfr_pattern (str): Glob pattern of the .fiducial tfrecord files.
        local_batch_size (int): Local batch size, will be multiplied with the number of deltas for the total batch size.
        params (list): List of the cosmological parameters with respect to which the perturbations to be used in
            training are returned, see the config for all possibilities. Defaults to None, then all are included
            according to the config.
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). Defaults to None.
        n_noise (int): Number of noise indices to include.
        noise_scale (float): Factor by which to multiply the shape noise. This could also be a tf.Variable to change
            it according to a schedule during training. Set to None to not include any shape noise. Defaults to 1.0.
        is_cached (bool): Whether to cache on the level on the deserialized tensors. This is only feasible if all of
            the fiducial .tfrecords fit into RAM. Defaults to False.
        n_readers (int, optional): Number of parallel readers, i.e. samples read out from different input files
            concurrently. This should be roughly less than a tenth of the number of files. Defaults to 8.
        n_prefetch (int, optional): Number of dataset elements to prefetch. Defaults to None, then tf.data.AUTOTUNE
            is used.
        file_name_shuffle_buffer (int, optional): Defaults to 128.
        examples_shuffle_buffer (int, optional): Defaults to 128.
        file_name_shuffle_seed (int, optional): Defaults to 17.
        examples_shuffle_seed (int, optional): Defaults to 67.
        is_eval (bool, optional): If this is True, then the dataset won't be shuffled repeatedly, such that one can go
            through it deterministically exactly once. Defaults to False.
        eval_seed (int, optional): Fixed seed for evaluation. Defaults to 32.
        input_context (tf.distribute.InputContext, optional): For distributed training, this is passed to the
            dataset_fn like in
            https://www.tensorflow.org/tutorials/distribute/input#tfdistributestrategydistribute_datasets_from_function
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
    if file_name_shuffle_buffer is not None:
        file_name_shuffle_buffer = file_name_shuffle_buffer // n_noise

    if examples_shuffle_buffer is not None:
        examples_shuffle_buffer = examples_shuffle_buffer // n_noise

    dset = tf.data.Dataset.sample_from_datasets(
        [
            get_fiducial_dset(
                tfr_pattern=tfr_pattern,
                local_batch_size=local_batch_size,
                params=params,
                conf=conf,
                i_noise=i_noise,
                noise_scale=noise_scale,
                is_cached=is_cached,
                n_readers=n_readers,
                n_prefetch=0,
                file_name_shuffle_buffer=file_name_shuffle_buffer,
                examples_shuffle_buffer=examples_shuffle_buffer,
                file_name_shuffle_seed=file_name_shuffle_seed + i_noise,
                examples_shuffle_seed=examples_shuffle_seed + i_noise,
                input_context=input_context,
            )
            for i_noise in range(n_noise)
        ]
    )

    # prefetch
    if n_prefetch is None:
        n_prefetch = tf.data.AUTOTUNE
    dset = dset.prefetch(n_prefetch)

    LOGGER.info(
        f"Successfully generated the fiducial training set with element_spec {dset.element_spec} for i_noise in"
        f" [0, {n_noise}]"
    )
    return dset
