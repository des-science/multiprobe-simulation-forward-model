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

from msfm.utils import logger, tfrecords, parameters
from msfm.utils.base_pipeline import MSFMpipeline

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


class GridPipeline(MSFMpipeline):
    """
    Sets up a tf.data.Dataset for the grid cosmologies.
    """

    def __init__(
        self,
        conf: dict = None,
        # cosmology
        params: list = None,
        with_lensing: bool = True,
        with_clustering: bool = True,
        # format
        apply_norm: bool = True,
        with_padding: bool = True,
        z_bin_inds: list = None,
    ):
        """Set up the physics parameters of the pipeline.

        Args:
            conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
                passed through) or None (the default config is loaded). Defaults to None.
            params (list): List of the cosmological parameters of interest. Fiducial: perturbations, grid: labels.
            with_lensing (bool, optional): Whether to include the kappa maps. Defaults to True.
            with_clustering (bool, optional): Whether to include the delta maps. Defaults to True.
            apply_norm (bool, optional): Whether to rescale the maps to approximate unit range. Defaults to True.
            with_padding (bool, optional): Whether to include the padding of the data vectors (the healpy DeepSphere \
                networks) need this. Defaults to True.
            z_bin_inds (list, optional): Specify the indices of the redshift bins to be included. Note that this is
                mainly meant for testing purposes and is inefficient, since all redshift bins are loaded from the
                .tfrecords nonetheless. Defaults to None, then all redshift bins are kept.
        """
        super().__init__(
            conf=conf,
            params=params,
            with_lensing=with_lensing,
            with_clustering=with_clustering,
            apply_norm=apply_norm,
            with_padding=with_padding,
            z_bin_inds=z_bin_inds,
            # these are fixed in the .tfrecord files
            apply_m_bias=False,
            shape_noise_scale=1.0,
            poisson_noise_scale=1.0,
        )

        # used to return the correct labels
        self.all_params = parameters.get_parameters(conf=conf)

        # used to reshape the stored tensors, and for nothing else
        self.n_all_params = len(self.all_params)

    def get_dset(
        self,
        tfr_pattern: str,
        local_batch_size: int,
        n_noise: int = 1,
        # performance
        n_readers: int = 8,
        n_prefetch: int = tf.data.AUTOTUNE,
        # training
        is_eval: bool = True,
        file_name_shuffle_buffer: int = 128,
        examples_shuffle_buffer: int = 128,
        file_name_shuffle_seed: int = 11,
        examples_shuffle_seed: int = 12,
        # distribution
        input_context: tf.distribute.InputContext = None,
    ) -> tf.data.Dataset:
        """Builds the tf.data.Dataset from the given file name pattern and performance related parameters.

        Args:
            tfr_pattern (str): Glob pattern of the .fiducial tfrecord files.
            local_batch_size (int): Local batch size. Can also be the string "cosmo". Then, every batch contains all of
                the realisations of exactly one cosmology.
            n_noise (int, optional): Number of noise realizations to return, where the noise index always runs from 0
                to n_noise - 1. Defaults to 1.
            n_readers (int, optional): Number of parallel readers, i.e. samples read out from different input files
                concurrently. This should be roughly less than a tenth of the number of files. Defaults to 8.
            n_prefetch (int, optional): Number of dataset elements to prefetch.
            input_context (Union[tf.distribute.InputContext, deep_lss.utils.distribute.HorovodStrategy], optional):
                Custom input_context attribute of my HorovodStrategy class or when using the TensorFlow builtin
                distribution strategies, this is passed to the dataset_fn like in
                https://www.tensorflow.org/tutorials/distribute/input#tfdistributestrategydistribute_datasets_from_function
                Then, the dataset is sharded. Defaults to None for a non distributed dataset.

                Example usage:
                    def dataset_fn(input_context):
                        dset = fiducial_pipeline.get_grid_dset(
                            tfr_pattern,
                            local_batch_size,
                            input_context=input_context,
                        )

        Returns:
            tf.data.Dataset: A deterministic dataset that goes through the grid cosmologies in the order of the sobol
                seeds. The output is a tuple like (data_vectors, cosmo, index), where data_vectors is a tensor of shape
            (batch_size, n_pix, n_z_metacal + n_z_maglim), cosmo is a label distributed on the Sobol sequence and index
            is a tuple containing (i_sobol, i_noise, i_example).
        """
        assert n_noise >= 1, f"n_noise = {n_noise} must be >= 1"

        # get the file names and dataset them
        dset = tf.data.Dataset.list_files(tfr_pattern, shuffle=(not is_eval), seed=file_name_shuffle_seed)

        # shard for distributed evaluation
        if input_context is not None:
            # NOTE that for the builtin MirroredStrategy, input_context.num_input_pipelines = 1 and
            # input_context.input_pipeline_id = 0, indicating that no sharding happens
            # NOTE My HorovodStrategy is written to be compatible with this

            # Taken from https://www.tensorflow.org/tutorials/distribute/input#usage_2
            dset = dset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
            LOGGER.info(f"Sharding the dataset over the .tfrecord files according to the input context")

        # repeat and shuffle the files
        if not is_eval:
            dset = dset.repeat()
            dset = dset.shuffle(file_name_shuffle_buffer, seed=file_name_shuffle_seed)
            LOGGER.info(f"Shuffling file names with shuffle_buffer = {file_name_shuffle_buffer}")

        # interleave, block_length is the number of files every reader reads
        if local_batch_size == "cosmo":
            assert n_readers == 1, f"Can only read from a single file concurrently when local_batch_size = 'cosmo'"
            assert is_eval, f"The 'cosmo' batching is only for validation"
        dset = dset.interleave(
            tf.data.TFRecordDataset,
            cycle_length=n_readers,
            block_length=1,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=is_eval,
        )
        LOGGER.info(f"Interleaving with n_readers = {n_readers}")

        # parse, output signature (data_vectors, index), where data_vectors is a dict
        dset = dset.map(
            lambda serialized_example: tfrecords.parse_inverse_grid(
                serialized_example,
                n_noise,
                # dimensions
                self.n_dv_pix,
                self.n_z_metacal,
                self.n_z_maglim,
                self.n_all_params,
                # map types
                self.with_lensing,
                self.with_clustering,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # map a single example to n_noise examples corresponding to different noise realizations
        dset = dset.interleave(
            lambda data_vectors: self._split_noise_realizations(data_vectors, n_noise),
            cycle_length=1,
            block_length=1,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )

        # shuffle the examples
        if not is_eval:
            dset = dset.shuffle(examples_shuffle_buffer, seed=examples_shuffle_seed)
            LOGGER.info(f"Shuffling examples with shuffle_buffer = {examples_shuffle_buffer}")

        # batch (first, for vectorization)
        if local_batch_size == "cosmo":
            n_patches = self.conf["analysis"]["n_patches"]
            n_perms_per_cosmo = self.conf["analysis"]["grid"]["n_perms_per_cosmo"]
            local_batch_size = n_patches * n_perms_per_cosmo * n_noise
            LOGGER.info(f"The dset is batched by cosmology")
        dset = dset.batch(local_batch_size, drop_remainder=False)
        LOGGER.info(f"Batching into {local_batch_size} elements locally")

        # augmentations (all in one function, to make parallelization faster)
        dset = dset.map(
            self._augmentations,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # prefetch
        if n_prefetch != 0:
            if n_prefetch is None:
                n_prefetch = tf.data.AUTOTUNE
            dset = dset.prefetch(n_prefetch)
            LOGGER.info(f"Prefetching {n_prefetch} elements")

        LOGGER.info(
            f"Successfully generated the grid validation set with element_spec {dset.element_spec} for i_noise in"
            f" [0, {n_noise})"
        )
        return dset

    def _split_noise_realizations(self, data_vectors: dict, n_noise: int) -> tf.data.Dataset:
        """Split the dictionary stored within the .tfrecord files into the separate noise realizations stored within.
        In this way, a single element of the dataset is mapped to a new dataset. Therefore, this function should be
        applied as flat_map or interleave.

        Args:
            data_vectors (dict): Full dictionary containing all noisy kg and dg maps, i_sobol and i_example indices.

        Returns:
            tf.data.Dataset: Dataset containing the separate noise realizations.
        """

        # separate the noise realizations
        if self.with_lensing:
            kg = []
            i_noise = []
            for i in range(n_noise):
                kg.append(data_vectors.pop(f"kg_{i}"))
                i_noise.append(i)

        if self.with_clustering:
            dg = []
            i_noise = []
            for i in range(n_noise):
                dg.append(data_vectors.pop(f"dg_{i}"))
                i_noise.append(i)

        # repeat the signal as often as there are different noise realizations
        for key in data_vectors.keys():
            data_vectors[key] = tf.repeat(tf.expand_dims(data_vectors[key], axis=0), n_noise, axis=0)

        # update the dictionary
        if self.with_lensing:
            data_vectors["kg"] = kg
        if self.with_clustering:
            data_vectors["dg"] = dg
        data_vectors["i_noise"] = i_noise

        # return a dataset containing n_examples elements
        return tf.data.Dataset.from_tensor_slices(data_vectors)

    def _augmentations(self, data_vectors: dict) -> tf.Tensor:
        """Applies random augmentations and general pre-processing to the maps. This includes in order:

        lensing
        - Add the chosen shape noise realization to the kappa maps
        - Reversibly normalize to roughly unit values
        - Mask the resulting data vector

        clustering
        - Reversibly normalize to roughly unit values
        - Mask the resulting data vector

        Concatenate both along the z bin axis.

        Args:
            data_vectors (dict): Depending on with_clustering and with_lensing, contains the tensors kg (sum of signal
                and intrinsic alignment) and sn (single realization) of shape (n_pix, n_z_metacal) and dg of shape
                (n_pix, n_z_maglim).
            index (tuple): A tuple of two integers (i_sobol, i_noise).

        Returns:
            tuple: (out_tensor, cosmo, index) the elements of the dataset, where out_tensor has shape
            (batch_size, n_pix, n_z_metacal + n_z_maglim), cosmo is a label distributed on the Sobol sequence and index
            is a tuple containing (i_sobol, i_noise).
        """
        LOGGER.warning(f"Tracing _augmentations")
        LOGGER.info(f"Running on the data_vectors.keys() = {data_vectors.keys()}")

        # to be explicit
        with tf.device("/CPU:0"):
            # label, cosmo params
            cosmo = data_vectors.pop("cosmo")
            cosmo = tf.gather(cosmo, [self.all_params.index(param) for param in self.params], axis=1)

            if self.with_lensing:
                # normalization
                if self.apply_norm:
                    data_vectors["kg"] = self.normalize_lensing(data_vectors["kg"])

                # masking
                data_vectors["kg"] *= self.masks_metacal

                out_tensor = data_vectors["kg"]

            if self.with_clustering:
                # normalization
                if self.apply_norm:
                    data_vectors["dg"] = self.normalize_clustering(data_vectors["dg"])

                # masking
                data_vectors["dg"] *= self.masks_maglim

                out_tensor = data_vectors["dg"]

            if self.with_lensing and self.with_clustering:
                # concatenate along the tomography axis
                out_tensor = tf.concat([data_vectors["kg"], data_vectors["dg"]], axis=-1)

            if not self.with_padding:
                LOGGER.info(f"Removing the padding")
                out_tensor = tf.boolean_mask(out_tensor, self.mask_total, axis=1)

        # potentially discard the unwanted redshift bins
        if self.z_bin_inds is not None:
            LOGGER.warning(f"Discarding all redshift bins except {self.z_bin_inds}")
            out_tensor = tf.gather(out_tensor, self.z_bin_inds, axis=-1)

        # gather the indices
        i_sobol = data_vectors.pop("i_sobol")
        i_example = data_vectors.pop("i_example")
        i_noise = data_vectors.pop("i_noise")

        return out_tensor, cosmo, (i_sobol, i_example, i_noise)
