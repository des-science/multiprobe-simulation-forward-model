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

from msfm.utils import logger, tfrecords
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
        with_padding: bool = True,
        # format
        apply_norm: bool = True,
    ):
        """Set up the physics parameters of the pipeline.

        Args:
            conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
                passed through) or None (the default config is loaded). Defaults to None.
            params (list): List of the cosmological parameters of interest. Fiducial: perturbations, grid: labels.
            with_lensing (bool, optional): Whether to include the kappa maps. Defaults to True.
            with_clustering (bool, optional): Whether to include the delta maps. Defaults to True.
            apply_norm (bool, optional): Whether to rescale the maps to approximate unit range. Defaults to True.
            apply_m_bias (bool, optional): Whether to include the multiplicative shear bias. Defaults to True.
            shape_noise_scale (float, optional): Factor by which to multiply the shape noise. This could also be a
                tf.Variable to change it according to a schedule during training. Set to None to not include any shape
                noise. Defaults to 1.0.
        """
        super().__init__(
            conf=conf,
            params=params,
            with_lensing=with_lensing,
            with_clustering=with_clustering,
            apply_norm=apply_norm,
            with_padding=with_padding,
            # these are fixed in the .tfrecord files
            apply_m_bias=False,
            shape_noise_scale=1.0,
            poisson_noise_scale=1.0,
        )

        # used to return the correct labels
        self.all_params = (
            self.conf["analysis"]["params"]["cosmo"]
            + self.conf["analysis"]["params"]["ia"]
            + self.conf["analysis"]["params"]["bg"]
        )
        # used to reshape the stored tensors, and for nothing else
        self.n_all_params = len(self.all_params)

    def get_dset(
        self,
        tfr_pattern: str,
        local_batch_size: int,
        i_noise: int = 0,
        # performance
        n_readers: int = 8,
        n_prefetch: int = tf.data.AUTOTUNE,
        # distribution
        input_context: tf.distribute.InputContext = None,
    ) -> tf.data.Dataset:
        """Builds the tf.data.Dataset from the given file name pattern and performance related parameters.

        Args:
            tfr_pattern (str): Glob pattern of the .fiducial tfrecord files.
            local_batch_size (int): Local batch size. Can also be the string "cosmo". Then, every batch contains all of
                the realisations of exactly one cosmology.
            i_noise (int): Index for the shape noise realizations. This has to be fixed and can't be a tf.Variable or
                other tensor (like randomly sampled).
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
            (batch_size, n_pix, n_z_metacal + n_z_maglim), cosmo is a label distributed on the Sobol sequence and index
            is a tuple containing (i_sobol, i_noise, i_example).
        """

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
        LOGGER.info(f"Interleaving with n_readers = {n_readers}")

        # parse, output signature (data_vectors, index), where data_vectors is a dict
        dset = dset.map(
            lambda serialized_example: tfrecords.parse_inverse_grid(
                serialized_example,
                i_noise,
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

        # batch (first, for vectorization)
        if local_batch_size == "cosmo":
            n_patches = self.conf["analysis"]["n_patches"]
            n_perms_per_cosmo = self.conf["analysis"]["grid"]["n_perms_per_cosmo"]
            local_batch_size = n_patches * n_perms_per_cosmo
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
            f"Successfully generated the grid set with element_spec {dset.element_spec} for i_noise = {i_noise}"
        )
        return dset

    def get_multi_noise_dset(
        self,
        tfr_pattern: str,
        local_batch_size: int,
        n_noise: int = 1,
        # performance
        n_readers: int = 8,
        n_prefetch: int = tf.data.AUTOTUNE,
        # distribution
        input_context: tf.distribute.InputContext = None,
    ) -> tf.data.Dataset:
        """Like get_dset, but for one of n random noise realizations (instead of fixed one).

        Args:
            n_noise (int, optional): Number of noise indices to include. This starts at zero, so if it is set to 2 for
                training, a value of i_noise=2 in self.get_dset would yield an unseen validation set.

        Returns:
            tf.data.Dataset: A deterministic dataset that goes through the grid cosmologies in the order of the sobol
                seeds. The output is a tuple like (data_vectors, cosmo, index), where data_vectors is a tensor of shape
            (batch_size, n_pix, n_z_metacal + n_z_maglim), cosmo is a label distributed on the Sobol sequence and index
            is a tuple containing (i_sobol, i_noise, i_example).
        """

        # larger values take up more RAM, so when multiple dsets are generated like here, care must be taken
        n_readers = n_readers // n_noise

        dset = self.get_dset(
            tfr_pattern=tfr_pattern,
            local_batch_size=local_batch_size,
            i_noise=0,
            n_readers=n_readers,
            n_prefetch=0,
            input_context=input_context,
        )

        for i_noise in range(1, n_noise):
            dset_single = self.get_dset(
                tfr_pattern=tfr_pattern,
                local_batch_size=local_batch_size,
                i_noise=i_noise,
                n_readers=n_readers,
                n_prefetch=0,
                input_context=input_context,
            )
            dset = dset.concatenate(dset_single)

        # prefetch
        if n_prefetch != 0:
            if n_prefetch is None:
                n_prefetch = tf.data.AUTOTUNE
            dset = dset.prefetch(n_prefetch)
            LOGGER.info(f"Prefetching {n_prefetch} elements")

        LOGGER.info(
            f"Successfully generated the grid set with element_spec {dset.element_spec} for i_noise in [0, {n_noise}]"
        )
        return dset

    def _augmentations(self, data_vectors: dict, index: tuple) -> tf.Tensor:
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

        return out_tensor, cosmo, index
