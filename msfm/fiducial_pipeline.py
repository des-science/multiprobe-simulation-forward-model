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


class FiducialPipeline(MSFMpipeline):
    """
    Sets up a tf.data.Dataset for the fiducial cosmology and its per parameter perturbations.
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
        # noise
        apply_m_bias: bool = True,
        shape_noise_scale: float = 1.0,
        poisson_noise_scale: float = 1.0,
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
            apply_m_bias (bool, optional): Whether to include the multiplicative shear bias. Defaults to True.
            shape_noise_scale (float, optional): Factor by which to multiply the shape noise. This could also be a
                tf.Variable to change it according to a schedule during training. Set to None to not include any shape
                noise. Defaults to 1.0.
            poisson_noise_scale (float, optional): Factor by which to multiply the Poisson noise. This could also be a
                tf.Variable to change it according to a schedule during training. Set to None to not include any 
                Poisson noise. Defaults to 1.0.
        """
        super().__init__(
            conf=conf,
            # cosmology
            params=params,
            with_lensing=with_lensing,
            with_clustering=with_clustering,
            # format
            apply_norm=apply_norm,
            with_padding=with_padding,
            z_bin_inds=z_bin_inds,
            # noise
            apply_m_bias=apply_m_bias,
            shape_noise_scale=shape_noise_scale,
            poisson_noise_scale=poisson_noise_scale,
        )

        # perturbations of cosmo, ia, and bg parameters
        self.pert_labels = parameters.get_fiducial_perturbation_labels(self.params)

    def get_dset(
        self,
        tfr_pattern: str,
        local_batch_size: int,
        n_noise: int = 1,
        # performance
        is_cached: bool = False,
        n_readers: int = 8,
        n_prefetch: int = None,
        file_name_shuffle_buffer: int = 16,
        examples_shuffle_buffer: int = 64,
        # random seeds
        file_name_shuffle_seed: int = 17,
        examples_shuffle_seed: int = 67,
        is_eval: bool = False,
        eval_seed: int = 32,
        # distribution
        input_context: tf.distribute.InputContext = None,
    ) -> tf.data.Dataset:
        """Builds the tf.data.Dataset from the given file name pattern and performance related parameters.

        Args:
            tfr_pattern (str): Glob pattern of the fiducial .tfrecord files.
            local_batch_size (int): Local batch size, will be multiplied with the number of deltas for the total batch
                size.
            n_noise (int, optional): Number of noise realizations to return, where the noise index always runs from 0
                to n_noise - 1. Defaults to 1.
            is_cached (bool): Whether to cache on the level on the deserialized tensors. This is only feasible if all of
                the fiducial .tfrecords fit into RAM. Defaults to False.
            n_readers (int, optional): Number of parallel readers, i.e. different input files read concurrently. This
                should be roughly less than a tenth of the number of files. Large values cost a lot of RAM, especially
                in the distributed setting. Defaults to 4.
            n_prefetch (int, optional): Number of dataset elements to prefetch. Defaults to None, then tf.data.AUTOTUNE
                is used.
            file_name_shuffle_buffer (int, optional): Size of the shuffle buffer for the .tfrecord files. Defaults to
                128.
            examples_shuffle_buffer (int, optional): Size of the shuffle buffer for the non-batched examples. Defaults
                to 128.
            file_name_shuffle_seed (int, optional): Defaults to 17.
            examples_shuffle_seed (int, optional): Defaults to 67.
            is_eval (bool, optional): If this is True, then the dataset won't be shuffled repeatedly, such that one can
                go through it deterministically exactly once. Defaults to False.
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
            tf.data.Dataset: A dataset that returns samples with a given batchsize in the right ordering for the delta
            loss. The index label consists of (i_example, i_noise)
        """
        assert n_noise >= 1, f"n_noise = {n_noise} must be >= 1"

        if is_eval:
            tf.random.set_seed(eval_seed)

            # parameters that are not used
            file_name_shuffle_seed = None
            examples_shuffle_buffer = None
            file_name_shuffle_seed = None
            examples_shuffle_seed = None

            LOGGER.warning(
                f"Evaluation mode is activated: the random seed is fixed, the shuffle arguments ignored, and the "
                f"dataset is not repeated"
            )

        # get the file names
        dset = tf.data.Dataset.list_files(tfr_pattern, shuffle=False)

        # shard for distributed training
        if input_context is not None:
            # NOTE Taken from https://www.tensorflow.org/tutorials/distribute/input#usage_2. This is black magic since
            # print(input_context.num_input_pipelines) yields 1, so I don't know how the sharding happens, but it does,
            # see distributed_sharding.ipynb
            dset = dset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
            LOGGER.info(f"Sharding the dataset over the .tfrecord files according to the input_context")

        # repeat and shuffle the files
        if not is_eval and not is_cached:
            dset = dset.repeat()
            dset = dset.shuffle(file_name_shuffle_buffer, seed=file_name_shuffle_seed)
            LOGGER.info(f"Shuffling file names with shuffle_buffer = {file_name_shuffle_buffer}")

        # interleave, block_length is the number of files every reader reads
        if is_eval:
            dset = dset.interleave(tf.data.TFRecordDataset, cycle_length=n_readers, block_length=1, deterministic=True)
        else:
            dset = dset.interleave(
                tf.data.TFRecordDataset,
                cycle_length=n_readers,
                block_length=1,
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=False,
            )
        LOGGER.info(f"Interleaving with n_readers = {n_readers}")

        # parse, output signature (data_vectors,)
        dset = dset.map(
            lambda serialized_example: tfrecords.parse_inverse_fiducial(
                serialized_example,
                self.pert_labels,
                n_noise,
                # dimensions
                self.n_dv_pix,
                self.n_z_metacal,
                self.n_z_maglim,
                # map types
                self.with_lensing,
                self.with_clustering,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        if is_cached:
            dset = dset.cache()
            dset = dset.repeat()
            LOGGER.warning(f"Caching the dataset")

        # map a single example to n_noise examples corresponding to different noise realizations
        dset = dset.interleave(
            lambda data_vectors: self._split_noise_realizations(data_vectors, n_noise),
            cycle_length=1,
            block_length=1,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=is_eval,
        )

        # shuffle the examples
        if (not is_eval) and (examples_shuffle_buffer is not None) and (examples_shuffle_buffer != 0):
            dset = dset.shuffle(examples_shuffle_buffer, seed=examples_shuffle_seed)
            LOGGER.info(f"Shuffling examples with shuffle_buffer = {examples_shuffle_buffer}")
        elif not is_eval:
            LOGGER.warning(f"Examples are not shuffled, which is underisable for is_eval = {is_eval}")

        # batch (first, for vectorization)
        if not is_eval:
            dset = dset.batch(local_batch_size, drop_remainder=True)
        if is_eval:
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
            f"Successfully generated the fiducial training set with element_spec {dset.element_spec} for i_noise in"
            f" [0, {n_noise})"
        )
        return dset

    def _split_noise_realizations(self, data_vectors: dict, n_noise: int) -> tf.data.Dataset:
        """Split the dictionary stored within the .tfrecord files into the separate noise realizations stored within.
        For this, the signal maps are copied in memory and paired with the noise realizations. So a single element
        of the dataset is mapped to a new dataset containing n_noise examples. Therefore, this function should be
        applied as flat_map or interleave.

        Args:
            data_vectors (dict): Full dictionary containing all kg perturbations, dg perturbations, i_example and
                i_noise indices.

        Returns:
            tf.data.Dataset: Dataset containing the separate noise realizations.
        """

        # separate the noise realizations
        sn = []
        pn = []
        i_noise = []
        for i in range(n_noise):
            sn.append(data_vectors.pop(f"sn_{i}"))
            pn.append(data_vectors.pop(f"pn_{i}"))
            i_noise.append(i)

        # repeat the signal as often as there are different noise realizations
        for key in data_vectors.keys():
            data_vectors[key] = tf.repeat(tf.expand_dims(data_vectors[key], axis=0), n_noise, axis=0)

        # update the dictionary
        data_vectors["sn"] = sn
        data_vectors["pn"] = pn
        data_vectors["i_noise"] = i_noise

        # return a dataset containing n_noise elements
        return tf.data.Dataset.from_tensor_slices(data_vectors)

    def _augmentations(self, data_vectors: dict) -> tf.Tensor:
        """This function wraps _lensing_augmentations and _clustering_augmentations and implements the appropriate case
        distinction.

        Args:
            data_vectors (dict): Full dictionary containing all kg perturbations, dg perturbations, i_example and
                i_noise indices.
            index (tuple): Label (i_example, i_noise), which is only passed through.

        Raises:
            ValueError: If neither lensing nor clustering maps are selected.

        Returns:
            tuple: (out_tensor, index) the elements of the dataset, where index is a tuple (i_example, i_noise).
        """
        LOGGER.warning(f"Tracing _augmentations")
        LOGGER.info(f"Running on the data_vectors.keys() = {data_vectors.keys()}")

        # to be explicit
        with tf.device("/CPU:0"):
            if self.with_lensing and self.with_clustering:
                kg_tensor = self._lensing_augmentations(data_vectors)
                dg_tensor = self._clustering_augmentations(data_vectors)

                # concatenate along the tomography axis
                out_tensor = tf.concat([kg_tensor, dg_tensor], axis=-1)

            elif self.with_lensing:
                assert not any(param in self.pert_labels for param in ["bg", "n_bg"])
                out_tensor = self._lensing_augmentations(data_vectors)

            elif self.with_clustering:
                assert not any(param in self.pert_labels for param in ["Aia", "n_Aia"])
                out_tensor = self._clustering_augmentations(data_vectors)

            else:
                raise ValueError(f"At least one of 'lensing' or 'clustering' maps need to be selected")

            if not self.with_padding:
                LOGGER.info(f"Removing the padding")
                out_tensor = tf.boolean_mask(out_tensor, self.mask_total, axis=1)

        # potentially discard the unwanted redshift bins
        if self.z_bin_inds is not None:
            LOGGER.warning(f"Discarding all redshift bins except {self.z_bin_inds}")
            out_tensor = tf.gather(out_tensor, self.z_bin_inds, axis=-1)

        # gather the indices
        i_example = data_vectors.pop("i_example")
        i_noise = data_vectors.pop("i_noise")

        return out_tensor, (i_example, i_noise)

    def _lensing_augmentations(self, data_vectors: dict) -> tf.Tensor:
        """Applies random augmentations and general pre-processing to the weak lensing maps (kg). This includes in
        order:
            - Load the fiducial for galaxy clustering perturbations
            - Add random multiplicative shear bias (the additive one is negligible) to the kappa maps
            - Add the chosen shape noise realization to the kappa maps
            - Reversibly normalize to roughly unit values
            - Mask the resulting data vector (this is only required if an addition like an additive shear bias is
                applied)
            - Concatenate the batch and perturbations along axis 0 (for compatibility with the delta loss)

        Args:
            data_vectors (dict): Has keys "kg_{pert_label}" and "sn", which contain tensors of shape
                (batch_size, n_pix, n_z_metacal).

        Returns:
            tf.tensor: data_vectors of shape (n_perts * batch_size, n_pix, n_z_metacal). The first batch_size elements
                correspond to the fiducial value, the second to the first perturbation, the third to the second
                perturbation, etc. (for compatibility with the delta loss).
        """
        LOGGER.warning(f"Tracing _lensing_augmentations")

        # shape noise
        sn = data_vectors.pop("sn")

        out_data_vectors = []
        for label in self.pert_labels:
            # galaxy bias perturbations
            if "bg" in label:
                # doesn't affect the convergence map
                data_vector = data_vectors[f"kg_fiducial"]

            # cosmology + intrinsic alignment perturbations
            else:
                data_vector = data_vectors[f"kg_{label}"]

            # shear bias
            if self.apply_m_bias and (self.m_bias_dist is not None):
                # only sample at the fiducial (which always comes first)
                if "fiducial" in label:
                    # shape (n_z_bins,)
                    m_bias = self.m_bias_dist.sample()

                # broadcast axis 0 of size n_pix
                data_vector *= 1.0 + m_bias
            else:
                LOGGER.warning(f"No multiplicative shear bias is applied")

            # shape noise
            if self.shape_noise_scale is not None:
                data_vector += self.shape_noise_scale * sn
            else:
                LOGGER.warning(f"No shape noise is added to the lensing maps")

            # normalization
            if self.apply_norm:
                data_vector = self.normalize_lensing(data_vector)

            # masking
            data_vector *= self.masks_metacal

            out_data_vectors.append(data_vector)

        # concatenate the perturbation axis
        return tf.concat(out_data_vectors, axis=0)

    def _clustering_augmentations(self, data_vectors: dict) -> tf.Tensor:
        """Applies random augmentations and general pre-processing to the clustering maps (dg). This includes in order:
            - Load the fiducial for intrinsic alignment perturbations
            - Mask the resulting data vector (just to be safe)
            - Concatenate the batch and perturbations along axis 0 (for compatibility with the delta loss)

        Args:
            data_vectors (dict): Has keys "dg_{pert_label}", which contain tensors of shape
                (batch_size, n_pix, n_z_maglim).

        Returns:
            tf.tensor: data_vectors of shape (n_perts * batch_size, n_pix, n_z_maglim). The first batch_size elements
                correspond to the fiducial value, the second to the first perturbation, the third to the second
                perturbation, etc. (for compatibility with the delta loss).
        """
        LOGGER.warning(f"Tracing _clustering_augmentations")

        # poisson noise
        pn = data_vectors.pop("pn")

        out_data_vectors = []
        for label in self.pert_labels:
            # intrinsic alignemnt perturbations
            if "Aia" in label:
                # doesn't affect the clustering map
                data_vector = data_vectors[f"dg_fiducial"]
            # cosmology perturbations
            else:
                data_vector = data_vectors[f"dg_{label}"]

            # poisson noise
            if self.poisson_noise_scale is not None:
                data_vector += self.poisson_noise_scale * pn
            else:
                LOGGER.warning(f"No poisson noise is added to the clustering maps")

            # normalization
            if self.apply_norm:
                data_vector = self.normalize_clustering(data_vector)

            # masking
            data_vector *= self.masks_maglim

            out_data_vectors.append(data_vector)

        # concatenate the perturbation axis
        return tf.concat(out_data_vectors, axis=0)
