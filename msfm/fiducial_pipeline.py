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

from msfm.utils import logger, tfrecords, parameters, files
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
            apply_m_bias (bool, optional): Whether to include the multiplicative shear bias. Defaults to True.
            shape_noise_scale (float, optional): Factor by which to multiply the shape noise. This could also be a
                tf.Variable to change it according to a schedule during training. Set to None to not include any shape
                noise. Defaults to 1.0.
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
        i_noise: int = 0,
        # performance
        is_cached: bool = False,
        n_readers: int = 8,
        n_prefetch: int = None,
        file_name_shuffle_buffer: int = 128,
        examples_shuffle_buffer: int = 16,
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
            i_noise (int): Index for the shape noise realizations. This has to be fixed and can't be a tf.Variable or
                other tensor (like randomly sampled).
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

        if is_eval:
            LOGGER.warning(f"Evaluation mode is activated, the random seed is fixed and the dataset is not repeated")
            tf.random.set_seed(eval_seed)

            # parameters that are not used
            file_name_shuffle_seed = None
            examples_shuffle_buffer = None
            file_name_shuffle_seed = None
            examples_shuffle_seed = None
            LOGGER.warning(f"In evaluation mode, the shuffle arguments are ignored")

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
            LOGGER.info(f"Shuffling file names")
            dset = dset.repeat()
            dset = dset.shuffle(file_name_shuffle_buffer, seed=file_name_shuffle_seed)

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

        # parse, output signature (data_vectors, (i_example, i_noise))
        dset = dset.map(
            lambda serialized_example: tfrecords.parse_inverse_fiducial(
                serialized_example,
                self.pert_labels,
                i_noise,
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
            LOGGER.warning(f"Caching the dataset")
            dset = dset.cache()
            dset = dset.repeat()

        # shuffle the tensors
        if not is_eval and examples_shuffle_buffer is not None:
            LOGGER.info(f"Shuffling examples")
            dset = dset.shuffle(examples_shuffle_buffer, seed=examples_shuffle_seed)

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
        if n_prefetch is None:
            n_prefetch = tf.data.AUTOTUNE
        dset = dset.prefetch(n_prefetch)

        LOGGER.info(
            f"Successfully generated the fiducial training set with element_spec {dset.element_spec} for"
            f" i_noise = {i_noise}"
        )
        return dset

    def get_multi_noise_dset(
        self,
        tfr_pattern: str,
        local_batch_size: int,
        n_noise: int = 1,
        # performance
        is_cached: bool = False,
        n_readers: int = 8,
        n_prefetch: int = None,
        file_name_shuffle_buffer: int = 128,
        examples_shuffle_buffer: int = 128,
        is_eval: bool = False,
        # random seeds
        file_name_shuffle_seed: int = 17,
        examples_shuffle_seed: int = 67,
        # distribution
        input_context: tf.distribute.InputContext = None,
    ) -> tf.data.Dataset:
        """Like get_dset, but for one of n random noise realizations (instead of fixed one).

        Args:
            n_noise (int, optional): Number of noise indices to include.

        Returns:
            tf.data.Dataset: A dataset that returns samples with a given batchsize in the right ordering for the delta
            loss. The index label consists of (i_example, i_noise)
        """

        dset_kwargs = {
            "tfr_pattern": tfr_pattern,
            "local_batch_size": local_batch_size,
            # performance
            "is_cached": is_cached,
            "n_readers": n_readers,
            "n_prefetch": 0,
            # random seeds
            "is_eval": is_eval,
            # distribution
            "input_context": input_context,
        }

        # deterministic loop over noise realizations
        if is_eval:
            dset = self.get_dset(
                **dset_kwargs,
                i_noise=0,
            )

            for i_noise in range(1, n_noise):
                dset_single = self.get_dset(
                    **dset_kwargs,
                    i_noise=i_noise,
                )
                dset = dset.concatenate(dset_single)

        # random noise realization
        else:
            if file_name_shuffle_buffer is not None:
                file_name_shuffle_buffer = file_name_shuffle_buffer // n_noise

            if examples_shuffle_buffer is not None:
                examples_shuffle_buffer = examples_shuffle_buffer // n_noise

            dset = tf.data.Dataset.sample_from_datasets(
                [
                    self.get_dset(
                        **dset_kwargs,
                        i_noise=i_noise,
                        file_name_shuffle_buffer=file_name_shuffle_buffer,
                        examples_shuffle_buffer=examples_shuffle_buffer,
                        file_name_shuffle_seed=file_name_shuffle_seed + i_noise,
                        examples_shuffle_seed=examples_shuffle_seed + i_noise,
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

    def _augmentations(self, data_vectors: dict, index: tuple) -> tf.Tensor:
        """This function wraps _lensing_augmentations and _clustering_augmentations and implements the appropriate case
        distinction.

        Args:
            data_vectors (dict): Full dictionary containing all kg and dg perturbations.
            index (tuple): Label (i_example, i_noise), which is only passed through.

        Raises:
            ValueError: If neither lensing nor clustering maps are selected.

        Returns:
            tuple: (out_tensor, index) the elements of the dataset.
        """
        LOGGER.warning(f"Tracing _augmentations")
        LOGGER.info(f"Running on the data_vectors.keys() = {data_vectors.keys()}")

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

        return out_tensor, index

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
