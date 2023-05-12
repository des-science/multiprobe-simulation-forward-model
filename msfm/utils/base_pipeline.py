# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created February 2023
Author: Arne Thomsen

Parent class of the fiducial and grid pipelines
"""

import tensorflow as tf
import healpy as hp

from msfm.utils import files, lensing


class MSFMpipeline:
    """Parent class of the fiducial and grid pipeline"""

    def __init__(
        self,
        conf: dict,
        params: list = None,
        with_lensing: bool = True,
        with_clustering: bool = True,
        apply_norm: bool = True,
        # noise
        apply_m_bias: bool = True,
        shape_noise_scale: float = 1.0,
    ):
        """Shared parameters are set up here.

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
        # general constants
        self.conf = files.load_config(conf)

        if params is None:
            self.params = (
                self.conf["analysis"]["params"]["cosmo"]
                + self.conf["analysis"]["params"]["ia"]
                + self.conf["analysis"]["params"]["bg"]
            )
        else:
            self.params = params

        self.apply_norm = apply_norm
        self.shape_noise_scale = shape_noise_scale

        self.n_z_metacal = len(self.conf["survey"]["metacal"]["z_bins"])
        self.n_z_maglim = len(self.conf["survey"]["maglim"]["z_bins"])

        # pixel file
        self.data_vec_pix, _, _, _ = files.load_pixel_file(self.conf)
        self.n_pix = len(self.data_vec_pix)
        self.masks_dict = files.get_tomo_masks(self.conf)
        self.masks_metacal = tf.constant(self.masks_dict["metacal"], dtype=tf.float32)
        self.masks_maglim = tf.constant(self.masks_dict["maglim"], dtype=tf.float32)

        # lensing
        self.with_lensing = with_lensing
        self.apply_m_bias = apply_m_bias
        if apply_m_bias:
            self.m_bias_dist = lensing.get_m_bias_distribution(self.conf)
        else:
            self.m_bias_dist = None
        self.tomo_kappa_std = tf.constant(self.conf["analysis"]["normalization"]["lensing"])
        self.normalize_lensing = lambda lensing_dv: lensing_dv / self.tomo_kappa_std

        # clustering
        self.with_clustering = with_clustering
        self.tomo_n_gal_maglim = tf.constant(self.conf["survey"]["maglim"]["n_gal"]) * hp.nside2pixarea(
            self.conf["analysis"]["n_side"], degrees=True
        )
        self.normalize_clustering = lambda clustering_dv: clustering_dv
        # self.normalize_clustering = lambda clustering_dv: clustering_dv / self.tomo_n_gal_maglim - 1.0
