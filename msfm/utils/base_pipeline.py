# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created February 2023
Author: Arne Thomsen

Parent class of the fiducial and grid pipelines
"""

import tensorflow as tf
import healpy as hp

from msfm.utils import analysis, lensing


class MSFMpipeline:
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
        # general constants
        self.params = params
        self.apply_norm = apply_norm
        self.shape_noise_scale = shape_noise_scale

        self.conf = analysis.load_config(conf)
        self.n_z_metacal = len(conf["survey"]["metacal"]["z_bins"])
        self.n_z_maglim = len(conf["survey"]["maglim"]["z_bins"])

        # pixel file
        self.data_vec_pix, _, _, _ = analysis.load_pixel_file(self.conf)
        self.n_pix = len(self.data_vec_pix)
        self.masks_dict = analysis.get_tomo_masks(self.conf)
        self.masks_metacal = tf.constant(self.masks_dict["metacal"], dtype=tf.float32)
        self.masks_maglim = tf.constant(self.masks_dict["maglim"], dtype=tf.float32)

        # lensing
        self.with_lensing = with_lensing
        self.apply_m_bias = apply_m_bias
        if apply_m_bias:
            self.m_bias_dist = lensing.get_m_bias_distribution(conf)
        else:
            self.m_bias_dist = None
        self.tomo_kappa_std = tf.constant(conf["analysis"]["normalization"]["lensing"])

        # clustering
        self.with_clustering = with_clustering
        self.tomo_n_gal_maglim = tf.constant(conf["survey"]["maglim"]["n_gal"]) * hp.nside2pixarea(
            conf["analysis"]["n_side"], degrees=True
        )
