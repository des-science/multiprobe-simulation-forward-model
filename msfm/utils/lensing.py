"""
Created on October 2022
Author: Arne Thomsen

Tools to handle the scale cuts, kaiser-squires transformation and multiplicative and additive shear biases.
"""

import numpy as np

from msfm.utils import files, logger, scales, imports

hp = imports.import_healpy()

LOGGER = logger.get_logger(__file__)


def get_kaiser_squires_factors(l_max):
    """Factors for a spherical Kaiser Squires transformation
    from eq. (11) in https://academic.oup.com/mnras/article/505/3/4626/6287258
    """
    l = hp.Alm.getlm(l_max)[0]

    kappa2gamma_fac = np.where(
        np.logical_and(l != 1, l != 0),
        -np.sqrt(((l + 2.0) * (l - 1)) / ((l + 1) * l)),
        0,
    )
    gamma2kappa_fac = np.where(
        np.logical_and(l != 1, l != 0),
        1 / kappa2gamma_fac,
        0,
    )
    l_mask_fac = np.where(np.logical_and(l != 1, l != 0), 1.0, 0.0)

    return kappa2gamma_fac, gamma2kappa_fac, l_mask_fac


def get_m_bias_distribution(conf=None):
    """Return a tensorflow probability distribution from which the (shear) multiplicative bias can be sampled.

    Args:
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). Defaults to None.

    Returns:
        tfp.distribution: Multiplicative bias.s
    """
    conf = files.load_config(conf)

    import tensorflow_probability as tfp

    m_bias_dist = tfp.distributions.MultivariateNormalDiag(
        loc=conf["survey"]["metacal"]["shear_bias"]["multiplicative"]["mu"],
        scale_diag=conf["survey"]["metacal"]["shear_bias"]["multiplicative"]["sigma"],
    )

    return m_bias_dist


def mode_removal(
    gamma1_patch,
    gamma2_patch,
    gamma2kappa_fac,
    n_side,
    hp_datapath=None,
    keep_b_mode=False,
    # deprecated
    apply_smoothing=False,
    l_min=None,
    l_max=None,
    make_grf=False,
    np_seed=None,
):
    """Takes in survey patches of gamma maps and puts out survey patches of kappa maps that only contain E-modes.

    Masking a pure-E shear field leaks power into the B-mode; the default (``keep_b_mode=False``) discards that
    B-mode as in the standard kappa-only forward model. When ``keep_b_mode=True`` the B-mode convergence patch is
    also reconstructed (from the *same* spin-2 alm transform, so no extra map2alm) and returned alongside the E-mode
    patch, for the B-mode information-loss Fisher study.

    Args:
        gamma1_patch (np.ndarray): Array of size n_pix, but only the survey patch is populated
        gamma2_patch (np.ndarray): Same
        gamma2kappa_fac (np.ndarray): Kaiser squires conversion factors
        n_side (int): Resolution of the map
        keep_b_mode (bool, optional): If True, also reconstruct and return the B-mode convergence patch. Defaults
            to False (E-mode only, back-compatible single-array return).
        apply_smoothing (bool, optional): Whether to apply smoothing to the kappa map. This is included here because
            the alm coefficients are already computed anyways for the mode removal. Defaults to False.
        l_min (int, optional): Minimal ell, this removes the large scales if smoothing is applied. Defaults to None.
        l_max (int, optional): Maximal ell, this smoothes the small scales if smoothing is applied. Defaults to None.
        make_grf (bool, optional): Whether to degrade the map to a Gaussian random field instead of a smoothed map.
            Defaults to False.
        hp_datapath (str, optional): Path to a healpy pixel weights file. Defaults to None.

    Returns:
        np.ndarray: E-mode kappa patch of size n_pix (only the survey patch is populated). If ``keep_b_mode`` is
            True, a tuple ``(kappa_E_patch, kappa_B_patch)`` of two such arrays is returned instead.
    """
    # gamma: map -> alm
    _, gamma_alm_E, gamma_alm_B = hp.map2alm(
        [np.zeros_like(gamma1_patch), gamma1_patch, gamma2_patch],
        pol=True,
        use_pixel_weights=True,
        datapath=hp_datapath,
    )
    # gamma -> kappa
    kappa_alm = gamma_alm_E * gamma2kappa_fac

    # kappa: alm -> map
    if apply_smoothing:
        assert not keep_b_mode, "keep_b_mode is not supported with the deprecated in-mode-removal smoothing"
        LOGGER.warning("Double check what you're doing, smoothing within the mode removal has been deprecated")
        if make_grf:
            kappa_patch = scales.alm_to_grf_map(kappa_alm, l_min, l_max, n_side, np_seed)
        else:
            kappa_patch = scales.alm_to_smoothed_map(kappa_alm, n_side, l_min, l_max, nest=False)
    else:
        kappa_patch = hp.alm2map(kappa_alm, n_side, pol=False).astype(np.float32)

    if keep_b_mode:
        # reuse the spin-2 alms already computed above: only the projection factor differs
        kappa_alm_B = gamma_alm_B * gamma2kappa_fac
        kappa_patch_B = hp.alm2map(kappa_alm_B, n_side, pol=False).astype(np.float32)
        return kappa_patch, kappa_patch_B

    return kappa_patch


# making this a tf.function doesn't speed things up because the seg_ids are always different
def noise_gen(counts, cat_dist, n_noise_per_signal, seed=None):
    """Generates shape noise from a map of galaxy counts and joint distribution of absolute shear values and their
    weights.

    Args:
        counts (np.ndarray): Array of shape (len(base_patch_pix),) that contains the galaxy count per pixel
        cat_dist (tfp.distributions): Distribution with samples of length 2 that contains the absolute magnitudes and
            weights
        n_noise_per_signal (int): Number of noise realizations to create, this dimension is included for vectorization
        seed (int, optional): Seed of the ellipticity and orientation draws. Defaults to None, which leaves the global
            tensorflow RNG alone. When given, the realization becomes a deterministic function of the seed and of the
            call itself, since tf.random.set_seed also resets the eager operation counter.

    Returns:
        np.ndarray: Arrays of shape (len(base_patch_pix, n_noise_per_signal) containing the two gamma components
    """

    import tensorflow as tf

    if seed is not None:
        tf.random.set_seed(seed)

    # indices to sum over all of the galaxies in the individual pixels. np.repeat rather than a Python loop over
    # ~23e6 galaxies, which dominated the runtime once the caller started drawing one count realization per noise
    # realization instead of one per patch
    seg_ids = np.repeat(np.arange(len(counts), dtype=np.int32), counts)

    # This forward model runs on CPU nodes, so pin the ops there like noise_gen_in_place already does rather than
    # letting an incidentally visible GPU pick them up. Beyond avoiding GPU OOM, it is what makes the seed above a
    # real reproducibility guarantee: segment_sum accumulates with atomics on GPU, so the summation order -- and with
    # it the float32 rounding -- varies from run to run even for identical draws.
    with tf.device("/CPU:0"):
        # make a tensor, this is important for performance
        seg_ids = tf.constant(seg_ids, dtype=tf.int32)

        # total number of galaxies in the patch
        n_gals_patch = len(seg_ids)

        # shape (n_gals_patch, n_noise_per_signal, 2)
        cat_samples = cat_dist.sample(sample_shape=(n_gals_patch, n_noise_per_signal))
        # shape (n_gals_patch, n_noise_per_signal)
        phase_samples = tf.random.uniform(
            shape=(
                n_gals_patch,
                n_noise_per_signal,
            ),
            minval=0,
            maxval=2 * np.pi,
        )

        # shape (n_gals_patch, n_noise_per_signal)
        g1_samples = tf.math.cos(phase_samples) * cat_samples[..., 0]
        g2_samples = tf.math.sin(phase_samples) * cat_samples[..., 0]
        w_samples = cat_samples[..., 1]

        # shape (n_gals_patch, n_noise_per_signal, 3)
        weighted_gamma_samples = tf.stack([g1_samples * w_samples, g2_samples * w_samples, w_samples], axis=-1)

        # len(base_patch_pix), unless the final pixels of the patch don't contain galaxies. Then, it's smaller
        sum_per_pix = tf.math.segment_sum(weighted_gamma_samples, seg_ids)

        # normalize with weights, set 0/0 equal to 0 instead of nan
        gamma_per_pix = tf.math.divide_no_nan(sum_per_pix[..., :2], tf.expand_dims(sum_per_pix[..., 2], axis=-1))

        # The condition means that the final pixel contains zero galaxies. Then, its index is not included in the
        # seg_ids (multiplication with zero) and because it's the last, tensorflow has no way of knowing that it
        # should still take the segmented_sum over this index, which evaluates to zero. The while loop allows more
        # than one of the last pixels to be zero.
        n_final_zero_pix = 0
        while counts[-(n_final_zero_pix + 1)] == 0:
            n_final_zero_pix += 1

        if n_final_zero_pix > 0:
            # There is no galaxy in the final pixels, so the shape noise there is equal to zero
            zero_pix = tf.zeros((n_final_zero_pix, n_noise_per_signal, 2), dtype=tf.float32)
            gamma_per_pix = tf.concat((gamma_per_pix, zero_pix), axis=0)

    # shape (len(base_patch_pix), n_noise_per_signal)
    return gamma_per_pix[..., 0].numpy(), gamma_per_pix[..., 1].numpy()


def noise_gen_in_place(gamma_abs, w, pix, base_patch_pix, n_pix, n_noise_per_signal, seed=None):
    """Generates shape noise by rotating galaxies from the catalog in-place.

    Args:
        gamma_abs (np.ndarray or tf.Tensor): Absolute shear |e| for each catalog galaxy
        w (np.ndarray or tf.Tensor): Weight for each catalog galaxy
        pix (np.ndarray or tf.Tensor): Pixel index for each catalog galaxy in the full sky map
        base_patch_pix (np.ndarray): The pixels that make up the current footprint cutout
        n_pix (int): Total number of pixels in the healpy map
        n_noise_per_signal (int): Number of noise realizations
        seed (int, optional): Seed of the orientation draws, see noise_gen. Defaults to None, which leaves the global
            tensorflow RNG alone.

    Returns:
        np.ndarray: Arrays of shape (len(base_patch_pix), n_noise_per_signal) containing the two gamma components
    """
    import tensorflow as tf

    if seed is not None:
        tf.random.set_seed(seed)

    # Place operations on CPU to avoid GPU OOM on shared login nodes where GPU memory is highly restricted
    with tf.device("/CPU:0"):
        pix = tf.cast(pix, tf.int32)
        n_gals = tf.shape(gamma_abs)[0]

        # shape (n_gals, n_noise_per_signal)
        phase_samples = tf.random.uniform(
            shape=(
                n_gals,
                n_noise_per_signal,
            ),
            minval=0,
            maxval=2 * np.pi,
        )

        g1_samples = tf.math.cos(phase_samples) * tf.expand_dims(gamma_abs, axis=1)
        g2_samples = tf.math.sin(phase_samples) * tf.expand_dims(gamma_abs, axis=1)
        w_samples = tf.expand_dims(w, axis=1)

        weighted_g1 = g1_samples * w_samples
        weighted_g2 = g2_samples * w_samples

        sum_g1 = tf.math.unsorted_segment_sum(weighted_g1, pix, num_segments=n_pix)
        sum_g2 = tf.math.unsorted_segment_sum(weighted_g2, pix, num_segments=n_pix)
        sum_w = tf.math.unsorted_segment_sum(w_samples, pix, num_segments=n_pix)

        gamma1_per_pix = tf.math.divide_no_nan(sum_g1, sum_w)
        gamma2_per_pix = tf.math.divide_no_nan(sum_g2, sum_w)

        gamma1_patch = tf.gather(gamma1_per_pix, base_patch_pix)
        gamma2_patch = tf.gather(gamma2_per_pix, base_patch_pix)

    return gamma1_patch.numpy(), gamma2_patch.numpy()


def source_clustering_factor(delta, b_sc, floor=1e-3):
    """Per-pixel shape-noise modulation for the Gatti et al. (https://arxiv.org/abs/2307.13860)
    source-clustering model.

    The source galaxy density is modulated by the matter density field as (1 + b_sc * delta), so the
    shape noise of the pixel-averaged shear scales as f = 1 / sqrt(1 + b_sc * delta): overdense regions
    host more sources and therefore have less shape noise. This is the exact per-pixel reduction of
    eq. (5), since the modulation is constant within a pixel and factors out of the catalog binning.

    Args:
        delta (np.ndarray): Per-pixel simulation source-bin density contrast (delta = (n - <n>) / <n>).
        b_sc (float): Source-clustering bias of the tomographic bin.
        floor (float, optional): Lower clip on (1 + b_sc * delta) to keep it positive. Defaults to 1e-3.

    Returns:
        np.ndarray: Per-pixel modulation factor f, same shape as delta.
    """
    return 1.0 / np.sqrt(np.clip(1.0 + b_sc * delta, a_min=floor, a_max=None))


def shape_noise_variance_map(gamma_abs, w, pix, n_pix):
    """Per-pixel reference (no source-clustering) shape-noise variance of the weighted mean ellipticity.

    For a pixel with galaxies of absolute ellipticity |e| and weight w, randomly rotating the
    ellipticities gives a weighted-mean shear whose variance (e1 and e2 components summed) is
    sum(w**2 * |e|**2) / (sum(w))**2. This is cosmology-independent and is used for the kurtosis term
    of the Gatti calibration (see source_clustering_factor).

    Args:
        gamma_abs (np.ndarray): Absolute ellipticity |e| of each catalog galaxy.
        w (np.ndarray): Weight of each catalog galaxy.
        pix (np.ndarray): Full-sky pixel index of each catalog galaxy.
        n_pix (int): Total number of pixels in the healpy map.

    Returns:
        np.ndarray: Per-pixel reference shape-noise variance of shape (n_pix,).
    """
    pix = np.asarray(pix).astype(np.int64)
    num = np.bincount(pix, weights=w**2 * gamma_abs**2, minlength=n_pix)
    den = np.bincount(pix, weights=w, minlength=n_pix)

    var = np.zeros(n_pix, dtype=np.float64)
    mask = den > 0
    var[mask] = num[mask] / den[mask] ** 2

    return var
