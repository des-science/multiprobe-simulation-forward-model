"""
Created on May 2023
Author: Arne Thomsen

Tools to handle the scale cuts/Gaussian smoothing.
"""

import numpy as np
import os, logging

from msfm.utils import files, logger, imports

hp = imports.import_healpy(parallel=True)

LOGGER = logger.get_logger(__file__)


def rad_to_arcmin(theta):
    return theta / np.pi * (180 * 60)


def arcmin_to_rad(theta):
    return theta * np.pi / (60 * 180)


def ell_to_angle(ell, arcmin=False, method="naive"):
    # method like 6.2 of https://academic.oup.com/mnras/article/505/4/5714/6296446
    if method == "naive":
        theta = np.pi / ell

    elif method == "physical":
        theta = 1 / ell

    if arcmin:
        theta = rad_to_arcmin(theta)

    return theta


def angle_to_ell(theta, arcmin=False, method="naive"):
    # method like 6.2 of https://academic.oup.com/mnras/article/505/4/5714/6296446
    if arcmin:
        theta = arcmin_to_rad(theta)

    if method == "naive":
        ell = np.pi / theta

    elif method == "physical":
        ell = 1 / theta

    return ell


def cls_to_smoothed_cls(cls, l_min, l_max=None, theta_fwhm=None, arcmin=True):
    """
    Apply high-pass and low-pass filters to the input power spectrum to obtain a smoothed power spectrum. This is
    written to be consistent with smoothing and smoothalm from healpy.
    https://healpy.readthedocs.io/en/latest/generated/healpy.sphtfunc.smoothalm.html
    Note that cls can't be tomographic.

    Args:
        cls (ndarray): Input power spectra of shape (n_examples, n_ell) or (n_ell,).
        l_min (int): Minimum multipole moment to remove large scales.
        l_max (int, optional): Maximum multipole moment to remove small scales. Defaults to None.
        theta_fwhm (float, optional): Full width at half maximum (FWHM) of the Gaussian smoothing kernel. Defaults to
            None, then l_max has to be provided.
        arcmin (bool, optional): If True, theta_fwhm is in arcminutes. If False, theta_fwhm is in radians. Defaults
            to True.

    Returns:
        ndarray: Smoothed power spectrum.
    """

    assert not (l_max is None and theta_fwhm is None), "Either l_max or theta_fwhm must be specified"
    assert l_max is None or theta_fwhm is None, "Only one of l_max or theta_fwhm can be specified"

    l = np.arange(cls.shape[-1])

    # remove large scales (sigmoid), the sigmoid is close to zero for l < l_min and close to one for l > l_min
    sigmoid = lambda x, delta: 1 / (1 + np.exp(delta - x))
    high_pass_fact = sigmoid(l, delta=l_min) ** 2

    # remove small scales (Gaussian smoothing)
    if l_max is not None:
        theta_fwhm = ell_to_angle(l_max, arcmin)

    if arcmin:
        theta_fwhm = arcmin_to_rad(theta_fwhm)

    sigma = theta_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    # extra square compared to
    # https://github.com/healpy/healpy/blob/be5d47b0720d2de69d422f661d75cd3577327d5a/healpy/sphtfunc.py#L974C13-L975C1
    # because we're smoothing Cls, not alms
    low_pass_fact = np.exp(-0.5 * l * (l + 1) * sigma**2) ** 2

    if cls.ndim == 1:
        cls = cls * high_pass_fact * low_pass_fact
    elif cls.ndim == 2:
        cls = cls * high_pass_fact[np.newaxis, :] * low_pass_fact[np.newaxis, :]
    elif cls.ndim == 3:
        raise NotImplementedError("cls.ndim == 3 not implemented yet")

    return cls


def alm_to_smoothed_map(alm, n_side, l_min, l_max=None, theta_fwhm=None, arcmin=True, nest=False):
    """Takes in alm coefficients and returns a map that has been smoothed according to l_min and l_max or theta_fwhm.

    Args:
        alm (np.array): Single spherical harmonics decomposition.
        n_side (int): Healpix nside of the output map.
        l_min (int): Largest scale.
        l_max (int): Smallest scale, specified as an ell.
        theta_fwhm (float): Smallest scale, specified as an angle, which is used as the FWHM of a Gaussian.
        arcmin (bool, optional): Whether the smallest scale is specified as an angle in arcmin, otherwise it is in
            radian.
        nest (bool, optional): Whether the (full sky) output map should be returned in NEST ordering.

    Returns:
        np.array: Healpy map of shape (n_pix,)
    """

    assert not (l_max is None and theta_fwhm is None), "Either l_max or theta_fwhm must be specified"
    assert l_max is None or theta_fwhm is None, "Only one of l_max or theta_fwhm can be specified"

    # alm are computed for the standard l_max = 3 * n_side - 1
    l = hp.Alm.getlm(3 * n_side - 1)[0]

    # remove large scales (sigmoid), the sigmoid is close to zero for l < l_min and close to one for l > l_min
    if l_min is not None:
        sigmoid = lambda x, delta: 1 / (1 + np.exp(delta - x))
        alm = sigmoid(l, delta=l_min) * alm
        # alm[l < l_min] = 0.0

    # remove small scales (Gaussian smoothing)
    if l_max is not None:
        theta_fwhm = ell_to_angle(l_max, arcmin)

    if arcmin:
        theta_fwhm = arcmin_to_rad(theta_fwhm)

    alm = hp.smoothalm(alm, fwhm=theta_fwhm, pol=False)

    full_map = hp.alm2map(alm, nside=n_side, pol=False)

    if nest:
        full_map = hp.reorder(full_map, r2n=True)

    return full_map, alm


def map_to_smoothed_map(full_map, n_side, l_min, l_max=None, theta_fwhm=None, arcmin=True, nest=False):
    """Takes in a (multiple) full sky healpy map(s) and returns a (multiple) map(s) that has (have) been smoothed
    according to l_min and l_max. The input can either be a single map, or a stack of multiple tomographic bins along
    the final axis.

    Args:
        full_map (np.array): Full sky healpy map(s) of the appropriate n_side and shape (n_pix,) or (n_pix, n_z_bins).
        n_side (int): Healpix nside of the output map.
        l_min (Union[int, list]): Largest scale(s).
        l_max (Union[int, list]): Smallest scale(s).
        theta_fwhm (Union[float, list]): Smallest scale(s), specified as an angle, which is used as the FWHM of a
            Gaussian.
        arcmin (bool, optional): Whether the smallest scale is specified as an angle in arcmin, otherwise it is in
            radian.
        nest (bool, optional): Whether the (full sky) input map is in NEST ordering. The output map is returned in the
            same ordering as the input map.

    Returns:
        np.array: Smoothed (full sky) healpy map(s) of shape (n_pix,) or (n_pix, n_z_bins).
    """

    # healpy path
    conf = files.load_config()
    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    hp_datapath = os.path.join(repo_dir, conf["files"]["healpy_data"])

    # multiple tomographic bins along final axis
    if full_map.ndim == 2:
        n_z_bins = full_map.shape[1]

        if isinstance(l_min, list) and isinstance(l_max, list):
            assert n_z_bins == len(l_min) == len(l_max)
        elif isinstance(l_min, list) and isinstance(theta_fwhm, list):
            assert n_z_bins == len(l_min) == len(theta_fwhm)
        else:
            raise ValueError(f"For tomographic inputs, l_min and l_max or theta_fwhm must be lists of length n_z_bins")

        alms = []
        for i_tomo in range(n_z_bins):
            current_map = full_map[:, i_tomo]
            if nest:
                current_map = hp.reorder(current_map, n2r=True)

            alm = hp.map2alm(current_map, pol=False, use_pixel_weights=True, datapath=hp_datapath)

            if l_max is not None:
                full_map[:, i_tomo], alm = alm_to_smoothed_map(
                    alm, n_side, l_min[i_tomo], l_max=l_max[i_tomo], arcmin=False, nest=nest
                )
            elif theta_fwhm is not None:
                full_map[:, i_tomo], alm = alm_to_smoothed_map(
                    alm, n_side, l_min[i_tomo], theta_fwhm=theta_fwhm[i_tomo], arcmin=arcmin, nest=nest
                )
            else:
                raise ValueError(f"Either l_max or theta_fwhm must be specified")

            alms.append(alm)

        alm = np.stack(alms, axis=1)

    # single map
    elif full_map.ndim == 1:
        assert (isinstance(l_min, int) and isinstance(l_max, int)) or (
            isinstance(l_min, int) and isinstance(theta_fwhm, float)
        )

        if nest:
            full_map = hp.reorder(full_map, n2r=True)

        alm = hp.map2alm(full_map, pol=False, use_pixel_weights=True, datapath=hp_datapath)
        full_map, alm = alm_to_smoothed_map(alm, n_side, l_min, l_max, theta_fwhm, arcmin, nest=nest)

    else:
        raise ValueError(f"Unknown full_map.ndim: {full_map.ndim}, must be 1 or 2")

    return full_map, alm


def data_vector_to_smoothed_data_vector(
    data_vector, data_vec_pix, n_side, l_min, l_max=None, theta_fwhm=None, arcmin=True
):
    """Takes in a (multiple) padded data vector(s) and returns a (multiple) data vectors(s) that has (have) been
    smoothed according to l_min and l_max. The input can either be a single map, or a stack of multiple tomographic
    bins along axis = 1 = -1.

    Args:
        data_vector (np.array): Partial sky padded data vector(s) of shape (len(data_vec_pix),) or
            (len(data_vec_pix), n_z_bins).
        data_vec_pix (np.array): Indices of the (padded) data vector in NEST ordering.
        n_side (int): Healpix nside of the output map.
        l_min (Union[int, list]): Largest scale(s).
        l_max (Union[int, list]): Smallest scale(s).
        theta_fwhm (Union[float, list]): Smallest scale(s), specified as an angle, which is used as the FWHM of a
            Gaussian.
        arcmin (bool, optional): Whether the smallest scale is specified as an angle in arcmin, otherwise it is in
            radian.

    Returns:
        np.array: Smoothed data vector(s) of shape (len(data_vec_pix),) or (len(data_vec_pix), n_z_bins).
    """

    n_pix = hp.nside2npix(n_side)

    # multiple tomographic bins along final axis
    if data_vector.ndim == 2:
        n_z_bins = data_vector.shape[1]

        full_map = np.zeros((n_pix, n_z_bins), dtype=np.float32)
        full_map[data_vec_pix] = data_vector

    # single map
    elif data_vector.ndim == 1:
        full_map = np.zeros(n_pix, dtype=np.float32)
        full_map[data_vec_pix] = data_vector

    else:
        raise ValueError(f"Unknown data_vector.ndim: {data_vector.ndim}, must be 1 or 2")

    full_map, alm = map_to_smoothed_map(full_map, n_side, l_min, l_max, theta_fwhm, arcmin, nest=True)

    data_vector = full_map[data_vec_pix]

    return data_vector, alm


# Gaussian Random Fields ##############################################################################################


def data_vector_to_grf_data_vector(
    np_seed, data_vector, data_vec_pix, n_side, l_min, l_max=None, theta_fwhm=None, arcmin=True
):
    """Takes in a (multiple) padded data vector(s) and returns a (multiple) data vectors(s) that has (have) been
    smoothed according to l_min and l_max and transformed to a Gaussian Random Field. This destroys all non-Gaussian
    information and is meant for testing purposes only, to be compared with Cls. The input can either be a single map,
    or a stack of multiple tomographic bins along axis = 1 = -1.

    Args:
        np_seed (int): A numpy random seed used in the (intrinsically random) generation alm -> map. It's important
        data_vector (np.array): Partial sky padded data vector(s) of shape (len(data_vec_pix),) or
            (len(data_vec_pix), n_z_bins).
        data_vec_pix (np.array): Indices of the (padded) data vector in NEST ordering.
        n_side (int): Healpix nside of the output map.
            that this is the same for all tomographic bins and maps that are added later on the GRF level (like signal
            and noise maps, both for weak lensing and galaxy clustering).
        l_min (Union[int, list]): Largest scale(s).
        l_max (Union[int, list]): Smallest scale(s).
        theta_fwhm (Union[float, list]): Smallest scale(s), specified as an angle, which is used as the FWHM of a
            Gaussian.
        arcmin (bool, optional): Whether the smallest scale is specified as an angle in arcmin, otherwise it is in
            radian.

    Returns:
        np.array: Smoothed data vector(s) of shape (len(data_vec_pix),) or (len(data_vec_pix), n_z_bins).
    """

    n_pix = hp.nside2npix(n_side)

    # alm are computed for the standard l_max = 3 * n_side - 1
    l = hp.Alm.getlm(3 * n_side - 1)[0]

    # healpy path
    conf = files.load_config()
    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    hp_datapath = os.path.join(repo_dir, conf["files"]["healpy_data"])

    # multiple tomographic bins along final axis
    if data_vector.ndim == 2:
        n_z_bins = data_vector.shape[1]

        if isinstance(l_min, list) and isinstance(l_max, list):
            assert n_z_bins == len(l_min) == len(l_max)
        elif isinstance(l_min, list) and isinstance(theta_fwhm, list):
            assert n_z_bins == len(l_min) == len(theta_fwhm)
        else:
            raise ValueError(f"For tomographic inputs, l_min and l_max or theta_fwhm must be lists of length n_z_bins")

        alms = []
        for i_tomo in range(n_z_bins):
            full_map = np.zeros((n_pix), dtype=np.float32)
            full_map[data_vec_pix] = data_vector[:, i_tomo]
            full_map = hp.reorder(full_map, n2r=True)

            alm = hp.map2alm(full_map, pol=False, use_pixel_weights=True, datapath=hp_datapath)

            # remove large scales (hard cut)
            alm[l < l_min[i_tomo]] = 0.0

            # remove small scales (Gaussian smoothing)
            if l_max is not None:
                current_theta_fwhm = ell_to_angle(l_max[i_tomo], arcmin)
            elif theta_fwhm is not None:
                current_theta_fwhm = theta_fwhm[i_tomo]
            else:
                raise ValueError(f"Either l_max or theta_fwhm must be specified")

            if arcmin:
                current_theta_fwhm = arcmin_to_rad(current_theta_fwhm)

            alm = hp.smoothalm(alm, fwhm=current_theta_fwhm, pol=False)

            # make a Gaussian Random Field
            cl = hp.alm2cl(alm)
            np.random.seed(np_seed)
            grf = hp.synfast(cl, nside=n_side, pol=False).astype(np.float32)
            grf = hp.reorder(grf, r2n=True)

            # padding is populated too
            data_vector[:, i_tomo] = grf[data_vec_pix]

            alms.append(alm)

        alm = np.stack(alms, axis=1)

    # single map
    elif data_vector.ndim == 1:
        assert (isinstance(l_min, int) and isinstance(l_max, int)) or (
            isinstance(l_min, int) and isinstance(theta_fwhm, float)
        )

        full_map = np.zeros(n_pix, dtype=np.float32)
        full_map[data_vec_pix] = data_vector
        full_map = hp.reorder(full_map, n2r=True)

        alm = hp.map2alm(full_map, pol=False, use_pixel_weights=True, datapath=hp_datapath)

        # remove large scales (hard cut)
        alm[l < l_min] = 0.0

        # remove small scales (Gaussian smoothing)
        if l_max is not None:
            theta_fwhm = ell_to_angle(l_max, arcmin)

        if arcmin:
            theta_fwhm = arcmin_to_rad(theta_fwhm)

        alm = hp.smoothalm(alm, fwhm=theta_fwhm, pol=False)

        # make a Gaussian Random Field
        cl = hp.alm2cl(alm)
        np.random.seed(np_seed)
        grf = hp.synfast(cl, nside=n_side, pol=False).astype(np.float32)
        grf = hp.reorder(grf, r2n=True)

        # padding is populated too
        data_vector = grf[data_vec_pix]

    else:
        raise ValueError(f"Unknown data_vector.ndim: {data_vector.ndim}, must be 1 or 2")

    return data_vector, alm


def alm_to_grf_map(alm, n_side, l_min, l_max, np_seed):
    """NOTE this function has not been tested yet in conjunction with run_datavectors.py and is deprecated.

    Take in an alm vector and return a full sky Gaussian Random Field in ring ordering. This is for testing purposes
    only and enables a comparison of the networks to power spectra. Note that it's important that the different
    tomographic bins are all generated according to the same np.random.seed, otherwise the tomographic information
    gets lost too.

    Args:
        alm (np.ndarray): Vector of complex alm coefficients. Only a single tomographic bin at a time is supported
            by this function, unlike some of the above in this file.
        n_side (int): Healpix nside of the output map.
        l_min (Union[int, list]): Largest scale(s).
        l_max (Union[int, list]): Smallest scale(s).
        np_seed (int): A numpy random seed used in the (intrinsically random) generation alm -> map.

    Returns:
        np.array: Smoothed full sky healpy map for a single tomographic bin of shape (n_pix,).
    """
    cl = hp.alm2cl(alm)

    # remove large scales
    cl[np.arange(0, l_min)] = 0

    # remove small scales and make a Gaussian Random Field
    np.random.seed(np_seed)
    grf = hp.synfast(cl, nside=n_side, pol=False, fwhm=ell_to_angle(l_max)).astype(np.float32)

    return grf
