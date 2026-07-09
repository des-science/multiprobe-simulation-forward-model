# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created February 2023
Author: Arne Thomsen

Functions to handle the configuration and read in the survey files on the data vector pixels, masks and noise
"""

import os, h5py, warnings
import numpy as np

from msfm.utils import logger, input_output, filenames, scales, maps, imports

hp = imports.import_healpy()

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


def load_config(conf=None):
    """Loads or passes through a config

    Args:
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). Defaults to None.

    Raises:
        ValueError: When an invalid conf is passed

    Returns:
        dict: A configuration dictionary
    """
    # load the default config within this repo
    if conf is None:
        file_dir = os.path.dirname(__file__)
        repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
        conf = os.path.join(repo_dir, "configs/config.yaml")
        LOGGER.warning(f"Loading the default config from {conf}")
        conf = input_output.read_yaml(conf)

    # load a config specified by a path
    elif isinstance(conf, str):
        conf = input_output.read_yaml(conf)

    # pass through an existing config
    elif isinstance(conf, dict):
        pass

    else:
        raise ValueError(f"conf {conf} must be None, a str specifying the path to the .yaml file, or the read dict")

    return conf


def load_pixel_file(conf=None):
    """Loads the .h5 file that contains the pixel indices associated with the survey like the different patches. That
    file is generated in notebooks/survey_file_gen/pixel_file.ipynb. If the conf argument is not passed, the default
    within the directory where this file resides is used.

    Args:
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). The relative paths are stored here. Defaults to
            None.

    Returns:
        data_vec_pix: data vector pixels including padding in NEST ordering (non-tomographic).
        patches_pix_dict: For "metacal" (tomographic) and "maglim" (non-tomographic), four patch indices in RING
            ordering to cut out from the full sky maps.
        corresponding_pix_dict: For "metacal" (tomographic) and "maglim" (non-tomographic), needed to convert the
            pixels in RING ordering to NEST inside the datavector.
        gamma2_signs: Signs for gamma2 that come from mirroring the survey patch, needed for Metacal only.
    """
    conf = load_config(conf)

    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    pixel_file = os.path.join(repo_dir, conf["files"]["pixels"])

    with h5py.File(pixel_file, "r") as f:
        # pixel indices of padded data vector
        data_vec_pix = f["data_vec"][:]

        # Metacal sample: weak lensing
        metacal_tomo_patches_pix = []
        metacal_tomo_corresponding_pix = []
        for z_bin in conf["survey"]["metacal"]["z_bins"]:
            # shape (4, pix_in_bin)
            patches_pix = f[f"metacal/patches/{z_bin}"][:]
            # shape (pix_in_bin,)
            corresponding_pix = f[f"metacal/patch_to_data_vec/{z_bin}"][:]

            metacal_tomo_patches_pix.append(patches_pix)
            metacal_tomo_corresponding_pix.append(corresponding_pix)

        # to correct the shear for patch cut outs that have been mirrored
        gamma2_signs = f["metacal/gamma_2_sign"][:]

        # Maglim sample: galaxy clustering
        maglim_tomo_patches_pix = []
        maglim_tomo_corresponding_pix = []
        for z_bin in conf["survey"]["maglim"]["z_bins"]:
            patches_pix = f[f"maglim/patches/{z_bin}"][:]
            corresponding_pix = f[f"maglim/patch_to_data_vec/{z_bin}"][:]

            maglim_tomo_patches_pix.append(patches_pix)
            maglim_tomo_corresponding_pix.append(corresponding_pix)

    LOGGER.debug(f"Loaded the pixel file {pixel_file}")

    # package into dictionaries
    patches_pix_dict = {}
    patches_pix_dict["metacal"] = metacal_tomo_patches_pix
    patches_pix_dict["maglim"] = maglim_tomo_patches_pix

    corresponding_pix_dict = {}
    corresponding_pix_dict["metacal"] = metacal_tomo_corresponding_pix
    corresponding_pix_dict["maglim"] = maglim_tomo_corresponding_pix

    return data_vec_pix, patches_pix_dict, corresponding_pix_dict, gamma2_signs


def get_clustering_systematics(conf=None, pixel_type="data_vector", apply_smoothing=False):
    """Per (maglim) tomographic bin survey systematics maps packaged as data vectors, such that the maps can be
    multiplied on that level.

    Args:
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). Defaults to None.
        pixel_type (str, optional): Either "map" or "data_vector", determines whether the systematics map is returned
            as a full sky healpy map or in data vector format.

    Returns:
        list: len = n_z_maglim
    """
    assert pixel_type in ["map", "data_vector"]

    conf = load_config(conf)

    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    pixel_file = os.path.join(repo_dir, conf["files"]["pixels"])

    with h5py.File(pixel_file, "r") as f:
        tomo_sys = []
        for z_bin in conf["survey"]["maglim"]["z_bins"]:
            tomo_sys.append(f[f"maglim/systematics/{pixel_type}/{z_bin}"][:])

    if apply_smoothing:
        # constants
        data_vec_pix, patches_pix_dict, _, _ = load_pixel_file(conf)
        n_side = conf["analysis"]["n_side"]
        n_pix = hp.nside2npix(n_side)
        tomo_l_min = conf["analysis"]["scale_cuts"]["maglim"]["l_min"]
        tomo_theta_fwhm = conf["analysis"]["scale_cuts"]["maglim"]["theta_fwhm"]

        for sys, l_min, theta_fwhm in zip(tomo_sys, tomo_l_min, tomo_theta_fwhm):
            if pixel_type == "map":
                # populate the survey footprint
                base_patch_pix = patches_pix_dict["maglim"][0]
                sys_map = np.zeros(n_pix)
                sys_map[base_patch_pix] = sys
                sys = scales.map_to_smoothed_map(sys_map, n_side, l_min, theta_fwhm=theta_fwhm)

            elif pixel_type == "data_vector":
                sys = scales.data_vector_to_smoothed_data_vector(
                    sys, data_vec_pix, n_side, l_min, theta_fwhm=theta_fwhm
                )

            else:
                raise ValueError(f"Unsupported pixel_type = {pixel_type}")

    # shape (n_pix, n_z_maglim)
    return np.stack(tomo_sys, axis=-1)


def get_tomo_dv_masks(conf=None):
    """Masks the data vectors for the different tomographic bins. (NEST ordering)

    Args:
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). Defaults to None.

    Returns:
        dict: For "metacal" (tomographic) and "maglim" (non-tomographic), mask array of shape (n_pix, n_z_bins) that
            is zero for the padding and one for the data.
    """
    data_vec_pix, _, corresponding_pix_dict, _ = load_pixel_file(conf)

    masks_metacal = []
    # loop over the tomographic bins
    for pix in corresponding_pix_dict["metacal"]:
        mask = np.zeros(len(data_vec_pix), dtype=np.int32)
        # loop over individual pixels
        for p in pix:
            mask[p] = 1
        masks_metacal.append(mask)

    masks_maglim = []
    # loop over the tomographic bins
    for pix in corresponding_pix_dict["maglim"]:
        mask = np.zeros(len(data_vec_pix), dtype=np.int32)
        # loop over individual pixels
        for p in pix:
            mask[p] = 1
        masks_maglim.append(mask)

    masks_dict = {
        "metacal": np.array(masks_metacal).T,
        "maglim": np.array(masks_maglim).T,
    }

    return masks_dict


def get_dv_mask(conf=None):
    masks_dict = get_tomo_dv_masks(conf)

    assert np.all(masks_dict["metacal"] == masks_dict["maglim"]), "The masks for metacal and maglim should be the same"
    assert np.all(
        masks_dict["metacal"] == masks_dict["metacal"][:, 0][:, None]
    ), "The mask should be the same for all tomographic bins"

    return masks_dict["metacal"][:, 0].astype(bool)


def get_tomo_masks(conf=None, nest_out=True):
    conf = load_config(conf)

    n_pix = hp.nside2npix(conf["analysis"]["n_side"])
    data_vec_pix, _, _, _ = load_pixel_file(conf)
    dv_masks_dict = get_tomo_dv_masks(conf)

    masks_dict = {}
    for sample in dv_masks_dict.keys():
        dv_masks = dv_masks_dict[sample]
        masks = np.zeros((n_pix, dv_masks.shape[-1]))
        masks[data_vec_pix] = dv_masks

        if nest_out == False:
            masks = maps.tomographic_reorder(masks, n2r=True)

        masks_dict[sample] = masks

    return masks_dict


def get_mask(conf=None, nest_out=True):
    masks_dict = get_tomo_masks(conf, nest_out)

    assert np.all(masks_dict["metacal"] == masks_dict["maglim"]), "The masks for metacal and maglim should be the same"
    assert np.all(
        masks_dict["metacal"] == masks_dict["metacal"][:, 0][:, None]
    ), "The mask should be the same for all tomographic bins"

    return masks_dict["metacal"][:, 0].astype(bool)


def load_noise_file(conf=None):
    """Loads the .h5 file that contains the noise information of the survey. That
    file is generated in notebooks/survey_file_gen/noise_file.ipynb

    Args:
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). The relative paths are stored here. Defaults to
            None.

    Returns:
        tomo_gamma_cat: list for the tomographic bins containing all of the gamma values for the galaxies in the survey
    """
    conf = load_config(conf)

    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    noise_file = os.path.join(repo_dir, conf["files"]["noise"])

    with h5py.File(noise_file, "r") as f:
        tomo_gamma_cat = []
        for z_bin in conf["survey"]["metacal"]["z_bins"]:
            # shape (n_gal, 4) with e1, e2, w, pix (pix is the full-sky pixel index per galaxy,
            # used by the 'rotate' and 'gatti' source-clustering modes)
            gamma_cat = f[f"{z_bin}/cat"][:]

            tomo_gamma_cat.append(gamma_cat)
    LOGGER.info(f"Loaded the noise file")

    return tomo_gamma_cat


def load_redshift_distributions(galaxy_sample, conf=None):
    """Load the redshift distributions from disk to memory.

    Args:
        galaxy_sample (str): Either "metacal" or "maglim".
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). The relative paths are stored here. Defaults to
            None.

    Returns:
        list: Per redshift bin z an nz values of the distribution.
    """
    assert galaxy_sample in ["maglim", "metacal"]

    conf = load_config(conf)

    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    redshift_dir = os.path.join(repo_dir, conf["dirs"]["redshift_distributions"])

    n_z_bins = len(conf["survey"][galaxy_sample]["z_bins"])

    tomo_z = []
    tomo_nz = []
    for i_tomo in range(1, n_z_bins + 1):
        z_dist_file = filenames.get_filename_z_distribution(redshift_dir, galaxy_sample, i_tomo)
        z_dist = np.loadtxt(z_dist_file)

        tomo_z.append(z_dist[:, 0])
        tomo_nz.append(z_dist[:, 1])

    return tomo_z, tomo_nz


def read_metacal_bias(key, conf=None):
    conf = load_config(conf)

    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    metacal_bias_file = os.path.join(repo_dir, conf["files"]["metacal_bias"])
    with h5py.File(metacal_bias_file, "r") as f:
        metacal_bias = f[key][:]

    return np.array(metacal_bias)


def get_shape_noise(conf=None):
    """Parse and validate the shape-noise model configuration block.

    The block lives at conf["analysis"]["modelling"]["lensing"]["shape_noise"] and disentangles two
    orthogonal choices:
      - method: how the shape noise is generated / whether it models source clustering
          "in_place" -> rotate galaxies in place (no source-clustering bias)
          "gatti"    -> calibrated Gatti et al. (https://arxiv.org/abs/2307.13860) density modulation
          "count"    -> count-based Poisson resampling of the catalog
      - bias: where the per-bin source-clustering bias b_sc comes from (ignored for "in_place")
          "fixed" -> gatti: cosmology-independent `fixed_bsc` from the config
                     count: per-cosmology bias read from files.metacal_bias
          "prior" -> b_sc sampled from the Latin hypercube (params.sc = [bsc])

    Raises a clear ValueError for any unexpected/old-string form so a bad config fails loudly
    everywhere the shape-noise model is read.

    Args:
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary
            (the config is passed through) or None (the default config is loaded). Defaults to None.

    Returns:
        tuple: (method, bias, fixed_bsc). `bias` is None for method "in_place"; `fixed_bsc` is the
            per metacal bin np.ndarray used only for method "gatti" with bias "fixed" (else None).
    """
    conf = load_config(conf)

    sn_conf = conf["analysis"]["modelling"]["lensing"]["shape_noise"]
    if not isinstance(sn_conf, dict):
        raise ValueError(
            f"shape_noise config must be a nested block with a 'method' (and 'bias') key, got {sn_conf!r}"
        )

    method = sn_conf.get("method")
    valid_methods = ("in_place", "gatti", "count")
    if method not in valid_methods:
        raise ValueError(f"shape_noise method must be one of {valid_methods}, got {method!r}")

    # in_place rotates galaxies in place and has no notion of a source-clustering bias
    if method == "in_place":
        return method, None, None

    bias = sn_conf.get("bias")
    valid_biases = ("fixed", "prior")
    if bias not in valid_biases:
        raise ValueError(
            f"shape_noise bias must be one of {valid_biases} for method {method!r}, got {bias!r}"
        )

    fixed_bsc = None
    if method == "gatti" and bias == "fixed":
        n_z = len(conf["survey"]["metacal"]["z_bins"])
        fixed_bsc = sn_conf.get("fixed_bsc")
        if fixed_bsc is None or len(fixed_bsc) != n_z:
            raise ValueError(
                f"shape_noise fixed_bsc must be a length-{n_z} list for method 'gatti' with bias 'fixed', "
                f"got {fixed_bsc!r}"
            )
        fixed_bsc = np.asarray(fixed_bsc, dtype=np.float64)

    return method, bias, fixed_bsc


def read_sc_calibration(conf, b_sc):
    """Load the Gatti source-clustering calibration factors and evaluate them at the bias b_sc.

    The calibration restores the correct shape-noise variance and kurtosis after the per-pixel
    modulation f = 1 / sqrt(1 + b_sc * delta) (see lensing.source_clustering_factor). The factors
    (corr_variance, A_corr, coeff_kurtosis) are stored as linear fits in b_sc, one per metacal bin,
    and determined in a separate calibration notebook. The file is an .npy dict of the form
        {"corr_variance": {"slope": [...], "intercept": [...]},
         "A_corr":        {"slope": [...], "intercept": [...]},
         "coeff_kurtosis":{"slope": [...], "intercept": [...]}}
    where each array has length n_z_metacal.

    If no calibration file is configured or it does not exist on disk, a no-op calibration
    (corr_variance, A_corr, coeff_kurtosis) = (1.0, 1.0, 0.0) is returned for every bin, so the
    pipeline runs (pure f-modulation) before the calibration has been determined.

    Args:
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary
            (the config is passed through) or None (the default config is loaded).
        b_sc (array-like): Per metacal bin source-clustering bias at which to evaluate the fits.

    Returns:
        list: Per metacal bin tuple (corr_variance, A_corr, coeff_kurtosis).
    """
    conf = load_config(conf)

    b_sc = np.atleast_1d(np.asarray(b_sc, dtype=np.float64))
    n_z = len(conf["survey"]["metacal"]["z_bins"])

    sc_calib_path = conf["files"].get("sc_calibration", None)

    if sc_calib_path is None:
        LOGGER.warning("No sc_calibration file configured, using no-op source-clustering calibration")
        return [(1.0, 1.0, 0.0)] * n_z

    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    sc_calib_file = os.path.join(repo_dir, sc_calib_path)

    if not os.path.exists(sc_calib_file):
        LOGGER.warning(
            f"sc_calibration file {sc_calib_file} not found, using no-op source-clustering calibration"
        )
        return [(1.0, 1.0, 0.0)] * n_z

    fits = np.load(sc_calib_file, allow_pickle=True).item()
    LOGGER.info(f"Loaded source-clustering calibration from {sc_calib_file}")

    def _eval(name, i):
        return fits[name]["slope"][i] * b_sc[i] + fits[name]["intercept"][i]

    return [(_eval("corr_variance", i), _eval("A_corr", i), _eval("coeff_kurtosis", i)) for i in range(n_z)]
