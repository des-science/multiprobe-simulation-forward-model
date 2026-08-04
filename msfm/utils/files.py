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


# the rotation of the imaging systematics maps into the footprint frame costs about a second, while
# get_metacal_systematics is called once per tomographic bin and permutation. Keyed by the file path
_METACAL_SYSTEMATICS_CACHE = {}


def get_metacal_systematics(conf=None, full_sky=False, dataset="contamination"):
    """Per (metacal) tomographic bin DES Y3 imaging systematics contamination factor of the source galaxy density.

    The maps are produced by the lss_sys repository, see its output/README.md. They quantify the spurious density
    modulation that the survey properties (depth, seeing, stellar density, ...) imprint on the observed source counts
    and that the simulations do not have. This function returns the "contamination" dataset <1/W>, which is what a
    clean model density is multiplied by to impose the DES imprint. It is deliberately NOT the "weight" dataset <W>,
    which decontaminates observed counts and is not the reciprocal of this one (they differ by up to 12.7%, since
    averaging over sub pixels does not commute with inverting).

    Two conventions have to be bridged. The maps are stored full sky in the RING scheme in celestial coordinates,
    whereas the forward model works in the rotated footprint frame of Fig. 4 of https://arxiv.org/pdf/2511.04681, so
    they are rotated with the very same hp.Rotator that notebooks/pixel_file.ipynb used to build the mask and the
    maglim systematics maps. And they carry unit mean over their own footprint, which is slightly wider than the
    thresholded analysis mask, so they are renormalized to unit mean over the base patch. That makes the
    contamination conserve the total galaxy count over exactly the footprint that n_bar is measured on.

    Args:
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). Defaults to None.
        full_sky (bool, optional): Whether to scatter the base patch into a full sky map that is 1 (i.e. no
            contamination) everywhere else, for the full sky consumers. Defaults to False.
        dataset (str, optional): Either "contamination" (<1/W>, imprint on a clean model) or "weight" (<W>,
            decontaminate an observation). Only "contamination" belongs in the forward model, "weight" is there for
            diagnostics of the observation itself. Defaults to "contamination".

    Returns:
        np.ndarray: (len(base_patch_pix), n_z_metacal) contamination factor in the base patch frame, or
            (n_pix, n_z_metacal) if full_sky is set.
    """
    assert dataset in ("contamination", "weight"), f"Unknown dataset {dataset!r}"

    conf = load_config(conf)

    n_side = conf["analysis"]["n_side"]
    n_pix = hp.nside2npix(n_side)
    n_z = len(conf["survey"]["metacal"]["z_bins"])

    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    sys_file = os.path.join(repo_dir, conf["files"]["metacal_systematics"])

    cache_key = (sys_file, n_side, n_z, dataset)
    if cache_key not in _METACAL_SYSTEMATICS_CACHE:
        # deliberately inside the cache guard: load_pixel_file has no cache of its own and reads a ~65 MB file, while
        # postprocess_shape_noise calls this function once per (permutation, tomographic bin)
        _, patches_pix_dict, _, _ = load_pixel_file(conf)
        base_patch_pix = patches_pix_dict["metacal"][0][0]

        # the footprint rotation of notebooks/pixel_file.ipynb, which defines the frame of the mask and the patches
        y_rad = conf["analysis"]["footprint"]["rotation"]["y_rad"]
        z_rad = conf["analysis"]["footprint"]["rotation"]["z_rad"]
        rotator = hp.Rotator(rot=(0, -y_rad, z_rad), eulertype="Y", deg=False)

        tomo_sys = []
        covered_ref = None
        with h5py.File(sys_file, "r") as f:
            assert str(f.attrs["ordering"]) == "RING", f"Expected RING ordered maps, got {f.attrs['ordering']!r}"
            assert int(f.attrs["n_bins"]) == n_z, f"Expected {n_z} tomographic bins, got {f.attrs['n_bins']}"
            n_side_sys = int(f.attrs["nside"])
            zero_off_footprint = str(f.attrs.get("off_footprint", "unseen")) == "zero"
            LOGGER.info(
                f"Reading the {dataset} maps of the imaging systematics run {str(f.attrs['label'])!r} at nside "
                f"{n_side_sys} (lss_sys {str(f.attrs['git_sha'])[:8]})"
            )

            for i_z in range(n_z):
                sys_map = f[f"bin{i_z}/{dataset}"][:].astype(np.float64)

                # off the footprint there is no correction defined, which is not the same as a correction of zero.
                # Setting it to 1 before the rotation also keeps the interpolation from bleeding into the footprint
                covered = (sys_map != 0.0) if zero_off_footprint else ~hp.mask_bad(sys_map)
                sys_map = np.where(covered, sys_map, 1.0)

                # the footprint is a property of the lss_sys run rather than of the bin, so one coverage check below
                # covers all of them -- as long as that actually holds
                if covered_ref is None:
                    covered_ref = covered
                else:
                    assert np.array_equal(covered, covered_ref), (
                        f"Tomographic bin {i_z} of {sys_file} has a different footprint than bin 0, so the coverage "
                        f"of the analysis mask has to be checked per bin"
                    )

                if n_side_sys != n_side:
                    sys_map = hp.ud_grade(sys_map, nside_out=n_side, order_in="RING", order_out="RING")

                # celestial to the rotated footprint frame
                sys_map = rotator.rotate_map_pixel(sys_map)

                # the maps carry unit mean over their own (wider) footprint, renormalize to the analysis one
                sys_patch = sys_map[base_patch_pix]
                sys_patch /= np.mean(sys_patch)
                tomo_sys.append(sys_patch)

                LOGGER.debug(
                    f"Bin {i_z + 1}: contamination in "
                    f"[{sys_patch.min():.3f}, {sys_patch.max():.3f}], std {sys_patch.std():.4f}"
                )

        # Off the systematics footprint the correction is undefined and was set to 1 above, so any analysis pixel
        # landing there would carry no imprint while still entering the unit-mean renormalization below. That the
        # rotation puts the whole analysis mask inside the (wider) systematics footprint is the frame invariant that
        # this function, postprocessing and the bias fit all rely on, and it is exactly the kind of thing that broke
        # silently once already in the observation, so assert it rather than trust the pixel counts
        covered_rot = covered_ref.astype(np.float64)
        if n_side_sys != n_side:
            covered_rot = hp.ud_grade(covered_rot, nside_out=n_side, order_in="RING", order_out="RING")
        covered_rot = rotator.rotate_map_pixel(covered_rot)

        n_uncovered = int(np.sum(covered_rot[base_patch_pix] < 0.5))
        assert n_uncovered == 0, (
            f"{n_uncovered} of the {len(base_patch_pix)} analysis footprint pixels fall outside the imaging "
            f"systematics footprint of {sys_file} once rotated into the forward model frame. Those pixels would go "
            f"uncorrected and dilute the renormalization"
        )
        LOGGER.debug(f"The rotated systematics footprint covers all {len(base_patch_pix)} analysis mask pixels")

        # shape (len(base_patch_pix), n_z_metacal)
        _METACAL_SYSTEMATICS_CACHE[cache_key] = np.stack(tomo_sys, axis=-1)

    tomo_sys = _METACAL_SYSTEMATICS_CACHE[cache_key]

    if full_sky:
        # the only consumer that still needs the pixel file on a cache hit
        _, patches_pix_dict, _, _ = load_pixel_file(conf)
        base_patch_pix = patches_pix_dict["metacal"][0][0]

        sys_full = np.ones((n_pix, n_z))
        sys_full[base_patch_pix] = tomo_sys
        return sys_full

    return tomo_sys


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
      - survey_systematics: whether to imprint the DES Y3 imaging systematics of
          files.metacal_systematics on the model source density before it is resampled, see
          files.get_metacal_systematics. Only "count" acts on the source density, so it is False for
          the other methods. Absent means False, which keeps the v16/v17 configs parsing.

    Raises a clear ValueError for any unexpected/old-string form so a bad config fails loudly
    everywhere the shape-noise model is read.

    Args:
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary
            (the config is passed through) or None (the default config is loaded). Defaults to None.

    Returns:
        tuple: (method, bias, fixed_bsc, survey_systematics). `bias` is None for method "in_place";
            `fixed_bsc` is the per metacal bin np.ndarray used only for method "gatti" with bias
            "fixed" (else None); `survey_systematics` is a bool that is only ever True for "count".
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
        return method, None, None, False

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

    survey_systematics = bool(sn_conf.get("survey_systematics", False))
    if survey_systematics and method != "count":
        raise ValueError(
            f"shape_noise survey_systematics is only defined for method 'count', which acts on the source "
            f"density, got method {method!r}"
        )

    return method, bias, fixed_bsc, survey_systematics


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
