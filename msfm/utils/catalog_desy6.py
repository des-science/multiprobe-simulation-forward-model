"""Reader for the public DES Y6 3x2pt catalogs, the Y6 counterpart of catalog.py.

The files are those of https://desdr-server.ncsa.illinois.edu/despublic/y6a2_files/y6_psf/final_version/, found in
dirs.catalog. They are the fiducial KP selection already: the metadetection catalog is cut (mdet_flags == 0) and
masked, and every MagLim object passes totmaskgood and shear_v2_extramask_good, so no cuts are applied here.
DES's 3x2pt_pipeline_public.ipynb in the same directory is the reference for the response, the weights and the
shape-noise statistics.

The maps hold every survey bin along axis 1, in survey-bin order (tomo_bin_k is survey bin k + 1); the config's z_bins
select from them. Positions and shears are rotated into the footprint frame exactly as for Y3 (catalog.py).

The joint mask is read with healsparse (mask_area, rotated_mask_coverage), imported only there, since it is in neither
tf_env nor torch_env yet.
"""

import os

import h5py
import numpy as np
import healpy as hp

from msfm.utils import catalog, logger

LOGGER = logger.get_logger(__file__)

# files of the release, relative to dirs.catalog
RELEASE_FILES = {
    "3x2pt": "3x2pt_2025-07-28-20h_UNBLINDED_wmask_nob2.fits",
    "metadetect": "metadetect_2024-11-07.hdf5",
    "maglim": "maglim_2024-11-07.hdf5",
    "mask": "jointmask_DESY6_2024-06-04.hsp",
    "source_redshift": "desy6_source_redshift.h5",
}
N_SOURCE_BINS = 4
N_LENS_BINS = 6

# shear step of the metadetection images, deltag in 3x2pt_pipeline_public.ipynb
DELTA_G = 0.01
# mean response per source bin, KP Table 2 (https://arxiv.org/abs/2601.14559); the full catalog must reproduce it
RESPONSE_KP = [0.856, 0.869, 0.819, 0.692]
RESPONSE_KP_ATOL = 5e-3
# mean m per source bin, Table III of the methods paper as quoted in 3x2pt_pipeline_public.ipynb
M_MEAN_DES = [-3.4e-3, 6.46e-3, 15.94e-3, 1.7e-3]
M_MEAN_DES_ATOL = 1e-5

# the joint mask is a bit-packed boolean HealSparse map
MASK_NSIDE = 131072
# len(mask.valid_pixels) * pixarea in 3x2pt_pipeline_public.ipynb, in deg^2
AREA_DES = 4031.036
AREA_DES_ATOL = 1e-3
# valid_pixels of the whole mask needs ~450 GB (DES), so the area is counted a few coverage pixels at a time
MASK_COV_PIXELS_PER_READ = 16
# analysis pixels per batch when sampling the mask
MASK_SAMPLE_BATCH = 16384


def release_path(conf, key):
    """Absolute path of a release file in dirs.catalog."""
    path = os.path.join(conf["dirs"]["catalog"], RELEASE_FILES[key])
    if not os.path.isfile(path):
        raise FileNotFoundError(f"DES Y6 release file {path} (dirs.catalog) does not exist")
    return path


def m_prior(conf):
    """Mean and standard deviation of the multiplicative shear bias per source bin, over the M_BIN samples.

    Args:
        conf (dict): A DES Y6 config.

    Returns:
        tuple: mu (n_bins,), sigma (n_bins,)
    """
    path = release_path(conf, "source_redshift")
    with h5py.File(path, "r") as f:
        keys = sorted(key for key in f.keys() if key.startswith("M_BIN"))
        assert keys == [f"M_BIN{i}" for i in range(1, N_SOURCE_BINS + 1)], f"{path}: {keys}"
        m = [f[key][:] for key in keys]
    assert all(np.all(np.isfinite(x)) for x in m), f"{path}: m is not finite"

    mu, sigma = np.array([np.mean(x) for x in m]), np.array([np.std(x) for x in m])
    LOGGER.info(f"m prior from {m[0].size} samples: mu = {mu}, sigma = {sigma} (DES mu {M_MEAN_DES})")
    assert np.all(np.abs(mu - M_MEAN_DES) < M_MEAN_DES_ATOL), f"mean m {mu} differs from DES's {M_MEAN_DES}"
    return mu, sigma


def mask_area(conf):
    """Area of the joint mask in deg^2, counted at its full resolution as DES does.

    Args:
        conf (dict): A DES Y6 config.

    Returns:
        float: The area in deg^2, equal to DES's value within AREA_DES_ATOL.
    """
    import healsparse

    path = release_path(conf, "mask")
    coverage = healsparse.HealSparseCoverage.read(path)
    assert coverage.nside_sparse == MASK_NSIDE, f"{path}: nside {coverage.nside_sparse}, not {MASK_NSIDE}"

    cov_pix = np.flatnonzero(coverage.coverage_mask)
    n_valid = 0
    for chunk in np.array_split(cov_pix, int(np.ceil(cov_pix.size / MASK_COV_PIXELS_PER_READ))):
        n_valid += healsparse.HealSparseMap.read(path, pixels=chunk.tolist()).valid_pixels.size

    area = n_valid * hp.nside2pixarea(MASK_NSIDE, degrees=True)
    LOGGER.info(
        f"joint mask: {cov_pix.size} coverage pixels, {n_valid} valid pixels, {area:.3f} deg^2 (DES {AREA_DES})"
    )
    assert abs(area - AREA_DES) < AREA_DES_ATOL, f"the mask area {area:.4f} deg^2 differs from DES's {AREA_DES}"
    return area


def rotated_mask_coverage(conf, sample_nside):
    """Fraction of every analysis.n_side pixel of the rotated footprint frame that the joint mask covers.

    Each pixel is sampled at the centres of its sub-pixels at sample_nside. The centres are rotated back to the sky
    with catalog.survey_pix_to_angles, the inverse of the rotation the catalog maps use, and looked up in the
    full-resolution mask. Only pixels near the mask's coverage pixels are sampled; all others stay at zero.

    Args:
        conf (dict): A DES Y6 config.
        sample_nside (int): Resolution of the sampling points, a multiple of analysis.n_side.

    Returns:
        np.ndarray: coverage (n_pix,) in RING ordering, between 0 and 1.
    """
    import healsparse

    n_side = conf["analysis"]["n_side"]
    n_pix = hp.nside2npix(n_side)
    assert sample_nside % n_side == 0, f"sample nside {sample_nside} is no multiple of {n_side}"
    n_sub = (sample_nside // n_side) ** 2

    path = release_path(conf, "mask")
    # kept bit-packed in memory, a few GB
    mask = healsparse.HealSparseMap.read(path)
    assert mask.nside_sparse == MASK_NSIDE, f"{path}: nside {mask.nside_sparse}, not {MASK_NSIDE}"
    assert np.dtype(mask.dtype) == np.bool_, f"{path}: {mask.dtype} map, not boolean"

    # candidates: pixels whose centre lies in a coverage pixel of the mask or in one of its neighbours
    nside_cov = mask.nside_coverage
    covered = mask.coverage_mask.copy()
    neighbours = hp.get_all_neighbours(nside_cov, np.flatnonzero(covered), nest=True).ravel()
    covered[neighbours[neighbours >= 0]] = True
    ra, dec = catalog.survey_pix_to_angles(conf, np.arange(n_pix), n_side)
    candidates = np.flatnonzero(covered[hp.ang2pix(nside_cov, ra, dec, nest=True, lonlat=True)])

    coverage = np.zeros(n_pix)
    for batch in np.array_split(candidates, int(np.ceil(candidates.size / MASK_SAMPLE_BATCH))):
        # the NEST children of a pixel tile it exactly
        sub = (hp.ring2nest(n_side, batch)[:, None] * n_sub + np.arange(n_sub)).ravel()
        ra, dec = catalog.survey_pix_to_angles(conf, hp.nest2ring(sample_nside, sub), sample_nside)
        inside = mask.get_values_pos(ra, dec, lonlat=True)
        coverage[batch] = inside.reshape(batch.size, n_sub).mean(axis=1)

    LOGGER.info(
        f"rotated mask: sampled {candidates.size} pixels at nside {sample_nside} ({n_sub} points each), "
        f"{np.sum(coverage > 0)} touch the mask"
    )
    return coverage


def _read(group, column, max_objects=None):
    """A column of a catalog group, or its first max_objects rows."""
    return group[column][:max_objects]


def _noshear(mdet, i_bin, max_objects=None):
    """Positions, ellipticities and weights of source bin i_bin in the unsheared catalog: ra, dec, e1, e2, w."""
    group = mdet[f"noshear/tomo_bin_{i_bin}"]
    columns = [_read(group, col, max_objects) for col in ("ra", "dec", "gauss_g_1", "gauss_g_2", "w")]
    assert all(np.all(np.isfinite(x)) for x in columns), f"source bin {i_bin + 1}: not finite"
    assert np.all(columns[-1] > 0), f"source bin {i_bin + 1}: non-positive weights"
    return columns


def metadetect_response(mdet, i_bin, max_objects=None):
    """Weighted mean shear response (R11, R22) of source bin i_bin, from the four sheared catalogs as DES computes it.

    Args:
        mdet (h5py.File): The open metadetection file.
        i_bin (int): Index of the tomo_bin group, survey bin i_bin + 1.
        max_objects (int, optional): Use only the first rows of each catalog, for smoke tests. Defaults to None.

    Returns:
        tuple: (R11, R22)
    """
    mean = {}
    for step, comp in (("1p", 1), ("1m", 1), ("2p", 2), ("2m", 2)):
        group = mdet[f"{step}/tomo_bin_{i_bin}"]
        mean[step] = np.average(_read(group, f"gauss_g_{comp}", max_objects), weights=_read(group, "w", max_objects))

    R11 = (mean["1p"] - mean["1m"]) / (2 * DELTA_G)
    R22 = (mean["2p"] - mean["2m"]) / (2 * DELTA_G)
    return R11, R22


def build_metacal_maps(conf, sign_e1=1, sign_e2=-1, apply_shear_rotation=True, max_objects=None):
    """Shear and count maps of every metadetection source bin, in the rotated footprint frame at analysis.n_side.

    The shear map is sum(w e) / (R sum(w)) per pixel, as eq. (10) of https://arxiv.org/pdf/2403.02314, with the scalar
    R = (R11 + R22) / 2 of metadetect_response. No mean shear is subtracted (plan D8). The signs and the spin-2
    rotation follow build_metacal_map_from_cat for Y3.

    Args:
        conf (dict): A DES Y6 config.
        sign_e1 (int, optional): Factor applied to e1. Defaults to 1.
        sign_e2 (int, optional): Factor applied to e2 (healpy convention). Defaults to -1.
        apply_shear_rotation (bool, optional): Rotate the shears into the local frame of the rotated positions.
            Defaults to True.
        max_objects (int, optional): Use only the first rows of each catalog, for smoke tests. Defaults to None.

    Returns:
        tuple: gamma (n_pix, n_bins, 2), count (n_pix, n_bins) int32, response (n_bins,)
    """
    n_side = conf["analysis"]["n_side"]
    n_pix = hp.nside2npix(n_side)
    area_arcmin = conf["survey"]["Aeff"] * 60**2

    gamma = np.zeros((n_pix, N_SOURCE_BINS, 2))
    count = np.zeros((n_pix, N_SOURCE_BINS), dtype=np.int32)
    response = np.zeros(N_SOURCE_BINS)
    with h5py.File(release_path(conf, "metadetect"), "r") as mdet:
        for step in ("noshear", "1p", "1m", "2p", "2m"):
            assert sorted(mdet[step].keys()) == [f"tomo_bin_{i}" for i in range(N_SOURCE_BINS)], f"{step} bins"

        for i in range(N_SOURCE_BINS):
            R11, R22 = metadetect_response(mdet, i, max_objects)
            response[i] = 0.5 * (R11 + R22)

            ra, dec, e1, e2, w = _noshear(mdet, i, max_objects)
            pix, psi = catalog.survey_angles_to_pix_and_shear_rotation(conf, ra, dec, n_side)
            e1_rot, e2_rot = catalog.rotate_shear(e1, e2, psi) if apply_shear_rotation else (e1, e2)

            w_map = np.bincount(pix, weights=w, minlength=n_pix)
            seen = w_map > 0
            for comp, (sign, e) in enumerate(((sign_e1, e1_rot), (sign_e2, e2_rot))):
                e_map = np.bincount(pix, weights=sign * e * w, minlength=n_pix)
                gamma[seen, i, comp] = e_map[seen] / (w_map[seen] * response[i])
            count[:, i] = np.bincount(pix, minlength=n_pix)

            # the statistics of 3x2pt_pipeline_public.ipynb, section 6
            mean_e1, mean_e2 = np.average(e1, weights=w), np.average(e2, weights=w)
            sum_w, sum_w2, sum_wr = np.sum(w), np.sum(w**2), np.sum(w) * response[i]
            var = (np.sum(w**2 * (e1 - mean_e1) ** 2) + np.sum(w**2 * (e2 - mean_e2) ** 2)) / sum_wr**2
            sigma_e = np.sqrt(0.5 * var * sum_w**2 / sum_w2)
            n_eff = sum_w**2 / sum_w2 / area_arcmin
            LOGGER.info(
                f"source bin {i + 1}: N = {e1.size}, n_eff = {n_eff:.3f} arcmin^-2, sigma_e = {sigma_e:.4f}, "
                f"R11 = {R11:.4f}, R22 = {R22:.4f}, R = {response[i]:.4f} (KP {RESPONSE_KP[i]}), "
                f"mean e = ({mean_e1:.2e}, {mean_e2:.2e})"
            )

    if max_objects is None:
        off = np.abs(response - RESPONSE_KP) > RESPONSE_KP_ATOL
        assert not np.any(off), f"response {response} differs from KP Table 2 {RESPONSE_KP} by > {RESPONSE_KP_ATOL}"

    return gamma, count, response


def metacal_shapes(conf, i_bin, response, sign_e1=1, sign_e2=-1, apply_shear_rotation=True, max_objects=None):
    """Per-galaxy shear estimates of source bin i_bin for the shape-noise file, as catalog.get_shapes_from_cat for Y3.

    The signs and the spin-2 rotation are those of build_metacal_maps, and response is the R that calibrated the
    shear maps.

    Args:
        conf (dict): A DES Y6 config.
        i_bin (int): Index of the tomo_bin group, survey bin i_bin + 1.
        response (float): The mean shear response R of the bin.
        sign_e1 (int, optional): Factor applied to e1. Defaults to 1.
        sign_e2 (int, optional): Factor applied to e2 (healpy convention). Defaults to -1.
        apply_shear_rotation (bool, optional): Rotate the shears into the local frame of the rotated positions.
            Defaults to True.
        max_objects (int, optional): Use only the first rows of the catalog, for smoke tests. Defaults to None.

    Returns:
        np.ndarray: (n_gal, 4) float32 with columns (sign_e1 e1 / R, sign_e2 e2 / R, w, pix), pix the RING pixel of
            the rotated position at analysis.n_side.
    """
    n_side = conf["analysis"]["n_side"]
    # float32 holds every pixel index exactly
    assert hp.nside2npix(n_side) <= 2**24, f"nside {n_side} pixel indices do not fit into float32"

    with h5py.File(release_path(conf, "metadetect"), "r") as mdet:
        ra, dec, e1, e2, w = _noshear(mdet, i_bin, max_objects)
    pix, psi = catalog.survey_angles_to_pix_and_shear_rotation(conf, ra, dec, n_side)
    if apply_shear_rotation:
        e1, e2 = catalog.rotate_shear(e1, e2, psi)

    cat = np.empty((w.size, 4), dtype=np.float32)
    cat[:, 0] = sign_e1 * e1 / response
    cat[:, 1] = sign_e2 * e2 / response
    cat[:, 2] = w
    cat[:, 3] = pix
    return cat


def build_maglim_maps(conf, max_objects=None):
    """Raw and weighted count maps of every MagLim bin, in the rotated footprint frame at analysis.n_side.

    The weight is the per-galaxy ENET survey-systematics weight of the release, normalized to a mean of 1 per bin.

    Args:
        conf (dict): A DES Y6 config.
        max_objects (int, optional): Use only the first rows of each catalog, for smoke tests. Defaults to None.

    Returns:
        tuple: count (n_pix, n_bins) int32, weighted count (n_pix, n_bins)
    """
    n_side = conf["analysis"]["n_side"]
    n_pix = hp.nside2npix(n_side)
    area_arcmin = conf["survey"]["Aeff"] * 60**2

    count = np.zeros((n_pix, N_LENS_BINS), dtype=np.int32)
    weighted_count = np.zeros((n_pix, N_LENS_BINS))
    with h5py.File(release_path(conf, "maglim"), "r") as maglim:
        assert sorted(maglim.keys()) == [f"tomo_bin_{i}" for i in range(N_LENS_BINS)], "MagLim bins"

        for i in range(N_LENS_BINS):
            group = maglim[f"tomo_bin_{i}"]
            assert np.all(_read(group, "PHOTOZBIN", max_objects) == i + 1), f"tomo_bin_{i} is not survey bin {i + 1}"
            ra, dec = _read(group, "RA", max_objects), _read(group, "DEC", max_objects)
            w = _read(group, "weight", max_objects).astype(np.float64)
            assert all(np.all(np.isfinite(x)) for x in (ra, dec, w)), f"lens bin {i + 1}: not finite"
            assert np.all(w > 0), f"lens bin {i + 1}: non-positive weights"

            pix = catalog.survey_angles_to_pix(conf, ra, dec, n_side)
            count[:, i] = np.bincount(pix, minlength=n_pix)
            weighted_count[:, i] = np.bincount(pix, weights=w, minlength=n_pix)

            n_eff = np.sum(w) ** 2 / np.sum(w**2) / area_arcmin
            LOGGER.info(
                f"lens bin {i + 1}: N = {w.size}, mean weight = {np.mean(w):.5f}, n_eff = {n_eff:.4f} arcmin^-2"
            )
            if max_objects is None:
                assert abs(np.mean(w) - 1) < 1e-3, f"lens bin {i + 1}: the weights average {np.mean(w)}, not 1"

    return count, weighted_count
