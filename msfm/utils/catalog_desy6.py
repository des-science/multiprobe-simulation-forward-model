"""Reader for the public DES Y6 3x2pt catalogs, the Y6 counterpart of catalog.py.

The files are those of https://desdr-server.ncsa.illinois.edu/despublic/y6a2_files/y6_psf/final_version/, found in
dirs.catalog. They are the fiducial KP selection already: the metadetection catalog is cut (mdet_flags == 0) and
masked, and every MagLim object passes totmaskgood and shear_v2_extramask_good, so no cuts are applied here.
DES's 3x2pt_pipeline_public.ipynb in the same directory is the reference for the response, the weights and the
shape-noise statistics.

The maps hold every survey bin along axis 1, in survey-bin order (tomo_bin_k is survey bin k + 1); the config's z_bins
select from them. Positions and shears are rotated into the footprint frame exactly as for Y3 (catalog.py).

The joint mask is read with healsparse (mask_coverage), imported only there, since only torch_env has it.
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
# per-bin statistics that 3x2pt_pipeline_public.ipynb prints for the full catalogs (cells 7, 14, 25 and 26), as
# (values, tolerance), the tolerance being one unit of the last digit printed; the full catalogs must reproduce them
SOURCE_DES = {
    "n": ([33707071, 34580888, 34646416, 36727798], 0),
    "response": ([0.856143, 0.868506, 0.819427, 0.691769], 1e-6),
    "n_eff": ([2.047, 2.097, 2.138, 2.320], 1e-3),
    "sigma_e": ([0.2657, 0.2865, 0.2821, 0.3474], 1e-4),
}
LENS_DES = {
    "n": ([1852538, 1335294, 1413738, 1783834, 1391521, 1409280], 0),
    "n_eff": ([0.1276, 0.0920, 0.0974, 0.1228, 0.0959, 0.0970], 1e-4),
}
# mean m per source bin, Table III of the methods paper as quoted in 3x2pt_pipeline_public.ipynb
M_MEAN_DES = [-3.4e-3, 6.46e-3, 15.94e-3, 1.7e-3]
M_MEAN_DES_ATOL = 1e-5

# the joint mask is a bit-packed boolean HealSparse map
MASK_NSIDE = 131072
# len(mask.valid_pixels) * pixarea in 3x2pt_pipeline_public.ipynb, in deg^2
AREA_DES = 4031.036
AREA_DES_ATOL = 1e-3

# the MagLim ENet weights are a map at this nside (ENET_256fit_1024pred_19tpl in desy6kp_cats_2024-11-07.hdf5)
LENS_WEIGHT_NSIDE = 1024
# gates of maglim_weight_maps; measured: correlation 0.993 to 0.996, totals within 3e-5
LENS_WEIGHT_MIN_CORR = 0.98
LENS_WEIGHT_TOTAL_RTOL = 1e-3


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


def mask_coverage(conf, n_side):
    """Fraction of every n_side pixel that the joint mask covers, in celestial coordinates, and the mask area.

    healsparse counts the full-resolution pixels of the mask inside each n_side pixel, so the area is exact.

    Args:
        conf (dict): A DES Y6 config.
        n_side (int): Resolution of the coverage map.

    Returns:
        tuple: coverage (n_pix,) in RING ordering, between 0 and 1, and the area in deg^2
    """
    import healsparse

    path = release_path(conf, "mask")
    mask = healsparse.HealSparseMap.read(path)
    assert mask.nside_sparse == MASK_NSIDE, f"{path}: nside {mask.nside_sparse}, not {MASK_NSIDE}"
    assert np.dtype(mask.dtype) == np.bool_, f"{path}: {mask.dtype} map, not boolean"

    coverage = mask.fracdet_map(n_side).generate_healpix_map(nest=False).astype(np.float64)
    coverage[coverage == hp.UNSEEN] = 0.0
    assert np.all((coverage >= 0) & (coverage <= 1)), f"{path}: coverage outside [0, 1]"

    area = np.sum(coverage) * hp.nside2pixarea(n_side, degrees=True)
    LOGGER.info(
        f"joint mask at nside {n_side}: {np.sum(coverage > 0)} pixels touch it, {area:.3f} deg^2 (DES {AREA_DES})"
    )
    assert abs(area - AREA_DES) < AREA_DES_ATOL, f"the mask area {area:.4f} deg^2 differs from DES's {AREA_DES}"
    return coverage, area


def _check_against_des(sample, stats, reference):
    """Asserts that every per-bin statistic in stats agrees with DES's value in reference, within its tolerance."""
    for key, (des, atol) in reference.items():
        ours = np.asarray(stats[key])
        assert np.all(np.abs(ours - np.asarray(des)) <= atol), f"{sample} {key} {ours} differs from DES's {des}"
    LOGGER.info(f"{sample} {', '.join(reference)} agree with 3x2pt_pipeline_public.ipynb")


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
    R = (R11 + R22) / 2 of metadetect_response. No mean shear is subtracted, as in Y3. The signs and the spin-2
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
    # DES's area, not survey.Aeff, which step values writes after this step
    area_arcmin = AREA_DES * 60**2

    gamma = np.zeros((n_pix, N_SOURCE_BINS, 2))
    count = np.zeros((n_pix, N_SOURCE_BINS), dtype=np.int32)
    response = np.zeros(N_SOURCE_BINS)
    stats = {key: np.zeros(N_SOURCE_BINS) for key in ("n", "n_eff", "sigma_e")}
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
            stats["n"][i], stats["n_eff"][i], stats["sigma_e"][i] = e1.size, n_eff, sigma_e
            LOGGER.info(
                f"source bin {i + 1}: N = {e1.size}, n_eff = {n_eff:.3f} arcmin^-2, sigma_e = {sigma_e:.4f}, "
                f"R11 = {R11:.4f}, R22 = {R22:.4f}, R = {response[i]:.6f}, mean e = ({mean_e1:.2e}, {mean_e2:.2e})"
            )

    if max_objects is None:
        _check_against_des("source", {"response": response, **stats}, SOURCE_DES)

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
    """Raw count maps of every MagLim bin, in the rotated footprint frame at analysis.n_side.

    As for Y3, the counts are not weighted here: the ENet weight maps of the pixel file (maglim_weight_maps) multiply
    them on the observation path. The per-galaxy ENet weights, normalized to a mean of 1 per bin, only enter the
    n_eff that is checked against DES.

    Args:
        conf (dict): A DES Y6 config.
        max_objects (int, optional): Use only the first rows of each catalog, for smoke tests. Defaults to None.

    Returns:
        np.ndarray: count (n_pix, n_bins) int32
    """
    n_side = conf["analysis"]["n_side"]
    n_pix = hp.nside2npix(n_side)
    # DES's area, as in build_metacal_maps
    area_arcmin = AREA_DES * 60**2

    count = np.zeros((n_pix, N_LENS_BINS), dtype=np.int32)
    stats = {key: np.zeros(N_LENS_BINS) for key in ("n", "n_eff")}
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

            n_eff = np.sum(w) ** 2 / np.sum(w**2) / area_arcmin
            stats["n"][i], stats["n_eff"][i] = w.size, n_eff
            LOGGER.info(
                f"lens bin {i + 1}: N = {w.size}, mean weight = {np.mean(w):.5f}, n_eff = {n_eff:.4f} arcmin^-2"
            )
            if max_objects is None:
                assert abs(np.mean(w) - 1) < 1e-3, f"lens bin {i + 1}: the weights average {np.mean(w)}, not 1"

    if max_objects is None:
        _check_against_des("lens", stats, LENS_DES)

    return count


def _fill_from_neighbours(values, known, region, n_side):
    """Gives each unknown pixel of region the mean of its known neighbours, outwards until all are known.

    Returns:
        tuple: values on region, number of passes
    """
    full = np.zeros(hp.nside2npix(n_side))
    ok = np.zeros(full.size, dtype=bool)
    full[region[known]], ok[region[known]] = values[known], True
    todo = region[~known]
    n_pass = 0
    while todo.size:
        neighbours = hp.get_all_neighbours(n_side, todo)
        safe = np.where(neighbours >= 0, neighbours, 0)
        usable = (neighbours >= 0) & ok[safe]
        n_usable = usable.sum(axis=0)
        fill = n_usable > 0
        assert np.any(fill), f"{todo.size} pixels without a known neighbour"
        mean = np.sum(np.where(usable, full[safe], 0.0), axis=0) / np.maximum(n_usable, 1)
        full[todo[fill]], ok[todo[fill]] = mean[fill], True
        todo = todo[~fill]
        n_pass += 1
    return full[region], n_pass


def maglim_weight_maps(conf, patch, sub_nside, fill_rings=2):
    """Pixel-level ENet weights w = 1 / <F> of every MagLim bin on the base patch, to correct the observed counts.

    The release has the ENet weight per galaxy only. It is constant inside each nside-1024 pixel (checked), so every
    such pixel holding a galaxy gives the weight map there. Each patch pixel is split into its sub_nside children, whose
    centres are rotated back to the sky and looked up in that map, without interpolating across the empty pixels. The
    observed count of a pixel is the clean one times the area mean of the contamination F = 1 / w, so the weight is
    1 / <F> over the children with a value. Pixels without any take the mean <F> of their neighbours, which may lie up
    to fill_rings outside the patch, since some patch pixels have no neighbours in it.

    Two checks against the per-galaxy weights, in the frame of the count maps: the mean galaxy weight of each pixel
    follows w, and the counts times w sum to the per-galaxy weights. The weights keep the catalog's normalization.

    Args:
        conf (dict): A DES Y6 config.
        patch (np.ndarray): RING pixels of the base patch at analysis.n_side, in the rotated footprint frame.
        sub_nside (int): Resolution of the children the contamination is averaged over.
        fill_rings (int, optional): Rings of neighbours around the patch that may fill its empty pixels. Defaults to 2.

    Returns:
        tuple: weights (len(patch), N_LENS_BINS) and filled (len(patch), N_LENS_BINS) bool, in survey-bin order
    """
    n_side = conf["analysis"]["n_side"]
    n_pix = hp.nside2npix(n_side)

    region = patch
    for _ in range(fill_rings):
        neighbours = hp.get_all_neighbours(n_side, region).ravel()
        region = np.union1d(region, neighbours[neighbours >= 0])
    patch_in_region = np.searchsorted(region, patch)
    assert np.array_equal(region[patch_in_region], patch), "the patch is not inside the fill region"

    # the sky pixel under the centre of every child
    n_child = (sub_nside // n_side) ** 2
    children = hp.nest2ring(sub_nside, (hp.ring2nest(n_side, region)[:, None] * n_child + np.arange(n_child)).ravel())
    ra, dec = catalog.survey_pix_to_angles(conf, children, sub_nside)
    child_sky_pix = hp.ang2pix(LENS_WEIGHT_NSIDE, ra, dec, lonlat=True).reshape(region.size, n_child)
    del children, ra, dec

    weights = np.zeros((patch.size, N_LENS_BINS))
    filled = np.zeros((patch.size, N_LENS_BINS), dtype=bool)
    with h5py.File(release_path(conf, "maglim"), "r") as maglim:
        for i in range(N_LENS_BINS):
            group = maglim[f"tomo_bin_{i}"]
            ra, dec, w = group["RA"][:], group["DEC"][:], group["weight"][:].astype(np.float64)
            assert np.all(w > 0), f"lens bin {i + 1}: non-positive weights"

            sky_pix = hp.ang2pix(LENS_WEIGHT_NSIDE, ra, dec, lonlat=True)
            order = np.argsort(sky_pix, kind="stable")
            sky_pix_sorted, w_sorted = sky_pix[order], w[order]
            occupied, first = np.unique(sky_pix_sorted, return_index=True)
            spread = np.max(np.maximum.reduceat(w_sorted, first) / np.minimum.reduceat(w_sorted, first) - 1)
            assert spread < 1e-6, f"lens bin {i + 1}: the weight varies by {spread} inside an nside-1024 pixel"
            w_sky = np.zeros(hp.nside2npix(LENS_WEIGHT_NSIDE))
            w_sky[occupied] = w_sorted[first]

            w_child = w_sky[child_sky_pix]
            known = w_child > 0
            n_known = known.sum(axis=1)
            contamination = np.sum(np.where(known, 1 / np.where(known, w_child, 1.0), 0.0), axis=1)
            contamination /= np.maximum(n_known, 1)
            contamination, n_pass = _fill_from_neighbours(contamination, n_known > 0, region, n_side)
            weights[:, i] = 1 / contamination[patch_in_region]
            filled[:, i] = n_known[patch_in_region] == 0

            pix = catalog.survey_angles_to_pix(conf, ra, dec, n_side)
            count = np.bincount(pix, minlength=n_pix)[patch]
            weighted_count = np.bincount(pix, weights=w, minlength=n_pix)[patch]
            has = count > 0
            corr = np.corrcoef(weighted_count[has] / count[has], weights[has, i])[0, 1]
            total = np.sum(count * weights[:, i]) / np.sum(weighted_count)
            LOGGER.info(
                f"lens bin {i + 1}: {occupied.size} nside-{LENS_WEIGHT_NSIDE} pixels with a weight, "
                f"{np.mean(known[patch_in_region]):.3f} of the patch children; {filled[:, i].sum()} patch pixels "
                f"filled ({n_pass} passes); w in [{weights[:, i].min():.3f}, {weights[:, i].max():.3f}], std "
                f"{weights[:, i].std():.4f}; corr with the mean galaxy weight {corr:.4f}; sum N w / sum w_gal "
                f"{total:.6f}"
            )
            assert corr > LENS_WEIGHT_MIN_CORR, f"lens bin {i + 1}: w does not follow the galaxy weights ({corr})"
            assert abs(total - 1) < LENS_WEIGHT_TOTAL_RTOL, f"lens bin {i + 1}: N w sums to {total} of the weights"

    return weights, filled
