# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created July 2026
Author: Arne Thomsen (with Claude)

Fit the per-cosmology linear source galaxy clustering bias b_g,s of every metacal tomographic bin.

Following section 3 of https://arxiv.org/abs/2511.04681, the source galaxy number counts are modelled as

    n_g,si = <n_g,si> * (1 + b_g,si * (delta_m,si - <delta_m,si>) / <delta_m,si>)

and Poisson sampled. The bias b_g,si is then fixed per cosmology by matching the one-point function (the pixel
histogram of the galaxy counts) of the simulated counts to a reference observation. This absorbs the cosmology
dependence of the one-point source galaxy distribution into the bias, which the shape noise generator subsequently
uses to draw as many galaxies from the real DES Y3 shear catalog as there are galaxies in a given pixel.

The reference observation was Buzzard in the above paper and is DES Y3 here. The resulting bias table is consumed by
files.read_metacal_bias in the count based shape noise branch of postprocessing.postprocess_shape_noise, see
files.get_shape_noise (method "count", bias "fixed").

The DES Y3 count map contains imaging systematics (depth, seeing, stellar density, ...) that the simulations do not
have, which the fit would otherwise absorb into the bias. The correction therefore goes on the SIMULATION side: the
model density is contaminated with the DES imprint (files.get_metacal_systematics) before it is Poisson sampled,
while the target stays the raw, unmodified observation. Decontaminating the observation instead would fit a target
that the forward model is not asked to reproduce, and would not carry over into the pipeline.

This is an in-package reimplementation of
dev/scripts/source_galaxy_clustering_bias/jozef_code/optimize_biases_parallel_desy3.py by Jozef Bucko, which stays
unchanged as the reference. The deliberate deviations from it are marked with "deviation:" comments below.
"""

import os, warnings, multiprocessing

import numpy as np
import h5py
from scipy.optimize import minimize
from scipy.stats import wasserstein_distance

from msfm.utils import files, filenames, catalog, clustering, cosmogrid, imports, logger

hp = imports.import_healpy(parallel=False)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)

# one-point function: 200 bins covering 1 to 200 galaxies per pixel. Note that the lower edge of 0.5 means that empty
# pixels are excluded from the histogram, which is intentional and inherited from the reference implementation
N_BINS_1PT = 200
RANGE_1PT = (0.5, 200.5)

# two-point function: per tomographic bin ell range of the pseudo Cls of the count maps
L_MIN_2PT = 30
L_MAX_2PT = (589, 863, 1159, 1382)

# fixing the seed of every single cost function evaluation is what makes the Poisson sampled cost function
# deterministic in the bias, without which the Nelder-Mead simplex cannot converge
DEFAULT_SEED = 42

# the per cosmology fit diagnostics stored alongside the biases, which no consumer of the table reads
_DIAGNOSTIC_COMPRESSION = {"compression": "gzip", "compression_opts": 4}


def _count_statistic(counts, kind, i_z, mask=None):
    """One-point or two-point statistic of a single tomographic galaxy count map.

    Args:
        counts (np.ndarray): Galaxy counts, either of the footprint pixels only (kind "1pt") or of the full sky
            (kind "2pt").
        kind (str): Either "1pt" for the pixel count histogram or "2pt" for the pseudo Cls.
        i_z (int): Index of the tomographic bin, only used to select the ell range of kind "2pt".
        mask (np.ndarray, optional): Boolean full sky footprint mask, required for kind "2pt". Defaults to None.

    Raises:
        ValueError: If an unknown kind is passed.

    Returns:
        np.ndarray: The statistic, of length N_BINS_1PT or L_MAX_2PT[i_z] - L_MIN_2PT.
    """
    if kind == "1pt":
        return np.histogram(counts, bins=N_BINS_1PT, range=RANGE_1PT)[0]
    elif kind == "2pt":
        # pseudo Cls of the masked count map, no mode coupling deconvolution (as in the reference)
        return hp.anafast((counts * mask).astype(np.float64))[L_MIN_2PT : L_MAX_2PT[i_z]]
    else:
        raise ValueError(f"Unknown kind {kind!r}, use either '1pt' or '2pt'")


def _bin_centres_1pt():
    """Centres of the one-point function bins, the support of the pixel count distribution."""
    edges = np.linspace(RANGE_1PT[0], RANGE_1PT[1], N_BINS_1PT + 1)

    return 0.5 * (edges[:-1] + edges[1:])


def decontaminate_counts(counts, weight, renormalize=True):
    """Remove the imaging systematics from an observed galaxy count map, n_corrected = n_observed * W.

    DIAGNOSTIC ONLY, this is not part of the fit. The fit contaminates the simulation instead and leaves the
    observation alone, see the module docstring. This helper exists so that the notebook can reproduce the third
    column of the lss_sys sky images, which is the standard visual check that the correction map does what it says.

    Args:
        counts (np.ndarray): (n_fp, n_z) observed galaxy counts of the footprint pixels.
        weight (np.ndarray): (n_fp, n_z) ISD weight map W of the same pixels, see files.get_metacal_systematics with
            dataset "weight".
        renormalize (bool, optional): Whether to rescale each tomographic bin such that the total galaxy count is
            preserved. W has unit mean over the footprint PIXELS but not over the GALAXIES, which sit preferentially
            where the systematics inflated the counts, i.e. where W < 1. Without this the corrected counts lose up to
            8.7% of the galaxies of a bin, which would show up as a shift of n_bar rather than as the change of the
            shape of the count distribution that the correction is about. The absolute normalization of W carries no
            information anyway, only pixel ratios do. Defaults to True.

    Returns:
        np.ndarray: (n_fp, n_z) corrected galaxy counts, not integer valued.
    """
    corrected = counts.astype(np.float64) * weight

    if renormalize:
        scale = np.sum(counts, axis=0) / np.sum(corrected, axis=0)
        corrected = corrected * scale
        LOGGER.info(f"Rescaled the corrected counts by {np.array2string(scale, precision=4)} to conserve galaxies")

    return corrected


def build_target(conf=None, count_map=None, kind="1pt"):
    """Build the target statistic of the reference observation to optimize the bias against.

    Args:
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). Defaults to None.
        count_map (np.ndarray, optional): Full sky (n_pix, n_z) galaxy count map of the reference observation. Defaults
            to None, in which case the DES Y3 metacal count map is built from (or read from the cache of)
            catalog.build_metacal_map_from_cat.
        kind (str, optional): Either "1pt" or "2pt". Defaults to "1pt".

    Returns:
        tuple: (target, n_bar, mask) with the per tomographic bin list of target statistics, the (n_z,) mean number of
            galaxies per footprint pixel and the (n_pix,) boolean footprint mask in the RING scheme.
    """
    conf = files.load_config(conf)

    n_side = conf["analysis"]["n_side"]
    n_z = len(conf["survey"]["metacal"]["z_bins"])

    # RING to match the CosmoGrid full sky maps
    mask = files.get_mask(conf, nest_out=False)

    if count_map is None:
        # returns immediately from data/metacal_wl_count_map.npy if that cache exists
        _, count_map = catalog.build_metacal_map_from_cat(conf, debug=False)

    assert count_map.shape == (
        hp.nside2npix(n_side),
        n_z,
    ), f"Expected a full sky ({hp.nside2npix(n_side)}, {n_z}) count map, got {count_map.shape}"

    n_bar = np.sum(count_map[mask], axis=0) / np.sum(mask)

    # the forward model itself derives n_bar from the config instead, see postprocessing.postprocess_shape_noise.
    # Assert that the two agree so that a drift between the fit target and what is actually generated fails loudly
    n_bar_conf = np.array(conf["survey"]["metacal"]["n_gal"]) * hp.nside2pixarea(n_side, degrees=True)
    assert np.allclose(n_bar, n_bar_conf, rtol=0.05), (
        f"The empirical mean galaxy count per pixel {n_bar} disagrees with the config survey.metacal.n_gal "
        f"{n_bar_conf} by more than 5%"
    )
    LOGGER.info(
        f"Mean galaxy counts per pixel: {np.array2string(n_bar, precision=2)} "
        f"(config: {np.array2string(n_bar_conf, precision=2)})"
    )

    target = []
    for i_z in range(n_z):
        if kind == "1pt":
            # deviation: restricted to the footprint pixels explicitly. The reference zeroed the counts outside of the
            # footprint and histogrammed the full sky, which is equivalent since RANGE_1PT excludes empty pixels
            target.append(_count_statistic(count_map[mask, i_z], kind, i_z))
        else:
            target.append(_count_statistic(count_map[:, i_z].astype(np.float64), kind, i_z, mask=mask))

    return target, n_bar, mask


def cosmo_key(path_par):
    """Name of a CosmoGrid cosmology from its entry in the CosmoGridV1 metainfo file.

    Args:
        path_par (str, bytes): Parameter file path of a cosmology, as stored in the "path_par" field of
            cosmogrid.get_cosmo_params_info, e.g. "CosmoGrid/raw/grid/cosmo_000001/params.yml".

    Returns:
        str: The name of the cosmology directory, e.g. "cosmo_000001", which is also its key in the bias table.
    """
    return np.asarray(path_par).astype(str).item().split("/")[-2]


def get_cosmogrid_dirs(conf=None, cosmogrid_dir=None, simset="grid", with_fiducial=True):
    """Map every cosmology of a CosmoGrid simulation set to its directory, in the form fit_bias_table expects.

    Args:
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). Defaults to None.
        cosmogrid_dir (str): Root of the projected CosmoGrid maps, i.e. the directory that holds the "grid" and
            "fiducial" simulation sets.
        simset (str, optional): CosmoGrid simulation set to enumerate. Defaults to "grid".
        with_fiducial (bool, optional): Whether to append the fiducial cosmology under the key "fiducial". It is then
            fit directly, rather than aliased to the closest grid cosmology as in notebooks/metacal_bias.ipynb.
            Defaults to True.

    Returns:
        dict: Mapping of the bias table key to the directory of the corresponding cosmology.
    """
    conf = files.load_config(conf)
    assert cosmogrid_dir is not None, "The root directory of the projected CosmoGrid maps is required"

    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    meta_info_file = os.path.join(repo_dir, conf["files"]["meta_info"])

    params = cosmogrid.get_cosmo_params_info(meta_info_file, simset=simset)
    cosmo_dirs = {cosmo_key(p): os.path.join(cosmogrid_dir, simset, cosmo_key(p)) for p in params["path_par"]}

    LOGGER.info(
        f"Enumerated {len(cosmo_dirs)} cosmologies of the CosmoGrid {simset} simulation set"
        + (" plus the fiducial one" if with_fiducial else "")
    )

    if with_fiducial:
        cosmo_dirs["fiducial"] = os.path.join(cosmogrid_dir, "fiducial", "cosmo_fiducial")

    return cosmo_dirs


def read_density_contrast(conf, cosmo_dir, perm=0):
    """Read the normalized dark matter density contrast of the metacal source bins of a CosmoGrid cosmology.

    The h5 read and the ud_grade mirror the private postprocessing._read_full_sky_bin, and the normalization mirrors
    postprocessing.postprocess_shape_noise. They are reimplemented here so that this module stays free of the
    TensorFlow import that postprocessing pulls in at module scope.

    Args:
        conf (str, dict): Can be either a string (a config.yaml is read in), a dictionary (the config is passed
            through) or None (the default config is loaded).
        cosmo_dir (str): Directory of a single CosmoGrid cosmology, e.g. <...>/bary/grid/cosmo_000001.
        perm (int, optional): Permutation index of the cosmology. Defaults to 0.

    Returns:
        np.ndarray: (n_pix, n_z) normalized density contrast of the metacal source bins in the RING scheme.
    """
    conf = files.load_config(conf)

    n_side = conf["analysis"]["n_side"]
    n_pix = hp.nside2npix(n_side)
    z_bins = conf["survey"]["metacal"]["z_bins"]
    with_bary = conf["analysis"]["modelling"]["baryonified"]

    full_maps_file = filenames.get_filename_full_maps(os.path.join(cosmo_dir, f"perm_{perm:04d}"), with_bary=with_bary)

    delta = np.zeros((n_pix, len(z_bins)), dtype=np.float32)
    with h5py.File(full_maps_file, "r") as f:
        for i_z, z_bin in enumerate(z_bins):
            delta_full = f[f"map/dg/{z_bin}"][:]

            # the CosmoGrid maps are stored at a higher resolution than the analysis n_side
            if delta_full.shape[0] != n_pix:
                delta_full = hp.ud_grade(delta_full, nside_out=n_side, order_in="RING", order_out="RING", pess=True)

            # normalize to a number density contrast
            delta[:, i_z] = (delta_full - np.mean(delta_full)) / np.mean(delta_full)

    return delta


def counts_from_bias(n_bar, delta, bias, contamination=None, seed=DEFAULT_SEED):
    """Poisson sampled galaxy counts of a single tomographic bin for a given linear source clustering bias.

    Args:
        n_bar (float): Mean number of galaxies per pixel.
        delta (np.ndarray): Normalized density contrast, either of the footprint pixels only or of the full sky.
        bias (float): Linear source galaxy clustering bias b_g,s.
        contamination (np.ndarray, optional): DES Y3 imaging systematics contamination factor of the same shape as
            delta, see files.get_metacal_systematics. Defaults to None, i.e. a clean simulation.
        seed (int, optional): Seed of the Poisson realization. Must be held fixed across the cost function
            evaluations of a single fit, otherwise the objective is stochastic and Nelder-Mead cannot converge.
            Defaults to DEFAULT_SEED.

    Returns:
        np.ndarray: Poisson sampled galaxy counts, of the same shape as delta.
    """
    # identical clip and renormalize transform as the reference, see DeepLSS appendix E
    ng = clustering.galaxy_density_to_count(n_bar, delta, bias, contamination_map=contamination)

    return np.random.default_rng(seed).poisson(ng).astype(np.float32)


def _cost(bias, delta_z, n_bar_z, target_z, kind, i_z, mask, seed, contamination_z=None):
    """Distance between the statistic of the simulated counts and the target, minimized over the bias."""
    counts = counts_from_bias(n_bar_z, delta_z, bias, contamination=contamination_z, seed=seed)
    stat = _count_statistic(counts, kind, i_z, mask=mask)

    if kind == "1pt":
        if np.sum(stat) == 0:
            # every pixel fell outside of RANGE_1PT, the bias is far outside of any sensible range
            return np.inf

        # deviation: the reference passed the histogram heights themselves to wasserstein_distance, which is
        # permutation invariant across the bins and therefore blind to which count value has which frequency. This is
        # the actual 1D Wasserstein distance between the two pixel count distributions
        centres = _bin_centres_1pt()
        return wasserstein_distance(centres, centres, u_weights=stat, v_weights=target_z)
    else:
        return np.sum((np.log10(stat) - np.log10(target_z)) ** 2)


def fit_bias(delta, target, n_bar, mask=None, kind="1pt", b_init=1.0, seed=DEFAULT_SEED, contamination=None):
    """Fit the linear source clustering bias of every tomographic bin of a single cosmology.

    Args:
        delta (np.ndarray): (n_pix, n_z) normalized density contrast, as returned by read_density_contrast.
        target (list): Per tomographic bin target statistics, as returned by build_target.
        n_bar (np.ndarray): (n_z,) mean number of galaxies per footprint pixel, as returned by build_target.
        mask (np.ndarray, optional): (n_pix,) boolean footprint mask. Required for kind "1pt", where the fit is
            restricted to the footprint pixels, and for kind "2pt", where the count maps are masked. Defaults to None.
        kind (str, optional): Either "1pt" or "2pt". Defaults to "1pt".
        b_init (float, optional): Initial guess of the bias. Defaults to 1.0.
        seed (int, optional): Seed of the Poisson realizations. Defaults to DEFAULT_SEED.
        contamination (np.ndarray, optional): (n_fp, n_z) DES Y3 imaging systematics contamination factor of the
            footprint pixels, as returned by files.get_metacal_systematics. Its pixel ordering is the base patch,
            which is np.arange(n_pix)[mask] by construction and therefore aligned with delta[mask]. Defaults to None,
            i.e. a clean simulation.

    Returns:
        tuple: (biases, losses, stats) with the (n_z,) best fit biases, the (n_z,) minimal losses and the statistics
            of the best fit counts. The latter is an (n_z, N_BINS_1PT) array for kind "1pt" and a list of per bin
            arrays for kind "2pt", where L_MAX_2PT differs between the tomographic bins.
    """
    n_z = delta.shape[-1]
    assert mask is not None, "A footprint mask is required"

    biases = np.zeros(n_z)
    losses = np.zeros(n_z)
    stats = []
    for i_z in range(n_z):
        # deviation: for the one-point function only the footprint pixels enter, which is both ~10x faster and makes
        # the clip renormalization of galaxy_density_to_count conserve galaxies over exactly the region that n_bar was
        # measured on. The two-point function needs the full sky map to be masked instead
        delta_z = delta[mask, i_z] if kind == "1pt" else delta[:, i_z]

        contamination_z = None
        if contamination is not None:
            if kind == "1pt":
                contamination_z = contamination[:, i_z]
            else:
                # the two-point function works on the full sky, where there is no correction off the footprint
                contamination_z = np.ones(len(delta))
                contamination_z[mask] = contamination[:, i_z]

        result = minimize(
            lambda b: _cost(b[0], delta_z, n_bar[i_z], target[i_z], kind, i_z, mask, seed, contamination_z),
            x0=[b_init],
            method="Nelder-Mead",
        )
        biases[i_z] = result.x[0]
        losses[i_z] = result.fun

        counts = counts_from_bias(n_bar[i_z], delta_z, biases[i_z], contamination=contamination_z, seed=seed)
        stats.append(_count_statistic(counts, kind, i_z, mask=mask))

    # the two-point statistics are ragged, since L_MAX_2PT differs between the tomographic bins
    if len({stat.shape for stat in stats}) == 1:
        stats = np.array(stats)

    return biases, losses, stats


def fit_cosmology(
    conf, cosmo_dir, target, n_bar, mask, perm=0, kind="1pt", b_init=1.0, seed=DEFAULT_SEED, contamination=None
):
    """Read the density contrast of a single cosmology and fit its biases. The unit of work of fit_bias_table.

    Args:
        conf (str, dict): Configuration, see files.load_config.
        cosmo_dir (str): Directory of a single CosmoGrid cosmology.
        target (list): Per tomographic bin target statistics, as returned by build_target.
        n_bar (np.ndarray): (n_z,) mean number of galaxies per footprint pixel.
        mask (np.ndarray): (n_pix,) boolean footprint mask.
        perm (int, optional): Permutation index of the cosmology. Defaults to 0.
        kind (str, optional): Either "1pt" or "2pt". Defaults to "1pt".
        b_init (float, optional): Initial guess of the bias. Defaults to 1.0.
        seed (int, optional): Seed of the Poisson realizations. Defaults to DEFAULT_SEED.
        contamination (np.ndarray, optional): Imaging systematics contamination factor, see fit_bias. Defaults to None.

    Returns:
        tuple: (biases, losses, stats), see fit_bias.
    """
    delta = read_density_contrast(conf, cosmo_dir, perm=perm)

    return fit_bias(delta, target, n_bar, mask=mask, kind=kind, b_init=b_init, seed=seed, contamination=contamination)


# module level state of the pool workers, set once per worker by _pool_init to avoid pickling the target and the mask
# for every single cosmology
_POOL_STATE = {}


def _pool_init(conf, target, n_bar, mask, perm, kind, b_init, seed, contamination):
    _POOL_STATE.update(
        conf=conf,
        target=target,
        n_bar=n_bar,
        mask=mask,
        perm=perm,
        kind=kind,
        b_init=b_init,
        seed=seed,
        contamination=contamination,
    )


def _pool_worker(item):
    key, cosmo_dir = item
    biases, losses, stats = fit_cosmology(
        _POOL_STATE["conf"],
        cosmo_dir,
        _POOL_STATE["target"],
        _POOL_STATE["n_bar"],
        _POOL_STATE["mask"],
        perm=_POOL_STATE["perm"],
        kind=_POOL_STATE["kind"],
        b_init=_POOL_STATE["b_init"],
        seed=_POOL_STATE["seed"],
        contamination=_POOL_STATE["contamination"],
    )

    return key, biases, losses, stats


def fit_bias_table(
    conf,
    cosmo_dirs,
    out_file,
    target=None,
    n_bar=None,
    mask=None,
    count_map=None,
    n_proc=None,
    perm=0,
    kind="1pt",
    b_init=1.0,
    seed=DEFAULT_SEED,
    contamination=None,
    systematics_label=None,
):
    """Fit the source clustering biases of many cosmologies in parallel and store them as an h5 bias table.

    The biases are stored as top level datasets keyed by the cosmology, such that the resulting file can be used as
    conf["files"]["metacal_bias"] and read with files.read_metacal_bias without any changes. The losses and the
    statistics of the best fit counts are stored alongside them in the "loss" and "stat" groups, the latter as an
    (n_z, N_BINS_1PT) dataset for kind "1pt" and as one dataset per tomographic bin for the ragged kind "2pt".

    Args:
        conf (str, dict): Configuration, see files.load_config.
        cosmo_dirs (dict): Mapping of the h5 key (e.g. "cosmo_000001" or "fiducial") to the directory of the
            corresponding CosmoGrid cosmology.
        out_file (str): Path of the h5 file to write. Any existing file is overwritten.
        target (list, optional): Target statistics, see build_target. Defaults to None, in which case build_target is
            called.
        n_bar (np.ndarray, optional): Mean galaxy counts per pixel, see build_target. Defaults to None.
        mask (np.ndarray, optional): Footprint mask, see build_target. Defaults to None.
        count_map (np.ndarray, optional): Reference observation count map, only used if target is None. Defaults to
            None.
        n_proc (int, optional): Number of worker processes. Defaults to None, i.e. all available CPUs.
        perm (int, optional): Permutation index of the cosmologies. Defaults to 0.
        kind (str, optional): Either "1pt" or "2pt". Defaults to "1pt".
        b_init (float, optional): Initial guess of the bias. Defaults to 1.0.
        seed (int, optional): Seed of the Poisson realizations. Defaults to DEFAULT_SEED.
        contamination (np.ndarray, optional): Imaging systematics contamination factor, see fit_bias. Defaults to
            None, i.e. a clean simulation.
        systematics_label (str, optional): Name of the imaging systematics run that contamination came from, stored
            as a root attribute so that a bias table records which forward model it was fit against. Defaults to
            None, which is stored as "none".

    Returns:
        dict: Mapping of the h5 key to the (n_z,) best fit biases.
    """
    conf = files.load_config(conf)

    if target is None or n_bar is None or mask is None:
        target, n_bar, mask = build_target(conf, count_map=count_map, kind=kind)

    if n_proc is None:
        try:
            n_proc = len(os.sched_getaffinity(0))
        except AttributeError:
            n_proc = os.cpu_count()
    n_proc = min(n_proc, len(cosmo_dirs))

    LOGGER.info(f"Fitting the {kind} source clustering bias of {len(cosmo_dirs)} cosmologies on {n_proc} processes")
    LOGGER.timer.start("fit_bias_table")

    biases, losses, stats = {}, {}, {}
    init_args = (conf, target, n_bar, mask, perm, kind, b_init, seed, contamination)
    with multiprocessing.Pool(n_proc, initializer=_pool_init, initargs=init_args) as pool:
        results = pool.imap_unordered(_pool_worker, list(cosmo_dirs.items()))
        for key, bias, loss, stat in LOGGER.progressbar(
            results, at_level="info", total=len(cosmo_dirs), desc="fitting biases"
        ):
            biases[key] = bias
            losses[key] = loss
            stats[key] = stat

    LOGGER.info(f"Fitted {len(biases)} cosmologies after {LOGGER.timer.elapsed('fit_bias_table')}")

    with h5py.File(out_file, "w") as f:
        # which forward model the biases were fit against. The consumer asserts that this matches its own config,
        # since a table fit against a clean model absorbs the imaging systematics into b and must not be combined
        # with a contaminated forward model, nor the other way around
        f.attrs["systematics_label"] = systematics_label if systematics_label is not None else "none"
        f.attrs["kind"] = kind

        keys = list(cosmo_dirs)
        # the row order of the stacked diagnostics below
        f.create_dataset("keys", data=np.array(keys, dtype=object), dtype=h5py.string_dtype())

        for key in keys:
            # one dataset per cosmology at the top level, such that files.read_metacal_bias(key, conf) works unchanged
            f.create_dataset(key, data=biases[key])

        # the diagnostics are STACKED instead of written per cosmology: 2500 tiny HDF5 datasets cost several times
        # more in object metadata than the data itself (measured 14.5 -> 4.4 MB for the full table), and the table has
        # to be small enough to live in the repository
        f.create_dataset("loss", data=np.array([losses[key] for key in keys]), **_DIAGNOSTIC_COMPRESSION)

        if all(isinstance(stats[key], np.ndarray) for key in keys):
            stat = np.array([stats[key] for key in keys])
            # the one-point statistic is a histogram of pixel counts, which fits comfortably into int32
            if np.issubdtype(stat.dtype, np.integer):
                stat = stat.astype(np.int32)
            f.create_dataset("stat", data=stat, **_DIAGNOSTIC_COMPRESSION)
        else:
            # the two-point statistics are ragged across the tomographic bins, so they cannot be stacked
            stat_group = f.create_group("stat")
            for key in keys:
                for i_z, stat in enumerate(stats[key]):
                    stat_group.create_dataset(f"{key}/bin_{i_z}", data=stat, **_DIAGNOSTIC_COMPRESSION)
    LOGGER.info(f"Stored the bias table of {len(biases)} cosmologies in {out_file}")

    return biases
