# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created November 2026
Author: Arne Thomsen

Derives the DES Y6 data products in data/desy6/ and the data-derived config values from the public Y6 release,
one step per product (data/desy6/README.md). The steps read the release from dirs.catalog, step systematics reads
an lss_sys export and step sc_bias the Y6 CosmoGrid projection (both --source). Every step writes only to the paths
its config keys name (under data/desy6/) and stops on a failed check. With --provenance, it requires committed code
and records its inputs, outputs and the msfm commit in files.provenance.

Steps, in order:
    nz           n(z) text files of all 4 source and 6 lens bins, from the 3x2pt FITS
    pixels       pixel file: the rotated nside-512 footprint, its four patches and the padded data vector, from the
                 mask, and the MagLim weight maps that correct the observed lens counts, from the catalog
    maps         shear and count maps of every source and lens bin, and the source response, in dirs.cache, where the
                 builders of msfm.utils.catalog load them
    values       n_gal, m prior, Aeff and the lens bias fiducials, written into the config lines marked
                 "# <- step values: <dotted.key>"
    noise        shape-noise file: (e1 / R, e2 / R, w, pix) per galaxy of every source bin
    systematics  source imaging-systematics maps, copied from the nside-512 export of an lss_sys run
    sc_bias      source-clustering bias table, fit per grid cosmology and on fiducial_bench, on perm_0000

Usage:
    python -m msfm.apps.products.desy6 nz --config configs/desy6/v1/v1.yaml
    python -m msfm.apps.products.desy6 pixels --config configs/desy6/v1/v1.yaml   # compute node, needs healsparse
    python -m msfm.apps.products.desy6 maps --config configs/desy6/v1/v1.yaml --max-objects 100000   # smoke test
    python -m msfm.apps.products.desy6 maps --config configs/desy6/v1/v1.yaml   # compute node, ~140M objects
    python -m msfm.apps.products.desy6 values --config configs/desy6/v1/v1.yaml
    python -m msfm.apps.products.desy6 values --config configs/desy6/v1/v1.yaml --check   # exit 1 if stale
    python -m msfm.apps.products.desy6 noise --config configs/desy6/v1/v1.yaml   # compute node, ~2 GB out
    python -m msfm.apps.products.desy6 systematics --config configs/desy6/v1/v1.yaml \
        --source <lss_sys>/output/desy6/<label>/<label>_512.h5
    python -m msfm.apps.products.desy6 sc_bias --config configs/desy6/v1/v1.yaml --max-cosmologies 1   # smoke
    python -m msfm.apps.products.desy6 sc_bias --config configs/desy6/v1/v1.yaml   # compute node, minutes

sc_bias reads the projection of the config's locations.yaml (msfm.utils.locations) unless --source names another.

--dry-run lists a step's inputs and outputs and writes nothing. --max-objects N runs a catalog step on the first N
rows of every catalog and writes nothing, so a partial map can never pass for the product. --max-cosmologies N does
the same for step sc_bias with the first N grid cosmologies and the fiducial one.
"""

import argparse
import copy
import os
import re
import shutil
import sys
import tempfile

import h5py
import healpy as hp
import numpy as np

from msfm.utils import catalog, catalog_desy6, configuration, files, filenames, locations, logger, products, redshift
from msfm.utils import source_clustering_bias as scb

LOGGER = logger.get_logger(__file__)

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SURVEY_NAME = "DESy6"
OUT_ROOT = "data/desy6"
CONFIG_ROOT = "configs/desy6"
STEP_ORDER = ["nz", "pixels", "maps", "values", "noise", "systematics", "sc_bias"]

# config sample key -> (n(z) HDU of the 3x2pt FITS, number of survey bins it must hold)
NZ_HDUS = {"metacal": ("nz_source", catalog_desy6.N_SOURCE_BINS), "maglim": ("nz_lens", catalog_desy6.N_LENS_BINS)}

# step pixels: the mask is rotated at this resolution, the one of the Y3 mask (notebooks/desy3/pixel_file.ipynb)
MASK_ROTATE_NSIDE = 4096
# patches 1-3 are the base patch rotated by 180 deg in phi and/or mirrored at the equator; mirroring flips gamma_2
PATCH_GAMMA_2_SIGN = [1, 1, -1, -1]

# step maps: cache file -> what it holds; every map covers all survey bins, axis 1 in survey-bin order. The names are
# the ones msfm.utils.catalog loads, as for Y3; it asks for the shear without the spin-2 rotation too
MAPS_FILES = {
    "metacal_gamma": "metacal_gamma_e1p_e2m.npy",
    "metacal_gamma_no_psi_rot": "metacal_gamma_no_psi_rot_e1p_e2m.npy",
    "metacal_count": "metacal_count.npy",
    "metacal_response": "metacal_response.npy",
    "maglim_count": "maglim_count.npy",
}

# step values: the lens bias fiducials are v18's power law (amplitude, exponent, pivot z0) on the lens n(z)
BG_FIDUCIAL_POWER_LAW = (1.5, 0.5, 0.62)

# step sc_bias: the permutation every cosmology is fit on, as for Y3, and the arms of the table with whether the
# forward model of each imprints the imaging systematics (shape_noise.survey_systematics)
SC_BIAS_PERM = 0
SC_BIAS_ARMS = {"clean": False, "contam": True}
# the key "fiducial", which the benchmark mocks read, is fit on the benchmark at the fiducial cosmology and resolution
SC_BIAS_FIDUCIAL_DIR = os.path.join("benchmarks", "fiducial_bench")


def _out_path(rel_path):
    """Absolute path of a config-named output, which must lie under data/desy6/."""
    rel_path = os.path.normpath(rel_path)
    if os.path.isabs(rel_path) or not (rel_path + os.sep).startswith(OUT_ROOT + os.sep):
        raise ValueError(f"DES Y6 outputs must lie under {OUT_ROOT}/, got {rel_path}")
    return os.path.join(REPO_DIR, rel_path)


def _dry_run(step, inputs, outputs):
    """Lists what a step would read and write. Returns None, so nothing is recorded."""
    return products.dry_run(step, inputs, outputs, rel=_rel)


def _maps_path(conf, key):
    """Path of a step maps file in dirs.cache, which must lie under data/desy6/."""
    _out_path(conf["dirs"]["cache"])
    return catalog.cache_file(conf, MAPS_FILES[key])


def _rel(path):
    """Repository-relative path for files inside the repository, absolute otherwise."""
    return products.rel_path(path, REPO_DIR)


def step_nz(conf, args):
    """Writes the fiducial n(z) of every source and lens bin as two columns (z_mid, n(z)), as for Y3.

    Returns:
        tuple: (input paths, output paths)
    """
    from astropy.io import fits

    fits_path = catalog_desy6.release_path(conf, "3x2pt")
    out_dir = _out_path(conf["dirs"]["redshift_distributions"])
    # the prefix files.load_redshift_distributions reads
    prefix = conf["survey"]["name"].lower()
    if args.dry_run:
        outputs = [
            filenames.get_filename_z_distribution(out_dir, sample, i_bin, prefix=prefix)
            for sample, (_, n_bins) in NZ_HDUS.items()
            for i_bin in range(1, n_bins + 1)
        ]
        return _dry_run("nz", [fits_path], outputs)
    os.makedirs(out_dir, exist_ok=True)

    outputs = []
    z_grid = None
    with fits.open(fits_path, memmap=False) as hdul:
        for sample, (hdu_name, n_bins) in NZ_HDUS.items():
            table = hdul[hdu_name].data
            z_low, z_mid, z_high = (np.asarray(table[col], dtype=np.float64) for col in ("Z_LOW", "Z_MID", "Z_HIGH"))

            # one contiguous grid of mid points, shared by both HDUs
            if z_grid is None:
                z_grid = (z_low, z_mid, z_high)
            assert all(np.array_equal(a, b) for a, b in zip(z_grid, (z_low, z_mid, z_high))), "z grids differ"
            assert np.all(np.diff(z_mid) > 0), f"{hdu_name}: Z_MID is not increasing"
            assert np.allclose(z_mid, 0.5 * (z_low + z_high), rtol=0, atol=1e-12), f"{hdu_name}: Z_MID is no mid point"
            assert np.allclose(
                z_low[1:], z_high[:-1], rtol=0, atol=1e-12
            ), f"{hdu_name}: the z bins are not contiguous"

            bin_cols = [col for col in table.columns.names if re.fullmatch(r"BIN\d+", col)]
            assert bin_cols == [f"BIN{i}" for i in range(1, n_bins + 1)], f"{hdu_name}: columns {bin_cols}"
            missing = {files.get_z_bin_number(z) for z in conf["survey"][sample]["z_bins"]} - set(range(1, n_bins + 1))
            assert not missing, f"survey.{sample}.z_bins names survey bins {missing} that {hdu_name} does not have"

            for i_bin in range(1, n_bins + 1):
                nz = np.asarray(table[f"BIN{i_bin}"], dtype=np.float64)
                assert np.all(np.isfinite(nz)) and np.all(nz >= 0), f"{hdu_name} BIN{i_bin}: negative or not finite"
                norm = np.sum(nz * (z_high - z_low))
                assert abs(norm - 1) < 1e-3, f"{hdu_name} BIN{i_bin}: integrates to {norm}"

                path = filenames.get_filename_z_distribution(out_dir, sample, i_bin, prefix=prefix)
                header = (
                    f"z_mid n(z): DES Y6 {hdu_name} BIN{i_bin} of {catalog_desy6.RELEASE_FILES['3x2pt']}, "
                    "written by msfm.apps.products.desy6 nz"
                )
                np.savetxt(path, np.column_stack([z_mid, nz]), header=header)

                # the text round trip is exact, so the file holds the FITS values
                back = np.loadtxt(path)
                assert np.array_equal(back[:, 0], z_mid) and np.array_equal(back[:, 1], nz), f"{path}: round trip"
                outputs.append(path)
                LOGGER.info(f"{os.path.relpath(path, REPO_DIR)}: mean z = {np.sum(nz * z_mid) / np.sum(nz):.4f}")

    return [fits_path], outputs


def _rotate_pix(pix, n_side, n_rot):
    """Rotates RING pixels by n_rot * 90 deg in phi, a pixel-exact symmetry of HEALPix."""
    theta, phi = hp.pix2ang(n_side, pix)
    return hp.ang2pix(n_side, theta, (phi + n_rot * np.pi / 2) % (2 * np.pi))


def _mirror_pix(pix, n_side):
    """Mirrors RING pixels at the equator, a pixel-exact symmetry of HEALPix."""
    theta, phi = hp.pix2ang(n_side, pix)
    return hp.ang2pix(n_side, np.pi - theta, phi)


def step_pixels(conf, args):
    """Writes the pixel file in the layout files.load_pixel_file reads, as notebooks/desy3/pixel_file.ipynb did for Y3.

    As in Y3, the mask is rotated into the footprint frame (the rotation of the catalog maps) at nside 4096, then
    downsampled to analysis.n_side. The Y6 mask comes at nside 131072, so it is first degraded to nside 4096, keeping
    the covered fraction of each pixel. A pixel enters the footprint when the mask covers at least
    analysis.footprint.coverage_threshold of it. Every survey bin gets the same patches, stored once and hard-linked
    under each bin label. The mask area, exact at full resolution, goes into the attributes, for step values.

    The group maglim/systematics holds the ENet weights of every lens bin as a pixel-level w = 1 / <F> on the base
    patch (catalog_desy6.maglim_weight_maps), in the two layouts files.get_clustering_systematics reads. As for Y3,
    they multiply the observed counts (apply_maglim_sys_map of observation.forward_model_observation_map) and stay out
    of the forward model.

    Returns:
        tuple: (input paths, output paths), or None with --dry-run
    """
    inputs = [catalog_desy6.release_path(conf, key) for key in ("mask", "maglim")]
    out_path = _out_path(conf["files"]["pixels"])
    if args.dry_run:
        return _dry_run("pixels", inputs, [out_path])

    n_side, n_side_down = conf["analysis"]["n_side"], conf["analysis"]["n_side_down"]
    assert conf["analysis"]["n_patches"] == len(PATCH_GAMMA_2_SIGN), "the patch symmetries make four patches"
    assert n_side % n_side_down == 0, f"n_side {n_side} is no multiple of n_side_down {n_side_down}"
    threshold = conf["analysis"]["footprint"]["coverage_threshold"]
    assert 0 < threshold <= 1, f"analysis.footprint.coverage_threshold {threshold} is not in (0, 1]"
    survey_bins = {"metacal": catalog_desy6.N_SOURCE_BINS, "maglim": catalog_desy6.N_LENS_BINS}
    for sample, n_bins in survey_bins.items():
        missing = {files.get_z_bin_number(z) for z in conf["survey"][sample]["z_bins"]} - set(range(1, n_bins + 1))
        assert not missing, f"survey.{sample}.z_bins names survey bins {missing} that Y6 does not have"
    pix_area = hp.nside2pixarea(n_side, degrees=True)

    coverage_sky, area = catalog_desy6.mask_coverage(conf, MASK_ROTATE_NSIDE)
    rotation = conf["analysis"]["footprint"]["rotation"]
    rotator = hp.Rotator(rot=(0, -rotation["y_rad"], rotation["z_rad"]), eulertype="Y", deg=False)
    assert np.allclose(rotator.mat, catalog._get_footprint_rotation_matrix(conf)), "not the catalog maps' rotation"
    coverage = hp.ud_grade(rotator.rotate_map_pixel(coverage_sky), n_side)
    del coverage_sky
    rotated_area = np.sum(coverage) * pix_area
    assert abs(rotated_area / area - 1) < 1e-2, f"the rotated mask covers {rotated_area:.1f} deg^2, not {area:.1f}"

    # the base patch, its rotation, its mirror image and the rotated mirror image
    patch_0 = np.flatnonzero(coverage >= threshold)
    mirrored = _mirror_pix(patch_0, n_side)
    patches = np.stack([patch_0, _rotate_pix(patch_0, n_side, 2), mirrored, _rotate_pix(mirrored, n_side, 2)])
    assert np.unique(patches).size == patches.size, "the four rotated patches overlap: choose new footprint.rotation"

    # the threshold is chosen as in Y3, so the footprint area matches the mask area
    area_threshold = np.sort(coverage)[::-1][int(round(area / pix_area)) - 1]
    thresholds = sorted({0.5, 0.6, 0.8, 0.9, threshold})
    areas = ", ".join(f"{t}: {np.sum(coverage >= t) * pix_area:.1f}" for t in thresholds)
    LOGGER.info(
        f"footprint at threshold {threshold}: {patch_0.size} pixels, {patch_0.size * pix_area:.1f} deg^2 "
        f"({patch_0.size * pix_area / area - 1:+.2%} vs the mask), mean coverage {np.mean(coverage[patch_0]):.4f}; "
        f"the mask area is preserved at threshold {area_threshold:.4f}; mask {area:.1f} deg^2, rotated "
        f"{rotated_area:.1f} deg^2; area by threshold {{{areas}}} deg^2"
    )

    # padded data vector: all n_side pixels of every n_side_down pixel the base patch touches, NEST
    n_children = (n_side // n_side_down) ** 2
    nest_patch_0 = hp.ring2nest(n_side, patch_0)
    data_vec = (np.unique(nest_patch_0 // n_children)[:, None] * n_children + np.arange(n_children)).ravel()
    patch_to_data_vec = np.searchsorted(data_vec, nest_patch_0)
    assert np.array_equal(data_vec[patch_to_data_vec], nest_patch_0), "base patch pixels missing from the data vector"
    LOGGER.info(f"data vector: {data_vec.size} pixels, {data_vec.size // n_children} at nside {n_side_down}")

    # averaged over children at the resolution the mask is rotated at
    lens_weights, lens_filled = catalog_desy6.maglim_weight_maps(conf, patch_0, MASK_ROTATE_NSIDE)

    with h5py.File(out_path, "w") as f:
        f.attrs["info"] = (
            f"DES Y6 {catalog_desy6.RELEASE_FILES['mask']}, rotated by analysis.footprint.rotation, at nside {n_side}. "
            "Written by msfm.apps.products.desy6 pixels"
        )
        f.attrs["area_deg2"] = area
        f.attrs["coverage_threshold"] = threshold
        f.attrs["rotate_nside"] = MASK_ROTATE_NSIDE
        f.attrs["rotation_y_rad"] = conf["analysis"]["footprint"]["rotation"]["y_rad"]
        f.attrs["rotation_z_rad"] = conf["analysis"]["footprint"]["rotation"]["z_rad"]

        f.create_dataset("data_vec", data=data_vec, dtype="i4").attrs["info"] = "padded data vector pixels, NEST"
        f.create_dataset("footprint/coverage", data=coverage, dtype="f4", compression="gzip").attrs["info"] = (
            "fraction of each pixel the mask covers, RING"
        )
        f.create_dataset("footprint/patches", data=patches, dtype="i4").attrs["info"] = (
            "(n_patches, n_pix) pixels of the four patches, RING; patch 0 is the rotated footprint"
        )
        f.create_dataset("footprint/patch_to_data_vec", data=patch_to_data_vec, dtype="i4").attrs["info"] = (
            "position of each base-patch pixel in data_vec"
        )
        f.create_dataset("metacal/gamma_2_sign", data=PATCH_GAMMA_2_SIGN, dtype="i4").attrs["info"] = (
            "factor for gamma_2 per patch: -1 for the mirrored ones"
        )

        # every survey bin, so a change of z_bins needs no new file
        for sample, n_bins in survey_bins.items():
            for name in ("patches", "patch_to_data_vec"):
                group = f.require_group(f"{sample}/{name}")
                for i_bin in range(1, n_bins + 1):
                    group[f"{sample}{i_bin}"] = f[f"footprint/{name}"]

        sys_group = f.create_group("maglim/systematics")
        sys_group.attrs["info"] = (
            f"ENet weights w = 1 / <F> of {catalog_desy6.RELEASE_FILES['maglim']} on the base patch, from the "
            "per-galaxy weights (catalog_desy6.maglim_weight_maps). They multiply the observed counts. map: "
            "base-patch order; data_vector: data_vec order, 0 in the padding; filled: no galaxy in the pixel, "
            "neighbour mean"
        )
        for i_bin in range(1, catalog_desy6.N_LENS_BINS + 1):
            dv = np.zeros(data_vec.size, dtype=np.float32)
            dv[patch_to_data_vec] = lens_weights[:, i_bin - 1]
            sys_group.create_dataset(f"map/maglim{i_bin}", data=lens_weights[:, i_bin - 1], dtype="f4")
            sys_group.create_dataset(f"data_vector/maglim{i_bin}", data=dv)
            sys_group.create_dataset(f"filled/maglim{i_bin}", data=lens_filled[:, i_bin - 1])

    # the pipeline's loaders read the file back
    data_vec_back, patches_back, patch_to_data_vec_back, signs_back = files.load_pixel_file(conf)
    assert np.array_equal(data_vec_back, data_vec) and np.array_equal(signs_back, PATCH_GAMMA_2_SIGN), "read back"
    for sample in survey_bins:
        assert all(np.array_equal(p, patches) for p in patches_back[sample]), f"{sample} patches read back"
        assert all(np.array_equal(p, patch_to_data_vec) for p in patch_to_data_vec_back[sample]), f"{sample} read back"
    lens_columns = [files.get_z_bin_number(z) - 1 for z in conf["survey"]["maglim"]["z_bins"]]
    used_weights = lens_weights[:, lens_columns].astype(np.float32)
    sys_map = files.get_clustering_systematics(conf, pixel_type="map")
    sys_dv = files.get_clustering_systematics(conf, pixel_type="data_vector")
    assert np.array_equal(sys_map, used_weights), "maglim systematics map read back"
    assert np.array_equal(sys_dv[patch_to_data_vec], used_weights), "maglim systematics data vector read back"
    assert np.count_nonzero(sys_dv) == used_weights.size, "maglim systematics data vector: padding not 0"

    return inputs, [out_path]


def step_maps(conf, args):
    """Builds the shear and count maps of every source and lens bin, and saves the source response that calibrated
    the shear maps, so that step values takes R from the same numbers. The shear is also saved without the spin-2
    rotation, from a second pass over the catalog. The lens counts are raw, as for Y3, and float64 like the Y3 cache;
    the weight maps of step pixels correct them on the observation path.

    Returns:
        tuple: (input paths, output paths), or None with --dry-run or --max-objects
    """
    inputs = [catalog_desy6.release_path(conf, key) for key in ("metadetect", "maglim")]
    paths = {key: _maps_path(conf, key) for key in MAPS_FILES}
    if args.dry_run:
        return _dry_run("maps", inputs, list(paths.values()))

    gamma, source_count, response = catalog_desy6.build_metacal_maps(conf, max_objects=args.max_objects)
    gamma_no_rot, source_count_no_rot, response_no_rot = catalog_desy6.build_metacal_maps(
        conf, apply_shear_rotation=False, max_objects=args.max_objects
    )
    assert np.array_equal(source_count, source_count_no_rot) and np.array_equal(response, response_no_rot)
    lens_count = catalog_desy6.build_maglim_maps(conf, max_objects=args.max_objects)
    maps = {
        "metacal_gamma": gamma,
        "metacal_gamma_no_psi_rot": gamma_no_rot,
        "metacal_count": source_count,
        "metacal_response": response,
        "maglim_count": lens_count.astype(np.float64),
    }
    for key, values in maps.items():
        LOGGER.info(f"{os.path.basename(paths[key])}: shape {values.shape}, dtype {values.dtype}")

    if args.max_objects is not None:
        LOGGER.info(f"--max-objects {args.max_objects}: smoke test done, nothing is written")
        return None

    for key, values in maps.items():
        np.save(paths[key], values)
    return inputs, list(paths.values())


def step_values(conf, args):
    """Writes the data-derived values into the config lines marked "# <- step values: <dotted.key>", and nothing else.

    n_gal is the count inside the footprint (patch 0 of the pixel file) over the footprint's area, in deg^-2, as
    notebooks/desy3/config.ipynb did for Y3. For MagLim the count is the raw one times the weight maps of the pixel
    file, which is the observation the clean simulations are compared to, so the two match in expectation. The m prior
    is the mean and std of the M_BIN samples, and Aeff is the full-resolution mask area stored in the pixel file. The
    lens bias fiducials are BG_FIDUCIAL_POWER_LAW averaged over the n(z) of step nz, as for v18. Per-bin values follow
    the config's z_bins.

    With --check, writes nothing and exits with 1 if any marked value is stale.

    Returns:
        tuple: (input paths, output paths), or None with --dry-run, --check, or when every value is up to date and
            --provenance is not set
    """
    conf_path = os.path.abspath(args.config)
    if not (_rel(conf_path) + os.sep).startswith(CONFIG_ROOT + os.sep):
        raise ValueError(f"step values only writes configs under {CONFIG_ROOT}/, got {_rel(conf_path)}")
    pixel_path = _out_path(conf["files"]["pixels"])
    maps = {key: _maps_path(conf, key) for key in ("metacal_count", "maglim_count")}
    m_path = catalog_desy6.release_path(conf, "source_redshift")
    nz_dir = _out_path(conf["dirs"]["redshift_distributions"])
    lens_bins = conf["survey"]["maglim"]["z_bins"]
    nz_paths = [
        filenames.get_filename_z_distribution(nz_dir, "maglim", files.get_z_bin_number(z), prefix=SURVEY_NAME.lower())
        for z in lens_bins
    ]
    inputs = [pixel_path, *maps.values(), m_path, *nz_paths]
    if args.dry_run:
        return _dry_run("values", inputs, [conf_path])
    for path in inputs:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{_rel(path)} does not exist, run the steps nz, pixels and maps first")

    n_side = conf["analysis"]["n_side"]
    footprint_conf = conf["analysis"]["footprint"]
    with h5py.File(pixel_path, "r") as f:
        made_with = (f.attrs["rotation_y_rad"], f.attrs["rotation_z_rad"], f.attrs["coverage_threshold"])
        expected = (footprint_conf["rotation"]["y_rad"], footprint_conf["rotation"]["z_rad"])
        assert made_with == (*expected, footprint_conf["coverage_threshold"]), f"stale pixel file: {made_with}"
        footprint = f["footprint/patches"][0]
        coverage = f["footprint/coverage"][:].astype(np.float64)
        area = float(f.attrs["area_deg2"])
        # in base-patch order, which is the order of the footprint pixels
        lens_weights = np.stack(
            [f[f"maglim/systematics/map/maglim{i}"][:] for i in range(1, catalog_desy6.N_LENS_BINS + 1)], axis=-1
        ).astype(np.float64)
    footprint_area = footprint.size * hp.nside2pixarea(n_side, degrees=True)
    # the galaxies follow the mask, so the footprint should hold about the same fraction of both
    mask_fraction = np.sum(coverage[footprint]) / np.sum(coverage)

    values = {"survey.Aeff": area}
    for sample, key, n_bins in (
        ("metacal", "metacal_count", catalog_desy6.N_SOURCE_BINS),
        ("maglim", "maglim_count", catalog_desy6.N_LENS_BINS),
    ):
        count = np.load(maps[key])
        assert count.shape == (hp.nside2npix(n_side), n_bins), f"{_rel(maps[key])}: shape {count.shape}"
        inside = np.sum(count[footprint], axis=0, dtype=np.float64)
        fraction = inside / np.sum(count, axis=0, dtype=np.float64)
        assert np.all(
            np.abs(fraction - mask_fraction) < 0.02
        ), f"{sample}: {fraction} of the galaxies lie in the footprint, but {mask_fraction:.4f} of the mask"
        if sample == "maglim":
            # the observation: raw counts times the weight maps
            inside = np.sum(count[footprint] * lens_weights, axis=0)
        n_gal = inside / footprint_area
        values[f"survey.{sample}.n_gal"] = _configured_bins(conf, sample, n_gal)
        LOGGER.info(
            f"{sample} ({key}): n_gal = {n_gal / 60**2} arcmin^-2, in the footprint {fraction} "
            f"(mask {mask_fraction:.4f})"
        )

    mu, sigma = catalog_desy6.m_prior(conf)
    values["survey.metacal.shear_bias.multiplicative.mu"] = _configured_bins(conf, "metacal", mu)
    values["survey.metacal.shear_bias.multiplicative.sigma"] = _configured_bins(conf, "metacal", sigma)
    LOGGER.info(f"Aeff = {area:.3f} deg^2, footprint {footprint_area:.1f} deg^2")

    # untruncated, as redshift.get_tomo_amplitudes_according_to_config treats the lens sample
    bg, n_bg, z0 = BG_FIDUCIAL_POWER_LAW
    z, nz = files.load_redshift_distributions("maglim", conf)
    bg_fiducial = redshift.get_tomo_amplitudes(bg, n_bg, z, nz, z0=z0, truncate_nz=False)
    bg_params = conf["analysis"]["params"]["bg"]["linear"]
    assert bg_params == [f"bg{files.get_z_bin_number(z)}" for z in lens_bins], f"{bg_params} do not follow {lens_bins}"
    for param, value in zip(bg_params, bg_fiducial):
        # the shortest decimal that gives back the float32 amplitude
        values[f"analysis.fiducial.{param}"] = float(str(value))
    LOGGER.info(
        f"bias fiducials {[values[f'analysis.fiducial.{p}'] for p in bg_params]}, power law (bg, n_bg, z0) "
        f"{BG_FIDUCIAL_POWER_LAW}"
    )

    written = products.write_marked_values(conf, conf_path, values, args.check, rel=_rel)
    # with --provenance an up-to-date config is recorded too: these inputs at this commit give its values
    if args.check or not (written or args.provenance):
        return None
    return inputs, [conf_path]


def _configured_bins(conf, sample, per_survey_bin):
    """The values of the configured bins, in z_bins order, from the values of every survey bin."""
    return [float(per_survey_bin[files.get_z_bin_number(z) - 1]) for z in conf["survey"][sample]["z_bins"]]


def step_noise(conf, args):
    """Writes the shape-noise file in the layout files.load_noise_file reads, as notebooks/desy3/noise_file.ipynb did
    for Y3: per source bin, metacal<k>/cat holds (n_gal, 4) float32 with columns (e1 / R, e2 / R, w, pix).

    The ellipticities carry the signs and spin-2 rotation of the shear maps, and R is the response saved by step
    maps. Every survey bin is written, so a change of z_bins needs no new file.

    Returns:
        tuple: (input paths, output paths), or None with --dry-run or --max-objects
    """
    response_path = _maps_path(conf, "metacal_response")
    count_path = _maps_path(conf, "metacal_count")
    inputs = [catalog_desy6.release_path(conf, "metadetect"), response_path, count_path]
    out_path = _out_path(conf["files"]["noise"])
    if args.dry_run:
        return _dry_run("noise", inputs, [out_path])
    for path in (response_path, count_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{_rel(path)} does not exist, run step maps first")

    n_bins, n_pix = catalog_desy6.N_SOURCE_BINS, hp.nside2npix(conf["analysis"]["n_side"])
    response, count = np.load(response_path), np.load(count_path)
    assert response.shape == (n_bins,) and count.shape == (n_pix, n_bins), f"{response.shape}, {count.shape}"
    cats = (catalog_desy6.metacal_shapes(conf, i, response[i], max_objects=args.max_objects) for i in range(n_bins))

    def log(i, cat):
        rms = np.sqrt(np.average(cat[:, :2].astype(np.float64) ** 2, axis=0, weights=cat[:, 2]))
        LOGGER.info(
            f"source bin {i + 1}: N = {cat.shape[0]}, R = {response[i]:.4f}, weighted rms (e1, e2) / R = {rms}"
        )

    if args.max_objects is not None:
        for i, cat in enumerate(cats):
            log(i, cat)
        LOGGER.info(f"--max-objects {args.max_objects}: smoke test done, nothing is written")
        return None

    with h5py.File(out_path, "w") as f:
        f.attrs["info"] = (
            f"DES Y6 metadetection shapes per source bin from {catalog_desy6.RELEASE_FILES['metadetect']}. "
            "Written by msfm.apps.products.desy6 noise"
        )
        for i, cat in enumerate(cats):
            # the same galaxies in the same pixels as the count map of step maps
            pix_count = np.bincount(cat[:, 3].astype(np.int64), minlength=n_pix)
            assert np.array_equal(
                pix_count, count[:, i]
            ), f"source bin {i + 1}: the galaxies differ from the count map"
            log(i, cat)
            f.create_dataset(f"metacal{i + 1}/cat", data=cat).attrs["info"] = (
                "(n_gal, 4) with columns (e1 / R, e2 / R, w, pix): e1 and e2 rotated into the footprint frame with "
                f"signs (+1, -1), R = {response[i]}, pix the RING pixel at nside {conf['analysis']['n_side']}"
            )

    # the pipeline's loader reads the configured bins back
    for z_bin, cat in zip(conf["survey"]["metacal"]["z_bins"], files.load_noise_file(conf)):
        survey_bin = files.get_z_bin_number(z_bin)
        expected = (int(np.sum(count[:, survey_bin - 1])), 4)
        assert cat.shape == expected and cat.dtype == np.float32, f"metacal{survey_bin}: {cat.shape}, {cat.dtype}"
    return inputs, [out_path]


def step_systematics(conf, args):
    """Copies the source imaging-systematics maps of an lss_sys run, its nside-512 export (--source), to
    files.metacal_systematics, the file files.get_metacal_systematics reads. Y3's file was copied by hand. lss_sys
    keeps its output out of git, so this copy and its provenance record tie the forward model to the run.

    The export is checked before anything is written: the layout the loader expects, the mask of step pixels as its
    footprint, and the run label in the file name, since configuration.py matches that label against the bias table.
    It must also read through files.get_metacal_systematics, which asserts that the run covers the whole footprint.
    Last, the observed source counts of step maps must rise with the contamination C in every bin.

    Returns:
        tuple: (input paths, output paths), or None with --dry-run
    """
    source = os.path.abspath(args.source)
    pixel_path = _out_path(conf["files"]["pixels"])
    count_path = _maps_path(conf, "metacal_count")
    inputs = [source, pixel_path, count_path]
    out_path = _out_path(conf["files"]["metacal_systematics"])
    if args.dry_run:
        return _dry_run("systematics", inputs, [out_path])
    for path in inputs:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{_rel(path)} does not exist")

    n_side, n_bins = conf["analysis"]["n_side"], catalog_desy6.N_SOURCE_BINS
    n_pix = hp.nside2npix(n_side)
    # the loader reads bin{i} for the i-th configured bin
    survey_bins = [files.get_z_bin_number(z) for z in conf["survey"]["metacal"]["z_bins"]]
    assert survey_bins == list(range(1, n_bins + 1)), f"survey.metacal.z_bins must be all {n_bins} survey bins"

    with h5py.File(source, "r") as f:
        label = str(f.attrs["label"])
        assert str(f.attrs["ordering"]) == "RING", f"ordering {f.attrs['ordering']}"
        assert int(f.attrs["nside"]) == n_side, f"nside {f.attrs['nside']}, not analysis.n_side {n_side}"
        assert int(f.attrs["n_bins"]) == n_bins, f"{f.attrs['n_bins']} bins"
        assert str(f.attrs["off_footprint"]) == "zero", f"off_footprint {f.attrs['off_footprint']}"
        assert catalog_desy6.RELEASE_FILES["mask"] in str(f.attrs["footprint"]), f"footprint {f.attrs['footprint']}"
        assert set(f) == {f"bin{i}" for i in range(n_bins)}, f"groups {sorted(f)}"
        for i in range(n_bins):
            for dataset in ("contamination", "weight"):
                values = f[f"bin{i}/{dataset}"][:]
                assert values.shape == (n_pix,), f"bin{i}/{dataset}: shape {values.shape}"
                assert np.all(np.isfinite(values)) and np.all(values >= 0), f"bin{i}/{dataset}: negative or NaN"
        LOGGER.info(f"lss_sys run {label!r}, lss_sys {str(f.attrs['git_sha'])[:8]}")

    # <survey>_metacal_systematics_<run label token>_<nside>.h5, as Y3's desy3_metacal_systematics_STD32_512.h5
    expected = f"{conf['survey']['name'].lower()}_metacal_systematics_{label.rsplit('_', 1)[-1]}_{n_side}.h5"
    assert os.path.basename(out_path) == expected, f"files.metacal_systematics must be named {expected} for {label}"

    # the source itself goes through the loader, so a failed check writes nothing
    source_conf = copy.deepcopy(conf)
    source_conf["files"]["metacal_systematics"] = source
    contamination = files.get_metacal_systematics(source_conf)
    weight = files.get_metacal_systematics(source_conf, dataset="weight")

    # base-patch order, as the loader returns it; the count map is in the same footprint frame
    _, patches, _, _ = files.load_pixel_file(conf)
    count = np.load(count_path)
    assert count.shape == (n_pix, n_bins), f"{_rel(count_path)}: shape {count.shape}"
    count = count[patches["metacal"][0][0]].astype(np.float64)
    for i in range(n_bins):
        c, raw = contamination[:, i], count[:, i] / np.mean(count[:, i])
        corrected = count[:, i] * weight[:, i] / np.mean(count[:, i] * weight[:, i])
        r = np.corrcoef(c, raw)[0, 1]
        LOGGER.info(
            f"source bin {i + 1}: C in [{c.min():.3f}, {c.max():.3f}], std {c.std():.4f}; counts vs C: r = {r:.3f}, "
            f"slope {np.polyfit(c, raw, 1)[0]:.3f}, W-corrected {np.polyfit(c, corrected, 1)[0]:.3f}"
        )
        assert r > 0, f"source bin {i + 1}: the observed counts fall with the contamination (r = {r:.3f})"

    shutil.copyfile(source, out_path)
    assert products.sha256(out_path) == products.sha256(source), f"{_rel(out_path)}: the copy differs from {source}"
    return inputs, [out_path]


def step_sc_bias(conf, args):
    """Fits the source-clustering bias table files.metacal_bias, as notebooks/desy3/sc_bias_fit_count.ipynb did for
    Y3: per cosmology and source bin, the b whose Poisson-sampled counts on the projection's dg/<bin> at perm_0000
    match the one-point function of the observed counts of step maps. One HDF5 group per arm: clean, without imaging
    systematics, and contam, with the contamination of step systematics imprinted. files.metacal_bias_arm picks one.

    --source is the projection root, which holds grid/ and benchmarks/. The key "fiducial", which
    run_single_postprocessing reads for every benchmark, is fit on SC_BIAS_FIDUCIAL_DIR, since Y6 has no fiducial
    projection. The table is written to a temporary file and checked first: every fit finite and positive, and a
    config pointing at it passes files.read_metacal_bias and configuration.print_and_check_modeling_in_config, which
    matches each arm's systematics label against the forward model. --max-cosmologies N fits the first N grid
    cosmologies and the fiducial one, runs the same checks and writes nothing.

    Returns:
        tuple: (input paths, output paths), or None with --dry-run or --max-cosmologies
    """
    source = os.path.abspath(args.source)
    pixel_path = _out_path(conf["files"]["pixels"])
    count_path = _maps_path(conf, "metacal_count")
    sys_path = _out_path(conf["files"]["metacal_systematics"])
    meta_path = os.path.join(REPO_DIR, conf["files"]["meta_info"])
    inputs = [source, pixel_path, count_path, sys_path, meta_path]
    out_path = _out_path(conf["files"]["metacal_bias"])

    # every grid cosmology, or the first N, and the fiducial one last
    cosmo_dirs = scb.get_cosmogrid_dirs(conf, source, with_fiducial=False)
    if args.max_cosmologies is not None:
        cosmo_dirs = dict(list(cosmo_dirs.items())[: args.max_cosmologies])
    cosmo_dirs["fiducial"] = os.path.join(source, SC_BIAS_FIDUCIAL_DIR)
    with_bary = conf["analysis"]["modelling"]["baryonified"]
    missing = [
        key
        for key, cosmo_dir in cosmo_dirs.items()
        if not os.path.isfile(
            filenames.get_filename_full_maps(os.path.join(cosmo_dir, f"perm_{SC_BIAS_PERM:04d}"), with_bary=with_bary)
        )
    ]
    LOGGER.info(f"{len(cosmo_dirs) - len(missing)} of {len(cosmo_dirs)} cosmologies have perm_{SC_BIAS_PERM:04d}")
    if args.dry_run:
        return _dry_run("sc_bias", inputs, [out_path])
    if missing:
        raise FileNotFoundError(f"{len(missing)} cosmologies have no projection under {source}, e.g. {missing[:3]}")
    for path in inputs[1:]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{_rel(path)} does not exist")

    # the target, the same for both arms. The histogram has to reach the largest observed count, or it drops pixels
    _, count_map = catalog.build_metacal_map_from_cat(conf, debug=False)
    target, n_bar, mask = scb.build_target(conf, count_map=count_map)
    max_count = count_map[mask].max(axis=0)
    assert np.all(
        max_count <= len(target[0])
    ), f"the largest footprint counts {max_count} exceed shape_noise.bias_fit_max_count {len(target[0])}"
    LOGGER.info(f"largest footprint count per source bin {max_count}, histogram up to {len(target[0])}")

    with h5py.File(sys_path, "r") as f:
        sys_label = str(f.attrs["label"])
    # the fit pairs the contamination, in base-patch order, with the footprint pixels in ascending order
    contamination = files.get_metacal_systematics(conf)
    _, patches, _, _ = files.load_pixel_file(conf)
    assert np.array_equal(patches["metacal"][0][0], np.flatnonzero(mask)), "the base patch is not the sorted mask"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, os.path.basename(out_path))
        biases = {}
        for i, (arm, systematics) in enumerate(SC_BIAS_ARMS.items()):
            biases[arm] = scb.fit_bias_table(
                conf,
                cosmo_dirs,
                tmp_path,
                target=target,
                n_bar=n_bar,
                mask=mask,
                perm=SC_BIAS_PERM,
                contamination=contamination if systematics else None,
                systematics_label=sys_label if systematics else None,
                group=arm,
                mode="w" if i == 0 else "a",
            )
        _check_bias_table(conf, tmp_path, biases)

        if args.max_cosmologies is not None:
            LOGGER.info(f"--max-cosmologies {args.max_cosmologies}: the table passed every check, nothing is written")
            return None
        shutil.copyfile(tmp_path, out_path)
        assert products.sha256(out_path) == products.sha256(
            tmp_path
        ), f"{_rel(out_path)}: the copy differs from the checked table"
    return inputs, [out_path]


def _check_bias_table(conf, path, biases):
    """Every fit is finite and positive (0 is the lower bound of the fit), every loss finite, and each arm reads
    back unchanged through the loader and passes the checks the pipelines run on its config."""
    z_bins = conf["survey"]["metacal"]["z_bins"]
    table, loss = {}, {}
    with h5py.File(path, "r") as f:
        # the row order of the stacked losses
        keys = list(f["clean"]["keys"].asstr()[:])
        for arm in SC_BIAS_ARMS:
            assert list(f[arm]["keys"].asstr()[:]) == keys and set(biases[arm]) == set(keys), f"{arm}: keys differ"
            table[arm] = np.array([biases[arm][key] for key in keys])
            loss[arm] = f[arm]["loss"][:]
            assert np.all(np.isfinite(table[arm])) and np.all(table[arm] > 0), f"{arm}: a bias is not positive"
            assert np.all(np.isfinite(loss[arm])), f"{arm}: a loss is not finite"

    for arm, systematics in SC_BIAS_ARMS.items():
        check_conf = copy.deepcopy(conf)
        check_conf["files"]["metacal_bias"] = path
        check_conf["files"]["metacal_bias_arm"] = arm
        check_conf["analysis"]["modelling"]["lensing"]["shape_noise"]["survey_systematics"] = systematics
        for key in keys:
            read = files.read_metacal_bias(key, check_conf)
            assert np.array_equal(read, biases[arm][key]), f"{arm}/{key}: read back {read}, fit {biases[arm][key]}"
        configuration.print_and_check_modeling_in_config(check_conf)

    for i, z_bin in enumerate(z_bins):
        clean, contam = table["clean"][:, i], table["contam"][:, i]
        LOGGER.info(
            f"{z_bin}: b clean {np.median(clean):.3f} [{clean.min():.3f}, {clean.max():.3f}], "
            f"contam {np.median(contam):.3f} [{contam.min():.3f}, {contam.max():.3f}], "
            f"median contam / clean {np.median(contam / clean):.3f}, "
            f"loss clean max {loss['clean'][:, i].max():.3g}, contam max {loss['contam'][:, i].max():.3g}"
        )
    LOGGER.info(f"bias table of {len(keys)} cosmologies checked")


STEPS = {
    "nz": step_nz,
    "pixels": step_pixels,
    "maps": step_maps,
    "values": step_values,
    "noise": step_noise,
    "systematics": step_systematics,
    "sc_bias": step_sc_bias,
}
CATALOG_STEPS = {"maps", "noise"}
# the steps that read an input from outside the release, given as --source; a step with a locations key defaults to
# that path of the config's locations.yaml
SOURCE_STEPS = {"systematics", "sc_bias"}
SOURCE_LOCATIONS = {"sc_bias": "projection"}
# the code a step runs: changes elsewhere (data/desy6, notebooks) don't change a product
CODE_PATHS = ("msfm", "configs")


def _clean_commit():
    """Commit of this checkout. Refuses uncommitted changes in CODE_PATHS, so the commit is what runs."""
    return products.clean_commit(REPO_DIR, CODE_PATHS)


def _record_provenance(conf, conf_path, step, commit, inputs, outputs):
    """Replaces the record of `step` in files.provenance. One record per step; git keeps the history."""
    prov_path = _out_path(conf["files"]["provenance"])
    record = products.provenance_record(
        command="python -m msfm.apps.products.desy6 " + " ".join(sys.argv[1:]),
        config=_rel(conf_path),
        commits={"msfm_commit": commit},
        inputs=inputs,
        outputs=outputs,
        rel=_rel,
    )
    about = "Written by msfm.apps.products.desy6, one record per step: how each file in data/desy6/ was made."
    products.record_provenance(prov_path, step, record, STEP_ORDER, about)
    LOGGER.info(f"recorded step {step} in {os.path.relpath(prov_path, REPO_DIR)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=list(STEPS), help="the product to derive")
    parser.add_argument("--config", required=True, help="DES Y6 config, e.g. configs/desy6/v1/v1.yaml")
    parser.add_argument("--check", action="store_true", help="values only: write nothing, exit 1 if a value is stale")
    parser.add_argument("--dry-run", action="store_true", help="list the inputs and outputs, write nothing")
    parser.add_argument(
        "--max-objects",
        type=int,
        default=None,
        help="catalog steps: use the first N rows of each catalog, write nothing",
    )
    parser.add_argument(
        "--max-cosmologies",
        type=int,
        default=None,
        help="sc_bias only: fit the first N grid cosmologies and the fiducial one, write nothing",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="systematics: the nside-512 export of the lss_sys run; sc_bias: the projection root, which holds grid/ "
        "and benchmarks/ (default: the projection of the config's locations.yaml)",
    )
    parser.add_argument(
        "--provenance",
        action="store_true",
        help="for production runs: refuse uncommitted msfm/ or configs/, record the step in files.provenance",
    )
    args = parser.parse_args(argv)
    if args.check and args.step != "values":
        parser.error("--check only applies to the step values")
    if args.max_objects is not None and (args.step not in CATALOG_STEPS or args.max_objects < 1):
        parser.error(f"--max-objects takes a positive number and applies to {sorted(CATALOG_STEPS)}")
    if args.max_cosmologies is not None and (args.step != "sc_bias" or args.max_cosmologies < 1):
        parser.error("--max-cosmologies takes a positive number and applies to the step sc_bias")
    if args.source is not None and args.step not in SOURCE_STEPS:
        parser.error(f"--source applies to the steps {sorted(SOURCE_STEPS)} only")
    if args.source is None and args.step in SOURCE_LOCATIONS:
        args.source = locations.get(args.config, SOURCE_LOCATIONS[args.step])
        LOGGER.info(f"--source {args.source}, the {SOURCE_LOCATIONS[args.step]} of the config's locations.yaml")
    if args.source is None and args.step in SOURCE_STEPS:
        parser.error(f"the step {args.step} needs --source")

    conf = files.load_config(args.config)
    if conf["survey"]["name"] != SURVEY_NAME:
        raise ValueError(f"{args.config} is a {conf['survey']['name']} config, these products are {SURVEY_NAME}")
    configuration.check_survey_paths(conf)

    # checked before the step runs, so a dry run with --provenance catches uncommitted code too
    commit = _clean_commit() if args.provenance else None
    result = STEPS[args.step](conf, args)
    if result is not None:
        for path in result[1]:
            LOGGER.info(f"wrote {os.path.abspath(path)}")
        if args.provenance:
            _record_provenance(conf, args.config, args.step, commit, *result)


if __name__ == "__main__":
    main()
