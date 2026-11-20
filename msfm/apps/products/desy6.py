# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created November 2026
Author: Arne Thomsen

Derives the DES Y6 data products in data/desy6/ and the data-derived config values from the public Y6 release,
one step per product (data/desy6/README.md). Every step reads its inputs from dirs.catalog,
writes only to the paths its config keys name (under data/desy6/) and stops on a failed check. With --provenance, it
requires committed code and records its inputs, outputs and the msfm commit in files.provenance.

Steps, in order:
    nz       n(z) text files of all 4 source and 6 lens bins, from the 3x2pt FITS
    pixels   pixel file: the rotated nside-512 footprint, its four patches and the padded data vector, from the mask,
             and the MagLim weight maps that correct the observed lens counts, from the catalog
    maps     shear, count and weighted-count maps of every source and lens bin, and the source response, in dirs.cache
    values   n_gal, response, m prior and Aeff, per bin label, in files.survey_values
    config   writes files.survey_values into the config lines marked "# <- desy6 values: <dotted.key>"
    noise    shape-noise file: (e1 / R, e2 / R, w, pix) per galaxy of every source bin
    sc_bias  source-clustering bias table                                          (not implemented yet)

Usage:
    python -m msfm.apps.products.desy6 nz --config configs/desy6/v1/default.yaml
    python -m msfm.apps.products.desy6 pixels --config configs/desy6/v1/default.yaml   # compute node, needs healsparse
    python -m msfm.apps.products.desy6 maps --config configs/desy6/v1/default.yaml --max-objects 100000   # smoke test
    python -m msfm.apps.products.desy6 maps --config configs/desy6/v1/default.yaml   # compute node, ~140M objects
    python -m msfm.apps.products.desy6 values --config configs/desy6/v1/default.yaml
    python -m msfm.apps.products.desy6 config --config configs/desy6/v1/default.yaml
    python -m msfm.apps.products.desy6 config --config configs/desy6/v1/default.yaml --check   # exit 1 if stale
    python -m msfm.apps.products.desy6 noise --config configs/desy6/v1/default.yaml   # compute node, ~2 GB out

--dry-run lists a step's inputs and outputs and writes nothing. --max-objects N runs a catalog step on the first N
rows of every catalog and writes nothing, so a partial map can never pass for the product.
"""

import argparse
import copy
import datetime
import hashlib
import os
import re
import subprocess
import sys

import h5py
import healpy as hp
import numpy as np
import yaml

from msfm.utils import catalog, catalog_desy6, files, filenames, logger

LOGGER = logger.get_logger(__file__)

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SURVEY_NAME = "DESy6"
FILE_PREFIX = "desy6"
OUT_ROOT = "data/desy6"
CONFIG_ROOT = "configs/desy6"
STEP_ORDER = ["nz", "pixels", "maps", "values", "config", "noise", "sc_bias"]

# a config line whose value the step config writes: "<key>: <value> # <- desy6 values: <dotted.key>"
VALUE_MARKER = re.compile(
    r"^(?P<head>\s*(?P<key>[A-Za-z_]\w*):\s*)(?P<value>[^#]*?)(?P<gap>\s*)"
    r"(?P<marker>#\s*<-\s*desy6 values:\s*(?P<dotted>[A-Za-z_][\w.]*)\s*)$"
)

# config sample key -> (n(z) HDU of the 3x2pt FITS, number of survey bins it must hold)
NZ_HDUS = {"metacal": ("nz_source", catalog_desy6.N_SOURCE_BINS), "maglim": ("nz_lens", catalog_desy6.N_LENS_BINS)}

# step pixels: the mask is rotated at this resolution, the one of the Y3 mask (notebooks/desy3/pixel_file.ipynb)
MASK_ROTATE_NSIDE = 4096
# patches 1-3 are the base patch rotated by 180 deg in phi and/or mirrored at the equator; mirroring flips gamma_2
PATCH_GAMMA_2_SIGN = [1, 1, -1, -1]

# step maps: cache file -> what it holds; every map covers all survey bins, axis 1 in survey-bin order
MAPS_FILES = {
    "metacal_gamma": f"{FILE_PREFIX}_metacal_gamma_e1p_e2m.npy",
    "metacal_count": f"{FILE_PREFIX}_metacal_count.npy",
    "metacal_response": f"{FILE_PREFIX}_metacal_response.npy",
    "maglim_count": f"{FILE_PREFIX}_maglim_count.npy",
    "maglim_weighted_count": f"{FILE_PREFIX}_maglim_weighted_count.npy",
}


def _out_path(rel_path):
    """Absolute path of a config-named output, which must lie under data/desy6/."""
    rel_path = os.path.normpath(rel_path)
    if os.path.isabs(rel_path) or not (rel_path + os.sep).startswith(OUT_ROOT + os.sep):
        raise ValueError(f"DES Y6 outputs must lie under {OUT_ROOT}/, got {rel_path}")
    return os.path.join(REPO_DIR, rel_path)


def _dry_run(step, inputs, outputs):
    """Lists what a step would read and write. Returns None, so nothing is recorded."""
    LOGGER.info(f"dry run of step {step}, nothing is written")
    for path in inputs:
        LOGGER.info(f"  reads  {_rel(path)} {'(exists)' if os.path.exists(path) else '(MISSING)'}")
    for path in outputs:
        LOGGER.info(f"  writes {_rel(path)} {'(exists, would be replaced)' if os.path.exists(path) else ''}")
    return None


def _survey_bins(conf, sample):
    """Survey bin numbers of the configured bins, from labels like maglim3."""
    bins = []
    for label in conf["survey"][sample]["z_bins"]:
        match = re.fullmatch(rf"{sample}(\d+)", label)
        if match is None:
            raise ValueError(f"survey.{sample}.z_bins label {label} does not end in a survey bin number")
        bins.append(int(match.group(1)))
    return bins


def _rel(path):
    """Repository-relative path for files inside the repository, absolute otherwise."""
    rel = os.path.relpath(os.path.abspath(path), REPO_DIR)
    return path if rel.startswith("..") else rel


def step_nz(conf, args):
    """Writes the fiducial n(z) of every source and lens bin as two columns (z_mid, n(z)), as for Y3.

    Returns:
        tuple: (input paths, output paths)
    """
    from astropy.io import fits

    fits_path = catalog_desy6.release_path(conf, "3x2pt")
    out_dir = _out_path(conf["dirs"]["redshift_distributions"])
    if args.dry_run:
        outputs = [
            filenames.get_filename_z_distribution(out_dir, sample, i_bin, prefix=FILE_PREFIX)
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
            missing = set(_survey_bins(conf, sample)) - set(range(1, n_bins + 1))
            assert not missing, f"survey.{sample}.z_bins names survey bins {missing} that {hdu_name} does not have"

            for i_bin in range(1, n_bins + 1):
                nz = np.asarray(table[f"BIN{i_bin}"], dtype=np.float64)
                assert np.all(np.isfinite(nz)) and np.all(nz >= 0), f"{hdu_name} BIN{i_bin}: negative or not finite"
                norm = np.sum(nz * (z_high - z_low))
                assert abs(norm - 1) < 1e-3, f"{hdu_name} BIN{i_bin}: integrates to {norm}"

                path = filenames.get_filename_z_distribution(out_dir, sample, i_bin, prefix=FILE_PREFIX)
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
        missing = set(_survey_bins(conf, sample)) - set(range(1, n_bins + 1))
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
    used_weights = lens_weights[:, np.array(_survey_bins(conf, "maglim")) - 1].astype(np.float32)
    sys_map = files.get_clustering_systematics(conf, pixel_type="map")
    sys_dv = files.get_clustering_systematics(conf, pixel_type="data_vector")
    assert np.array_equal(sys_map, used_weights), "maglim systematics map read back"
    assert np.array_equal(sys_dv[patch_to_data_vec], used_weights), "maglim systematics data vector read back"
    assert np.count_nonzero(sys_dv) == used_weights.size, "maglim systematics data vector: padding not 0"

    return inputs, [out_path]


def step_maps(conf, args):
    """Builds the shear, count and weighted-count maps of every source and lens bin, and saves the source response
    that calibrated the shear maps, so that step values takes R from the same numbers.

    Returns:
        tuple: (input paths, output paths), or None with --dry-run or --max-objects
    """
    inputs = [catalog_desy6.release_path(conf, key) for key in ("metadetect", "maglim")]
    cache_dir = _out_path(conf["dirs"]["cache"])
    paths = {key: os.path.join(cache_dir, name) for key, name in MAPS_FILES.items()}
    if args.dry_run:
        return _dry_run("maps", inputs, list(paths.values()))

    gamma, source_count, response = catalog_desy6.build_metacal_maps(conf, max_objects=args.max_objects)
    lens_count, lens_weighted_count = catalog_desy6.build_maglim_maps(conf, max_objects=args.max_objects)
    maps = {
        "metacal_gamma": gamma,
        "metacal_count": source_count,
        "metacal_response": response,
        "maglim_count": lens_count,
        "maglim_weighted_count": lens_weighted_count,
    }
    for key, values in maps.items():
        LOGGER.info(f"{MAPS_FILES[key]}: shape {values.shape}, dtype {values.dtype}")

    if args.max_objects is not None:
        LOGGER.info(f"--max-objects {args.max_objects}: smoke test done, nothing is written")
        return None

    os.makedirs(cache_dir, exist_ok=True)
    for key, values in maps.items():
        np.save(paths[key], values)
    return inputs, list(paths.values())


def step_values(conf, args):
    """Writes files.survey_values, the data-derived config values, with per-bin values keyed by survey bin label.

    n_gal is the count inside the footprint (patch 0 of the pixel file) over the footprint's area, in deg^-2, as
    notebooks/desy3/config.ipynb did for Y3: raw counts for the sources, weighted counts for MagLim, whose per-galaxy
    weights correct the survey systematics. The response is the one that calibrated the shear maps, the m prior is the
    mean and std of the M_BIN samples, and Aeff is the full-resolution mask area stored in the pixel file.

    Returns:
        tuple: (input paths, output paths), or None with --dry-run
    """
    pixel_path = _out_path(conf["files"]["pixels"])
    cache_dir = _out_path(conf["dirs"]["cache"])
    maps = {key: os.path.join(cache_dir, MAPS_FILES[key]) for key in ("metacal_count", "maglim_weighted_count")}
    response_path = os.path.join(cache_dir, MAPS_FILES["metacal_response"])
    m_path = catalog_desy6.release_path(conf, "source_redshift")
    out_path = _out_path(conf["files"]["survey_values"])
    inputs = [pixel_path, *maps.values(), response_path, m_path]
    if args.dry_run:
        return _dry_run("values", inputs, [out_path])
    for path in inputs:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{_rel(path)} does not exist, run the steps pixels and maps first")

    n_side = conf["analysis"]["n_side"]
    footprint_conf = conf["analysis"]["footprint"]
    with h5py.File(pixel_path, "r") as f:
        made_with = (f.attrs["rotation_y_rad"], f.attrs["rotation_z_rad"], f.attrs["coverage_threshold"])
        expected = (footprint_conf["rotation"]["y_rad"], footprint_conf["rotation"]["z_rad"])
        assert made_with == (*expected, footprint_conf["coverage_threshold"]), f"stale pixel file: {made_with}"
        footprint = f["footprint/patches"][0]
        coverage = f["footprint/coverage"][:].astype(np.float64)
        area = float(f.attrs["area_deg2"])
    footprint_area = footprint.size * hp.nside2pixarea(n_side, degrees=True)
    # the galaxies follow the mask, so the footprint should hold about the same fraction of both
    mask_fraction = np.sum(coverage[footprint]) / np.sum(coverage)

    survey = {"Aeff": area, "metacal": {}, "maglim": {}}
    for sample, key, n_bins in (
        ("metacal", "metacal_count", catalog_desy6.N_SOURCE_BINS),
        ("maglim", "maglim_weighted_count", catalog_desy6.N_LENS_BINS),
    ):
        count = np.load(maps[key])
        assert count.shape == (hp.nside2npix(n_side), n_bins), f"{_rel(maps[key])}: shape {count.shape}"
        inside = np.sum(count[footprint], axis=0, dtype=np.float64)
        fraction = inside / np.sum(count, axis=0, dtype=np.float64)
        assert np.all(
            np.abs(fraction - mask_fraction) < 0.02
        ), f"{sample}: {fraction} of the galaxies lie in the footprint, but {mask_fraction:.4f} of the mask"
        n_gal = inside / footprint_area
        survey[sample]["n_gal"] = {f"{sample}{i + 1}": float(n_gal[i]) for i in range(n_bins)}
        LOGGER.info(
            f"{sample} ({key}): n_gal = {n_gal / 60**2} arcmin^-2, in the footprint {fraction} "
            f"(mask {mask_fraction:.4f})"
        )

    response = np.load(response_path)
    assert response.shape == (catalog_desy6.N_SOURCE_BINS,) and np.all(response > 0), f"response {response}"
    survey["metacal"]["response"] = {f"metacal{i + 1}": float(r) for i, r in enumerate(response)}

    mu, sigma = catalog_desy6.m_prior(conf)
    survey["metacal"]["shear_bias"] = {
        "multiplicative": {
            "mu": {f"metacal{i + 1}": float(x) for i, x in enumerate(mu)},
            "sigma": {f"metacal{i + 1}": float(x) for i, x in enumerate(sigma)},
        }
    }
    LOGGER.info(f"Aeff = {area:.3f} deg^2, footprint {footprint_area:.1f} deg^2, response = {response}")

    values = {
        "about": (
            "Written by msfm.apps.products.desy6 values; step config copies these into the config lines marked "
            "'# <- desy6 values: <dotted.key>'. Per-bin values are keyed by survey bin label."
        ),
        "survey": survey,
    }
    with open(out_path, "w") as f:
        yaml.safe_dump(values, f, sort_keys=False, width=119)
    with open(out_path) as f:
        assert yaml.safe_load(f) == values, f"{_rel(out_path)}: the YAML round trip is not exact"
    return inputs, [out_path]


def _dotted_get(tree, dotted):
    node = tree
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted)
        node = node[part]
    return node


def _dotted_set(tree, dotted, value):
    *parents, leaf = dotted.split(".")
    for part in parents:
        tree = tree[part]
    tree[leaf] = value


def _number(value, where):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{where}: {value!r} is not a finite number")
    return float(value)


def _marked_value(conf, values, dotted):
    """The value of a marked config key: a scalar as it is, a per-bin mapping as a list in z_bins order."""
    try:
        value = _dotted_get(values, dotted)
    except KeyError:
        raise KeyError(f"files.survey_values has no value for the marked config key {dotted}") from None

    if not isinstance(value, dict):
        return _number(value, dotted)

    # per-bin values are keyed by bin label, so the config's z_bins pick the bins and their order
    parts = dotted.split(".")
    if len(parts) < 3 or parts[0] != "survey" or "z_bins" not in conf["survey"].get(parts[1], {}):
        raise ValueError(f"{dotted}: per-bin values must sit under survey.<sample>.")
    labels = conf["survey"][parts[1]]["z_bins"]
    missing = [label for label in labels if label not in value]
    if missing:
        raise KeyError(f"files.survey_values {dotted} has no value for the bins {missing}")
    return [_number(value[label], f"{dotted}.{label}") for label in labels]


def _yaml_number(x):
    # PyYAML only parses a float with a dot in the mantissa: "1e-05" would be read as a string
    text = repr(x)
    mantissa, _, exponent = text.partition("e")
    return f"{mantissa}.0e{exponent}" if exponent and "." not in mantissa else text


def _yaml_value(value):
    if isinstance(value, list):
        return "[" + ", ".join(_yaml_number(x) for x in value) + "]"
    return _yaml_number(value)


def step_config(conf, args):
    """Writes the values of files.survey_values into the marked lines of the config, and nothing else.

    With --check, writes nothing and exits with 1 if any marked value differs from files.survey_values.

    Returns:
        tuple: (input paths, output paths), or None with --check or when every value is up to date
    """
    conf_path = os.path.abspath(args.config)
    rel_conf = os.path.relpath(conf_path, REPO_DIR)
    if not (rel_conf + os.sep).startswith(CONFIG_ROOT + os.sep):
        raise ValueError(f"step config only writes configs under {CONFIG_ROOT}/, got {rel_conf}")
    values_path = _out_path(conf["files"]["survey_values"])
    if not os.path.isfile(values_path):
        raise FileNotFoundError(f"{_rel(values_path)} (files.survey_values) does not exist, run step values first")
    with open(values_path) as f:
        values = yaml.safe_load(f)
    with open(conf_path) as f:
        lines = f.read().splitlines(keepends=True)

    new_lines, expected, stale = [], {}, []
    for i_line, line in enumerate(lines, start=1):
        body = line.rstrip("\r\n")
        match = VALUE_MARKER.match(body)
        if match is None:
            # comment-only lines (the header explains the marker) are no markers
            if "desy6 values:" in body and not body.lstrip().startswith("#"):
                raise ValueError(f"{rel_conf}:{i_line}: malformed value marker: {body.strip()}")
            new_lines.append(line)
            continue

        dotted = match["dotted"]
        if dotted.split(".")[-1] != match["key"]:
            raise ValueError(f"{rel_conf}:{i_line}: the marker {dotted} sits on the key {match['key']}")
        if dotted in expected:
            raise ValueError(f"{rel_conf}:{i_line}: {dotted} is marked twice")
        expected[dotted] = _marked_value(conf, values, dotted)

        new_body = match["head"] + _yaml_value(expected[dotted]) + (match["gap"] or " ") + match["marker"]
        if new_body != body:
            stale.append(dotted)
        new_lines.append(new_body + line[len(body) :])
    if not expected:
        raise ValueError(f"{rel_conf} has no lines marked '# <- desy6 values: <dotted.key>'")

    # the result parses, each marker sat on the line of its key, and no other value changed
    new_text = "".join(new_lines)
    new_conf = yaml.safe_load(new_text)
    reference = copy.deepcopy(conf)
    for dotted, value in expected.items():
        if _dotted_get(new_conf, dotted) != value:
            raise ValueError(f"{dotted}: the rewritten config reads {_dotted_get(new_conf, dotted)!r}, not {value!r}")
        _dotted_set(reference, dotted, value)
    assert new_conf == reference, "the rewrite changed a config value that carries no marker"

    if args.check:
        for dotted in stale:
            LOGGER.error(
                f"{rel_conf}: {dotted} = {_dotted_get(conf, dotted)!r}, files.survey_values: {expected[dotted]!r}"
            )
        if stale:
            raise SystemExit(1)
        LOGGER.info(f"{rel_conf}: all {len(expected)} marked values agree with {_rel(values_path)}")
        return None

    if not stale:
        # nothing written, so the provenance record of the last write stays
        LOGGER.info(f"{rel_conf}: all {len(expected)} marked values are up to date")
        return None
    with open(conf_path, "w") as f:
        f.write(new_text)
    LOGGER.info(f"{rel_conf}: wrote {len(stale)} of {len(expected)} marked values: {stale}")
    return [values_path], [conf_path]


def step_noise(conf, args):
    """Writes the shape-noise file in the layout files.load_noise_file reads, as notebooks/desy3/noise_file.ipynb did
    for Y3: per source bin, metacal<k>/cat holds (n_gal, 4) float32 with columns (e1 / R, e2 / R, w, pix).

    The ellipticities carry the signs and spin-2 rotation of the shear maps, and R is the response saved by step
    maps. Every survey bin is written, so a change of z_bins needs no new file.

    Returns:
        tuple: (input paths, output paths), or None with --dry-run or --max-objects
    """
    cache_dir = _out_path(conf["dirs"]["cache"])
    response_path = os.path.join(cache_dir, MAPS_FILES["metacal_response"])
    count_path = os.path.join(cache_dir, MAPS_FILES["metacal_count"])
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
    for survey_bin, cat in zip(_survey_bins(conf, "metacal"), files.load_noise_file(conf)):
        expected = (int(np.sum(count[:, survey_bin - 1])), 4)
        assert cat.shape == expected and cat.dtype == np.float32, f"metacal{survey_bin}: {cat.shape}, {cat.dtype}"
    return inputs, [out_path]


STEPS = {
    "nz": step_nz,
    "pixels": step_pixels,
    "maps": step_maps,
    "values": step_values,
    "config": step_config,
    "noise": step_noise,
}
CATALOG_STEPS = {"maps", "noise"}
# the code a step runs: changes elsewhere (data/desy6, notebooks) don't change a product
CODE_PATHS = ("msfm", "configs")


def _git(*cmd):
    return subprocess.run(["git", "-C", REPO_DIR, *cmd], capture_output=True, text=True, check=True).stdout.strip()


def _clean_commit():
    """Commit of this checkout. Refuses uncommitted changes in CODE_PATHS, so the commit is what runs."""
    changes = _git("status", "--porcelain", "--", *CODE_PATHS)
    if changes:
        raise SystemExit(f"commit {' and '.join(CODE_PATHS)} before a step writes products:\n{changes}")
    return _git("rev-parse", "HEAD")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mtime(path):
    return datetime.datetime.fromtimestamp(os.path.getmtime(path)).astimezone().isoformat(timespec="seconds")


def _record_provenance(conf, conf_path, step, commit, inputs, outputs):
    """Replaces the record of `step` in files.provenance. One record per step; git keeps the history."""
    prov_path = _out_path(conf["files"]["provenance"])
    steps = {}
    if os.path.isfile(prov_path):
        with open(prov_path) as f:
            steps = (yaml.safe_load(f) or {}).get("steps", {})

    steps[step] = {
        "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": "python -m msfm.apps.products.desy6 " + " ".join(sys.argv[1:]),
        "config": _rel(conf_path),
        "msfm_commit": commit,
        "inputs": [{"path": _rel(p), "bytes": os.path.getsize(p), "mtime": _mtime(p)} for p in inputs],
        "outputs": [{"path": _rel(p), "sha256": _sha256(p)} for p in outputs],
    }

    prov = {
        "about": "Written by msfm.apps.products.desy6, one record per step: how each file in data/desy6/ was made.",
        "steps": {name: steps[name] for name in [*STEP_ORDER, *steps] if name in steps},
    }
    with open(prov_path, "w") as f:
        yaml.safe_dump(prov, f, sort_keys=False, width=119)
    LOGGER.info(f"recorded step {step} in {os.path.relpath(prov_path, REPO_DIR)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=list(STEPS), help="the product to derive")
    parser.add_argument("--config", required=True, help="DES Y6 config, e.g. configs/desy6/v1/default.yaml")
    parser.add_argument("--check", action="store_true", help="config only: write nothing, exit 1 if a value is stale")
    parser.add_argument("--dry-run", action="store_true", help="list the inputs and outputs, write nothing")
    parser.add_argument(
        "--max-objects",
        type=int,
        default=None,
        help="catalog steps: use the first N rows of each catalog, write nothing",
    )
    parser.add_argument(
        "--provenance",
        action="store_true",
        help="for production runs: refuse uncommitted msfm/ or configs/, record the step in files.provenance",
    )
    args = parser.parse_args(argv)
    if args.check and args.step != "config":
        parser.error("--check only applies to the step config")
    if args.dry_run and args.step == "config":
        parser.error("the dry run of the step config is --check")
    if args.max_objects is not None and (args.step not in CATALOG_STEPS or args.max_objects < 1):
        parser.error(f"--max-objects takes a positive number and applies to {sorted(CATALOG_STEPS)}")

    conf = files.load_config(args.config)
    if conf["survey"]["name"] != SURVEY_NAME:
        raise ValueError(f"{args.config} is a {conf['survey']['name']} config, these products are {SURVEY_NAME}")

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
