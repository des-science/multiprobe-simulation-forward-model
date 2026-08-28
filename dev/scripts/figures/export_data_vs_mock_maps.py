# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created August 2026
Author: Arne Thomsen

Exports everything the "DES Y3 next to a CosmoGrid mock" figure needs into a single, self contained
.h5 file: the real DES Y3 observation and one realization of a benchmark fiducial mock, forward
modelled through the *same* function, smoothed with the scale cuts of the summary network and
projected onto the sky over the whole survey footprint.

The two sides are comparable by construction. The mock in data/<version>/<release>/obs/*.h5 was
written by msfm.apps.run_single_postprocessing with

    observation.forward_model_observation_map(..., apply_norm=False, with_padding=True, nest_in=False)

and the DES side of this script calls that function with exactly the same arguments, on the maps
that deep_lss.utils.evaluation.evaluate_obs_des feeds the network. So both are kappa (bin 1-4)
followed by the maglim galaxy counts (bin 1-4), on the same padded data vector, with the ell < 30
cut applied and nothing else.

NOTE the clustering channel is a galaxy *count* per pixel, not an overdensity: map_to_data_vec is
called without divide_by_mean on this path, and the tfrecords carry the same convention
(run_grid_postprocessing stores what galaxy_density_to_count returns). The ell < 30 cut takes out
most but not all of the monopole, which is why its footprint mean is a small positive number
rather than zero. Do not label it delta_g.

The smoothing on top is the network's, taken from a y3-deep-lss scales config: a Gaussian of the
per bin theta_fwhm, applied by msfm's own scales.data_vector_to_smoothed_data_vector so that the
figure shows what the network sees rather than a second implementation of it. The white noise that
HealpySmoothing adds afterwards is deliberately left out -- it is a regularizer, not part of the
data.

All eight channels are exported, so choosing a different tomographic bin is a plot-side edit. A
different smoothing scale is not: rerun this with another --scales_config.

Two projections come out of it: the whole footprint in celestial coordinates, and a square
gnomonic zoom of the same smoothed maps for the inset panel, whose border is traced back into the
footprint plane so the main panel can mark it.

Only the smoothed maps are projected. At the ~5 arcmin per pixel a 5000 deg^2 panel can hold, an
unsmoothed map is noise; the sparse healpix maps under maps/ carry both versions at full
resolution for anything the images cannot answer.

Runs on numpy/healpy/h5py only, no tensorflow and no GPU, but it does ~40 spherical harmonic
transforms at nside 512, so give it a compute node:

    srun -A a0158 --partition=debug --time=00:30:00 \
        uenv run --view=default pytorch/v2.9.1:v2 -- \
        ~/dlss/torch_env/bin/python3 dev/scripts/figures/export_data_vs_mock_maps.py
"""

import argparse
import os
import subprocess

import h5py
import numpy as np
import yaml

from msfm.utils import catalog, files, imports, observation, parameters, scales

# not parallel=True: that pins OMP_NUM_THREADS to every visible core, which a login node refuses
# to hand out. Set OMP_NUM_THREADS yourself; the transforms scale well up to a few dozen threads.
hp = imports.import_healpy()

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

# healpy marks unseen pixels with this value, and the projectors fill the area outside of the
# sphere with it as well
UNSEEN_THRESHOLD = -1e30


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # inputs
    parser.add_argument(
        "--msfm_config",
        default=os.path.join(REPO_DIR, "configs/v18/default.yaml"),
        help="msfm forward model config. Must be the one --mock_file was produced with",
    )
    parser.add_argument(
        "--scales_config",
        default="/users/athomsen/dlss/repos/y3-deep-lss/configs/scales/8wl,32gc.yaml",
        help="y3-deep-lss scale cut config, the source of the per bin smoothing kernels",
    )
    parser.add_argument(
        "--mock_file",
        # NOTE the absolute /iopsstor path, the $HOME/scratch symlink does not resolve inside a container
        default="/iopsstor/scratch/cscs/athomsen/deep_lss/data/v18/default/obs/fiducial_bench_obs_maps.h5",
        help="benchmark observation file, obs/maps holds its realizations",
    )
    parser.add_argument("--i_realization", type=int, default=0, help="which realization of --mock_file to export")
    # full footprint panel, in celestial coordinates
    parser.add_argument(
        "--footprint_reso", type=float, default=5.0, help="footprint panel resolution in arcmin per pixel"
    )
    parser.add_argument("--footprint_margin", type=float, default=2.0, help="blank margin around the footprint in deg")
    parser.add_argument("--graticule_step", type=float, default=20.0, help="spacing of the RA/Dec grid lines in deg")
    # square zoom inset, gnomonic, in the rotated frame the maps live in
    parser.add_argument("--zoom_lon", type=float, default=90.0, help="zoom centre longitude in deg, rotated frame")
    parser.add_argument("--zoom_lat", type=float, default=-30.0, help="zoom centre latitude in deg, rotated frame")
    parser.add_argument("--zoom_reso", type=float, default=1.0, help="zoom resolution in arcmin per pixel")
    parser.add_argument("--zoom_size", type=int, default=600, help="zoom size in pixels; size x reso sets its degrees")
    parser.add_argument(
        "--min_coverage",
        type=float,
        default=0.8,
        help="fail unless at least this fraction of the zoom is inside the footprint",
    )
    # output
    parser.add_argument(
        "--output",
        default=None,
        help="path of the .h5 file to write. Defaults to the paper_2_plotting cache, named after "
        "the two configs and the realization",
    )
    return parser.parse_args()


def default_output(args, msfm_conf_path, scales_conf_path):
    """paper_2_plotting/cache/<msfm version>_<scales>_<realization>.h5.

    The name carries everything that changes the numbers, because nothing downstream checks that a
    cache file was produced by the settings that found it.
    """
    version = os.path.basename(os.path.dirname(msfm_conf_path))
    release = os.path.splitext(os.path.basename(msfm_conf_path))[0]
    scales_name = os.path.splitext(os.path.basename(scales_conf_path))[0]
    mock = os.path.basename(args.mock_file).replace("_obs_maps.h5", "")
    name = f"data_vs_mock_maps_{version}-{release}_{scales_name}_{mock}-{args.i_realization}.h5"
    return os.path.abspath(os.path.join(REPO_DIR, "../deep_lss_paper/paper_2_plotting/cache", name))


def git_hash(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def load_desy3(msfm_conf):
    """The real DES Y3 observation as a padded data vector, in physical units.

    Mirrors deep_lss.utils.evaluation.evaluate_obs_des for the plain "DESy3" label: shear rotated
    into the frame of the rotated footprint, and the maglim counts corrected by the DES Y3 weight
    map. The catalog maps come from data/cache/, which is why this needs no access to the catalogs.
    """
    wl_gamma_map = catalog.build_metacal_map_from_cat(msfm_conf)[0]
    gc_count_map = catalog.build_maglim_map_from_cat(msfm_conf)

    data_vec, _, data_vec_pix = observation.forward_model_observation_map(
        wl_gamma_map=wl_gamma_map,
        gc_count_map=gc_count_map,
        conf=msfm_conf,
        apply_norm=False,
        with_padding=True,
        nest_in=False,
        apply_maglim_sys_map=True,
    )
    return np.asarray(data_vec, dtype=np.float32), np.asarray(data_vec_pix)


def load_mock(mock_file, i_realization, data_vec_len, n_channels):
    """One realization of a benchmark observation file, already in the same padded data vector."""
    with h5py.File(mock_file, "r") as f:
        n_realizations = f["obs/maps"].shape[0]
        assert 0 <= i_realization < n_realizations, f"--i_realization must be in [0, {n_realizations})"
        data_vec = f["obs/maps"][i_realization].astype(np.float32)

    assert data_vec.shape == (data_vec_len, n_channels), (
        f"{mock_file} holds maps of shape {data_vec.shape}, expected {(data_vec_len, n_channels)}. "
        "--mock_file and --msfm_config have to be the same forward model version"
    )
    return data_vec


def smooth(data_vec, data_vec_pix, msfm_conf, scales_conf, dv_masks):
    """Gaussian smooths a padded data vector with the network's per bin kernels.

    msfm's own smoothing, so this is the same code path the .tfrecords and the observations went
    through -- only the kernel widths come from the y3-deep-lss scales config instead of the msfm
    one, which leaves theta_fwhm null because the network smooths on the fly.

    l_min is deliberately left out and hard_cut off: the ell < 30 cut is already baked into both
    data vectors, and a second pass with the config's hard_cut=True would turn theta_fwhm into a
    top hat in ell (see scales.gaussian_low_pass_factor_alm) rather than the real space Gaussian
    that deepsphere's HealpySmoothing actually applies.
    """
    n_side = msfm_conf["analysis"]["n_side"]

    out = np.zeros_like(data_vec)
    i = 0
    for probe, sample in (("lensing", "metacal"), ("clustering", "maglim")):
        scale_conf = scales_conf["scale_cuts"][probe]
        n_z = len(scale_conf["theta_fwhm"])
        smoothed, _ = scales.data_vector_to_smoothed_data_vector(
            data_vec[:, i : i + n_z].copy(),
            data_vec_pix=data_vec_pix,
            n_side=n_side,
            l_min=None,
            theta_fwhm=list(scale_conf["theta_fwhm"]),
            arcmin=scales_conf["scale_cuts"]["arcmin"],
            mask=dv_masks[sample],
            hard_cut=False,
            conf=msfm_conf,
        )
        out[:, i : i + n_z] = smoothed
        i += n_z

    assert i == data_vec.shape[-1], f"the scales config covers {i} of {data_vec.shape[-1]} channels"
    return out


# --- the square zoom inset ---------------------------------------------------------------------
# Gnomonic, so a small field is undistorted, and cut in the rotated frame the maps live in. Its
# border is traced back into the footprint projection so the main panel can mark where it came from.


def project_zoom(values, patch_pix, n_side, zoom_proj):
    """Gnomonic image stack of footprint values, NaN outside the footprint and outside the sphere.

    Args:
        values (n_patch_pix, n_channels): map values on patch_pix, in the rotated frame.
        patch_pix (n_patch_pix,): footprint pixel indices in NEST ordering.

    Returns:
        (n_channels, ny, nx) float32
    """
    n_pix = hp.nside2npix(n_side)

    def vec2pix(x, y, z):
        return hp.vec2pix(n_side, x, y, z, nest=True)

    images = []
    for i in range(values.shape[-1]):
        full = np.full(n_pix, np.nan, dtype=np.float64)
        full[patch_pix] = values[:, i]
        img = np.asarray(zoom_proj.projmap(full, vec2pix), dtype=np.float32)
        # healpy fills everything it considers unseen with UNSEEN, we want NaN throughout
        images.append(np.where(img < UNSEEN_THRESHOLD, np.nan, img))

    return np.stack(images)


def zoom_outline(zoom_proj, foot_proj, rot_matrix, n_per_edge=200):
    """The zoom's border, as a closed polygon in the footprint projection plane.

    This is the square drawn on the main panel. The zoom is defined in the rotated frame, so its
    boundary is un-rotated before being projected the way the footprint images were.

    Returns:
        (4 * n_per_edge + 1, 2)
    """
    x_min, x_max, y_min, y_max = zoom_proj.get_extent()
    xs = np.linspace(x_min, x_max, n_per_edge)
    ys = np.linspace(y_min, y_max, n_per_edge)
    edges = [
        np.stack([xs, np.full(n_per_edge, y_min)], axis=-1),
        np.stack([np.full(n_per_edge, x_max), ys], axis=-1),
        np.stack([xs[::-1], np.full(n_per_edge, y_max)], axis=-1),
        np.stack([np.full(n_per_edge, x_min), ys[::-1]], axis=-1),
    ]
    boundary = np.concatenate(edges + [edges[0][:1]], axis=0)

    vec_rot = np.asarray(zoom_proj.xy2vec(boundary[:, 0], boundary[:, 1]))
    vec = np.dot(rot_matrix.T, vec_rot)
    x, y = foot_proj.vec2xy(vec[0], vec[1], vec[2])

    return np.stack([np.asarray(x), np.asarray(y)], axis=-1).astype(np.float64)


# --- the full footprint, back in celestial coordinates -----------------------------------------
# The forward model works in a rotated frame (the footprint is moved so that the four patch cut
# outs fit), so a figure that shows the survey as it sits on the sky has to undo that rotation.
# catalog.py owns the rotation; nothing is reimplemented here.


def celestial_source_pix(n_side, msfm_conf):
    """For every celestial RING pixel, the rotated-frame RING pixel holding its value.

    A gather rather than a scatter: rotating the map by pushing each source pixel to its
    destination leaves holes wherever two sources land on one destination, which shows up as
    speckle. Nearest neighbour, so no value is interpolated into existence.
    """
    ra, dec = hp.pix2ang(n_side, np.arange(hp.nside2npix(n_side)), lonlat=True)
    return catalog.survey_angles_to_pix(msfm_conf, ra, dec, n_side)


def footprint_center(patch_pix_ring, n_side, msfm_conf):
    """Celestial (ra, dec) of the footprint centroid, in degrees.

    Averaged as unit vectors, which is what makes it right across the RA = 0 wrap the DES
    footprint straddles.
    """
    ra, dec = catalog.survey_pix_to_angles(msfm_conf, patch_pix_ring, n_side)
    vec = np.asarray(hp.ang2vec(ra, dec, lonlat=True)).mean(axis=0)
    vec /= np.linalg.norm(vec)
    ra_c, dec_c = hp.vec2ang(vec, lonlat=True)
    # vec2ang always returns arrays; healpy's Rotator rejects a rot tuple holding them
    return float(ra_c[0]), float(dec_c[0])


def footprint_projector(patch_pix_ring, n_side, msfm_conf, reso, margin_deg):
    """A Lambert azimuthal equal-area projector framing the whole footprint.

    Equal area, because the figure is read as "how much sky, and what is in it". The plane
    coordinates of a point depend only on the projection centre, so the frame is sized by
    projecting the footprint first and then asking for enough pixels to hold it.

    Returns:
        (projector, (ra_center_deg, dec_center_deg))
    """
    ra_c, dec_c = footprint_center(patch_pix_ring, n_side, msfm_conf)
    ra, dec = catalog.survey_pix_to_angles(msfm_conf, patch_pix_ring, n_side)
    vec = np.asarray(hp.ang2vec(ra, dec, lonlat=True)).T

    def make(xsize, ysize):
        return hp.projector.AzimuthalProj(
            rot=(ra_c, dec_c, 0.0), lamb=True, xsize=int(xsize), ysize=int(ysize), reso=reso
        )

    probe = make(1000, 1000)
    x, y = probe.vec2xy(vec[0], vec[1], vec[2])
    x0, x1, y0, y1 = probe.get_extent()
    per_pixel = (x1 - x0) / 1000.0

    # the margin is in degrees of great circle; near the centre the Lambert plane is radians
    margin = np.radians(margin_deg)
    half_x = max(abs(np.nanmin(x)), abs(np.nanmax(x))) + margin
    half_y = max(abs(np.nanmin(y)), abs(np.nanmax(y))) + margin

    return make(2 * half_x / per_pixel, 2 * half_y / per_pixel), (float(ra_c), float(dec_c))


def project_celestial(values, patch_pix_ring, source_pix, n_side, proj):
    """Image stack of footprint values in celestial coordinates, NaN off the footprint.

    Args:
        values (n_patch_pix, n_channels): map values on patch_pix_ring, in the rotated frame.
        source_pix: from :func:`celestial_source_pix`.

    Returns:
        (n_channels, ny, nx) float32
    """
    n_pix = hp.nside2npix(n_side)

    def vec2pix(x, y, z):
        return hp.vec2pix(n_side, x, y, z)

    images = []
    for i in range(values.shape[-1]):
        rotated = np.full(n_pix, np.nan, dtype=np.float64)
        rotated[patch_pix_ring] = values[:, i]
        img = np.asarray(proj.projmap(rotated[source_pix], vec2pix), dtype=np.float32)
        images.append(np.where(img < UNSEEN_THRESHOLD, np.nan, img))

    return np.stack(images)


def crop_to_data(images, extent, margin_pix=8):
    """Trim the all-NaN border off a projected image stack, carrying the extent with it.

    The projector frames a rectangle around the footprint centre; the footprint is not a
    rectangle, so a tight crop is what keeps the panel from being mostly empty.

    Returns:
        (cropped stack, cropped extent)
    """
    finite = np.isfinite(images).any(axis=0)
    rows = np.flatnonzero(finite.any(axis=1))
    cols = np.flatnonzero(finite.any(axis=0))
    n_y, n_x = finite.shape

    i0 = max(int(rows[0]) - margin_pix, 0)
    i1 = min(int(rows[-1]) + margin_pix + 1, n_y)
    j0 = max(int(cols[0]) - margin_pix, 0)
    j1 = min(int(cols[-1]) + margin_pix + 1, n_x)

    x0, x1, y0, y1 = extent
    dx = (x1 - x0) / n_x
    dy = (y1 - y0) / n_y
    cropped_extent = (x0 + j0 * dx, x0 + j1 * dx, y0 + i0 * dy, y0 + i1 * dy)

    return images[:, i0:i1, j0:j1], cropped_extent


def graticule(proj, ra_range, dec_range, step_deg, n_samples=400, pad_deg=3.0):
    """Meridians and parallels of the celestial grid, as polylines in the projection plane.

    The graticule *is* the coordinate system of a footprint panel -- there are no meaningful
    numeric axes on an azimuthal projection -- so it is exported alongside the images rather than
    reconstructed by the plotting code, which has no healpy.

    Returns:
        dict with ``meridian_values`` (k,), ``meridians`` (k, n_samples, 2) and the same for
        parallels. Points behind the projection are NaN.
    """
    ra_lo, ra_hi = ra_range
    dec_lo, dec_hi = dec_range

    def line(ra, dec):
        vec = np.asarray(hp.ang2vec(ra, dec, lonlat=True)).T
        x, y = proj.vec2xy(vec[0], vec[1], vec[2])
        return np.stack([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)], axis=-1)

    ra_values = np.arange(np.ceil(ra_lo / step_deg), np.floor(ra_hi / step_deg) + 1) * step_deg
    dec_values = np.arange(np.ceil(dec_lo / step_deg), np.floor(dec_hi / step_deg) + 1) * step_deg

    ra_samples = np.linspace(ra_lo - pad_deg, ra_hi + pad_deg, n_samples)
    dec_samples = np.linspace(dec_lo - pad_deg, dec_hi + pad_deg, n_samples)

    return {
        "meridian_values": np.mod(ra_values, 360.0),
        "meridians": np.stack([line(np.full(n_samples, ra), dec_samples) for ra in ra_values]),
        "parallel_values": dec_values,
        "parallels": np.stack([line(ra_samples, np.full(n_samples, dec)) for dec in dec_values]),
    }


def main():
    args = parse_args()

    with open(args.msfm_config, "r") as f:
        msfm_conf_str = f.read()
    msfm_conf = yaml.safe_load(msfm_conf_str)
    with open(args.scales_config, "r") as f:
        scales_conf_str = f.read()
    scales_conf = yaml.safe_load(scales_conf_str)

    output = args.output or default_output(args, args.msfm_config, args.scales_config)

    n_side = msfm_conf["analysis"]["n_side"]
    n_z_wl = len(msfm_conf["survey"]["metacal"]["z_bins"])
    n_z_gc = len(msfm_conf["survey"]["maglim"]["z_bins"])
    n_channels = n_z_wl + n_z_gc

    # the footprint is the intersection of all per bin metacal and maglim masks; get_dv_mask
    # asserts that they agree, which is what makes a single patch_pix well defined
    data_vec_pix_all, _, _, _ = files.load_pixel_file(msfm_conf)
    dv_masks_dict = files.get_tomo_dv_masks(msfm_conf)
    footprint = files.get_dv_mask(msfm_conf)
    patch_pix = np.asarray(data_vec_pix_all, dtype=np.int64)[footprint]
    print(f"{len(patch_pix)} footprint pixels of {hp.nside2npix(n_side)}", flush=True)

    # -------------------------------------------------------------------------- the two datasets
    print("forward modeling the DES Y3 catalogs", flush=True)
    desy3_raw, data_vec_pix = load_desy3(msfm_conf)
    assert np.array_equal(data_vec_pix, data_vec_pix_all), "unexpected data vector pixels"

    print(f"reading realization {args.i_realization} of {args.mock_file}", flush=True)
    mock_raw = load_mock(args.mock_file, args.i_realization, len(data_vec_pix), n_channels)

    # ------------------------------------------------------------------------------- smoothing
    fwhm = np.concatenate([scales_conf["scale_cuts"][p]["theta_fwhm"] for p in ("lensing", "clustering")]).astype(
        np.float64
    )
    assert scales_conf["scale_cuts"]["arcmin"], "the scale cut config is expected to be in arcmin"
    print(f"smoothing with FWHM = {fwhm} arcmin", flush=True)

    desy3_smooth = smooth(desy3_raw, data_vec_pix, msfm_conf, scales_conf, dv_masks_dict)
    mock_smooth = smooth(mock_raw, data_vec_pix, msfm_conf, scales_conf, dv_masks_dict)

    maps_dict = {
        "desy3": {"raw": desy3_raw[footprint], "smoothed": desy3_smooth[footprint]},
        "mock": {"raw": mock_raw[footprint], "smoothed": mock_smooth[footprint]},
    }
    for source, versions in maps_dict.items():
        std = versions["smoothed"].std(axis=0)
        print(f"{source} smoothed std per channel = {np.array2string(std, precision=5)}", flush=True)

    # ------------------------------------------------------- the whole footprint, on the real sky
    # Only the smoothed maps: at the ~5 arcmin per pixel a 5000 deg^2 panel can hold, an unsmoothed
    # map is noise, and storing both would double a file that is already the biggest thing in
    # paper_2_plotting/cache.
    patch_pix_ring = hp.nest2ring(n_side, patch_pix)
    foot_proj, (ra_c, dec_c) = footprint_projector(
        patch_pix_ring, n_side, msfm_conf, args.footprint_reso, args.footprint_margin
    )
    print(f"footprint projection centred on (ra, dec) = ({ra_c:.2f}, {dec_c:.2f}) deg", flush=True)

    source_pix = celestial_source_pix(n_side, msfm_conf)
    sources = list(maps_dict)
    stacked = np.concatenate(
        [project_celestial(maps_dict[s]["smoothed"], patch_pix_ring, source_pix, n_side, foot_proj) for s in sources]
    )
    # cropped as one array, so both panels end up on the same frame -- cropping them separately
    # would silently shift one against the other if their footprints ever stopped agreeing
    stacked, footprint_extent = crop_to_data(stacked, foot_proj.get_extent())
    footprint_projections = dict(zip(sources, np.split(stacked, len(sources))))
    print(f"footprint panel is {stacked.shape[1:]} pixels", flush=True)

    # the graticule spans the footprint itself, so it is derived from where the footprint actually is
    ra_f, dec_f = catalog.survey_pix_to_angles(msfm_conf, patch_pix_ring, n_side)
    d_ra = (ra_f - ra_c + 180.0) % 360.0 - 180.0
    grat = graticule(
        foot_proj,
        (ra_c + d_ra.min(), ra_c + d_ra.max()),
        (dec_f.min(), dec_f.max()),
        args.graticule_step,
    )
    print(
        f"graticule: RA {np.array2string(grat['meridian_values'], precision=0)}, "
        f"Dec {np.array2string(grat['parallel_values'], precision=0)}",
        flush=True,
    )

    # ------------------------------------------------------------------------------- the zoom
    zoom_proj = hp.projector.GnomonicProj(
        rot=(args.zoom_lon, args.zoom_lat, 0.0), xsize=args.zoom_size, ysize=args.zoom_size, reso=args.zoom_reso
    )
    zoom_projections = {
        source: project_zoom(versions["smoothed"], patch_pix, n_side, zoom_proj)
        for source, versions in maps_dict.items()
    }
    covered = np.mean(np.isfinite(zoom_projections["desy3"][0]))
    print(f"the zoom is {covered:.1%} inside the footprint", flush=True)
    assert covered > args.min_coverage, "the zoom falls off the survey footprint, adjust --zoom_lon/--zoom_lat"

    # private, but it is the one function in this repo that owns the footprint rotation and this
    # script lives inside that repo; catalog.survey_pix_to_angles only inverts it for pixel centres
    rot_matrix = catalog._get_footprint_rotation_matrix(msfm_conf)
    outline_xy = zoom_outline(zoom_proj, foot_proj, rot_matrix)

    # ------------------------------------------------------------------------------------ write
    channel_names = [f"kappa_{i + 1}" for i in range(n_z_wl)] + [f"n_g_{i + 1}" for i in range(n_z_gc)]
    channel_labels = [rf"$\kappa^{i + 1}$" for i in range(n_z_wl)] + [rf"$n_g^{i + 1}$" for i in range(n_z_gc)]

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    print(f"writing {output}", flush=True)
    with h5py.File(output, "w") as f:
        ds = dict(compression="gzip", compression_opts=4, shuffle=True)

        f.attrs["description"] = (
            "the DES Y3 observation and one realization of a CosmoGridV1 benchmark fiducial mock, forward "
            "modelled identically and smoothed with the scale cuts of the summary network"
        )
        f.attrs["msfm_config"] = os.path.relpath(args.msfm_config, REPO_DIR)
        f.attrs["msfm_git_hash"] = git_hash(REPO_DIR)
        f.attrs["scales_config"] = os.path.basename(args.scales_config)
        f.attrs["mock_file"] = args.mock_file
        f.attrs["i_realization"] = args.i_realization
        f.attrs["nside"] = n_side
        f.attrs["ordering"] = "NEST"
        f.attrs["coordinates"] = (
            "the maps live in the rotated frame of the forward model, not in celestial coordinates. "
            "The original DES footprint is rotated around the y axis by footprint_rotation_y_rad and then "
            "around the z axis by footprint_rotation_z_rad"
        )
        f.attrs["footprint_rotation_y_rad"] = msfm_conf["analysis"]["footprint"]["rotation"]["y_rad"]
        f.attrs["footprint_rotation_z_rad"] = msfm_conf["analysis"]["footprint"]["rotation"]["z_rad"]
        f.attrs["sources"] = ["desy3", "mock"]
        f.attrs["source_labels"] = ["DES Y3", "CosmoGrid mock"]
        f.attrs["probes"] = ["lensing", "clustering"]
        f.attrs["n_z_lensing"] = n_z_wl
        f.attrs["n_z_clustering"] = n_z_gc
        f.attrs["channel_names"] = channel_names
        f.attrs["channel_labels"] = channel_labels
        f.attrs["theta_fwhm_arcmin"] = fwhm

        # ------------------------------------------------------------------------------ pixels
        g = f.create_group("pixels")
        g.attrs["description"] = (
            "healpix NEST indices of the survey footprint. All map arrays are indexed by these pixels, scatter "
            "them onto a full sky map with full[patch_pix] = values"
        )
        g.create_dataset("patch_pix", data=patch_pix.astype(np.int32), **ds)
        theta, phi = hp.pix2ang(n_side, patch_pix, nest=True)
        g.create_dataset("longitude_deg", data=np.degrees(phi).astype(np.float32), **ds)
        g.create_dataset("latitude_deg", data=(90.0 - np.degrees(theta)).astype(np.float32), **ds)
        g.attrs["pixel_area_deg2"] = hp.nside2pixarea(n_side, degrees=True)
        g.attrs["footprint_area_deg2"] = len(patch_pix) * hp.nside2pixarea(n_side, degrees=True)

        # -------------------------------------------------------------------------------- maps
        g = f.create_group("maps")
        g.attrs["description"] = (
            "map values on pixels/patch_pix, in physical units (convergence, then galaxy counts per pixel), for both "
            "sources. 'raw' carries only the ell < 30 cut of the forward model, 'smoothed' is additionally "
            "Gaussian smoothed with theta_fwhm_arcmin. Neither has the network's white noise added"
        )
        g.attrs["channel_names"] = channel_names
        for source, versions in maps_dict.items():
            sub = g.create_group(source)
            for name, values in versions.items():
                sub.create_dataset(name, data=np.asarray(values, dtype=np.float32), **ds)

        # ------------------------------------------------------------------------- projections
        g = f.create_group("projections")
        g.attrs["description"] = (
            "the survey as it sits on the sky: the rotation of the forward model undone, then projected "
            "Lambert azimuthal equal-area about the footprint centroid, one image stack per source. NaN "
            "off the footprint and off the sphere. The plane coordinates are not degrees and carry no "
            "meaningful ticks -- the graticule below is the coordinate system. Plot with "
            "plt.imshow(image, extent=extent, origin='lower')"
        )
        g.attrs["projection"] = "lambert azimuthal equal-area"
        g.attrs["center_radec_deg"] = np.array([ra_c, dec_c], dtype=np.float64)
        g.attrs["reso_arcmin"] = args.footprint_reso
        g.attrs["version"] = "smoothed"
        for source, images in footprint_projections.items():
            d = g.create_dataset(source, data=images, **ds)
            d.attrs["extent"] = np.array(footprint_extent, dtype=np.float64)

        sub = g.create_group("graticule")
        sub.attrs["description"] = (
            "celestial grid lines in the same plane coordinates as the images: one polyline per line, NaN "
            "where it falls behind the projection. *_values carry the RA / Dec each line is at, in degrees, "
            "for the labels"
        )
        for key, value in grat.items():
            sub.create_dataset(key, data=np.asarray(value, dtype=np.float64), **ds)

        sub = g.create_group("zoom")
        sub.attrs["description"] = (
            "a square gnomonic cutout of the same smoothed maps, for the inset panel. Plotted against "
            "extent_deg, the offset from its centre in degrees. Its centre is given in the rotated frame "
            "of the forward model, not in celestial coordinates"
        )
        sub.attrs["center_lonlat_deg"] = np.array([args.zoom_lon, args.zoom_lat], dtype=np.float64)
        sub.attrs["reso_arcmin"] = args.zoom_reso
        sub.attrs["size_pix"] = args.zoom_size
        sub.attrs["size_deg"] = args.zoom_size * args.zoom_reso / 60.0
        half_deg = 0.5 * args.zoom_size * args.zoom_reso / 60.0
        for source, images in zoom_projections.items():
            d = sub.create_dataset(source, data=images, **ds)
            d.attrs["extent"] = np.array(zoom_proj.get_extent(), dtype=np.float64)
            d.attrs["extent_deg"] = np.array([-half_deg, half_deg, -half_deg, half_deg], dtype=np.float64)

        d = g.create_dataset("zoom_outline_xy", data=outline_xy, **ds)
        d.attrs["description"] = (
            "closed polygon tracing the border of the zoom, in the plane coordinates of the footprint "
            "images. Draw it on the main panel to mark where the inset came from"
        )

        # --------------------------------------------------------------------------- cosmology
        g = f.create_group("cosmology")
        g.attrs["description"] = (
            "the cosmology the mock was drawn at, i.e. the fiducial of the forward model. NOTE that bary_Mc is "
            "in log10 units here, the convention of the config rather than of the label tables"
        )
        param_names = parameters.get_parameters(conf=msfm_conf)
        fiducial = msfm_conf["analysis"]["fiducial"]
        g.create_dataset("param_names", data=np.array(param_names, dtype=h5py.string_dtype()))
        g.create_dataset(
            "fiducial_values", data=np.array([fiducial.get(p, np.nan) for p in param_names], dtype=np.float64)
        )

        # ------------------------------------------------------------------------- redshifts
        g = f.create_group("redshift_distributions")
        g.attrs["description"] = "the DES Y3 n(z) of the two galaxy samples used by the forward model"
        for sample, probe in (("metacal", "lensing"), ("maglim", "clustering")):
            tomo_z, tomo_nz = files.load_redshift_distributions(sample, msfm_conf)
            sub = g.create_group(probe)
            sub.create_dataset("z", data=np.asarray(tomo_z, dtype=np.float64), **ds)
            sub.create_dataset("nz", data=np.asarray(tomo_nz, dtype=np.float64), **ds)
            sub.attrs["z_bins"] = msfm_conf["survey"][sample]["z_bins"]

        # ---------------------------------------------------------------------------- configs
        g = f.create_group("configs")
        g.attrs["description"] = "verbatim copies of the configs that this file was produced with"
        g.attrs["msfm"] = msfm_conf_str
        g.attrs["scales"] = scales_conf_str

    print(f"done, {os.path.getsize(output) / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
