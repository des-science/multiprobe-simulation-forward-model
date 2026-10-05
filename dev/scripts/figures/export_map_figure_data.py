# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created July 2026
Author: Arne Thomsen

Exports everything that is needed to make the "forward model maps" figure (see
data/figures/combined_moll+gnom.png) into a single, self contained .h5 file. The idea is that this
script runs once on the cluster (it needs tensorflow, msfm and the .tfrecords) and the resulting
file can be copied anywhere and plotted with plot_map_figure.py, which only needs numpy, h5py and
matplotlib.

The file contains, per tomographic bin of both probes:
    - the sparse full sky maps (footprint pixel values + their healpix NEST indices)
    - a smoothed version of the same maps, using the scale cuts of the summary network
    - ready made Mollweide and gnomonic projections of both, as 2d image arrays with NaN outside
      the footprint / the sphere
    - the survey mask, the cosmology label, the redshift distributions and the full configs

Run inside the tensorflow container, e.g.
    sbatch /iopsstor/scratch/cscs/athomsen/deep_lss/claude/jobs/export_map_figure_data.sh
"""

import argparse
import os
import subprocess

import h5py
import healpy as hp
import numpy as np
import tensorflow as tf
import yaml

from msfm.grid_pipeline import GridPipeline
from msfm.utils import files, parameters

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

# healpy marks unseen pixels with this value, and the projectors fill the area outside of the
# sphere with it as well
UNSEEN_THRESHOLD = -1e30

# lower bound on the smoothed mask that the smoothed maps are divided by, see smooth_masked
MASK_DECONVOLUTION_CLIP = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # inputs
    parser.add_argument(
        "--msfm_config",
        default=os.path.join(REPO_DIR, "configs/v17/baseline.yaml"),
        help="msfm forward model config that was used to write the .tfrecords",
    )
    parser.add_argument(
        "--scales_config",
        default="/users/athomsen/dlss/repos/y3-deep-lss/configs/scales/8wl,32gc.yaml",
        help="y3-deep-lss scale cut config, only used for the smoothing kernels of the pretty maps",
    )
    parser.add_argument(
        "--tfrecord",
        # NOTE the absolute /iopsstor path, the $HOME/scratch symlink does not resolve inside the container.
        # The number in the file name is the row of parameters/grid in CosmoGridV1_metainfo.h5, NOT the sobol
        # index (row 1954 -> i_sobol 114996), which is the grid point closest to the fiducial cosmology.
        default="/iopsstor/scratch/cscs/athomsen/deep_lss/data/v17/baseline/tfrecords/grid/DESy3_grid_dmb_1954.tfrecord",
        help="single grid .tfrecord file to take the example from",
    )
    parser.add_argument("--i_signal", type=int, default=0, help="index of the signal realization within the file")
    parser.add_argument("--i_noise", type=int, default=0, help="index of the noise realization within the example")
    # gnomonic cutout
    parser.add_argument("--gnom_lon", type=float, default=90.0, help="gnomonic center longitude in deg")
    parser.add_argument("--gnom_lat", type=float, default=-30.0, help="gnomonic center latitude in deg")
    parser.add_argument("--gnom_reso", type=float, default=1.0, help="gnomonic resolution in arcmin per pixel")
    parser.add_argument("--gnom_size", type=int, default=600, help="gnomonic cutout size in pixels")
    # mollweide projection
    parser.add_argument("--moll_xsize", type=int, default=2048, help="width of the Mollweide image in pixels")
    # output
    parser.add_argument(
        "--output",
        default="/iopsstor/scratch/cscs/athomsen/deep_lss/claude/figures/desy3_forward_model_maps.h5",
        help="path of the .h5 file to write",
    )
    return parser.parse_args()


def git_hash(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def load_example(args, conf_path):
    """Reads a single (signal, noise) realization of one grid cosmology from a .tfrecord.

    Returns:
        maps (n_patch_pix, n_z_metacal + n_z_maglim): raw, unnormalized data vector without padding,
            i.e. kappa (signal + IA + shape noise) followed by galaxy counts (signal + Poisson noise)
        cosmo (n_params,): the cosmology label of this grid point
        index (3,): (i_sobol, i_signal, i_noise)
        pipe: the pipeline instance, for the pixel indices and masks
    """
    pipe = GridPipeline(
        conf=conf_path,
        with_lensing=True,
        with_clustering=True,
        # we want the raw physical units and no zero padding for the plot
        apply_norm=False,
        with_padding=False,
        return_maps=True,
        return_cls=False,
    )

    dset = pipe.get_dset(
        tfr_pattern=args.tfrecord,
        local_batch_size=1,
        signal_indices=[args.i_signal],
        noise_indices=[args.i_noise],
        n_readers=1,
        n_prefetch=1,
        is_eval=True,
    )

    for element in dset.take(1):
        # (map_tensor, cl_tensor, cosmo, (i_sobol, i_signal, i_noise)), cl_tensor is None here
        maps, _, cosmo, index = element
        break

    maps = np.asarray(maps[0], dtype=np.float32)
    cosmo = np.asarray(cosmo[0], dtype=np.float64)
    index = np.array([int(i[0]) for i in index], dtype=np.int64)

    return maps, cosmo, index, pipe


def smooth_masked(maps, patch_pix_ring, nside, fwhm_arcmin, lmax):
    """Gaussian smooths maps that only live on the survey footprint.

    The map and the mask are smoothed with the same kernel and the result is divided by the smoothed
    mask, so that the footprint edge does not bleed into the map. This mimics what the summary
    network sees (deepsphere.healpy_layers.HealpySmoothing with the mask argument), but it is done
    in harmonic space here and the white noise that the network adds on top is left out.

    Args:
        maps (n_patch_pix, n_channels): values on the footprint pixels
        patch_pix_ring (n_patch_pix,): footprint pixel indices in RING ordering
        nside (int): healpix nside of the maps
        fwhm_arcmin (n_channels,): per channel FWHM of the Gaussian kernel in arcmin
        lmax (int): band limit of the smoothing

    Returns:
        (n_patch_pix, n_channels) smoothed values on the same footprint pixels
    """
    n_pix = hp.nside2npix(nside)
    n_channels = maps.shape[-1]

    mask = np.zeros(n_pix, dtype=np.float64)
    mask[patch_pix_ring] = 1.0
    alm_mask = hp.map2alm(mask, lmax=lmax, iter=1)

    out = np.zeros_like(maps)
    for i in range(n_channels):
        full = np.zeros(n_pix, dtype=np.float64)
        full[patch_pix_ring] = maps[:, i]

        fwhm_rad = np.radians(fwhm_arcmin[i] / 60.0)
        beam = hp.gauss_beam(fwhm_rad, lmax=lmax)

        smoothed = hp.alm2map(hp.almxfl(hp.map2alm(full, lmax=lmax, iter=1), beam), nside)
        norm = hp.alm2map(hp.almxfl(alm_mask.copy(), beam), nside)

        # The mask is ~1 well inside the footprint, so the division only matters close to the edge.
        # It is clipped there, otherwise dividing by a vanishing smoothed mask blows the outermost
        # pixels up into a bright rim. Below the clip the map instead fades towards zero, which is
        # what the zero padded smoothing of the network does anyway.
        smoothed = smoothed / np.maximum(norm, MASK_DECONVOLUTION_CLIP)

        out[:, i] = smoothed[patch_pix_ring]

    return out


def sparse_to_full(values, patch_pix, n_pix):
    """Scatters footprint values onto a full sky map, everything else is NaN."""
    full = np.full(n_pix, np.nan, dtype=np.float64)
    full[patch_pix] = values
    return full


def clean_projection(img):
    """healpy fills the area outside of the sphere with UNSEEN, we want NaN everywhere."""
    img = np.asarray(img, dtype=np.float32)
    img = np.where(img < UNSEEN_THRESHOLD, np.nan, img)
    return img


def project(full_maps, nside, moll_proj, gnom_proj):
    """Projects a stack of full sky maps with both projectors.

    Args:
        full_maps (n_channels, n_pix): full sky maps in NEST ordering, NaN outside the footprint

    Returns:
        (moll, gnom) image stacks of shape (n_channels, ny, nx)
    """

    def vec2pix(x, y, z):
        return hp.vec2pix(nside, x, y, z, nest=True)

    moll = np.stack([clean_projection(moll_proj.projmap(m, vec2pix)) for m in full_maps])
    gnom = np.stack([clean_projection(gnom_proj.projmap(m, vec2pix)) for m in full_maps])

    return moll, gnom


def gnomonic_outline_in_mollweide(gnom_proj, moll_proj, n_per_edge=200):
    """Traces the boundary of the gnomonic cutout in the Mollweide projection plane.

    This is what the little box and the dashed connector lines in the figure are drawn from.

    Returns:
        (4 * n_per_edge + 1, 2) array of (x, y) Mollweide coordinates, closed polygon
    """
    x_min, x_max, y_min, y_max = gnom_proj.get_extent()

    xs = np.linspace(x_min, x_max, n_per_edge)
    ys = np.linspace(y_min, y_max, n_per_edge)
    edges = [
        np.stack([xs, np.full(n_per_edge, y_min)], axis=-1),
        np.stack([np.full(n_per_edge, x_max), ys], axis=-1),
        np.stack([xs[::-1], np.full(n_per_edge, y_max)], axis=-1),
        np.stack([np.full(n_per_edge, x_min), ys[::-1]], axis=-1),
    ]
    boundary = np.concatenate(edges + [edges[0][:1]], axis=0)

    # projection plane -> 3d vector on the sphere -> Mollweide projection plane
    vec = gnom_proj.xy2vec(boundary[:, 0], boundary[:, 1])
    x, y = moll_proj.vec2xy(vec[0], vec[1], vec[2])

    return np.stack([np.asarray(x), np.asarray(y)], axis=-1).astype(np.float64)


def main():
    args = parse_args()

    with open(args.msfm_config, "r") as f:
        msfm_conf_str = f.read()
    msfm_conf = yaml.safe_load(msfm_conf_str)
    with open(args.scales_config, "r") as f:
        scales_conf_str = f.read()
    scales_conf = yaml.safe_load(scales_conf_str)

    nside = msfm_conf["analysis"]["n_side"]
    n_pix = hp.nside2npix(nside)
    lmax = 3 * nside - 1

    n_z_wl = len(msfm_conf["survey"]["metacal"]["z_bins"])
    n_z_gc = len(msfm_conf["survey"]["maglim"]["z_bins"])

    print(f"reading {args.tfrecord}", flush=True)
    maps, cosmo, index, pipe = load_example(args, args.msfm_config)
    patch_pix = np.asarray(pipe.patch_pix, dtype=np.int64)
    print(f"maps {maps.shape}, {len(patch_pix)} footprint pixels of {n_pix}", flush=True)
    assert maps.shape == (len(patch_pix), n_z_wl + n_z_gc), f"unexpected map shape {maps.shape}"

    param_names = parameters.get_parameters(conf=msfm_conf)
    print(f"i_sobol, i_signal, i_noise = {index}", flush=True)
    print(f"cosmology = {dict(zip(param_names, np.round(cosmo, 4)))}", flush=True)

    # ------------------------------------------------------------------ physical map definitions
    # both probes are taken verbatim from the .tfrecord, no rescaling of any kind. The lensing
    # channels are the convergence, the clustering channels are the galaxy overdensity that
    # maps.map_to_data_vec built with divide_by_mean, in both cases with the ell < 30 cut applied
    kappa = maps[:, :n_z_wl]
    delta_g = maps[:, n_z_wl:]
    print(f"kappa mean/std per bin = {np.round(kappa.mean(0), 5)} {np.round(kappa.std(0), 5)}", flush=True)
    print(f"delta_g mean/std per bin = {np.round(delta_g.mean(0), 3)} {np.round(delta_g.std(0), 3)}", flush=True)

    # ------------------------------------------------------------------------------- smoothing
    fwhm_wl = np.array(scales_conf["scale_cuts"]["lensing"]["theta_fwhm"], dtype=np.float64)
    fwhm_gc = np.array(scales_conf["scale_cuts"]["clustering"]["theta_fwhm"], dtype=np.float64)
    fwhm = np.concatenate([fwhm_wl, fwhm_gc])
    assert scales_conf["scale_cuts"]["arcmin"], "the scale cut config is expected to be in arcmin"

    patch_pix_ring = hp.nest2ring(nside, patch_pix)
    print(f"smoothing with FWHM = {fwhm} arcmin", flush=True)
    smoothed = smooth_masked(
        np.concatenate([kappa, delta_g], axis=-1).astype(np.float64), patch_pix_ring, nside, fwhm, lmax
    )
    smoothed = smoothed.astype(np.float32)

    # ------------------------------------------------------------------------------ projections
    moll_proj = hp.projector.MollweideProj(xsize=args.moll_xsize)
    gnom_proj = hp.projector.GnomonicProj(
        rot=(args.gnom_lon, args.gnom_lat, 0.0), xsize=args.gnom_size, ysize=args.gnom_size, reso=args.gnom_reso
    )

    raw_stack = np.concatenate([kappa, delta_g], axis=-1)
    full_raw = np.stack([sparse_to_full(raw_stack[:, i], patch_pix, n_pix) for i in range(raw_stack.shape[-1])])
    full_smooth = np.stack([sparse_to_full(smoothed[:, i], patch_pix, n_pix) for i in range(smoothed.shape[-1])])

    print("projecting", flush=True)
    moll_raw, gnom_raw = project(full_raw, nside, moll_proj, gnom_proj)
    moll_smooth, gnom_smooth = project(full_smooth, nside, moll_proj, gnom_proj)

    covered = np.mean(np.isfinite(gnom_smooth[0]))
    print(f"the gnomonic cutout is {covered:.1%} inside the footprint", flush=True)
    assert covered > 0.8, "the gnomonic cutout falls outside of the survey footprint, adjust gnom_lon/gnom_lat"

    outline_xy = gnomonic_outline_in_mollweide(gnom_proj, moll_proj)

    # ------------------------------------------------------------------------------------ write
    channel_names = [f"kappa_{i + 1}" for i in range(n_z_wl)] + [f"delta_g_{i + 1}" for i in range(n_z_gc)]
    channel_labels = [rf"$\kappa^{i + 1}$" for i in range(n_z_wl)] + [rf"$\delta_g^{i + 1}$" for i in range(n_z_gc)]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"writing {args.output}", flush=True)
    with h5py.File(args.output, "w") as f:
        ds = dict(compression="gzip", compression_opts=4, shuffle=True)

        f.attrs["description"] = (
            "DES Y3 like weak lensing and galaxy clustering maps from the multiprobe-simulation-forward-model "
            "(CosmoGridV1 grid simulation), everything needed for the forward model map figure"
        )
        f.attrs["source_tfrecord"] = args.tfrecord
        f.attrs["msfm_config"] = os.path.relpath(args.msfm_config, REPO_DIR)
        f.attrs["msfm_git_hash"] = git_hash(REPO_DIR)
        f.attrs["nside"] = nside
        f.attrs["n_pix_full_sky"] = n_pix
        f.attrs["ordering"] = "NEST"
        f.attrs["coordinates"] = (
            "the maps live in the rotated frame of the forward model, not in celestial coordinates. "
            "The original DES footprint is rotated around the y axis by footprint_rotation_y_rad and then around "
            "the z axis by footprint_rotation_z_rad"
        )
        f.attrs["footprint_rotation_y_rad"] = msfm_conf["analysis"]["footprint"]["rotation"]["y_rad"]
        f.attrs["footprint_rotation_z_rad"] = msfm_conf["analysis"]["footprint"]["rotation"]["z_rad"]
        f.attrs["probes"] = ["lensing", "clustering"]
        f.attrs["n_z_lensing"] = n_z_wl
        f.attrs["n_z_clustering"] = n_z_gc
        f.attrs["channel_names"] = channel_names
        f.attrs["channel_labels"] = channel_labels
        f.attrs["i_sobol"] = index[0]
        f.attrs["i_signal"] = index[1]
        f.attrs["i_noise"] = index[2]

        # ------------------------------------------------------------------------------ pixels
        g = f.create_group("pixels")
        g.attrs["description"] = (
            "healpix NEST indices of the survey footprint at nside. All map arrays are indexed by these pixels, "
            "scatter them onto a full sky map with full[patch_pix] = values"
        )
        g.create_dataset("patch_pix", data=patch_pix.astype(np.int32), **ds)
        g.create_dataset("patch_pix_ring", data=patch_pix_ring.astype(np.int32), **ds)
        theta, phi = hp.pix2ang(nside, patch_pix, nest=True)
        g.create_dataset("longitude_deg", data=np.degrees(phi).astype(np.float32), **ds)
        g.create_dataset("latitude_deg", data=(90.0 - np.degrees(theta)).astype(np.float32), **ds)
        g.attrs["pixel_area_deg2"] = hp.nside2pixarea(nside, degrees=True)
        g.attrs["footprint_area_deg2"] = len(patch_pix) * hp.nside2pixarea(nside, degrees=True)
        g.attrs["sky_fraction"] = len(patch_pix) / n_pix

        # ------------------------------------------------------------------------------- masks
        g = f.create_group("mask")
        g.attrs["description"] = (
            "the footprint mask is implicitly given by pixels/patch_pix, which is the intersection of the per "
            "tomographic bin metacal and maglim masks. The full sky boolean masks are stored here for convenience "
            "(NEST ordering), the padding mask of the data vector is what the networks are fed with"
        )
        mask_full = np.zeros(n_pix, dtype=bool)
        mask_full[patch_pix] = True
        g.create_dataset("full_sky", data=mask_full, **ds)
        tomo_masks = files.get_tomo_masks(msfm_conf, nest_out=True)
        g.create_dataset("metacal_tomo", data=tomo_masks["metacal"].astype(bool), **ds)
        g.create_dataset("maglim_tomo", data=tomo_masks["maglim"].astype(bool), **ds)

        # -------------------------------------------------------------------------------- maps
        g = f.create_group("maps")
        g.attrs["description"] = (
            "map values on the pixels/patch_pix footprint. 'raw' is verbatim what is stored in the .tfrecord "
            "(including shape and Poisson noise, with the ell < 30 cut baked in), 'smoothed' is additionally "
            "Gaussian smoothed with the per bin scale cuts of the summary network"
        )
        g.attrs["channel_names"] = channel_names
        g.attrs["theta_fwhm_arcmin"] = fwhm
        g.attrs["scales_config"] = os.path.basename(args.scales_config)
        g.create_dataset("raw", data=raw_stack.astype(np.float32), **ds)
        g.create_dataset("smoothed", data=smoothed, **ds)
        g.create_dataset("kappa", data=kappa.astype(np.float32), **ds)
        g.create_dataset("delta_g", data=delta_g.astype(np.float32), **ds)
        g["kappa"].attrs["description"] = "convergence, signal + intrinsic alignments + shape noise"
        g["delta_g"].attrs["description"] = (
            "galaxy overdensity (n - n_bar) / n_bar with n_bar the mean over the footprint, including Poisson "
            "noise. This is the clustering data vector of the forward model, not a galaxy count"
        )
        g["raw"].attrs["description"] = "[kappa, delta_g] concatenated along the tomographic bin axis"

        # ------------------------------------------------------------------------- projections
        g = f.create_group("projections")
        g.attrs["description"] = (
            "ready made 2d images, NaN outside of the footprint and outside of the sphere. Plot them with "
            "plt.imshow(image, extent=extent, origin='lower')"
        )

        for name, moll, gnom in [("raw", moll_raw, gnom_raw), ("smoothed", moll_smooth, gnom_smooth)]:
            sub = g.create_group(name)
            d = sub.create_dataset("mollweide", data=moll, **ds)
            d.attrs["extent"] = np.array(moll_proj.get_extent(), dtype=np.float64)
            d.attrs["description"] = (
                "full sky Mollweide projection, the sphere is the ellipse x^2 / 4 + y^2 = 1 in these coordinates"
            )
            d = sub.create_dataset("gnomonic", data=gnom, **ds)
            d.attrs["extent"] = np.array(gnom_proj.get_extent(), dtype=np.float64)
            d.attrs["extent_arcmin"] = np.array(
                [
                    -0.5 * args.gnom_size * args.gnom_reso,
                    0.5 * args.gnom_size * args.gnom_reso,
                    -0.5 * args.gnom_size * args.gnom_reso,
                    0.5 * args.gnom_size * args.gnom_reso,
                ],
                dtype=np.float64,
            )
            d.attrs["description"] = "gnomonic cutout, the white holes are masked pixels inside the footprint"

        g.attrs["mollweide_xsize"] = args.moll_xsize
        g.attrs["gnomonic_center_lonlat_deg"] = np.array([args.gnom_lon, args.gnom_lat])
        g.attrs["gnomonic_reso_arcmin"] = args.gnom_reso
        g.attrs["gnomonic_size_pix"] = args.gnom_size
        g.attrs["gnomonic_size_deg"] = args.gnom_size * args.gnom_reso / 60.0
        d = g.create_dataset("gnomonic_outline_mollweide_xy", data=outline_xy, **ds)
        d.attrs["description"] = (
            "closed polygon tracing the border of the gnomonic cutout in the coordinates of the Mollweide image, "
            "draw this as the little box and anchor the connector lines to its corners"
        )

        # -------------------------------------------------------------------------- cosmology
        g = f.create_group("cosmology")
        g.attrs["description"] = (
            "the CosmoGrid label of this grid point. NOTE that bary_Mc is stored in raw units here, the same "
            "convention as in the .tfrecords and label tables, downstream it is used as log10(bary_Mc)"
        )
        g.create_dataset("param_names", data=np.array(param_names, dtype=h5py.string_dtype()))
        g.create_dataset("param_values", data=cosmo)
        fid = msfm_conf["analysis"]["fiducial"]
        g.create_dataset(
            "fiducial_values",
            data=np.array([fid.get(p, np.nan) for p in param_names], dtype=np.float64),
        )
        g["fiducial_values"].attrs["description"] = "the fiducial cosmology of the analysis, bary_Mc in log10 units"

        # ------------------------------------------------------------------------- redshifts
        g = f.create_group("redshift_distributions")
        g.attrs["description"] = "the DES Y3 n(z) of the two galaxy samples used by the forward model"
        for sample, key in [("metacal", "lensing"), ("maglim", "clustering")]:
            tomo_z, tomo_nz = files.load_redshift_distributions(sample, msfm_conf)
            sub = g.create_group(key)
            sub.create_dataset("z", data=np.asarray(tomo_z, dtype=np.float64), **ds)
            sub.create_dataset("nz", data=np.asarray(tomo_nz, dtype=np.float64), **ds)
            sub.attrs["z_bins"] = msfm_conf["survey"][sample]["z_bins"]
            sub.attrs["n_gal_per_deg2"] = msfm_conf["survey"][sample]["n_gal"]
        g["clustering"].attrs["z_lims"] = np.array(msfm_conf["survey"]["maglim"]["z_lims"], dtype=np.float64)

        # ---------------------------------------------------------------------------- configs
        g = f.create_group("configs")
        g.attrs["description"] = "verbatim copies of the configs that this file was produced with"
        g.attrs["msfm"] = msfm_conf_str
        g.attrs["scales"] = scales_conf_str

    print(f"done, {os.path.getsize(args.output) / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
