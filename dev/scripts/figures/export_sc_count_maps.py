# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created September 2026
Author: Arne Thomsen (with Claude)

Projects the forward modelled metacal source galaxy count maps of one CosmoGrid cosmology onto the
sky for the paper_2 figure that shows what the DES Y3 imaging-systematics correction does to them:
two arms, clean and contaminated, each at its own fitted source-clustering bias.

This is the map level companion of export_sc_bias_grid.py. That export carries the *fit* -- b_g,s
of 2500 grid cosmologies with and without the imprint -- and this one carries the maps the fit was
performed on, for one cosmology, so that the same statement can be read off the sky:

    n_g,si = clip_renorm[ <n_g,si> (1 + b_g,si delta_i) C_i ]   ->   Poisson

built exactly as postprocessing.postprocess_shape_noise builds it for the base patch, with the
contamination C = <1/W> of files.get_metacal_systematics and the per cosmology bias of
data/desy3_metacal_bias.h5. The two arms use the bias fitted *under their own contamination*, which
is the pair the pipeline actually chooses between: the clean arm is what the forward model would
consume with no correction, the contaminated arm is what it does consume.

Both the Poisson rate and one Poisson realization at the fit's own seed are exported, all four
tomographic bins of each. The rate is the noise free imprint and is what a figure about the
correction wants; the realization is what the shape noise generator is handed, and it carries
~12% shot noise per pixel against a contamination of 5.7, 8.2, 9.7, 22.2% in the four bins -- so
only in metacal4 is the imprint visible by eye in a realization. Which of the two, and which bin,
is therefore a plot side choice and both are in the file.

What is projected is smoothed to --smooth_fwhm first, by default the 12.6 arcmin scale cut of the
last metacal bin, which is the scale at which this sample enters the network. That is not
cosmetic: since the bias is fitted to the one-point function, the two arms agree per pixel by
construction and only separate once smoothed. The healpix values under maps/ are always the
unsmoothed ones the forward model itself produces.

Everything is derived from data/ and the projected CosmoGrid maps; there is no dependence on a
trained network or a run directory. Cheap: 17 spherical harmonic transforms at nside 512 for the
smoothing, and the CosmoGrid read is one .h5. A login node is fine.

    OMP_NUM_THREADS=4 ~/dlss/torch_env/bin/python3 dev/scripts/figures/export_sc_count_maps.py

The projection is the one every paper_2 sky map uses (footprint_projection.py): the forward
model's footprint rotation undone, then Lambert azimuthal equal-area about the footprint centroid,
with a celestial graticule instead of numeric axes.
"""

import argparse
import os
import subprocess

import h5py
import numpy as np

from msfm.utils import catalog, clustering, files, imports, source_clustering_bias as scb

# sibling module in this directory, shared with the other figure exports: the footprint projection
# and its graticule, so that every paper_2 sky map is the same projection
import footprint_projection

hp = imports.import_healpy()

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

#: The two arms of the fit, in the order they are read: clean first, so a figure drawing them left
#: to right shows "what the imprint does to it" rather than the reverse.
ARMS = ("clean", "contam")

#: What is built per arm. "rate" is the Poisson rate, i.e. the noise free expectation; "counts" is
#: one Poisson realization of it at the seed the bias fit itself used.
VERSIONS = ("rate", "counts")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        default=os.path.join(REPO_DIR, "configs/v18/default.yaml"),
        help="msfm forward model config. Must be the one --bias_file was fitted under",
    )
    parser.add_argument(
        "--bias_file",
        default=os.path.join(REPO_DIR, "data/desy3_metacal_bias.h5"),
        help="the tracked bias table of notebooks/sc_bias_fit_count.ipynb, one HDF5 group per arm",
    )
    parser.add_argument(
        "--cosmo_dir",
        # NOTE the absolute /iopsstor path, the $HOME/scratch symlink does not resolve inside a container
        default="/iopsstor/scratch/cscs/athomsen/CosmoGrid/v11desy3/bary/fiducial/cosmo_fiducial",
        help="directory of the single CosmoGrid cosmology to forward model",
    )
    parser.add_argument(
        "--cosmo_key",
        default="fiducial",
        help="its key in --bias_file. 'fiducial' for the fiducial, the directory name for a grid cosmology",
    )
    parser.add_argument("--perm", type=int, default=0, help="permutation index of --cosmo_dir")
    parser.add_argument("--reso", type=float, default=5.0, help="panel resolution in arcmin per pixel")
    parser.add_argument(
        "--smooth_fwhm",
        type=float,
        default=12.6,
        help="Gaussian FWHM in arcmin to smooth the maps with before projecting, 0 for none. The "
        "default is the analysis scale cut of the last metacal bin (8 Mpc/h, the lensing block of "
        "y3-deep-lss configs/scales/8wl,32gc.yaml), i.e. the scale at which this sample enters the "
        "network -- the sample picks the scale, not the fact that the quantity here is a count. "
        "Smoothing at all is not cosmetic: the bias is fitted to the one-point function, so "
        "unsmoothed the two arms match by construction and the figure shows nothing. It does cost "
        "separation, though, and knowing how much is the point of having the flag: the arms' rms "
        "ratio in the last bin is 0.98 unsmoothed, 1.17 here, and 1.72 at the 57 arcmin clustering "
        "cut. The healpix values under maps/ are always the unsmoothed ones",
    )
    parser.add_argument("--margin", type=float, default=2.0, help="blank margin around the footprint in deg")
    parser.add_argument("--graticule_step", type=float, default=20.0, help="spacing of the RA/Dec grid lines in deg")
    parser.add_argument("--output", default=None, help="path of the .h5 to write; defaults to the plotting cache")
    return parser.parse_args()


def default_output(config, cosmo_key, smooth_fwhm):
    """paper_2_plotting/cache/sc_count_maps_<version>-<release>_<cosmo key>[_smooth<fwhm>].h5.

    The name carries everything that changes the numbers, because nothing downstream checks that a
    cache file was produced by the settings that found it -- the smoothing included, since a
    smoothed and an unsmoothed panel look different enough to argue about.
    """
    version = os.path.basename(os.path.dirname(config))
    release = os.path.splitext(os.path.basename(config))[0]
    suffix = "" if smooth_fwhm <= 0 else f"_smooth{smooth_fwhm:g}"
    name = f"sc_count_maps_{version}-{release}_{cosmo_key}{suffix}.h5"
    return os.path.abspath(os.path.join(REPO_DIR, "../deep_lss_paper/paper_2_plotting/cache", name))


def git_hash(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def read_biases(path, cosmo_key):
    """The fitted b_g,s of one cosmology in both arms, plus each arm's systematics label.

    Returns:
        ({arm: (n_z,) biases}, {arm: label})
    """
    biases, labels = {}, {}
    with h5py.File(path, "r") as f:
        for arm in ARMS:
            group = f[arm]
            assert cosmo_key in group, f"{path}/{arm} has no cosmology {cosmo_key!r}"
            biases[arm] = group[cosmo_key][:].astype(np.float64)
            labels[arm] = str(group.attrs["systematics_label"])
    return biases, labels


def forward_model(n_bar, delta, biases, contamination):
    """The count maps of both arms and all tomographic bins, on the footprint pixels.

    The clean arm passes no contamination at all rather than a map of ones: that is the branch the
    forward model takes without the correction, and it is also what the clean bias was fitted
    under.

    Args:
        n_bar (n_z,): mean galaxies per pixel of each tomographic bin.
        delta (n_footprint_pix, n_z): normalized density contrast of the cosmology.
        biases (dict): from :func:`read_biases`.
        contamination (n_footprint_pix, n_z): <1/W>, from files.get_metacal_systematics.

    Returns:
        {arm: {version: (n_footprint_pix, n_z)}}
    """
    n_z = delta.shape[-1]
    maps = {}
    for arm in ARMS:
        contam = None if arm == "clean" else contamination
        columns = {version: [] for version in VERSIONS}
        for i in range(n_z):
            contam_i = None if contam is None else contam[:, i]
            columns["rate"].append(
                clustering.galaxy_density_to_count(n_bar[i], delta[:, i], biases[arm][i], contamination_map=contam_i)
            )
            # the fit's own seed, so that the realization shown is the one the fit was scored on
            columns["counts"].append(
                scb.counts_from_bias(n_bar[i], delta[:, i], biases[arm][i], contamination=contam_i)
            )
        maps[arm] = {version: np.stack(values, axis=-1).astype(np.float32) for version, values in columns.items()}
    return maps


def smooth(values, mask, n_side, fwhm_arcmin):
    """Gaussian-smooth maps that only exist on the footprint, dividing the footprint back out.

    Smoothing zeros-outside-the-footprint would pull the whole survey edge towards zero over a
    kernel width. Smoothing the mask alongside and dividing recovers the local mean of the map
    instead, which is what the eye should see at the boundary.

    Args:
        values (n_footprint_pix, n_z): map values on the footprint.
        mask (n_pix,): boolean footprint mask, in the same ordering as ``values`` is indexed by.

    Returns:
        (n_footprint_pix, n_z) the smoothed values
    """
    n_pix = hp.nside2npix(n_side)
    fwhm = np.radians(fwhm_arcmin / 60.0)

    weight = np.zeros(n_pix)
    weight[mask] = 1.0
    den = hp.smoothing(weight, fwhm=fwhm)[mask]

    out = np.empty_like(values)
    for i in range(values.shape[-1]):
        full = np.zeros(n_pix)
        full[mask] = values[:, i]
        out[:, i] = hp.smoothing(full, fwhm=fwhm)[mask] / den
    return out


def celestial_images(values, mask, source_pix, n_side, proj):
    """Image stack of footprint values, the footprint rotation undone. NaN off the footprint.

    The same operation as ``export_data_vs_mock_maps.project_celestial``, minus its NEST/RING
    conversion: the maps here are built on the RING mask of ``files.get_mask`` to begin with, which
    is the frame ``scb.read_density_contrast`` and ``files.get_metacal_systematics`` agree in.

    Args:
        values (n_footprint_pix, n_z): map values on the footprint, in the rotated frame.
        mask (n_pix,): boolean footprint mask, RING, in the rotated frame.
        source_pix (n_pix,): for every celestial RING pixel, the rotated-frame RING pixel holding
            its value.

    Returns:
        (n_z, ny, nx) float32
    """
    n_pix = hp.nside2npix(n_side)
    images = []
    for i in range(values.shape[-1]):
        rotated = np.full(n_pix, np.nan, dtype=np.float64)
        rotated[mask] = values[:, i]
        images.append(footprint_projection.project(rotated[source_pix], n_side, proj))
    return np.stack(images)


def main():
    args = parse_args()
    output = args.output or default_output(args.config, args.cosmo_key, args.smooth_fwhm)

    with open(args.config, "r") as f:
        config_str = f.read()
    conf = files.load_config(args.config)

    n_side = conf["analysis"]["n_side"]
    n_pix = hp.nside2npix(n_side)
    z_bins = list(conf["survey"]["metacal"]["z_bins"])
    n_z = len(z_bins)

    # ------------------------------------------------------------------- the forward model inputs
    # n_bar comes from the config rather than from the observation, which is what the forward model
    # itself does; the contamination is returned on the base patch, which is the footprint of this
    # very mask in np.arange(n_pix) order and is therefore aligned with the CosmoGrid maps
    mask = files.get_mask(conf, nest_out=False)
    n_bar = np.array(conf["survey"]["metacal"]["n_gal"]) * hp.nside2pixarea(n_side, degrees=True)
    contamination = files.get_metacal_systematics(conf)
    base_patch_pix = files.load_pixel_file(conf)[1]["metacal"][0][0]
    assert np.array_equal(base_patch_pix, np.arange(n_pix)[mask]), "the base patch is not the footprint of the mask"
    assert contamination.shape == (int(mask.sum()), n_z), contamination.shape

    area = int(mask.sum()) * hp.nside2pixarea(n_side, degrees=True)
    print(f"{int(mask.sum())} footprint pixels at nside {n_side}, {area:.0f} deg^2", flush=True)

    biases, labels = read_biases(args.bias_file, args.cosmo_key)
    print(f"biases of {args.cosmo_key!r}, systematics {labels['clean']!r} vs {labels['contam']!r}", flush=True)
    for arm in ARMS:
        print(f"  {arm:6s} b_g,s = {np.array2string(biases[arm], precision=3)}", flush=True)
    print(f"n_bar             = {np.array2string(n_bar, precision=2)} galaxies per pixel", flush=True)
    print(f"shot noise        = {np.array2string(100 / np.sqrt(n_bar), precision=1)} % per pixel", flush=True)
    print(
        f"contamination rms = {np.array2string(100 * contamination.std(axis=0), precision=1)} % per pixel", flush=True
    )

    print(f"reading {args.cosmo_dir}", flush=True)
    delta = scb.read_density_contrast(conf, args.cosmo_dir, perm=args.perm)[mask]

    # --------------------------------------------------------------------------- the count maps
    maps = forward_model(n_bar, delta, biases, contamination)
    print("\nper pixel rms of the footprint counts, in % of the mean", flush=True)
    print("           " + "".join(f"   {z:>10s}" for z in z_bins), flush=True)
    for arm in ARMS:
        for version in VERSIONS:
            values = maps[arm][version]
            rms = 100 * values.std(axis=0) / values.mean(axis=0)
            print(f"  {arm:6s} {version:6s}" + "".join(f"   {v:10.2f}" for v in rms), flush=True)

    # ------------------------------------------------------------------------------------ project
    ra_f, dec_f = _celestial_footprint(conf, mask, n_side)
    proj, (ra_c, dec_c) = footprint_projection.projector(ra_f, dec_f, args.reso, args.margin)
    print(f"\nprojection centred on (ra, dec) = ({ra_c:.2f}, {dec_c:.2f}) deg", flush=True)

    source_pix = _celestial_source_pix(conf, n_side)
    keys = [(arm, version) for arm in ARMS for version in VERSIONS]

    # only what is projected is smoothed; maps/ keeps the values the forward model itself produced
    shown = {key: maps[key[0]][key[1]] for key in keys}
    if args.smooth_fwhm > 0:
        print(f"smoothing with FWHM = {args.smooth_fwhm:g} arcmin", flush=True)
        shown = {key: smooth(values, mask, n_side, args.smooth_fwhm) for key, values in shown.items()}
        print("per pixel rms after smoothing, in % of the mean", flush=True)
        for (arm, version), values in shown.items():
            rms = 100 * values.std(axis=0) / values.mean(axis=0)
            print(f"  {arm:6s} {version:6s}" + "".join(f"   {v:10.2f}" for v in rms), flush=True)

    stacked = np.concatenate([celestial_images(shown[key], mask, source_pix, n_side, proj) for key in keys])
    # cropped as one stack, so every panel of the figure lands on the same frame -- cropping them
    # separately would silently shift two panels of the same sky against each other
    stacked, extent = footprint_projection.crop_to_data(stacked, proj.get_extent())
    projections = dict(zip(keys, np.split(stacked, len(keys))))
    print(f"panel is {stacked.shape[1:]} pixels", flush=True)

    grat = footprint_projection.footprint_graticule(proj, ra_f, dec_f, ra_c, args.graticule_step)
    print(
        f"graticule: RA {np.array2string(grat['meridian_values'], precision=0)}, "
        f"Dec {np.array2string(grat['parallel_values'], precision=0)}",
        flush=True,
    )

    # ------------------------------------------------------------------------------------- write
    ds = {"compression": "gzip", "compression_opts": 4, "shuffle": True}
    os.makedirs(os.path.dirname(output), exist_ok=True)
    print(f"writing {output}", flush=True)

    with h5py.File(output, "w") as f:
        f.attrs["description"] = (
            "the forward modelled metacal source galaxy count maps of one CosmoGrid cosmology, clean and "
            "contaminated with the DES Y3 imaging-systematics imprint, each at its own fitted "
            "source-clustering bias. The map level companion of the bias grid of export_sc_bias_grid.py. "
            "Produced by msfm/dev/scripts/figures/export_sc_count_maps.py"
        )
        f.attrs["config"] = os.path.relpath(args.config, REPO_DIR)
        f.attrs["bias_file"] = os.path.abspath(args.bias_file)
        f.attrs["cosmo_dir"] = args.cosmo_dir
        f.attrs["cosmo_key"] = args.cosmo_key
        f.attrs["perm"] = args.perm
        f.attrs["msfm_git_sha"] = git_hash(REPO_DIR)
        f.attrs["nside"] = n_side
        f.attrs["n_bins"] = n_z
        f.attrs["bin_labels"] = z_bins
        f.attrs["arms"] = list(ARMS)
        f.attrs["versions"] = list(VERSIONS)
        f.attrs["footprint_area_deg2"] = area
        f.attrs["seed"] = scb.DEFAULT_SEED
        # the numbers a caption would quote, computed here so that a figure never re-derives one of
        # them from a different footprint
        f.attrs["n_bar"] = n_bar
        f.attrs["shot_noise_frac"] = 1.0 / np.sqrt(n_bar)
        f.attrs["contamination_rms"] = contamination.std(axis=0)
        for arm in ARMS:
            f.attrs[f"bias_{arm}"] = biases[arm]
            f.attrs[f"systematics_label_{arm}"] = labels[arm]

        g = f.create_group("projections")
        g.attrs["description"] = (
            "the survey as it sits on the sky: the rotation of the forward model undone, then projected "
            "Lambert azimuthal equal-area about the footprint centroid, one image stack per arm and "
            "version, in tomographic bin order. NaN off the footprint and off the sphere. The plane "
            "coordinates are not degrees and carry no meaningful ticks -- the graticule is the coordinate "
            "system. Plot with plt.imshow(image, extent=extent, origin='lower')"
        )
        g.attrs["projection"] = "lambert azimuthal equal-area"
        g.attrs["center_radec_deg"] = np.array([ra_c, dec_c], dtype=np.float64)
        g.attrs["reso_arcmin"] = args.reso
        g.attrs["smooth_fwhm_arcmin"] = args.smooth_fwhm
        for version in VERSIONS:
            sub = g.create_group(version)
            for arm in ARMS:
                d = sub.create_dataset(arm, data=projections[arm, version], **ds)
                d.attrs["extent"] = np.array(extent, dtype=np.float64)

        footprint_projection.write_graticule(g, grat, **ds)

        # the healpix values behind the images, for anything a picture cannot answer
        m = f.create_group("maps")
        m.attrs["description"] = (
            "galaxy counts per pixel on the footprint pixels of pixels/pix, RING order in the ROTATED "
            "frame of the forward model. 'rate' is the Poisson rate, 'counts' one realization of it at "
            "the seed of the bias fit"
        )
        for version in VERSIONS:
            sub = m.create_group(version)
            for arm in ARMS:
                sub.create_dataset(arm, data=maps[arm][version], **ds)
        m.create_dataset("contamination", data=contamination.astype(np.float32), **ds)
        m.create_dataset("delta", data=delta.astype(np.float32), **ds)

        g = f.create_group("pixels")
        g.attrs["description"] = (
            "healpix RING indices of the survey footprint in the rotated frame, i.e. the base patch. Every "
            "array under maps/ is indexed by these"
        )
        g.attrs["pixel_area_deg2"] = hp.nside2pixarea(n_side, degrees=True)
        g.create_dataset("pix", data=np.arange(n_pix)[mask].astype(np.int64), **ds)

        g = f.create_group("configs")
        g.attrs["description"] = "verbatim copy of the config this file was produced with"
        g.attrs["msfm"] = config_str

    print(f"done, {os.path.getsize(output) / 1e6:.1f} MB", flush=True)


# --- the rotated frame of the forward model ----------------------------------------------------
# The forward model works in the rotated frame of Fig. 4 of arXiv:2511.04681, so a figure that
# shows the survey where it actually is has to undo that rotation. catalog.py owns it; nothing is
# reimplemented here. Same two calls as export_data_vs_mock_maps makes.


def _celestial_footprint(conf, mask, n_side):
    """Celestial (ra, dec) of the footprint pixel centres, for framing the projection."""
    return catalog.survey_pix_to_angles(conf, np.arange(hp.nside2npix(n_side))[mask], n_side)


def _celestial_source_pix(conf, n_side):
    """For every celestial RING pixel, the rotated-frame RING pixel holding its value.

    A gather rather than a scatter: rotating a map by pushing each source pixel to its destination
    leaves holes wherever two sources land on one destination, which reads as speckle.
    """
    ra, dec = hp.pix2ang(n_side, np.arange(hp.nside2npix(n_side)), lonlat=True)
    return catalog.survey_angles_to_pix(conf, ra, dec, n_side)


if __name__ == "__main__":
    main()
