# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created August 2026
Author: Arne Thomsen

Projects the DES Y3 imaging-systematics decontamination (ISD) weight maps onto the sky for the
paper_2 figure, one image per metacal tomographic bin.

The input is the tracked deliverable of the lss_sys pipeline,
data/desy3_metacal_systematics_STD32_512.h5, which is the same file the forward model reads: W is
a per pixel multiplicative correction with unit mean over the footprint, and n_corrected = n_obs * W.

NOTE the figure shows the *weight*, not the contamination. The forward model consumes
`contamination` = <1/W>, because it imposes the imprint on a clean simulation rather than removing
it from the data, and at nside 512 that is not 1/W (fracdet weighted averaging does not commute
with inversion; they differ by up to 12.7% in bin 3). W is what the DES Y3 systematics papers
plot and what the templates were fitted to, so W is what belongs in a figure about the correction.

Its footprint is the DES Y3 gold joint mask -- ~4143 deg^2, larger than the forward model's own
footprint, which is the intersection of the per bin metacal and maglim masks. The projection is
therefore derived from this file's own pixels, not from the forward model's.

Cheap: no spherical harmonics, a handful of nside 512 maps. A login node is fine.

    OMP_NUM_THREADS=4 ~/dlss/torch_env/bin/python3 dev/scripts/figures/export_isd_correction.py

The companion export for the lower panel of the same figure -- the template fits behind these maps
-- lives in the repo that owns them, lss_sys/scripts/metacal_sources/export_paper_fits.py.
"""

import argparse
import os
import subprocess

import h5py
import numpy as np

from msfm.utils import imports

# sibling module in this directory, shared with the other figure exports: the footprint projection
# and its graticule, so that every paper_2 sky map is the same projection
import footprint_projection

hp = imports.import_healpy()

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

# the weight W, not the contamination <1/W> the forward model consumes, see the module docstring
DATASET = "weight"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--systematics_file",
        default=os.path.join(REPO_DIR, "data/desy3_metacal_systematics_STD32_512.h5"),
        help="the lss_sys deliverable, as the forward model reads it",
    )
    parser.add_argument("--reso", type=float, default=5.0, help="panel resolution in arcmin per pixel")
    parser.add_argument("--margin", type=float, default=2.0, help="blank margin around the footprint in deg")
    parser.add_argument("--graticule_step", type=float, default=20.0, help="spacing of the RA/Dec grid lines in deg")
    parser.add_argument("--output", default=None, help="path of the .h5 to write; defaults to the plotting cache")
    return parser.parse_args()


def default_output(systematics_file):
    """paper_2_plotting/cache/isd_weight_maps_<label>.h5."""
    label = os.path.splitext(os.path.basename(systematics_file))[0].replace("desy3_", "")
    name = f"isd_{DATASET}_maps_{label}.h5"
    return os.path.abspath(os.path.join(REPO_DIR, "../deep_lss_paper/paper_2_plotting/cache", name))


def git_hash(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def load_bins(path):
    """The per bin maps and the footprint they share.

    Returns:
        (values (n_bins, n_footprint_pix), footprint pixel indices (RING), nside, file attrs,
        per bin attrs)
    """
    with h5py.File(path, "r") as f:
        attrs = {key: _decode(value) for key, value in f.attrs.items()}
        n_bins = int(attrs["n_bins"])
        n_side = int(attrs["nside"])
        assert attrs["ordering"] == "RING", f"expected RING ordering, got {attrs['ordering']}"
        assert attrs["off_footprint"] == "zero", "the footprint is derived from the zeros of the maps"

        maps = np.stack([f[f"bin{b}"][DATASET][:] for b in range(n_bins)])
        bin_attrs = [
            {key: _decode(value) for key, value in f[f"bin{b}"][DATASET].attrs.items()} for b in range(n_bins)
        ]

    # every bin shares one footprint -- the joint gold mask -- and this asserts it rather than
    # assuming it, because a per bin footprint would make one colour scale meaningless
    footprint = maps[0] != 0.0
    for b, m in enumerate(maps):
        assert np.array_equal(m != 0.0, footprint), f"bin {b} has a different footprint than bin 0"

    pix = np.flatnonzero(footprint)
    return maps[:, pix], pix, n_side, attrs, bin_attrs


def _decode(value):
    if isinstance(value, bytes):
        return value.decode()
    return value


def main():
    args = parse_args()
    output = args.output or default_output(args.systematics_file)

    print(f"reading {args.systematics_file}", flush=True)
    values, pix, n_side, attrs, bin_attrs = load_bins(args.systematics_file)
    n_bins = values.shape[0]
    area = len(pix) * hp.nside2pixarea(n_side, degrees=True)
    print(f"{len(pix)} footprint pixels at nside {n_side}, {area:.0f} deg^2", flush=True)
    for b in range(n_bins):
        v = values[b]
        print(
            f"  bin {b}: mean {v.mean():.4f}, rms {v.std():.1%}, range [{v.min():.3f}, {v.max():.3f}], "
            f"{bin_attrs[b].get('applied_corrections', '')}",
            flush=True,
        )

    # ---------------------------------------------------------------------------------- project
    ra, dec = hp.pix2ang(n_side, pix, lonlat=True)
    proj, (ra_c, dec_c) = footprint_projection.projector(ra, dec, args.reso, args.margin)
    print(f"projection centred on (ra, dec) = ({ra_c:.2f}, {dec_c:.2f}) deg", flush=True)

    n_pix = hp.nside2npix(n_side)
    images = []
    for b in range(n_bins):
        full = np.full(n_pix, np.nan)
        full[pix] = values[b]
        images.append(footprint_projection.project(full, n_side, proj))

    # cropped as one stack, so all four bins land on the same frame
    images, extent = footprint_projection.crop_to_data(np.stack(images), proj.get_extent())
    print(f"panel is {images.shape[1:]} pixels", flush=True)

    grat = footprint_projection.footprint_graticule(proj, ra, dec, ra_c, args.graticule_step)
    print(
        f"graticule: RA {np.array2string(grat['meridian_values'], precision=0)}, "
        f"Dec {np.array2string(grat['parallel_values'], precision=0)}",
        flush=True,
    )

    # ------------------------------------------------------------------------------------ write
    ds = {"compression": "gzip", "compression_opts": 4}
    os.makedirs(os.path.dirname(output), exist_ok=True)
    print(f"writing {output}", flush=True)

    with h5py.File(output, "w") as f:
        f.attrs["description"] = (
            "the DES Y3 imaging-systematics correction on the sky, one image per metacal tomographic bin, "
            "for the paper_2 ISD figure. Produced by msfm/dev/scripts/figures/export_isd_correction.py "
            "from the lss_sys deliverable it names below"
        )
        f.attrs["dataset"] = DATASET
        f.attrs["source_file"] = os.path.abspath(args.systematics_file)
        f.attrs["msfm_git_sha"] = git_hash(REPO_DIR)
        f.attrs["n_bins"] = n_bins
        f.attrs["n_side"] = n_side
        f.attrs["footprint_area_deg2"] = area
        f.attrs["bin_labels"] = [f"metacal{b + 1}" for b in range(n_bins)]
        # the rms of the correction per bin: the one number that says how big it is, and the
        # caption's, computed here so the notebook never re-derives it from a different footprint
        f.attrs["rms"] = values.std(axis=1)
        f.attrs["applied_corrections"] = [bin_attrs[b].get("applied_corrections", "") for b in range(n_bins)]
        # everything the deliverable says about itself, so a figure can be traced without it
        for key in ("git_sha", "label", "normalisation", "footprint", "source_run_dir", "status", "thresholds"):
            if key in attrs:
                f.attrs[f"lss_sys_{key}"] = attrs[key]

        g = f.create_group("projections")
        g.attrs["description"] = (
            "Lambert azimuthal equal-area about the footprint centroid, NaN off the footprint and off the "
            "sphere. The plane coordinates are not degrees and carry no meaningful ticks -- the graticule "
            "is the coordinate system. Plot with plt.imshow(image, extent=extent, origin='lower')"
        )
        g.attrs["projection"] = "lambert azimuthal equal-area"
        g.attrs["center_radec_deg"] = np.array([ra_c, dec_c], dtype=np.float64)
        g.attrs["reso_arcmin"] = args.reso

        d = g.create_dataset(DATASET, data=images, **ds)
        d.attrs["extent"] = np.array(extent, dtype=np.float64)

        footprint_projection.write_graticule(g, grat, **ds)

    print("done", flush=True)


if __name__ == "__main__":
    main()
