# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created July 2026
Author: Arne Thomsen

Makes the "forward model maps" figure from the self contained .h5 file written by
export_map_figure_data.py. This script only needs numpy, h5py and matplotlib, so it runs anywhere,
no healpy, no tensorflow, no cluster.

    python plot_map_figure.py desy3_forward_model_maps.h5 -o forward_model_maps.pdf

Everything that is projected is already a 2d image inside the .h5 file, so the plot is a handful of
imshow calls. The sparse healpix maps are in there as well (see maps/ and pixels/), in case you want
to reproject them yourself.
"""

import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, Ellipse, Polygon


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="the .h5 file written by export_map_figure_data.py")
    parser.add_argument("-o", "--output", default="forward_model_maps.pdf", help="output figure path")
    parser.add_argument(
        "--maps", default="smoothed", choices=["smoothed", "raw"], help="which version of the maps to plot"
    )
    parser.add_argument(
        "--scaling",
        default="per_panel",
        choices=["per_panel", "per_probe"],
        help="whether the color scale is shared between the tomographic bins of a probe",
    )
    parser.add_argument("--percentiles", type=float, nargs=2, default=[1.0, 99.0], help="color scale percentiles")
    parser.add_argument("--cmaps", nargs=2, default=["viridis", "magma"], help="colormap per probe")
    parser.add_argument("--sphere_color", default="0.55", help="color of the part of the sky outside the footprint")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


PROBE_TITLES = {"lensing": "Weak Lensing", "clustering": "Galaxy Clustering"}


def color_limits(images, percentiles):
    """Robust color limits over one or several images, ignoring the NaN padding."""
    values = np.concatenate([img[np.isfinite(img)].ravel() for img in images])
    return np.percentile(values, percentiles)


def plot_mollweide(ax, image, extent, cmap, vmin, vmax, outline_xy, sphere_color):
    """Full sky panel: a gray sphere with the survey footprint painted on top."""
    ax.add_patch(Ellipse((0, 0), width=4, height=2, facecolor=sphere_color, edgecolor="k", lw=1.5, zorder=0))
    ax.imshow(image, extent=extent, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, zorder=1, interpolation="nearest")
    # the sphere outline again, so that it is not covered by the map
    ax.add_patch(Ellipse((0, 0), width=4, height=2, facecolor="none", edgecolor="k", lw=1.5, zorder=3))
    # where the cutouts below are taken from
    ax.add_patch(Polygon(outline_xy, closed=True, facecolor="none", edgecolor="k", lw=1.0, zorder=4))

    ax.set(xlim=(-2.05, 2.05), ylim=(-1.05, 1.05), aspect="equal")
    ax.axis("off")


def plot_cutout(ax, image, extent, cmap, vmin, vmax, label):
    ax.imshow(image, extent=extent, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.set_title(label, fontsize=20, pad=8)


def main():
    args = parse_args()

    with h5py.File(args.input, "r") as f:
        probes = [p for p in f.attrs["probes"]]
        n_z = {probe: int(f.attrs[f"n_z_{probe}"]) for probe in probes}
        labels = [str(s) for s in f.attrs["channel_labels"]]

        group = f[f"projections/{args.maps}"]
        moll = group["mollweide"][:]
        moll_extent = group["mollweide"].attrs["extent"]
        gnom = group["gnomonic"][:]
        gnom_extent = group["gnomonic"].attrs["extent_arcmin"] / 60.0  # deg
        outline_xy = f["projections/gnomonic_outline_mollweide_xy"][:]

        cutout_deg = float(f["projections"].attrs["gnomonic_size_deg"])
        area = float(f["pixels"].attrs["footprint_area_deg2"])

    # channel slices per probe, the channels are ordered [lensing bins, clustering bins]
    slices, start = {}, 0
    for probe in probes:
        slices[probe] = slice(start, start + n_z[probe])
        start += n_z[probe]

    n_cols = sum(n_z.values())
    fig = plt.figure(figsize=(4.0 * n_cols, 13.0))
    grid = fig.add_gridspec(
        nrows=2, ncols=n_cols, height_ratios=[1.6, 1.0], hspace=0.22, wspace=0.12, left=0.02, right=0.98
    )

    for i_probe, probe in enumerate(probes):
        sl = slices[probe]
        cmap = args.cmaps[i_probe]
        columns = range(sl.start, sl.stop)

        if args.scaling == "per_probe":
            limits = [color_limits(gnom[sl], args.percentiles)] * n_z[probe]
        else:
            limits = [color_limits([gnom[i]], args.percentiles) for i in columns]

        # full sky panel of the first tomographic bin, spanning the columns of this probe
        ax_moll = fig.add_subplot(grid[0, sl])
        plot_mollweide(
            ax_moll,
            moll[sl.start],
            moll_extent,
            cmap,
            *limits[0],
            outline_xy,
            args.sphere_color,
        )
        ax_moll.set_title(PROBE_TITLES.get(probe, probe), fontsize=30, pad=2)
        ax_moll.text(
            0.62, 0.5, labels[sl.start], transform=ax_moll.transAxes, fontsize=34, ha="center", va="center", zorder=5
        )

        # the tomographic bins as gnomonic cutouts
        cutout_axes = []
        for i_bin, i_channel in enumerate(columns):
            ax = fig.add_subplot(grid[1, i_channel])
            plot_cutout(ax, gnom[i_channel], gnom_extent, cmap, *limits[i_bin], labels[i_channel])
            cutout_axes.append(ax)

        # dashed lines from the box on the sphere to the outermost cutouts
        box_left = outline_xy[np.argmin(outline_xy[:, 0])]
        box_right = outline_xy[np.argmax(outline_xy[:, 0])]
        box_bottom = outline_xy[np.argmin(outline_xy[:, 1])]
        for box_xy, ax, corner in [
            (box_left, cutout_axes[0], (gnom_extent[0], gnom_extent[3])),
            (box_bottom, cutout_axes[0], (gnom_extent[1], gnom_extent[3])),
            (box_bottom, cutout_axes[-1], (gnom_extent[0], gnom_extent[3])),
            (box_right, cutout_axes[-1], (gnom_extent[1], gnom_extent[3])),
        ]:
            fig.add_artist(
                ConnectionPatch(
                    xyA=tuple(box_xy),
                    coordsA=ax_moll.transData,
                    xyB=corner,
                    coordsB=ax.transData,
                    lw=0.8,
                    ls="--",
                    color="0.35",
                    zorder=0,
                )
            )

    fig.suptitle(
        f"DES Y3 like forward model, {area:.0f} deg$^2$ footprint, {cutout_deg:.0f}$^\\circ$ cutouts",
        fontsize=16,
        y=0.06,
    )

    fig.savefig(args.output, bbox_inches="tight", dpi=args.dpi)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
