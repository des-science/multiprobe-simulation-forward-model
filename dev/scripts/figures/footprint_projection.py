# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created August 2026
Author: Arne Thomsen

The DES footprint on the page, the way every paper_2 sky map does it: Lambert azimuthal equal-area
about the footprint centroid, cropped to the data, with a celestial graticule exported next to the
image.

Shared by the export scripts in this directory, so that two figures of the same survey come out
the same shape and can be read against each other. Equal area, because these panels are read as
"how much sky, and what is in it"; azimuthal about the centroid, because a 5000 deg^2 wedge 50 deg
off the equator is badly served by any cylindrical projection -- plate carree stretches its
southern half by a factor cos(dec) and turns the footprint into a different shape than it is.

An azimuthal projection has no meaningful numeric axes, which is what :func:`graticule` is for: the
RA / Dec grid *is* the coordinate system of the panel. It is exported with the image and drawn over
it by ``dlss_plot.maps.graticule``. (The plotting package can project a *direction* itself --
``dlss_plot.hierarchy.footprint_frame`` rebuilds this plane from what is stored beside the image --
but the image is this module's job: it needs the rotated frame and a full-sky resampling.)

Nothing here knows about the forward model. Every function takes celestial coordinates and a
healpix map in *celestial* RING order; undoing the rotated frame the forward model works in is the
caller's job, see ``export_data_vs_mock_maps.celestial_source_pix``.
"""

import numpy as np

from msfm.utils import imports

hp = imports.import_healpy()

#: healpy marks unseen pixels with this value, and the projectors fill the area outside of the
#: sphere with it as well.
UNSEEN_THRESHOLD = -1e30


def center(ra, dec):
    """Celestial ``(ra, dec)`` of the centroid of a set of directions, in degrees.

    Averaged as unit vectors, which is what makes it right across the RA = 0 wrap the DES
    footprint straddles.
    """
    vec = np.asarray(hp.ang2vec(ra, dec, lonlat=True)).mean(axis=0)
    vec /= np.linalg.norm(vec)
    ra_c, dec_c = hp.vec2ang(vec, lonlat=True)
    # vec2ang always returns arrays; healpy's Rotator rejects a rot tuple holding them
    return float(ra_c[0]), float(dec_c[0])


def projector(ra, dec, reso, margin_deg):
    """A Lambert azimuthal equal-area projector framing everything at ``ra``, ``dec``.

    The plane coordinates of a point depend only on the projection centre, so the frame is sized
    by projecting the footprint first and then asking for enough pixels to hold it.

    Args:
        ra, dec: celestial coordinates of the footprint pixels, in degrees.
        reso (float): arcmin per pixel of the output image.
        margin_deg (float): blank margin around the footprint, in degrees of great circle.

    Returns:
        (projector, (ra_center_deg, dec_center_deg))
    """
    ra_c, dec_c = center(ra, dec)
    vec = np.asarray(hp.ang2vec(ra, dec, lonlat=True)).T

    def make(xsize, ysize):
        return hp.projector.AzimuthalProj(
            rot=(ra_c, dec_c, 0.0), lamb=True, xsize=int(xsize), ysize=int(ysize), reso=reso
        )

    probe = make(1000, 1000)
    x, y = probe.vec2xy(vec[0], vec[1], vec[2])
    x0, x1, _, _ = probe.get_extent()
    per_pixel = (x1 - x0) / 1000.0

    # the margin is in degrees of great circle; near the centre the Lambert plane is radians
    margin = np.radians(margin_deg)
    half_x = max(abs(np.nanmin(x)), abs(np.nanmax(x))) + margin
    half_y = max(abs(np.nanmin(y)), abs(np.nanmax(y))) + margin

    return make(2 * half_x / per_pixel, 2 * half_y / per_pixel), (float(ra_c), float(dec_c))


def project(celestial_map, n_side, proj):
    """One healpix map, in celestial RING order, as an image. NaN stays NaN.

    Args:
        celestial_map (n_pix,): full sky RING map, NaN (or UNSEEN) outside the footprint.
        n_side (int): its nside.
        proj: from :func:`projector`.

    Returns:
        (ny, nx) float32, NaN off the footprint and off the sphere.
    """

    def vec2pix(x, y, z):
        return hp.vec2pix(n_side, x, y, z)

    img = np.asarray(proj.projmap(np.asarray(celestial_map, dtype=np.float64), vec2pix), dtype=np.float32)
    return np.where(img < UNSEEN_THRESHOLD, np.nan, img)


def crop_to_data(images, extent, margin_pix=8):
    """Trim the all-NaN border off a projected image stack, carrying the extent with it.

    The projector frames a rectangle around the footprint centre; the footprint is not a
    rectangle, so a tight crop is what keeps the panel from being mostly empty. Crop a stack in
    one call rather than image by image -- separate crops would silently shift two panels of the
    same sky against each other if their footprints ever stopped agreeing.

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


def footprint_graticule(proj, ra, dec, ra_center, step_deg, **kwargs):
    """:func:`graticule`, spanning where the footprint actually is.

    The RA range is measured as a signed offset from the projection centre, so that a footprint
    straddling RA = 0 gets one continuous range instead of the [0, 360) one its raw coordinates
    would suggest.
    """
    d_ra = (np.asarray(ra) - ra_center + 180.0) % 360.0 - 180.0
    return graticule(
        proj,
        (ra_center + d_ra.min(), ra_center + d_ra.max()),
        (np.min(dec), np.max(dec)),
        step_deg,
        **kwargs,
    )


def write_graticule(group, grat, **kwargs):
    """Write the graticule into an open h5 file, under the names the plotting package reads.

    An export owns its own layout, but the frame is read by shared code
    (``dlss_plot.maps.graticule``), so these four datasets always live in a ``graticule`` subgroup
    and the image's ``extent`` always sits on the image dataset itself.
    """
    sub = group.create_group("graticule")
    sub.attrs["description"] = (
        "celestial grid lines in the same plane coordinates as the images: one polyline per line, NaN "
        "where it falls behind the projection. *_values carry the RA / Dec each line is at, in degrees, "
        "for the labels"
    )
    for key, value in grat.items():
        sub.create_dataset(key, data=np.asarray(value, dtype=np.float64), **kwargs)
    return sub
