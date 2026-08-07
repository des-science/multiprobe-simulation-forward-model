"""Derive theta_fwhm and white_noise_sigma for a scale-cut config.

Ports the derivation in ``notebooks/scale_cuts.ipynb`` so a new cut does not have to be
re-improvised each time; the previous throwaway version of this
(``scratch/deep_lss/noise_lmax1024.py``, referenced from ``configs/scales/lmax_1024.yaml``)
was lost.

Two cut styles:

* ``--lmax L``   uniform harmonic cut. One Gaussian beam for every z-bin, from
  ``low_pass^2(L) = threshold``.
* ``--mpc A,B``  fixed transverse comoving scale (A Mpc/h lensing, B Mpc/h clustering),
  giving a different beam -- and so a different l_max -- per bin. This is how
  ``8wl,32gc.yaml`` was built.

The noise is flat white noise matching the *smoothed* signal at the cut::

    sigma_i = sqrt(high_pass^2(l_max) * low_pass^2(l_max) * C_l^signal(l_max) / pixarea)

which needs the mean signal power spectrum of the fiducial maps -- an anafast over
``--n-perms`` permutations, a few minutes. Cache it with ``--cache`` to iterate on the
threshold or the beam without paying for it again.

Run it on a compute node (healpy + the projected maps); the login node has neither::

    NAME=scalecut ENV=tf REPO=multiprobe-simulation-forward-model \\
        PAYLOAD=<abs path to this file> ARGS="--lmax 1024" \\
        /users/athomsen/dlss/repos/.claude/bin/run_job.sh

Regression check against the committed ``configs/scales/lmax_1024.yaml`` lensing row, run
2026-08-07 (job 3029171). ``theta_fwhm`` reproduces exactly at 16.96 arcmin; the sigmas agree to
**~1.4%**::

    committed (10 perms):  [1.8074e-04, 3.4800e-04, 5.5118e-04, 6.6987e-04]
    --lmax 1024 (1 perm):  [1.8284e-04, 3.5287e-04, 5.5900e-04, 6.6997e-04]

The residual is the permutation average, not a discrepancy: only ``perm_0000`` still has its maps
on scratch, so the mean C_l comes from one realization instead of ten. Restage more permutations
if you need the sigmas to better than a couple of percent.
"""

import argparse
import os

import astropy.units as u
import h5py
import healpy as hp
import numpy as np
from astropy.cosmology import FlatLambdaCDM

from msfm.utils import clustering, files, redshift, scales

# The projected maps and the healpy pixel weights, as canonical paths: the
# /users/athomsen/scratch symlink does not resolve inside the tensorflow container.
FIDU_DIR = "/iopsstor/scratch/cscs/athomsen/deep_lss/data/projected/fiducial/cosmo_fiducial"
MAP_FILE = "projected_probes_maps_v11dmb.h5"
HP_DATAPATH = "/users/athomsen/dlss/repos/multiprobe-simulation-forward-model/data/healpy_data"
L_MAX_GLOBAL = 1535
L_MIN = 30


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cut = p.add_mutually_exclusive_group(required=True)
    cut.add_argument("--lmax", type=int, help="uniform harmonic cut: one beam for every z-bin")
    cut.add_argument("--mpc", type=str, help="transverse comoving cut as 'lensing,clustering' in Mpc/h, e.g. 8,32")
    p.add_argument("--config", default="configs/v18/default.yaml", help="msfm config (repo-relative or absolute)")
    p.add_argument("--n-perms", type=int, default=10, help="fiducial permutations to average the Cls over")
    p.add_argument("--threshold", type=float, default=0.01, help="low_pass^2 suppression defining the cut")
    p.add_argument("--cache", default=None, help="npz to read/write mean_cls, to skip the anafast on a re-run")
    return p.parse_args()


def load_setup(conf):
    """Fiducial cosmology, tomographic bins and the per-bin IA / bias amplitudes."""
    fid = conf["analysis"]["fiducial"]
    cosmo = FlatLambdaCDM(Om0=fid["Om"], Ob0=fid["Ob"], H0=fid["H0"], Tcmb0=2.7255, Neff=3.046, m_nu=0.02 * u.eV)
    metacal_bins = conf["survey"]["metacal"]["z_bins"]
    maglim_bins = conf["survey"]["maglim"]["z_bins"]
    tomo_z_mc, tomo_nz_mc = files.load_redshift_distributions("metacal", conf)
    tomo_z_ml, tomo_nz_ml = files.load_redshift_distributions("maglim", conf)

    n_side = conf["analysis"]["n_side"]
    tomo_Aia = redshift.get_tomo_amplitudes(
        fid["Aia"], fid["n_Aia"], tomo_z_mc, tomo_nz_mc, conf["survey"]["metacal"]["z0"]
    )
    tomo_bg = [fid[f"bg{i + 1}"] for i in range(len(maglim_bins))]
    n_gal_ml = np.array(conf["survey"]["maglim"]["n_gal"]) * hp.nside2pixarea(n_side, degrees=True)

    z_bar_mc = [np.sum(z * nz) for z, nz in zip(tomo_z_mc, tomo_nz_mc)]
    z_bar_ml = [np.sum(z * nz) for z, nz in zip(tomo_z_ml, tomo_nz_ml)]
    return dict(
        cosmo=cosmo,
        h=fid["H0"] / 100,
        n_side=n_side,
        metacal_bins=metacal_bins,
        maglim_bins=maglim_bins,
        tomo_Aia=tomo_Aia,
        tomo_bg=tomo_bg,
        n_gal_ml=n_gal_ml,
        z_bar=(z_bar_mc, z_bar_ml),
    )


def available_perms(n_perms):
    """The requested permutations that still have their .h5 on scratch.

    Most `perm_*` directories under FIDU_DIR are empty skeletons -- the maps were pruned and
    only perm_0000 survives (checked 2026-08-07). Averaging over fewer permutations is safe
    here: the cut sits at high l, where a full-sky C_l already averages over thousands of
    modes, so the mean is cosmic-variance-limited only at low l which the cut discards. Report
    the count rather than silently changing the estimator.
    """
    found = [i for i in range(n_perms) if os.path.exists(os.path.join(FIDU_DIR, f"perm_{i:04d}", MAP_FILE))]
    if not found:
        raise SystemExit(f"no fiducial maps under {FIDU_DIR}/perm_*/{MAP_FILE} -- restage the data")
    if len(found) < n_perms:
        print(f"WARNING: only {len(found)} of {n_perms} permutations have maps on disk; using {found}")
    return found


def mean_signal_cls(setup, perms):
    """Mean C_l of the fiducial signal maps: kg + Aia*ia for lensing, galaxy counts for clustering.

    This is the expensive step (~30 s per permutation); everything else is arithmetic.
    """
    n_side = setup["n_side"]
    n_z = len(setup["metacal_bins"]) + len(setup["maglim_bins"])
    raw_cls = np.zeros((len(perms), L_MAX_GLOBAL + 1, n_z))

    for i, perm in enumerate(perms):
        path = os.path.join(FIDU_DIR, f"perm_{perm:04d}", MAP_FILE)
        with h5py.File(path, "r") as f:
            kg = np.stack([hp.ud_grade(f[f"map/kg/{b}"], n_side) for b in setup["metacal_bins"]], axis=-1)
            ia = np.stack([hp.ud_grade(f[f"map/ia/{b}"], n_side) for b in setup["metacal_bins"]], axis=-1)
            wl = kg + setup["tomo_Aia"] * ia

            dg = np.stack([hp.ud_grade(f[f"map/dg/{b}"], n_side) for b in setup["maglim_bins"]], axis=-1)
            gc = clustering.galaxy_density_to_count(
                setup["n_gal_ml"], (dg - np.mean(dg)) / np.mean(dg), setup["tomo_bg"]
            )
        fidu_map = np.concatenate([wl, gc], axis=-1)
        for j in range(n_z):
            raw_cls[i, :, j] = hp.anafast(
                fidu_map[:, j], lmax=L_MAX_GLOBAL, pol=False, datapath=HP_DATAPATH, use_pixel_weights=True
            )
        print(f"  perm {perm:04d} done ({i + 1}/{len(perms)})", flush=True)

    return np.mean(raw_cls, axis=0)


def beams_from_mpc(setup, mpc_lensing, mpc_clustering):
    """Per-bin FWHM in arcmin for a fixed transverse comoving scale (the 8wl,32gc style)."""
    z_bar_mc, z_bar_ml = setup["z_bar"]
    out = []
    for mpc, z_bar in ((mpc_lensing, z_bar_mc), (mpc_clustering, z_bar_ml)):
        theta = (mpc * u.Mpc / setup["h"]) / setup["cosmo"].comoving_distance(z_bar)
        out += list(scales.rad_to_arcmin(theta.value))
    return np.array(out)


def beam_from_lmax(lmax, threshold):
    """The single FWHM whose low_pass^2 hits `threshold` exactly at `lmax`.

    low_pass^2(l) = exp(-l(l+1) sigma^2) with sigma = fwhm / (2 sqrt(2 ln 2)), so
    inverting at l = lmax is closed-form -- no search needed.
    """
    sigma = np.sqrt(-np.log(threshold) / (lmax * (lmax + 1)))
    return scales.rad_to_arcmin(sigma * 2 * np.sqrt(2 * np.log(2)))


def derive(mean_cls, fwhms, setup, threshold):
    """Per-bin (l_max, sigma) for the given beams, following scale_cuts.ipynb cell 9."""
    ell = np.arange(0, L_MAX_GLOBAL + 1)
    high_pass = scales.gaussian_high_pass_factor_alm(ell, l_min=L_MIN) ** 2
    pixarea = hp.nside2pixarea(setup["n_side"])

    l_maxs, sigmas = [], []
    for i, fwhm in enumerate(fwhms):
        low_pass = scales.gaussian_low_pass_factor_alm(ell, theta_fwhm=fwhm, arcmin=True) ** 2
        below = ell[low_pass < threshold]
        l_max = int(below[0]) if len(below) else L_MAX_GLOBAL
        cl_noise = high_pass[l_max] * low_pass[l_max] * mean_cls[l_max, i]
        l_maxs.append(l_max)
        sigmas.append(np.sqrt(cl_noise / pixarea))
    return np.array(l_maxs), np.array(sigmas)


def main():
    args = parse_args()
    conf = files.load_config(args.config)
    setup = load_setup(conf)
    n_mc = len(setup["metacal_bins"])

    if args.cache and os.path.exists(args.cache):
        print(f"mean_cls from cache: {args.cache}")
        mean_cls = np.load(args.cache)["mean_cls"]
    else:
        perms = available_perms(args.n_perms)
        print(f"computing mean_cls over {len(perms)} permutation(s) (anafast, ~30 s each)...")
        mean_cls = mean_signal_cls(setup, perms)
        if args.cache:
            np.savez(args.cache, mean_cls=mean_cls)
            print(f"cached to {args.cache}")

    if args.lmax:
        fwhm = beam_from_lmax(args.lmax, args.threshold)
        fwhms = np.full(mean_cls.shape[1], fwhm)
        label = f"uniform l_max = {args.lmax}"
    else:
        mpc_l, mpc_c = (float(x) for x in args.mpc.split(","))
        fwhms = beams_from_mpc(setup, mpc_l, mpc_c)
        label = f"{mpc_l:g} Mpc/h lensing, {mpc_c:g} Mpc/h clustering"

    l_maxs, sigmas = derive(mean_cls, fwhms, setup, args.threshold)

    fmt = {"float": "{:0.4e}".format}
    print(f"\n{label}   (threshold = {args.threshold:.1%}, n_side = {setup['n_side']})")
    for name, sl in (("lensing", slice(None, n_mc)), ("clustering", slice(n_mc, None))):
        with np.printoptions(formatter=fmt):
            print(f"\n  {name}:")
            print(f"    theta_fwhm: {np.round(fwhms[sl], 2).tolist()}")
            print(f"    white_noise_sigma: {sigmas[sl]}")
            # The threshold CROSSING, one above the requested cut for a --lmax run: the config's
            # own l_max field carries the requested value (1024), the crossing (1025) the comment.
            print(f"    l_max: {l_maxs[sl].tolist()}   # threshold crossing")
            print(f"    l_min: {[L_MIN] * len(l_maxs[sl])}")

    print(
        "\nSanity-check against configs/scales/8wl,32gc.yaml: at a higher l_max than a bin's"
        "\n8wl,32gc l_max there is less smoothing, so sigma should come out LOWER, and vice versa."
    )


if __name__ == "__main__":
    main()
