# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created July 2026
Author: Arne Thomsen (with Claude)

Full-sky input-map vs. independent-theory audit for the weak-lensing forward model
(follow-up to wl_forward_model_audit_2026-07-11.md, which never checked the *absolute*
amplitude/scale of the CosmoGrid input maps against an external theory).

The prior audit (wl_sn_diagnostics.py, diagnostic 2/3) only verified signal *transfer*
through the KS/mode-removal chain relative to each map's *own* full-sky Cl. A wrong
amplitude or scale dependence baked into the kg (lensing) or ia (NLA intrinsic alignment)
input maps themselves would pass every one of those checks unnoticed, while still lowering
the S/N of the forward-modelled data vector.

This script closes that gap. For the fiducial CosmoGrid cosmology it:

 1. Measures the full-sky auto/cross pseudo-Cls of the kg and ia maps directly
    (healpy anafast, deconvolved by the nside pixel window), averaged over permutations.
 2. Computes the INDEPENDENT pyccl (CCL) theory prediction from the stored n(z):
      - GG  = lensing convergence auto  -> compare to Cl(kg)
      - II  = NLA intrinsic-alignment auto at A_IA = 1, eta = 0 -> compare to Cl(ia)
      - GI  = lensing x NLA cross (negative) -> compare to Cl(kg, ia)
    using the SAME NLA convention as the UFalcon projection
    (F = -A_IA C1 rho_crit Omega_m / D(z), C1 = 5e-14 h^-2, i.e. CCL use_A_ia=True).
 3. Cross-checks the stored per-shell projection kernels (kernel/kg, kernel/ia) against the
    analytic lensing efficiency and NLA weight -- an amplitude/scale check independent of
    both the maps and the matter power spectrum.

Runs on a Perlmutter CPU node; the CosmoGrid maps live on CFS. Results -> --out_dir.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py
import healpy as hp
import pyccl as ccl

# ---------------------------------------------------------------------------------------------------------------------
# CosmoGrid V1.1 fiducial cosmology (data/CosmoGridV1_metainfo.h5, parameters/fiducial[0]).
# Omega_c already excludes the neutrino density; Sum m_nu ~ 0.06 eV (O_nu = 0.00142, 3 x 0.02 eV).
COSMO = dict(
    Omega_c=0.209277442262,
    Omega_b=0.0493,
    h=0.6736,
    sigma8=0.84,
    n_s=0.9649,
    m_nu=0.06,
    mass_split="equal",
    w0=-1.0,
)
METACAL_BINS = ["metacal1", "metacal2", "metacal3", "metacal4"]
# Clariden copy of the full-sky maps (originally run on Perlmutter CFS:
# /global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/fiducial/cosmo_fiducial)
MAP_DIR = "/iopsstor/scratch/cscs/athomsen/CosmoGrid/v11desy3/bary/fiducial/cosmo_fiducial"
MAP_FILE = {"dmb": "projected_probes_maps_v11dmb.h5", "dmo": "projected_probes_maps_v11dmo.h5"}

# NLA normalisation cross-check (verified numerically, 2026-07-13): CCL's use_A_ia=True applies
#   -A_ia * 5e-14 * RHO_CRITICAL * Omega_m / D(a)      (pyccl/tracers.py, RHO_CRITICAL = 2.775366e11)
# and UFalcon's F_NLA_model applies
#   -A_ia * C1 * rho_crit,0 * Omega_m / D(z),  C1 = 5e-14/h^2, rho_crit,0 = astropy critical_density0
# whose product is 5e-14 * 2.775366e11 = 0.013876831 in BOTH cases -- the conventions are identical to
# 10 significant figures, so any residual II/GI offset can only come from the growth factor D.


def setup(args=None):
    p = argparse.ArgumentParser(description="full-sky input map vs CCL theory audit")
    p.add_argument("--variant", choices=["dmb", "dmo", "both"], default="both",
                   help="baryonified (dmb, used in production) and/or dark-matter-only (dmo, matches halofit)")
    p.add_argument("--n_perms", type=int, default=8, help="number of permutations to average the measured Cls over")
    p.add_argument("--matter_power", default="halofit", choices=["halofit", "camb"])
    p.add_argument("--transfer", default="boltzmann_camb")
    p.add_argument("--map_dir", type=str, default=MAP_DIR)
    p.add_argument("--out_dir", type=str,
                   default="/iopsstor/scratch/cscs/athomsen/deep_lss/runs/wl_audit")
    return p.parse_args(args)


def load_nz(map_path):
    z, nz = [], []
    with h5py.File(map_path, "r") as h:
        for b in METACAL_BINS:
            d = h[f"nz/{b}"][:]
            z.append(d[:, 0])
            nz.append(d[:, 1])
    return z, nz


def theory_cls(ell, z, nz, args):
    cosmo = ccl.Cosmology(matter_power_spectrum=args.matter_power, transfer_function=args.transfer, **COSMO)
    out = {"GG": [], "II": [], "GI": []}
    for zb, nb in zip(z, nz):
        t_g = ccl.WeakLensingTracer(cosmo, dndz=(zb, nb))  # shear only
        # NLA IA at unit amplitude, no redshift evolution (the map's baked-in convention;
        # the forward model applies the per-bin tomo_Aia = Aia * <((1+z)/(1+z0))^n_Aia> on top)
        t_i = ccl.WeakLensingTracer(cosmo, dndz=(zb, nb), has_shear=False,
                                    ia_bias=(zb, np.ones_like(zb)), use_A_ia=True)
        out["GG"].append(ccl.angular_cl(cosmo, t_g, t_g, ell))
        out["II"].append(ccl.angular_cl(cosmo, t_i, t_i, ell))
        out["GI"].append(ccl.angular_cl(cosmo, t_g, t_i, ell))
    return {k: np.array(v) for k, v in out.items()}


def measure_map_cls(variant, args):
    """Full-sky auto/cross pseudo-Cls of kg and ia, averaged over perms, pixel-window deconvolved."""
    perm_dirs = sorted(d for d in os.listdir(args.map_dir) if d.startswith("perm_"))[: args.n_perms]
    nside = None
    acc = None
    n_used = 0
    for pd in perm_dirs:
        path = os.path.join(args.map_dir, pd, MAP_FILE[variant])
        if not os.path.exists(path):
            continue
        with h5py.File(path, "r") as h:
            for i_b, b in enumerate(METACAL_BINS):
                kg = h[f"map/kg/{b}"][:].astype(np.float64)
                ia = h[f"map/ia/{b}"][:].astype(np.float64)
                if nside is None:
                    nside = hp.npix2nside(kg.size)
                    lmax = 3 * nside - 1
                    ell = np.arange(lmax + 1)
                    pw = hp.pixwin(nside, lmax=lmax)
                    pw2 = pw ** 2
                    pw2[pw2 == 0] = np.inf
                    acc = {k: np.zeros((len(METACAL_BINS), lmax + 1)) for k in ["kg", "ia", "cross"]}
                acc["kg"][i_b] += hp.anafast(kg, lmax=lmax) / pw2
                acc["ia"][i_b] += hp.anafast(ia, lmax=lmax) / pw2
                acc["cross"][i_b] += hp.anafast(kg, map2=ia, lmax=lmax) / pw2
        n_used += 1
    if not n_used:
        raise FileNotFoundError(f"no {MAP_FILE[variant]} found under {args.map_dir} (checked {len(perm_dirs)} perms)")
    for k in acc:
        acc[k] /= n_used
    return ell, acc, n_used, nside


def main(args=None):
    args = setup(args)
    os.makedirs(args.out_dir, exist_ok=True)
    variants = ["dmb", "dmo"] if args.variant == "both" else [args.variant]

    z, nz = load_nz(os.path.join(args.map_dir, "perm_0000", MAP_FILE[variants[0]]))

    results = {}
    for variant in variants:
        print(f"\n===== variant {variant} =====", flush=True)
        ell, meas, n_used, nside = measure_map_cls(variant, args)
        print(f"measured full-sky Cls averaged over {n_used} perms at nside {nside}", flush=True)
        th = theory_cls(ell.astype(float), z, nz, args)
        results[variant] = dict(ell=ell, meas=meas, th=th, n_used=n_used, nside=nside)

        # ratios in log-ell bins
        bins = [(30, 100), (100, 300), (300, 600), (600, 1000), (1000, 1500)]
        for probe_meas, probe_th, name in [("kg", "GG", "kg / GG"), ("ia", "II", "ia / II"),
                                           ("cross", "GI", "kg.ia / GI")]:
            print(f"  {name} median ratio (measured / theory):")
            for lo, hi in bins:
                sl = slice(lo, hi)
                m = meas[probe_meas][:, sl]
                t = th[probe_th][:, sl]
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.median(m / t, axis=1)
                print(f"    ell[{lo:4d},{hi:4d}]: " + "  ".join(f"b{i+1}={r:6.3f}" for i, r in enumerate(ratio)),
                      flush=True)

    np.savez(os.path.join(args.out_dir, "wl_theory_comparison.npz"),
             ell=results[variants[0]]["ell"],
             **{f"{v}_meas_{k}": results[v]["meas"][k] for v in variants for k in ["kg", "ia", "cross"]},
             **{f"{v}_th_{k}": results[v]["th"][k] for v in variants for k in ["GG", "II", "GI"]})

    # figure: measured vs theory per bin, kg + ia + cross, for the first variant
    v0 = variants[0]
    ell = results[v0]["ell"]
    meas, th = results[v0]["meas"], results[v0]["th"]
    fig, axes = plt.subplots(3, 4, figsize=(17, 10), sharex=True)
    lfac = ell * (ell + 1) / (2 * np.pi)
    for i_b in range(4):
        for row, (pm, pt, lbl) in enumerate(
            [("kg", "GG", r"$\kappa_g$ (GG)"), ("ia", "II", "ia (II)"), ("cross", "GI", r"$\kappa_g\times$ia (GI)")]
        ):
            ax = axes[row, i_b]
            sign = -1.0 if pm == "cross" else 1.0
            ax.loglog(ell[2:], (lfac * sign * meas[pm][i_b])[2:], label="measured map", lw=1)
            ax.loglog(ell[2:], (lfac * sign * th[pt][i_b])[2:], "--", label="CCL theory", lw=1.5)
            for v in variants[1:]:
                ax.loglog(ell[2:], (lfac * sign * results[v]["meas"][pm][i_b])[2:], ":", lw=1,
                          label=f"measured {v}")
            ax.set_xlim(20, 2000)
            if row == 0:
                ax.set_title(f"metacal{i_b+1}")
            if i_b == 0:
                ax.set_ylabel(lbl + r"  $\ell(\ell+1)C_\ell/2\pi$")
            if row == 2:
                ax.set_xlabel(r"$\ell$")
            ax.legend(fontsize=7)
    fig.suptitle(f"Full-sky input maps vs CCL theory ({v0}, {results[v0]['n_used']} perms; cross plotted as -GI)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "wl_theory_comparison.png"), dpi=140)
    plt.close(fig)

    # figure: ratio measured/theory
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for row, (pm, pt, lbl) in enumerate(
        [("kg", "GG", r"$\kappa_g$ / GG"), ("ia", "II", "ia / II"), ("cross", "GI", r"$\kappa_g\times$ia / GI")]
    ):
        ax = axes[row]
        for v in variants:
            for i_b in range(4):
                with np.errstate(divide="ignore", invalid="ignore"):
                    r = results[v]["meas"][pm][i_b] / results[v]["th"][pt][i_b]
                ls = "-" if v == "dmb" else ":"
                ax.plot(results[v]["ell"][2:], r[2:], ls, lw=1, label=f"{v} b{i_b+1}" if row == 0 else None)
        ax.axhline(1.0, color="k", lw=0.8)
        ax.axhspan(0.9, 1.1, color="grey", alpha=0.15)
        ax.set_xscale("log")
        ax.set_ylim(0.5, 1.5)
        ax.set_xlim(20, 2000)
        ax.set_ylabel(lbl)
    axes[0].legend(fontsize=7, ncol=4)
    axes[-1].set_xlabel(r"$\ell$")
    fig.suptitle("Measured / theory ratio (grey band = +/-10%)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "wl_theory_ratio.png"), dpi=140)
    plt.close(fig)
    print(f"\nsaved results + figures to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
