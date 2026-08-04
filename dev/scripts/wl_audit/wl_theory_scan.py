# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created July 2026
Author: Arne Thomsen (with Claude)

Cosmology scan of the full-sky input-map vs. independent-theory check (companion to
wl_theory_comparison.py, which does the fiducial). Repeats the measured-vs-CCL comparison at 3
values of S8 and 3 values of w0, drawn from the CosmoGrid grid (LHC), to confirm that the kg / ia
input-map amplitude and scale stay correct away from the fiducial cosmology.

Only the baryonified maps (v11dmb) exist on the grid (dark-matter-only is fiducial-only), so the
clean amplitude/scale test is ell in [30, 300] where baryons are negligible; the high-ell droop
below the halofit-DMO theory is the (cosmology-dependent) baryonic suppression set by each point's
bary_Mc / bary_nu, not a misspecification.

Each panel's title carries the cosmological parameters of its map. Theory is computed at the exact
grid cosmology of every point (from CosmoGridV1_metainfo.h5), so map and theory always share the
same cosmology. Runs on a Perlmutter CPU node; maps on CFS. Results -> --out_dir.
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

METACAL_BINS = ["metacal1", "metacal2", "metacal3", "metacal4"]
# Clariden copy of the full-sky grid maps (originally run on Perlmutter CFS:
# /global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/grid)
GRID_ROOT = "/iopsstor/scratch/cscs/athomsen/CosmoGrid/v11desy3/bary/grid"
METAINFO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..", "data/CosmoGridV1_metainfo.h5")
MAP_FILE = "projected_probes_maps_v11dmb.h5"

# Selected grid cosmologies, keyed by SOBOL INDEX (the CosmoGrid directory index = path_par =
# cosmo_{sobol_index:06d}; note sobol_index != id_param).
S8_SCAN = [194878, 168548, 82075]  # S8 ~ 0.58, 0.79, 0.98 at w0 ~ -1
W0_SCAN = [2, 16147, 964]          # w0 ~ -1.58, -1.24, -0.56 at S8 ~ 0.75-0.85


def setup(args=None):
    p = argparse.ArgumentParser(description="cosmology scan of the input-map vs CCL theory check")
    p.add_argument("--n_perms", type=int, default=3)
    p.add_argument("--grid_root", default=GRID_ROOT)
    p.add_argument("--out_dir", default="/iopsstor/scratch/cscs/athomsen/deep_lss/runs/wl_audit")
    return p.parse_args(args)


def grid_params(sobol):
    """Full cosmology of the grid point with sobol_index == sobol (the CosmoGrid directory index)."""
    with h5py.File(METAINFO, "r") as h:
        g = h["parameters/grid"][:]
    r = g[g["sobol_index"] == sobol][0]
    h_ = float(r["H0"]) / 100.0
    O_nu = float(r["O_nu"])
    m_nu_tot = O_nu * h_ ** 2 * 93.14  # eV (degenerate)
    S8 = float(r["s8"]) * np.sqrt(float(r["Om"]) / 0.3)
    return dict(
        sobol=int(sobol), path=r["path_par"].decode().rstrip("/"),
        Omega_c=float(r["O_cdm"]), Omega_b=float(r["Ob"]), h=h_, sigma8=float(r["s8"]),
        # CosmoGrid V1 fixes Sum m_nu ~ 0.06 eV, so the floor is a guard that never binds here; it
        # would silently clamp a future variable-m_nu grid, so revisit if O_nu ever varies
        n_s=float(r["ns"]), m_nu=max(m_nu_tot, 0.06), w0=float(r["w0"]), Om=float(r["Om"]),
        S8=S8, log10Mc=float(np.log10(r["bary_Mc"])), bnu=float(r["bary_nu"]),
    )


def title_str(p):
    return (f"{p['path']}:  S8={p['S8']:.3f}  $\\sigma_8$={p['sigma8']:.3f}  "
            f"$\\Omega_m$={p['Om']:.3f}  $w_0$={p['w0']:+.3f}  $n_s$={p['n_s']:.3f}  "
            f"$\\Omega_b$={p['Omega_b']:.3f}  h={p['h']:.3f}  $\\log_{{10}}M_c$={p['log10Mc']:.2f}  "
            f"$b_\\nu$={p['bnu']:+.2f}")


def theory_cls(ell, z, nz, p):
    cosmo = ccl.Cosmology(Omega_c=p["Omega_c"], Omega_b=p["Omega_b"], h=p["h"], sigma8=p["sigma8"],
                          n_s=p["n_s"], m_nu=p["m_nu"], mass_split="equal", w0=p["w0"],
                          matter_power_spectrum="halofit", transfer_function="boltzmann_camb")
    out = {"GG": [], "II": [], "GI": []}
    for zb, nb in zip(z, nz):
        t_g = ccl.WeakLensingTracer(cosmo, dndz=(zb, nb))
        t_i = ccl.WeakLensingTracer(cosmo, dndz=(zb, nb), has_shear=False,
                                    ia_bias=(zb, np.ones_like(zb)), use_A_ia=True)
        out["GG"].append(ccl.angular_cl(cosmo, t_g, t_g, ell))
        out["II"].append(ccl.angular_cl(cosmo, t_i, t_i, ell))
        out["GI"].append(ccl.angular_cl(cosmo, t_g, t_i, ell))
    return {k: np.array(v) for k, v in out.items()}


def measure(path, n_perms):
    """dmb kg/ia auto + cross full-sky Cls, perm-averaged, pixel-window deconvolved (one SHT per map)."""
    cdir = f"{GRID_ROOT}/{path}"
    perms = sorted(d for d in os.listdir(cdir) if d.startswith("perm_"))[:n_perms]
    nside = None
    acc = None
    n_used = 0
    z, nz = [], []
    for pd in perms:
        path = f"{cdir}/{pd}/{MAP_FILE}"
        with h5py.File(path, "r") as h:
            if not z:
                for b in METACAL_BINS:
                    d = h[f"nz/{b}"][:]
                    z.append(d[:, 0]); nz.append(d[:, 1])
            for i_b, b in enumerate(METACAL_BINS):
                kg = h[f"map/kg/{b}"][:].astype(np.float64)
                ia = h[f"map/ia/{b}"][:].astype(np.float64)
                if nside is None:
                    nside = hp.npix2nside(kg.size)
                    lmax = 3 * nside - 1
                    ell = np.arange(lmax + 1)
                    pw2 = hp.pixwin(nside, lmax=lmax) ** 2
                    pw2[pw2 == 0] = np.inf
                    acc = {k: np.zeros((4, lmax + 1)) for k in ["kg", "ia", "cross"]}
                alm_kg = hp.map2alm(kg, lmax=lmax)
                alm_ia = hp.map2alm(ia, lmax=lmax)
                acc["kg"][i_b] += hp.alm2cl(alm_kg) / pw2
                acc["ia"][i_b] += hp.alm2cl(alm_ia) / pw2
                acc["cross"][i_b] += hp.alm2cl(alm_kg, alm_ia) / pw2
        n_used += 1
    for k in acc:
        acc[k] /= n_used
    return ell, acc, z, nz, n_used


def run_point(sobol, n_perms):
    p = grid_params(sobol)
    ell, meas, z, nz, n_used = measure(p["path"], n_perms)
    th = theory_cls(ell.astype(float), z, nz, p)
    print(f"\n{p['path']}  S8={p['S8']:.3f} sigma8={p['sigma8']:.3f} Om={p['Om']:.3f} w0={p['w0']:+.3f} "
          f"({n_used} perms)", flush=True)
    for pm, pt, name in [("kg", "GG", "kg/GG"), ("ia", "II", "ia/II"), ("cross", "GI", "kg.ia/GI")]:
        line = f"  {name:8s}"
        for lo, hi in [(30, 100), (100, 300), (300, 600), (600, 1000), (1000, 1500)]:
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.median((meas[pm][:, lo:hi] / th[pt][:, lo:hi]))
            line += f"  [{lo},{hi}]={r:5.3f}"
        print(line, flush=True)
    return dict(p=p, ell=ell, meas=meas, th=th, n_used=n_used)


def plot_scan(results, scan_name, out_dir):
    """One figure per scan: rows = cosmologies (title = params), cols = probes (kg/GG, ia/II, GI)."""
    probes = [("kg", "GG", r"$\kappa_g$ / GG"), ("ia", "II", "ia / II"), ("cross", "GI", r"$\kappa_g\times$ia / GI")]
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(16, 3.1 * n), squeeze=False)
    for i_c, res in enumerate(results):
        ell = res["ell"]
        for j, (pm, pt, lbl) in enumerate(probes):
            ax = axes[i_c][j]
            for i_b in range(4):
                with np.errstate(divide="ignore", invalid="ignore"):
                    r = res["meas"][pm][i_b] / res["th"][pt][i_b]
                ax.plot(ell[2:], r[2:], lw=0.9, label=f"metacal{i_b+1}")
            ax.axhline(1.0, color="k", lw=0.8)
            ax.axhspan(0.9, 1.1, color="grey", alpha=0.15)
            ax.axvspan(30, 300, color="green", alpha=0.06)  # clean amplitude/scale band (baryon-free)
            ax.set_xscale("log")
            ax.set_ylim(0.5, 1.5)
            ax.set_xlim(20, 2000)
            ax.set_ylabel(lbl)
            if i_c == n - 1:
                ax.set_xlabel(r"$\ell$")
            if j == 0 and i_c == 0:
                ax.legend(fontsize=7, ncol=2)
        # cosmology params as the title spanning the row (put on the middle column axis)
        axes[i_c][1].set_title(title_str(res["p"]), fontsize=10)
    fig.suptitle(f"Input map (v11dmb) / CCL theory  --  {scan_name}  "
                 f"(green band ell<300 = baryon-free amplitude/scale test; grey band = +/-10%)",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    fp = os.path.join(out_dir, f"wl_theory_scan_{scan_name}.png")
    fig.savefig(fp, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {fp}", flush=True)


def main(args=None):
    args = setup(args)
    global GRID_ROOT
    GRID_ROOT = args.grid_root
    os.makedirs(args.out_dir, exist_ok=True)
    for scan_name, sobols in [("S8_scan", S8_SCAN), ("w0_scan", W0_SCAN)]:
        print(f"\n========== {scan_name} ==========", flush=True)
        results = [run_point(s, args.n_perms) for s in sobols]
        plot_scan(results, scan_name, args.out_dir)
        np.savez(os.path.join(args.out_dir, f"wl_theory_scan_{scan_name}.npz"),
                 ell=results[0]["ell"],
                 **{f"c{r['p']['sobol']}_meas_{k}": r["meas"][k] for r in results for k in ["kg", "ia", "cross"]},
                 **{f"c{r['p']['sobol']}_th_{k}": r["th"][k] for r in results for k in ["GG", "II", "GI"]},
                 params=np.array([str(r["p"]) for r in results], dtype=object))
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
