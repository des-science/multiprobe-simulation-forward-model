#!/usr/bin/env python
"""Phase 2 of the B-mode Cls information-loss Fisher study: turn the fiducial sample covariance
(fiducial_cls.h5) plus the finite-difference Jacobian (forecast_inputs.h5 from reduce.py) into a
Gaussian Fisher forecast, and report the change in sigma(S8) and the Om-S8 FoM when the B-mode Cls
columns are ADDED to the data vector.

This replicates, in numpy on the binned Cls data vector, the exact algorithm of
y3-deep-lss get_jac_and_cov_matrix:
  - sample covariance C of the fiducial realizations (ddof=1),
  - central finite-difference Jacobian J_i = (mean(Cl^{+i}) - mean(Cl^{-i})) / (2*offset_i),
  - Fisher F = J^T C^{-1} J, marginalized parameter covariance F^{-1}.
The only thing that changes between the two forecasts is whether the 42 B-mode columns are appended
to the 36 E/clustering columns of the data vector.

Covariance columns (h5 cls/binned, raw get_cls lexicographic order, no gather) line up one-to-one
with the Jacobian columns (reduce.py parses cl_{label} the same way) -- verified in reduce.py's
build against the h5 fiducial mean.

Notes / choices (all surfaced in the printout):
  - Hartlap debiasing of C^{-1} with the sample size n and data dimension p.
  - Each (signal example, noise realization) row of cls/binned is treated as one survey mock, so C
    is the full data covariance (cosmic variance + noise) -- the same convention as the delta-loss
    training batch.
  - Reported both without any prior (raw Fisher) and with weak Gaussian priors set to the flat
    prior width / sqrt(12) (regularizes the prior-dominated params Ob/H0/ns without biasing S8).
"""
import argparse
import os

import numpy as np
import h5py

from msfm.utils import files, parameters


def load_covariance_sample(fiducial_cls_h5):
    with h5py.File(fiducial_cls_h5, "r") as f:
        cls_E = f["cls/binned"][:]        # (n_real, n_bins, 36)
        cls_B = f["cls/bmode_binned"][:]  # (n_real, n_bins, 42)
    n_real = cls_E.shape[0]
    D_E = cls_E.reshape(n_real, -1)        # (n_real, n_bins*36)
    D_B = cls_B.reshape(n_real, -1)        # (n_real, n_bins*42)
    return D_E, D_B, n_real


def load_jacobian(forecast_inputs_h5):
    with h5py.File(forecast_inputs_h5, "r") as f:
        pE = f["pert_binned_E"][:]   # (n_lab, n_bins, 36)
        pB = f["pert_binned_B"][:]   # (n_lab, n_bins, 42)
        offsets = f["offsets"][:]
        fiducials = f["fiducials"][:]
        params = [p.decode() for p in f["params"][:]]
        labels = [l.decode() for l in f["labels"][:]]
    n_lab = pE.shape[0]
    JE = pE.reshape(n_lab, -1)   # (n_lab, n_bins*36)
    JB = pB.reshape(n_lab, -1)   # (n_lab, n_bins*42)
    return JE, JB, offsets, fiducials, params, labels


def build_jac(J_flat, params, labels, offsets):
    """Central differences over labels [fiducial, P0_m, P0_p, P1_m, P1_p, ...] -> (n_dim, n_params)."""
    n_params = len(params)
    jac = np.zeros((J_flat.shape[1], n_params), dtype=np.float64)
    for i in range(n_params):
        i_m = labels.index(f"delta_{params[i]}_m")
        i_p = labels.index(f"delta_{params[i]}_p")
        jac[:, i] = (J_flat[i_p] - J_flat[i_m]) / (2.0 * offsets[i])
    return jac


def fisher(cov, jac, n_real, hartlap=True):
    """F = J^T C^{-1} J via a stable solve, with optional Hartlap debiasing of C^{-1}."""
    p = cov.shape[0]
    # C^{-1} J  ->  solve C X = J
    X = np.linalg.solve(cov, jac)
    F = jac.T @ X
    if hartlap:
        alpha = (n_real - p - 2.0) / (n_real - 1.0)
        assert alpha > 0, f"Hartlap factor <= 0: n_real={n_real}, p={p} (too few realizations)"
        F = alpha * F
    cond = np.linalg.cond(cov)
    return F, cond


def add_gaussian_priors(F, params, conf):
    """Add diag(1/sigma_prior^2) with sigma_prior = flat-prior-width / sqrt(12)."""
    intervals = parameters.get_prior_intervals(conf=conf)
    sig = (intervals[:, 1] - intervals[:, 0]) / np.sqrt(12.0)
    return F + np.diag(1.0 / sig**2), sig


def s8_propagation(cov_param, params, fiducials):
    """sigma(S8) and Om-S8 FoM from the marginalized (Om, s8) 2x2 block. S8 = s8*(Om/0.3)^0.5."""
    iOm, is8 = params.index("Om"), params.index("s8")
    Om, s8 = fiducials[iOm], fiducials[is8]
    dS8_dOm = 0.5 * s8 * (0.3) ** -0.5 * Om ** -0.5
    dS8_ds8 = (Om / 0.3) ** 0.5
    C2 = cov_param[np.ix_([iOm, is8], [iOm, is8])]
    g = np.array([dS8_dOm, dS8_ds8])
    var_S8 = g @ C2 @ g
    # transform (Om, s8) -> (Om, S8):  M = [[1,0],[dS8_dOm, dS8_ds8]]
    M = np.array([[1.0, 0.0], [dS8_dOm, dS8_ds8]])
    C_OmS8 = M @ C2 @ M.T
    fom = 1.0 / np.sqrt(np.linalg.det(C_OmS8))
    return np.sqrt(var_S8), fom, np.sqrt(C2[0, 0]), np.sqrt(C2[1, 1])


def report(name, cov, jac, n_real, params, fiducials, conf, use_priors):
    F, cond = fisher(cov, jac, n_real)
    tag = ""
    if use_priors:
        F, _ = add_gaussian_priors(F, params, conf)
        tag = " + priors"
    Cp = np.linalg.inv(F)
    sig = np.sqrt(np.diag(Cp))
    sS8, fom, sOm, ss8 = s8_propagation(Cp, params, fiducials)
    print(f"\n=== {name}{tag} ===  (data dim p={cov.shape[0]}, cov cond={cond:.2e})")
    print(f"  sigma(Om) = {sOm:.5f}   sigma(s8) = {ss8:.5f}   sigma(S8) = {sS8:.5f}   FoM(Om,S8) = {fom:.2f}")
    return dict(F=F, Cp=Cp, sig=sig, sS8=sS8, fom=fom, sOm=sOm, ss8=ss8, cond=cond)


def load_covariance_sample_E(fiducial_cls_h5):
    """E-only covariance sample (cls/binned) -- for the ell_min=30 baseline tree with no B block."""
    with h5py.File(fiducial_cls_h5, "r") as f:
        cls_E = f["cls/binned"][:]
    n_real = cls_E.shape[0]
    return cls_E.reshape(n_real, -1), n_real


def load_jacobian_E(forecast_inputs_h5):
    """E-only Jacobian source (pert_binned_E) -- baseline forecast_inputs.h5 has no B block."""
    with h5py.File(forecast_inputs_h5, "r") as f:
        pE = f["pert_binned_E"][:]
        offsets = f["offsets"][:]
        fiducials = f["fiducials"][:]
        params = [p.decode() for p in f["params"][:]]
        labels = [l.decode() for l in f["labels"][:]]
    return pE.reshape(pE.shape[0], -1), offsets, fiducials, params, labels


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--fiducial_cls", required=True)
    ap.add_argument("--inputs", required=True, help="forecast_inputs.h5 from reduce.py")
    ap.add_argument("--baseline_fiducial_cls", default=None,
                    help="OPTIONAL ell_min=30 baseline fiducial_cls.h5 (E-only cov) for the l_min "
                         "ablation leg")
    ap.add_argument("--baseline_inputs", default=None,
                    help="OPTIONAL ell_min=30 baseline forecast_inputs.h5 (E-only Jacobian, from "
                         "reduce.py --e_only)")
    args = ap.parse_args()
    conf = files.load_config(args.config)

    D_E, D_B, n_real = load_covariance_sample(args.fiducial_cls)
    JE, JB, offsets, fiducials, params, labels = load_jacobian(args.inputs)
    print(f"n_realizations = {n_real}, n_params = {len(params)}")
    print(f"params: {params}")

    # data vectors: E-only and E+B
    D_EB = np.concatenate([D_E, D_B], axis=1)
    JE_flat = JE
    JEB_flat = np.concatenate([JE, JB], axis=1)

    cov_E = np.cov(D_E, rowvar=False)     # (nE, nE)
    cov_EB = np.cov(D_EB, rowvar=False)   # (nEB, nEB)

    jac_E = build_jac(JE_flat, params, labels, offsets)
    jac_EB = build_jac(JEB_flat, params, labels, offsets)

    for use_priors in (False, True):
        rE = report("E-only (36 cols)", cov_E, jac_E, n_real, params, fiducials, conf, use_priors)
        rEB = report("E+B (36+42 cols)", cov_EB, jac_EB, n_real, params, fiducials, conf, use_priors)
        print(f"\n  --> adding B-mode Cls: "
              f"sigma(S8) {rE['sS8']:.5f} -> {rEB['sS8']:.5f}  "
              f"({100*(rEB['sS8']/rE['sS8']-1):+.2f} %),   "
              f"FoM {rE['fom']:.2f} -> {rEB['fom']:.2f}  "
              f"({100*(rEB['fom']/rE['fom']-1):+.2f} %)")
        # per-parameter marginalized sigma change
        print("  per-param sigma ratio (E+B / E-only):")
        for i, p in enumerate(params):
            print(f"    {p:10s} {rE['sig'][i]:.4e} -> {rEB['sig'][i]:.4e}  ({rEB['sig'][i]/rE['sig'][i]:.3f})")

    # ---- optional l_min ablation: ell_min=30 baseline E-only vs this-config (l_min=None) E-only ----
    if args.baseline_fiducial_cls and args.baseline_inputs:
        print("\n" + "=" * 78)
        print("l_min ABLATION: what dropping the lensing ell_min=30 map high-pass buys (E-only)")
        print("=" * 78)
        Db_E, nb = load_covariance_sample_E(args.baseline_fiducial_cls)
        Jb_E, off_b, fid_b, par_b, lab_b = load_jacobian_E(args.baseline_inputs)
        assert par_b == params, f"param mismatch baseline vs bmode: {par_b} vs {params}"
        assert Db_E.shape[1] == D_E.shape[1], f"E-dim mismatch: {Db_E.shape[1]} vs {D_E.shape[1]}"
        cov_base = np.cov(Db_E, rowvar=False)
        jac_base = build_jac(Jb_E, par_b, lab_b, off_b)
        for use_priors in (False, True):
            rB = report("baseline  ell_min=30 (E-only)", cov_base, jac_base, nb,
                        params, fid_b, conf, use_priors)
            rN = report("this cfg  ell_min=None (E-only)", cov_E, jac_E, n_real,
                        params, fiducials, conf, use_priors)
            print(f"\n  --> dropping ell_min=30: "
                  f"sigma(S8) {rB['sS8']:.5f} -> {rN['sS8']:.5f}  "
                  f"({100*(rN['sS8']/rB['sS8']-1):+.2f} %),   "
                  f"FoM {rB['fom']:.2f} -> {rN['fom']:.2f}  "
                  f"({100*(rN['fom']/rB['fom']-1):+.2f} %)")


if __name__ == "__main__":
    main()
