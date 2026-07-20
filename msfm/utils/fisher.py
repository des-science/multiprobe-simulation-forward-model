# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created July 2026
Author: Arne Thomsen

Unified Gaussian Fisher forecasting on the stored tomographic Cls.

This is the offline, numpy counterpart of the training-time Fisher machinery
`deep_lss.utils.delta_loss.get_jac_and_cov_matrix` (y3-deep-lss): a sample covariance over the
fiducial realizations plus a central finite-difference Jacobian from the +/- perturbations, combined
into F = J^T C^-1 J and marginalized to parameter constraints. Where the training loss operates on the
network's learned summary inside the TF graph, this module operates on the raw binned Cls data vector
so any subset of Cl columns can be forecast and compared offline.

A forecast is fully specified by three things:
  - a COVARIANCE source: a `fiducial_cls.h5` (datasets like `cls/binned`, `cls/bmode_binned`),
  - a JACOBIAN source: a `forecast_inputs.h5` from `dev/bmode_fisher/reduce.py` (datasets
    `pert_binned_E` / `pert_binned_B`, plus `offsets`/`fiducials`/`params`/`labels`),
  - an ordered list of BLOCKS, each naming one (cov dataset, jac dataset) pair. The data vector is the
    per-realization concatenation of the chosen blocks; the Jacobian is the matching concatenation of
    the perturbation-mean blocks.

`BLOCKS_DEFAULT` maps the two blocks used today (E = 36-col E/clustering, B = 42-col B-touching). The
"standard E-mode-only" forecast is simply `blocks=["E"]`; adding the B-mode is `blocks=["E", "B"]`.
Different sources (e.g. an ell_min=30 baseline tree vs an ell_min=None tree) let the same block set be
compared across analysis choices.

All the linear-algebra choices (Hartlap debiasing, correlation-matrix-normalized robustness solve,
S8/FoM error propagation, Gaussian priors) live here; the `run_fisher_forecast` app is a thin driver
over a YAML spec.
"""
import numpy as np
import h5py

from msfm.utils import parameters


# Block registry: block name -> (covariance dataset in fiducial_cls.h5, jacobian dataset in inputs.h5).
# A block's covariance dataset has shape (n_real, n_bins, n_col); its jacobian dataset (n_lab, n_bins,
# n_col). Extend this dict to forecast further Cl sub-vectors (e.g. clustering-only) without touching
# the math below.
BLOCKS_DEFAULT = {
    "E": {"cov": "cls/binned", "jac": "pert_binned_E"},
    "B": {"cov": "cls/bmode_binned", "jac": "pert_binned_B"},
}


def load_covariance_sample(cov_h5, blocks, block_defs=BLOCKS_DEFAULT):
    """Per-realization data vector D (n_real, p) = concat of the chosen blocks' binned Cls.

    Each row is one (signal example, noise realization) survey mock, so cov(D) is the full data
    covariance (cosmic variance + shape/shot noise) -- the same convention as the delta-loss batch.
    """
    cols = []
    n_real = None
    with h5py.File(cov_h5, "r") as f:
        for b in blocks:
            ds = block_defs[b]["cov"]
            if ds not in f:
                raise KeyError(f"block '{b}': covariance dataset '{ds}' not in {cov_h5}")
            a = f[ds][:]  # (n_real, n_bins, n_col)
            n_real = a.shape[0] if n_real is None else n_real
            assert a.shape[0] == n_real, f"block '{b}' has {a.shape[0]} realizations, expected {n_real}"
            cols.append(a.reshape(n_real, -1))
    return np.concatenate(cols, axis=1), n_real


def load_jacobian_source(jac_h5, blocks, block_defs=BLOCKS_DEFAULT):
    """Flat perturbation-mean matrix J (n_lab, p) plus the metadata needed to difference it.

    Rows are the perturbation labels [fiducial, P0_m, P0_p, P1_m, P1_p, ...]; columns match
    `load_covariance_sample` block-for-block (same datasets, same order) so C and J align one-to-one.
    """
    cols = []
    with h5py.File(jac_h5, "r") as f:
        offsets = f["offsets"][:]
        fiducials = f["fiducials"][:]
        params = [p.decode() for p in f["params"][:]]
        labels = [l.decode() for l in f["labels"][:]]
        for b in blocks:
            ds = block_defs[b]["jac"]
            if ds not in f:
                raise KeyError(f"block '{b}': jacobian dataset '{ds}' not in {jac_h5}")
            a = f[ds][:]  # (n_lab, n_bins, n_col)
            cols.append(a.reshape(a.shape[0], -1))
    J = np.concatenate(cols, axis=1)
    return J, offsets, fiducials, params, labels


def build_jacobian(J_flat, params, labels, offsets):
    """Central-difference Jacobian dCl/dtheta of shape (p, n_params) from the labelled pert means."""
    n_params = len(params)
    jac = np.zeros((J_flat.shape[1], n_params), dtype=np.float64)
    for i in range(n_params):
        i_m = labels.index(f"delta_{params[i]}_m")
        i_p = labels.index(f"delta_{params[i]}_p")
        jac[:, i] = (J_flat[i_p] - J_flat[i_m]) / (2.0 * offsets[i])
    return jac


def hartlap_factor(n_real, p):
    """(n - p - 2)/(n - 1): debiases the inverse of a sample covariance (Hartlap et al. 2007)."""
    alpha = (n_real - p - 2.0) / (n_real - 1.0)
    assert alpha > 0, f"Hartlap factor <= 0: n_real={n_real}, p={p} (too few realizations)"
    return alpha


def fisher_matrix(cov, jac, n_real, hartlap=True, normalize=True):
    """F = J^T C^-1 J via a stable solve, with optional Hartlap debiasing.

    normalize=True solves on the correlation matrix (each column scaled by its std) instead of the raw
    covariance. This is mathematically identical in exact arithmetic but removes the huge dynamic range
    of the Cls (~1e-7..1e-12) that otherwise inflates the condition number to ~1e15; the reported
    `cond` then reflects true rank, not scaling. Returns (F, cond).
    """
    p = cov.shape[0]
    if normalize:
        s = np.sqrt(np.diag(cov))
        Cn = cov / np.outer(s, s)
        Jn = jac / s[:, None]
        X = np.linalg.solve(Cn, Jn)
        F = Jn.T @ X  # J^T C^-1 J == Jn^T Cn^-1 Jn under the diagonal rescaling C = S Cn S
        cond = np.linalg.cond(Cn)
    else:
        X = np.linalg.solve(cov, jac)
        F = jac.T @ X
        cond = np.linalg.cond(cov)
    if hartlap:
        F = hartlap_factor(n_real, p) * F
    return F, cond


def add_gaussian_priors(F, params, conf):
    """Add diag(1/sigma_prior^2) with sigma_prior = flat-prior-width / sqrt(12) (variance of a uniform).

    Regularizes the prior-dominated parameters (Ob/H0/ns) without biasing the well-measured ones.
    """
    intervals = parameters.get_prior_intervals(conf=conf)
    sig = (intervals[:, 1] - intervals[:, 0]) / np.sqrt(12.0)
    return F + np.diag(1.0 / sig**2)


def s8_propagation(cov_param, params, fiducials, om_name="Om", s8_name="s8", om_pivot=0.3):
    """sigma(S8) and the Om-S8 FoM from the marginalized (Om, s8) block. S8 = s8 * (Om/0.3)^0.5."""
    iOm, is8 = params.index(om_name), params.index(s8_name)
    Om, s8 = fiducials[iOm], fiducials[is8]
    dS8_dOm = 0.5 * s8 * om_pivot ** -0.5 * Om ** -0.5
    dS8_ds8 = (Om / om_pivot) ** 0.5
    C2 = cov_param[np.ix_([iOm, is8], [iOm, is8])]
    g = np.array([dS8_dOm, dS8_ds8])
    var_S8 = g @ C2 @ g
    M = np.array([[1.0, 0.0], [dS8_dOm, dS8_ds8]])  # (Om, s8) -> (Om, S8)
    C_OmS8 = M @ C2 @ M.T
    fom = 1.0 / np.sqrt(np.linalg.det(C_OmS8))
    return np.sqrt(var_S8), fom, np.sqrt(C2[0, 0]), np.sqrt(C2[1, 1])


def run_forecast(cov_h5, jac_h5, blocks, conf=None, priors=False, hartlap=True, normalize=True,
                 block_defs=BLOCKS_DEFAULT):
    """End-to-end single forecast -> dict of Fisher matrix, marginalized sigmas, S8/FoM, diagnostics.

    conf is only needed when priors=True (for the flat-prior widths). params/labels/offsets/fiducials
    come from the jacobian source, so every forecast is self-describing.
    """
    D, n_real = load_covariance_sample(cov_h5, blocks, block_defs)
    J, offsets, fiducials, params, labels = load_jacobian_source(jac_h5, blocks, block_defs)
    assert D.shape[1] == J.shape[1], f"cov/jac dim mismatch: {D.shape[1]} vs {J.shape[1]}"

    cov = np.cov(D, rowvar=False)
    jac = build_jacobian(J, params, labels, offsets)
    F, cond = fisher_matrix(cov, jac, n_real, hartlap=hartlap, normalize=normalize)
    if priors:
        if conf is None:
            raise ValueError("priors=True requires a config (for the flat-prior widths)")
        F = add_gaussian_priors(F, params, conf)

    Cp = np.linalg.inv(F)
    sig = np.sqrt(np.diag(Cp))
    sS8, fom, sOm, ss8 = s8_propagation(Cp, params, fiducials)
    return dict(
        blocks=list(blocks), p=D.shape[1], n_real=n_real, cond=cond, priors=priors,
        params=params, fiducials=np.asarray(fiducials), F=F, cov_param=Cp,
        sigma=sig, sigma_Om=sOm, sigma_s8=ss8, sigma_S8=sS8, fom_Om_S8=fom,
    )
