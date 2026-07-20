# B-mode Cls information-loss Fisher study — report

**Question.** `lensing.mode_removal` reconstructs κ from masked shear via Kaiser–Squires and keeps
only the E-mode alm, discarding the B-mode ([msfm/utils/lensing.py:99](../../msfm/utils/lensing.py#L99)).
The footprint is applied in *shear* space, which leaks E→B. **Does throwing away that B-mode inflate
the cosmological posterior?** We answer it with a sim-based multiprobe Gaussian Fisher forecast on the
tomographic Cls, feeding the *same* forecast machinery an augmented data vector `concat(cl_E, cl_B)`
and comparing constraints with vs. without the B columns.

## Result

All numbers are marginalized over the 14 parameters; sim covariance over 39960 fiducial realizations
(3996 signal × 10 noise), central finite-difference Jacobian from the ± perturbations, Hartlap-debiased
`F = JᵀC⁻¹J`. No-priors leg shown; adding weak Gaussian priors changes nothing qualitatively.

| Data vector | ℓ_min (lensing) | cols | σ(Ωm) | σ(S8) | FoM(Ωm,S8) |
|---|---|---|---|---|---|
| **Baseline** (current, published) | 30 | 36 E | 0.01190 | 0.01145 | 7714 |
| **+ low-ℓ** (drop ℓ_min) | None | 36 E | 0.01190 | 0.01144 | 7737 |
| **+ B-mode** | None | 78 E+B | 0.01190 | 0.01144 | 7740 |

- **Dropping the lensing ℓ_min=30 high-pass:** FoM **+0.30%**, σ(S8) −0.10%.
- **Adding the 42 B-mode columns on top:** FoM **+0.04%**, σ(S8) −0.00%. Every per-parameter σ moves
  ≤0.1%.

**Conclusion.** Both increments are negligible (combined ~+0.34% FoM, inside sim-Fisher noise). The
current **kappa-only, ℓ_min=30 baseline already captures essentially all the two-point cosmological
information**; discarding the Kaiser–Squires B-mode does **not** inflate the posterior at 2-point.

### Why B is empty at 2-point
The B-mode data vector is noise-dominated: per-column signal-to-noise (|mean|/std over realizations)
has median **0.005** for B vs **6.5** for E; only **9.5%** of B columns clear S/N>1 vs **84%** for E.

### Numerical robustness
The raw sample-covariance condition number looks alarming (~1e15) but that is pure **dynamic range**
(Cls span ~1e-7 down to ~1e-12), not rank deficiency. Repeating the solve on the **correlation matrix**
(normalize each column by its std) gives cond ~640 and reproduces σ(S8)/FoM to all printed digits — the
null result is physics, not a numerically washed-out signal.

### Caveat
This is a **Gaussian, two-point** statement. If the E→B leakage carries non-Gaussian information, Cls
cannot see it. The definitive follow-up would be a field-level κ_E+κ_B two-channel network test — but
the power spectra give no 2-point motivation to invest in it.

## Files

### Unified forecast infrastructure (repo `msfm/`, promoted from the dev prototype)
| File | Role |
|---|---|
| [msfm/utils/fisher.py](../../msfm/utils/fisher.py) | Core library. Loads a covariance sample + Jacobian source assembled from named Cl **blocks** (E, B, …), builds `F = JᵀC⁻¹J` (Hartlap + correlation-matrix normalization), and propagates σ(S8)/FoM. `run_forecast(...)` is one self-describing forecast. |
| [msfm/apps/run_fisher_forecast.py](../../msfm/apps/run_fisher_forecast.py) | Driver over a YAML spec: runs any set of named forecasts and prints requested pairwise comparisons. `--only`, `--priors`/`--no-priors` overrides. |
| [configs/v17/fisher_bmode.yaml](../../configs/v17/fisher_bmode.yaml) | The spec for this study: `baseline_E` (standard E-only), `bmode_E` (+low-ℓ), `bmode_EB` (+B), with the two comparisons. Running just `baseline_E` reproduces the standard forecast. |

### Jacobian-source production (this directory, `dev/bmode_fisher/`)
| File | Role |
|---|---|
| [reduce.py](reduce.py) | Streams the fiducial tfrecords once, averages each perturbation's raw Cls over all realizations, bins them exactly as `merge()` → the Jacobian source `forecast_inputs.h5`. `--e_only` handles the ℓ_min=30 baseline tree (no `cl_bmode` field). |
| [reduce.slurm](reduce.slurm) / [reduce_baseline.slurm](reduce_baseline.slurm) | Submit phase 1 over the baseline_bmode (E+B) and baseline (ℓ_min=30, `--e_only`) tfrecords. |
| [analyze.py](analyze.py) | Original one-off phase-2 script (E-only/E+B + ℓ_min ablation hard-coded). **Superseded by the unified `msfm` infrastructure above**; kept for provenance — it produced the numbers in this report and the two agree to all digits. |

### The dataset-producing code (repo `msfm/`, gated by config flag `analysis.modelling.lensing.b_mode_cls`, default off)
| File | Change |
|---|---|
| [msfm/utils/lensing.py](../../msfm/utils/lensing.py) | `mode_removal(keep_b_mode=)` — project the B-mode alm instead of E, reusing the one map2alm. |
| [msfm/utils/postprocessing.py](../../msfm/utils/postprocessing.py) | Parallel `kg_b`/`ia_b`/`sn_b` B-channel data-vector containers. |
| [msfm/apps/run_fiducial_postprocessing.py](../../msfm/apps/run_fiducial_postprocessing.py) | Threads the B channel through the same IA+m-bias+mask+smoothing; new `run_tfrecords_alm_to_cl_bmode` (12-channel stack → 78 cols, returns the 42 B-touching cols, asserts the 36 E/g cols reproduce the standard `cl`); `merge` bins the B-block into `cls/bmode_*`. |
| [msfm/utils/power_spectra.py](../../msfm/utils/power_spectra.py) | `run_tfrecords_alm_to_cl_bmode`; B-block binned via `smooth_and_bin_cls(with_cross=False)`. |
| [msfm/utils/tfrecords.py](../../msfm/utils/tfrecords.py) | Stores/parses `cl_bmode_{label}` (plain reshape — NOT the triangular bin_indices gather). |
| [configs/v17/baseline_bmode.yaml](../../configs/v17/baseline_bmode.yaml) | Clone of `baseline.yaml` with `b_mode_cls: True` and lensing `scale_cuts.l_min: Null`. |

> Note: the `msfm/` changes are staged in the working tree (uncommitted), gated so a normal run is
> byte-identical. `dev/bmode_fisher/` is untracked.

### Data products (`/pscratch/sd/a/athomsen/v11desy3/v17/`)
| Path | Contents |
|---|---|
| `baseline_bmode/cls/fiducial_cls.h5` (19 GB) | Covariance sample: `cls/binned` (39960,32,36) + `cls/bmode_binned` (39960,32,42). |
| `baseline_bmode/cls/forecast_inputs.h5` (596 KB) | Jacobian source: `pert_binned_E` (29,32,36) + `pert_binned_B` (29,32,42). |
| `baseline/cls/fiducial_cls.h5` (8.6 GB) | ℓ_min=30 baseline covariance: `cls/binned` (39960,32,36). |
| `baseline/cls/forecast_inputs.h5` (284 KB) | ℓ_min=30 baseline Jacobian (E-only). |

## Reproduce the forecast

```bash
# (data already produced; this is the ~seconds analysis step)
python -m msfm.apps.run_fisher_forecast --spec configs/v17/fisher_bmode.yaml            # no priors
python -m msfm.apps.run_fisher_forecast --spec configs/v17/fisher_bmode.yaml --priors   # + priors
python -m msfm.apps.run_fisher_forecast --spec configs/v17/fisher_bmode.yaml --only baseline_E  # standard forecast alone
```

Add a new forecast by editing the spec (a `{source, blocks}` entry); add a new Cl sub-vector by adding
a block to `fisher.BLOCKS_DEFAULT` (or the spec's `blocks:`). No code change for either.

To regenerate the Jacobian sources from scratch: `sbatch dev/bmode_fisher/reduce.slurm` and
`sbatch dev/bmode_fisher/reduce_baseline.slurm` (each ~20–40 min, IO-bound, one CPU node). The
covariance `fiducial_cls.h5` files come from the full `run_fiducial_postprocessing.py` production run
with the respective config (see the `bmode-cls-fisher-study` / `v17-packed-production-run` notes).
