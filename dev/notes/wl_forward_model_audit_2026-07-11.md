# Weak lensing forward-model audit (v16 `rot_in_place`)

**Date:** 2026-07-11
**Scope:** conceptual + numerical audit of the weak lensing part of `msfm/apps/run_grid_postprocessing.py`
with `configs/v16/rot_in_place.yaml`
**Motivation:** the WL part of the inference pipeline (both 2pt/Cls and map-level) is less constraining
than expected. Suspects going in: underestimated S/N, the delta-NLA (`bta`) model, the Kaiser-Squires
inversion, the mode-removal procedure.
**Artifacts:** diagnostics script `dev/scripts/wl_audit/wl_sn_diagnostics.py` (+ `.sh` sbatch wrapper),
results in `/iopsstor/scratch/cscs/athomsen/deep_lss/runs/wl_audit/` (job 2735114:
`wl_sn_diagnostics.npz`, `noise_level.png`, `signal_level.png`, `ds_amplitude.png`).

---

## Executive summary

1. **No bug was found in the Kaiser-Squires inversion, the mode removal, or the shape-noise model.**
   The signal, noise, and real-data paths apply the same operations in the same order, and the
   numerical diagnostics confirm the noise level and the signal transfer quantitatively.
2. **The delta-NLA `ds` template is unexpectedly large at low redshift** — Cl(ds) ≈ 1.2 × Cl(ia) in
   metacal1 — so marginalizing over `Aia · bta ∈ [−6, 6]` injects a large, cosmology-degenerate
   nuisance direction. This is now the **prime in-pipeline suspect** for the degraded WL constraints,
   additive with the wide-prior (w0, baryons, ns/Ob/H0) hypothesis.
3. The masked-KS/E-only convention costs an inherent ≈ 10 % power-level S/N penalty (signal transfer
   0.80 vs. noise transfer 0.89 relative to naive f_sky scaling). This is a design property applied
   identically to simulations and data, not a defect.

---

## 1. Data flow that was traced

```
CosmoGrid full-sky kg / ia / dg (metacal n(z)), ud_grade -> nside 512
    ├─ signal (kg, ia, ds each):  map2alm ──KS──> gamma full sky ──cut patch (gamma2 sign for
    │       mirrored patches)──> masked pseudo-E ──1/KS──> kappa ──dv cut, remove mean
    ├─ ds = (ia − ⟨ia⟩) · (dg − ⟨dg⟩)/⟨dg⟩          [postprocessing.py:271–274]
    └─ noise: DESY3_noise_v11.h5 catalog (e/(R_gamma+R_s), w, rotated pix) ── random in-place
            rotations, w-weighted per-pixel means at the base patch ──> same mode removal ──> dv
combine:  kg + tomo_Aia · (ia + bta · ds),  ×(1 + m),  ×mask,  hard ℓ ≥ 30,  ×mask
Cls:      alm(signal) + alm(noise) summed BEFORE squaring (cross terms included)
data:     catalog.build_metacal_map_from_cat + observation.forward_model_observation_map
          (identical estimator: Σw·e_rot/(R·Σw); identical mask/mode-removal/smoothing order)
```

## 2. Verified correct (static analysis)

| Item | Check |
|---|---|
| KS factors | `−√[(ℓ+2)(ℓ−1)/((ℓ+1)ℓ)]` matches Jeffrey+21 eq. (11); inverse is the exact reciprocal, ℓ = 0, 1 zeroed |
| Mirrored patches | γ2 sign flip for parity-flipped cutouts (correct spin-2 behavior); isotropic noise needs and gets none |
| B-mode handling | B discarded identically for sim signal, sim noise, and data — no S/N asymmetry |
| Noise calibration | full metacal response `e/(R_gamma + R_s)` applied when building the noise file (`catalog.get_shapes_from_cat`); real per-pixel depth via actual galaxy positions/weights — `in_place` *fixes* the uniform-depth issue of the older `count` method |
| m-bias | sims multiply (1 + m), data maps are *not* m-corrected (only R) — consistent; per-example resampling = implicit marginalization |
| Ordering | mask support → mode removal → per-patch mean removal → mask → hard ℓ ≥ 30 → mask, identical in `lensing_transform` and `forward_model_observation_map` |
| Cls | signal and noise alms summed before squaring → signal×noise cross terms present; pseudo-Cl convention identical to the observation |
| Bookkeeping | `(Aia, n_Aia, bta)` unpacking matches the Latin-hypercube construction order; Sobol cross-checks; `_verify_tfrecord` round trip; noise independent across z-bins; 80 signal × 5 noise structure |

## 3. Numerical diagnostics (job 2735114)

### 3.1 Shape-noise level — **calibrated, no S/N bug**

Empirical pseudo-Cl of `in_place` noise kappa_E maps vs. the analytic expectation
`N_ℓ = (4π/N_pix²) · Σ_patch [Σw²|e|²/(2(Σw)²)] · ℓ(ℓ+1)/((ℓ+2)(ℓ−1))`:

| bin | empirical / analytic (median, ℓ ∈ [300, 1500]) |
|---|---|
| metacal1 | 0.8910 |
| metacal2 | 0.8886 |
| metacal3 | 0.8894 |
| metacal4 | 0.8859 |

The uniform ≈ 11 % deficit is the footprint re-cut after mode removal discarding ringing that leaks
outside the patch — the data receives the identical treatment, so this is a consistent convention,
not a miscalibration. Cross-check: the high-ℓ (1200–1500) plateau of the *stored grid Cls*
sits at 1.07 / 1.01 / 1.02 / 1.01 × the analytic noise for bins 1–4, i.e. noise + a small
signal/IA contribution on top — the tfrecords carry the right noise level end-to-end.

### 3.2 Signal transfer through KS + mode removal — **no amplitude loss beyond known mechanisms**

Pseudo-Cl of a fiducial CosmoGrid kg map pushed through the exact pipeline chain, divided by
f_patch × full-sky Cl (median, ℓ ∈ [100, 1000]): **0.798 / 0.798 / 0.800 / 0.805** for bins 1–4.

Decomposition: ≈ 0.89 (footprint re-cut, same as the noise) × ≈ 0.90 (E→B leakage discarded).
Both apply identically to simulations and data. Net effect relative to an ideal estimator:
signal keeps 80 % of its pseudo-power while noise keeps 89 % → an inherent ≈ 10 % power-level
S/N penalty of the masked-KS E-only design. Worth knowing, not a bug.

### 3.3 delta-NLA `ds` template amplitude — **the surprise**

The static audit initially argued the 2D product-of-projections `ds = (ia − ⟨ia⟩) · δ_2D` should be
strongly LOS-suppressed and therefore inert. **The measurement says otherwise at low z**, because
δ_2D of the nearby structure is order unity:

| bin | Cl(ds)/Cl(ia) (median, ℓ ∈ [100, 1000]) | IA-term power change at bta = 2 |
|---|---|---|
| metacal1 | **1.23** | **× 7.7** |
| metacal2 | 0.22 | × 1.9 |
| metacal3 | 0.077 | × 0.83 |
| metacal4 | 0.033 | × 0.41 |

(The "power change" column includes the 2·bta·Cl(ia, ds) cross term; ia and ds are strongly
correlated since both trace the same structure.)

Implications:

- The grid marginalizes `Aia ∈ [−3, 3]`, `bta ∈ [0, 2]` → an effective amplitude `Aia·bta ∈ [−6, 6]`
  in front of a template whose power traces the matter field *squared*. In the low-z bins this is a
  large, cosmology-degenerate freedom — a credible mechanism for the observed loss of constraining
  power in **both** the 2pt and the map-level analyses.
- Coherence check: the stored grid Cl plateau excess over the analytic noise is 7 % in metacal1 vs.
  1–2 % in bins 2–4 — exactly where the ds power lives.
- Independently of the amplitude, the 2D product has the wrong scale dependence compared to the true
  projected 3D δ·s TATT term (2105.13544 eq. 19), so `bta` as implemented is not a faithful TATT
  extension either: real density-weighted IA in the data would not be absorbed correctly.

### 3.4 Follow-up (same day): bta = 0 conditioning on the existing DES chains — **no likelihood-level cost**

Executed recommended step 1 using the already-existing drop-and-fix chains of the 2pt lensing run
(`runs/v16/rot_in_place/cls/lensing/m1_default/ensemble_flow_1000000/`), comparing
`chain_{obs}_w0gt-1.npy` (params `[Om, s8, w0, Aia, n_Aia, bta]`) against
`chain_{obs}_w0gt-1_nla.npy` (bta dropped and fixed to 0, exact conditioning since bta is a flow
parameter). Script: scratchpad `compare_nla_chains.py` (1.02M samples per chain).

| | DESy3 | DESy3_no_psi_rot |
|---|---|---|
| σ(S8) free / bta=0 | 0.0380 / 0.0395 | 0.0385 / 0.0390 |
| σ(Om) free / bta=0 | 0.0671 / 0.0653 | 0.0654 / 0.0637 |
| FoM(Om,S8) ratio free/bta=0 | 1.004 | 0.975 |
| S8 mean shift | +0.0000 | −0.0006 |
| bta marginal | mean 0.79, std 0.567 (≈ prior U[0,2]) | mean 0.78, std 0.555 |
| corr(bta, S8) | −0.01 | −0.00 |
| corr(Aia·bta, S8) | +0.44 | +0.38 |

**Fixing bta = 0 changes nothing**: widths equal within MCMC noise, S8 mean shift zero, Om/s8 shift
±0.27σ along the banana. The bta marginal is essentially the prior (near-unidentifiable, as the
original static argument guessed), and its marginalization costs no Om–S8 constraining power *at the
likelihood level, given the current compression*. The corr(Aia·bta, S8) ≈ 0.4 is inherited from the
standard Aia–S8 degeneracy, not an independent bta direction.

Consequently the bta hypothesis survives **only in its compression-level form**: the summary network,
trained on data whose bin-1 IA power swings ×7.7 across the bta prior, may have discarded
cosmological information to cope with the ds variance — a loss that MCMC conditioning cannot recover.
The decisive test is a retrain on bta = 0 data; hypothesis 2 (prior volume) is otherwise rank 1.

**v17 config created** (`configs/v17/baseline.yaml`): identical to v16/rot_in_place except
`extended_nla: False`, the `dg -> ds` pair dropped from metacal `map_types` (the ds template is
never built — the `dg -> sn` pair stays; its input is ignored by the `in_place` noise method), and
**bta removed entirely** (`params.ia.tatt`, its grid prior, and its fiducial). Every code consumer
of the tatt/bta keys is behind an `extended_nla` (or `"bta" in params`) guard except one, which was
fixed: `msi/utils/plotting.py` now uses `.get("tatt", [])`. Because the y3-deep-lss probes configs
list the training params explicitly (they are NOT derived from the msfm config), bta-free variants
were added for the compressor: `configs/probes/lensing_nla.yaml`, `combined_nla.yaml`,
`lensing_ext_nla.yaml` (cross/2x2pt/3x2pt follow the same one-line pattern when needed). The
bta-containing probes configs fail fast on v17 data (`all_params.index('bta')` -> ValueError in
`grid_pipeline`/`cls_preprocessing`), and conversely the `_nla` configs must not be used on
v16-and-earlier data. Verified: all_params derivation, prior/fiducial lookups, probes column
gathers (in all_params order), and the config-checker invariants. Notes for the rerun:

- Only the **grid** needs regeneration. The fiducial/observation products are bit-identical to v16
  at the fiducial (bta = 0 makes `kg + Aia·(ia + 0·ds)` = standard NLA), so `obs/` and the fiducial
  tfrecords can be reused/symlinked from `data/v16/rot_in_place/`.
- The Latin hypercube drops from 7 to 6 dimensions, so **all** astro draws (Aia, n_Aia, bg1–4)
  re-randomize per cosmology — v16/v17 comparisons are ensemble-level, not example-level.
- `normalization.lensing` stds are fiducial-based (bta = 0) and stay valid.

## 4. Ranked hypotheses for the constraining-power gap

1. **delta-NLA `bta` marginalization** (this audit, Section 3.3) — large low-z template freedom.
2. **Prior volume**: w0 ∈ [−2, −1/3], bary_Mc/bary_nu, wide ns/Ob/H0, Aia/n_Aia — the DES Y3 SBI
   comparison papers (Gower-Street-based) fix w0 = −1, near-fix ns/Ωbh²/h, and do not marginalize
   baryons (standing hypothesis, pre-dating this audit).
3. Inherent masked-KS S/N penalty (Section 3.2) — real but only ≈ 10 % in power, affects any
   comparison against idealized forecasts, identical for all analysis variants.
4. *(ruled out)* shape-noise miscalibration, KS/mode-removal amplitude bugs, missing Cl cross terms,
   response/weighting inconsistencies between sims and data.

## 5. Recommended next steps

1. **Condition the inference on bta = 0** with the existing nla drop-and-fix variant in msi and
   compare the WL FoM (2pt first — no retraining needed). This directly measures the cost of `bta`.
2. If `bta` is confirmed as the driver: tighten its prior, or replace the map-level product with
   **shell-level δ-weighting** in the UFalcon projection — the
   `data/v16/rot_in_place/obs/fiducial_bench_nla_per_shell_obs_maps.h5` benchmark already exists for
   validation, and a comparison of Cl(ds) between the two implementations quantifies the
   approximation error directly.
3. Quantify the prior-volume contribution by conditioning the flow on w0 = −1 (and optionally
   narrow ns/Ob/H0, fixed baryons) — separates hypothesis 2 from hypothesis 1.
4. Optional 2pt cross-check: a Gaussian Fisher forecast from the mean/covariance of the binned Cls
   at the fiducial vs. the likelihood-flow posterior width, to isolate density-estimation losses
   from data-vector information content.

## 6. Reproducing the diagnostics

```bash
sbatch dev/scripts/wl_audit/wl_sn_diagnostics.sh
```

runs on one Clariden CPU allocation in ~1 minute (TF container + `~/dlss/tf_env`); inputs are the
repo data files (`DESY3_noise_v11.h5`, `DESY3_pixels_v11_fiducial_512.h5`), one fiducial CosmoGrid
permutation, and (optionally) the stored `grid_cls.h5`. All thresholds/paths are CLI-configurable —
see `wl_sn_diagnostics.py --help`.
