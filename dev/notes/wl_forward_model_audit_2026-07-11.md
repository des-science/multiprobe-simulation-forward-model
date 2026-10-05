# Weak lensing forward-model audit (v16 `rot_in_place` / v17 `baseline`)

**Date:** 2026-07-11, follow-ups through 2026-07-12
**Scope:** conceptual + numerical audit of the weak lensing part of `msfm/apps/run_grid_postprocessing.py`
with `configs/v16/rot_in_place.yaml`; §§3.7–3.8 re-audit the same data path under
`configs/v17/baseline.yaml` (bta dropped)
**Motivation:** the WL part of the inference pipeline (both 2pt/Cls and map-level) is less constraining
than expected. Suspects going in: underestimated S/N, the delta-NLA (`bta`) model, the Kaiser-Squires
inversion, the mode-removal procedure.
**Artifacts:** diagnostics script `dev/scripts/wl_audit/wl_sn_diagnostics.py` (+ `.sh` sbatch wrapper),
results in `/iopsstor/scratch/cscs/athomsen/deep_lss/runs/wl_audit/` (job 2735114:
`wl_sn_diagnostics.npz`, `noise_level.png`, `signal_level.png`, `ds_amplitude.png`).

---

## Executive summary (state as of 2026-07-13)

1. **No bug found anywhere in the WL data path.** KS inversion, mode removal, shape-noise
   calibration (§§3.1–3.2), scale cuts/smoothing and m-bias (§3.5), and the full v17 re-trace
   including exact NEST/RING verification, ℓ_min = 30 (essentially free), and the NLA `tomo_Aia`
   amplitudes (§§3.7–3.8) all check out. Signal, noise, and real-data paths apply the same
   operations in the same order. **The absolute amplitude/scale of the kg and ia *input* maps is
   also correct** (§3.9): their measured full-sky Cls match independent CCL GG/II/GI theory to
   ≈2–3 % over ℓ ∈ [100, 1500], and the NLA A_IA=1/η=0/C1 convention baked into the ia map is
   verified directly — closing the last untested WL data-side hypothesis.
2. **delta-NLA `bta`:** the `ds` template is large at low z (Cl(ds) ≈ 1.2 × Cl(ia) in metacal1,
   §3.3), but fixing bta = 0 on the existing chains changes nothing (§3.4) — the hypothesis
   survives only at **compression level** (network trained across the ×7.7 bin-1 power swing).
   Decisive test = the v17 (bta-free) retrain, still pending.
3. **Wide-prior hypothesis ruled out** (§3.6): an 11-param VMIM retrain + reference priors gains
   only +0.8 % FoM; the ns/Ob/H0/baryon information is simply absent from the lensing Cls. What
   *was* measured instead: the **VMIM target dimensionality** costs constraining power
   monotonically (3p/6p/11p targets → grid-mock FoM 491/404/204), and the fac2 decoupling test
   shows the 3→6 penalty is compression-side (MI target diluting into the IA dims), not flow-side
   — motivating a weighted/two-block GMM head.
4. **Masked-KS E-only penalty ≈ 9 % in σ** (signal pseudo-Cl transfer 0.80 vs noise 0.89, §§3.2,
   3.5c) — inherent, sim/data-consistent, and **shared by the J24/G24 map constructions** (§3.8b),
   so it only matters against idealized forecasts, not in the reference comparison. The more
   conservative joint metacal∩maglim mask costs at most another ≈ 4–6 % (§3.8c).
5. **Residual per-parameter gap vs J24 ≈ 1.26–1.33×** after accounting for the references'
   3D-target choice (~25 % in FoM). Remaining candidates are compression/inference-side, plus
   few-% data-side factors (reference mask width, Cl-estimator conventions).

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
  extension either: real density-weighted IA in the data would not be absorbed correctly. A
  per-shell δ-weighted UFalcon benchmark exists
  (`data/v16/rot_in_place/obs/fiducial_bench_nla_per_shell_obs_maps.h5`); its bench chain shifts
  the contours only minimally, so swapping the implementation is not a fix — v17 drops bta instead.

### 3.4 Follow-up (same day): bta = 0 conditioning on the existing DES chains — **no likelihood-level cost**

Executed the bta = 0 conditioning test using the already-existing drop-and-fix chains of the 2pt lensing run
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
were added for the compressor: `configs/probes/lensing_nla.yaml` and `combined_nla.yaml`
(cross/2x2pt/3x2pt follow the same one-line pattern when needed). *(Correction, same day: the msi
plotting fix and the `_nla` probes configs had not actually been persisted on Perlmutter — they were
(re-)applied in the follow-up session and are uncommitted in the msi and y3-deep-lss working trees.
A `lensing_ext_nla.yaml` mentioned in an earlier draft does not exist — there is no `lensing_ext`
base config.)* The
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

### 3.5 Follow-up (same day): three more hypotheses — smoothing/scale cuts, m-bias, mode removal

**(a) Scale-cut implementation and the unsmoothed ≈ smoothed puzzle — implementation correct, and
the similarity is *expected*.** Verified end to end: the tfrecords carry only the hard ℓ ≥ 30 step
filter (`scales.gaussian_high_pass_factor_alm` with `hard_cut`; `l_max: Null` → low-pass factor of
ones), the network-side cut is θ_FWHM Gaussian smoothing followed by white noise that is added
**unconditionally** (the custom `deepsphere.utils.GaussianNoiseLayer` fires at eval time too,
unlike Keras `GaussianNoise`) and re-masked, and `default_unsmoothed` genuinely built no smoothing
layer (`get_smoothing_kwargs` KeyError → None; no `smoothing/` kernel cache in the run dir, unlike
`t6_default`). The two runs therefore differ exactly as intended. The reason they give similar
constraints is S/N, not a bug: from the stored §3.1 pseudo-Cls, the per-mode S/N `Cl_s/Cl_n` of the
κ maps drops below 1 at ℓ ≈ 30/31/75/87 (bins 1–4; bin 1 never exceeds 1) — everything beyond the
8 Mpc/h l_max cuts (589/863/1159/1382) is deep in shape noise. Gaussian amplitude-Fisher on the
patch pseudo-Cls: F(≤l_max)/F(≤1535) = 0.87/0.91/0.93/0.97 per bin, jointly 0.94 → the expected
σ gain from removing the smoothing is only **≈ 3 %**, within run-to-run scatter. The intuition
"less smoothing must tighten contours" fails because the extra modes carry ~no information; the
null result is itself evidence that the analysis is shape-noise-limited at high ℓ, consistent with
hypothesis 2 (prior volume) rather than a lost-scales explanation.

**(b) Multiplicative shear bias — implemented correctly, magnitude negligible.** Grid: `m ~ N(mu,
sigma)` per bin (`MultivariateNormalDiag`, per-bin independent as in the DES Y3 prior) resampled
per signal example and baked into the tfrecords (`run_grid_postprocessing.py:545`), applied to
`kg` *after* the IA addition (correct — m scales the measured shear including IA), not applied to
the noise (correct — intrinsic-shape scatter is response-calibrated, m is a shear-response bias).
Fiducial pipeline samples once at the fiducial label and reuses the draw for all perturbations
(correct common-random-numbers for the delta loss). Data maps apply only R, never m, matching sims
that multiply (1+m). The per-example resampling is an implicit marginalization with the correct
prior. Cost: σ_m ≈ 0.8 % per bin is an ≲1 % amplitude uncertainty on κ against σ(S8)/S8 ≈ 5 % —
adds < 1 % to the error budget in quadrature. Not the problem.

**(c) Mode removal — does throw away information, but a known, bounded ≈ 9 % in σ.** The procedure
(masked γ → pol `map2alm` → keep E → ×`gamma2kappa` → `alm2map` → re-cut footprint) discards
(i) B-modes (signal-free in theory; the mask leaks E-signal into B, and that leaked part is lost)
and (ii) ringing outside the footprint. This is exactly the §3.2 transfer measurement: signal
pseudo-Cl transfer 0.80 vs noise 0.89. Propagated through the same amplitude-Fisher (undoing both
transfers as the no-loss counterfactual): F ratio 0.85 → **σ inflation ≈ 8.6 %**. Real, but
identical for sims and data (no bias), common to 2pt and map-level (consistent with both being
degraded equally), and inherent to masked KS E-only — avoiding it would mean feeding γ maps
directly to the network or an inpainting/iterative mass-mapping, not a bug fix.

### 3.6 Follow-up (2026-07-12): the wide-prior hypothesis is ruled out for the lensing Cls; a
VMIM-target-dimensionality penalty was measured instead

Hypothesis 2 (prior volume in the implicitly marginalized ns/Ob/H0/baryons) was tested twice on the
2pt lensing pipeline (`runs/v16/rot_in_place/cls/lensing/`, lmax_1024 scale cuts), with escalating
strength; both came back null. All numbers below are blinded (posterior sizes only);
FoM = det(Cov[Om, S8])^(−1/2), S8 = s8·√(Om/0.3).

**(a) Frozen-summary extended conditioning** (`run_inference.py --extend_params`, 2026-07-11): refit
the flow on the existing 6-param summaries but conditioned on all 11 grid params, then rerun the DES
chains with the Gower-Street reference priors (ns/Ωbh²/H0 near-delta Gaussians, baryons fixed at
fiducial; `chain_DESy3_*_refpriors.npy`). Gain: **+2 % (w0>−1+NLA), +5 % (ΛCDM+NLA)**. Ambiguous on
its own (the compression may have discarded the ξ-information before the flow ever saw it).

**(b) Full VMIM retrain** (run `lmax_1024_ext`, job 2735233; `configs/probes/lensing_ext.yaml` +
`configs/loss/vmim_ext.yaml`): the compression itself retrained with the mutual-information target
extended to all 11 grid params, so the summaries are directly rewarded for retaining
(Om,S8)–(ns,Ob,H0,baryon) degeneracy directions. Result: **refpriors gain +0.8 %** (w0>−1+NLA,
FoM 155.7 → 157.0). Direct evidence the information is absent: on 1000 grid mocks the
posterior/prior width ratios of the extension params are ns 0.995, Ob 0.992, H0 0.977, bary_Mc
0.986, bary_nu 0.992 — pure prior, while Om/s8/Aia come out constrained (0.70/0.38/0.27). The
lensing Cl vector at these scales simply does not carry the ns/Ob/H0/baryon information whose
marginalization was hypothesized to cost Om–S8 power. **Hypothesis 2 is dead for the 2pt lensing
analysis** (the map-level analysis inherits the physics argument but was not retested).

**Side discovery — the VMIM target dimension itself costs constraining power, monotonically.** The
11-param run degraded FoM(Om,S8) by ~2–2.7× uniformly (DES wCDM 148 vs baseline 352; grid-mock
median 204 vs 404) while every coverage test stayed acceptable. A 3-param control (run
`lmax_1024_min`, job 2737079; `configs/probes/lensing_min.yaml`, target (Om, s8, w0) only — the
space the DES Y3 SBI reference papers fit their flows in) then *beat* the 6-param baseline:
DES wCDM FoM 452/352/148 and grid-mock median 491/404/204 for the 3p/6p/11p targets. So 3→6 params
costs ~25 % FoM and 6→11 another ~2×. Mechanism: `dim_summary_fac: 1` ties dim(summary) = n_params,
so extending the target scales the hard dimension of *both* stages — the GMM head's modeled
posterior and the flow's random variable s in p(s|θ) (θ, being only a conditioning input, is the
cheap direction). Coverage tests cannot localize such a loss even in principle: a posterior that is
exact Bayes on a lossy statistic of s passes SBC/HPD/TARP identically to the true posterior —
coverage certifies self-consistency, not information content.

**Bottom line for the gap vs J24/G24/W26:** wide implicit priors contribute ~nothing; the
3D-flow-target choice of the references explains ~25 % in FoM; the closest like-for-like comparison
(min run, w0>−1: σ(Om) = 0.0555, σ(S8) = 0.0345 vs J24 Cl 0.044/0.026) leaves a **residual ≈
1.26×/1.33× per-parameter gap** (down from ≈1.5× quoted for the 6-param baseline). Candidates for
the remainder: Cl estimator details, effective ℓ-range, noise treatment, and the §3.2 masked-KS
penalty (≈ 9 % of it).

**Decoupling test (2026-07-12, job 2737601, run `lmax_1024_min_fac2`): the 3→6 target penalty is
entirely compression-side.** Same 3-param target as `lmax_1024_min` but `dim_summary_fac: 2`
(existing `configs/loss/vmim_fac2.yaml`), i.e. a 6-dim summary like the baseline — this separates
the two knobs that `dim_summary_fac: 1` ties together. Grid-mock median FoM2D(Om,S8):
**(3p target, 6-dim s) = 537 vs (3p, 3-dim s) = 491 vs (6p, 6-dim s) = 404.** At fixed 6-dim
summary, growing the target 3p→6p costs the full ~25 %; at fixed 3p target, growing the summary
3→6 dims *gains* ~9 % (matching a better trained MI bound: best head vali −3.93 vs −3.79 in the
same 3-dim θ space). So the NLE flow handles a 6-dim random variable fine in-distribution — the
penalty is what the MI target rewards: the 6-param target dilutes summary information into the IA
directions. This motivates a weighted/two-block GMM head (full-weight (Om,s8,w0) block +
down-weighted IA block) as the fix that keeps explicit IA posteriors; the 6→11 leg remains
unsplit (flow-side cost may well kick in at 11 dims). Caveats: (i) SBC ranks show the same mild
low-Om tilt as min/baseline (mean rank ≈ 0.477, shape shared by all three runs — not
fac2-specific), but **l-C2ST at the DES point rejects (p = 0.020)** for this run only, and the DES
FoM (wCDM 379) sits at 0.71× its own mock median vs 0.87–0.92× for the other runs — redundant
summary dimensions appear more fragile exactly at the real-data point (out-of-distribution
directions the 3-param target does not pin down), echoing the historical fac=2 overconfidence note
in `configs/loss/cls/vmim_flow_fac1.yaml`. `lmax_1024_min_fac2` is a test artifact, not a
production candidate; any future dim(s) > n_target design must check l-C2ST at the data point.

**Head-conditioning tests (2026-07-12, jobs 2737882 `lmax_1024_std` GMM + 2737883
`lmax_1024_flow_std` RealNVP head; both 6-param target, fac=1, standardize_theta): the mock-level
"target-dimensionality penalty" was mostly a HEAD-OPTIMIZATION artifact — but the recovered
information does NOT materialize on the DES data.** Standardizing θ inside the variational head is
affine-invariant for the MI bound, so any change is pure optimization conditioning (the physical
6-param target spans ~30× in scale, Om vs n_Aia); it was newly implemented for the flow head
(`nets/estimators/normalizing_flow.py` + `utils/mutual_info_loss.py`, mirroring the GMM's
physical-units log-Jacobian convention). Results:

- *Grid-mock median FoM2D:* flow_std **538**, std **526**, min_fac2 537, min 491, baseline 404.
  Three different interventions (6-dim s for a 3p target; std GMM 6p; std flow 6p) converge at
  ~535, +30 % over the baseline — the earlier 3p-vs-6p "dilution" reading is revised: the
  unstandardized head was under-training on the tight (Om, s8) directions, and the
  scale-homogeneous 3p target had escaped that penalty for free, mimicking a target-content
  effect. Head NLLs (physical units, same 6p target, directly comparable): flow_std −2.29 vs
  GMM-std −1.86 — with standardization the flow head trains stably (the historical instability
  was raw θ feeding the coupling MLPs directly) and reaches the *tighter* bound; mock FoM tracks
  the bound monotonically across all runs.
- *DES data point:* wCDM FoM2D flow_std 341 / std 361 / baseline 352 / min_fac2 379 / min 452 —
  the +30 % mock gain does NOT transfer. DES percentile within near-fiducial grid mocks
  (|ΔOm| < 0.05, |Δs8| < 0.10, n = 74): baseline 53rd, min 62nd, min_fac2 35th, std 24th,
  flow_std **12th** — a monotone slide as the compression extracts more information, while
  l-C2ST stays non-rejecting for std (p = 0.28) and flow_std (p = 0.41), i.e. the flows are
  locally calibrated and the DES posteriors are *genuinely* wider than mock ones. The extra
  Cl-vector features the improved summaries exploit yield constraining power on simulations but
  not on the real data — the expected signature of a mild data-vs-sim mismatch in fine Cl
  features (candidates: uniform sim noise depth, baryon/IA model residuals, data systematics)
  that the blunt baseline compression could not see. Consistently, the 3-dim min summary is the
  *best on data* (452) despite being the weakest of the improved variants on mocks — fewer
  summary dimensions expose less surface to off-manifold data features. Follow-up discriminator:
  PPC (`run_ppc.py`) on the std/flow_std summary spaces, or locating the DES summary within
  p(s|θ̂) along the extra summary directions.

**Flow-head ablation + productionization (2026-07-12, jobs 2738338–42, `lmax_1024_flow_{p6h128,
p4h128,np4h128,p6h64,p4h64}`):** grid over couplings {4,6} × permutation {on,off} × conditioner
width {64,128}, all fac = 1 + standardized (via the new default), vs the flow_std reference
(6×128, no perm). Head bound ranking (best vali NLL): p6h128 −2.312 > np6h128 −2.292 > p4h128
−2.210 ≈ np4h128 −2.206 > p6h64 −2.188 > p4h64 −2.124 — depth and width are worth ~0.1 nat each,
the roll-by-1 permutation between couplings (zero parameters, fixes the static half-split that
kept Om/s8/w0 from conditioning on each other directly) a small free gain. Downstream, however,
the mock median FoM2D is FLAT across the entire grid (526–547; l-C2ST clean everywhere, np4h128
marginal at p = 0.05) and uncorrelated with the bound — past the standardization fix, head
capacity is not a binding constraint on the 6-param target. DES wCDM FoM scatters 325–412 across
variants (seed noise at the low-percentile data point; consistent with the mismatch signature
above). Consequences: `standardize_theta` now defaults to true in BOTH training apps (Cls app:
empirical label stats; maps `run_training.py`: newly supported, analytic uniform-prior stats +
`density_estimator` pass-through — valid because the MI bound is affine-invariant),
`configs/loss/vmim.yaml` = flow head 6×128 `permute: true` (chosen for bound tightness/headroom
on harder targets, e.g. the 11p ext, not for FoM), old GMM default preserved as `vmim_gmm.yaml`;
redundant loss configs (vmim_std, vmim_flow_std, vmim_flow, vmim_ext, the `configs/loss/cls/`
duplicate subdir) removed and dev submission scripts repointed. The low-Om SBC tilt persists in
all standardized runs (KS p ~1e-3–1e-4, slightly stronger than baseline) — shared, not
architecture-specific. NOTE: maps benchmarks (`bench_t*`) trained via vmim.yaml after this date
use flow + standardized heads — head-loss values are not comparable to pre-flip runs.
Update (later same day): the vmim.yaml default was reduced to 4 couplings (flow 4×128 + permute)
on the flat-downstream-FoM justification; 6 couplings remain a documented opt-in for harder
targets.

**Target-dimensionality rerun with the flow+standardized head (2026-07-12, jobs 2738422
`lmax_1024_flow_min` 3p / 2738423 `lmax_1024_flow_ext` 11p; the 6p point reuses
`lmax_1024_flow_p4h128`, identical config): the dimensionality penalty is dead at the mock
level.** Grid-mock median FoM2D: 3p 490 / 6p 533 / 11p 488, vs the GMM-era 491 / 404 / 204 —
the catastrophic 11-param collapse (§3.6, "COSTS ~2–2.7x") was entirely head conditioning; the
3p target never suffered it because it is scale-homogeneous. With a well-conditioned head the
MI target's content barely matters on mocks (6p even mildly best). On DES (blinded, sizes only)
the ordering is different: wCDM FoM2D 3p 409 / 6p 383 / 11p 298 (GMM-era 452 / 352 / 148) —
the ext run doubles but still lags on data while matching on mocks, i.e. the data-vs-sim
mismatch penalty of §3.6 *grows with target dimensionality* (more summary directions exposed to
off-manifold features); the 3-param compression remains the most robust on data. λCDM FoM2D:
585 / 629 / 474. Coverage: l-C2ST clean for both new runs (min p = 0.060, ext p = 0.530);
flow_ext SBC is the only run where the low-Om tilt VANISHES (p = 0.52) but bta fails hard
(p = 4e-5; the least-constrained, likelihood-flat parameter) with H0/Aia marginal
(0.023/0.046); flow_min keeps the Om tilt (p = 0.011). Head NLLs (−3.921 min, −3.177 ext) are
not comparable across targets. All `lmax_1024_flow_*` runs are test artifacts, not production.

Caveat discovered en route (cost one wasted run, `lmax_1024_ext.broken_raw_mc`): the stored label
tables keep `bary_Mc` RAW (1e12–1e15) while configs/priors/inference use log10(Mc), and there are
TWO independent label-gather sites (`deep_lss/utils/cls_preprocessing.py` for Cls runs,
`msi/utils/preprocessing.py` for map runs). Both now convert raw→log10 behind a
`min > 1e10` assert. Failure signature if ever regressed: compression vali_mse ~1e27 (or, with
theta standardization, a silently absorbed +33-nat loss offset), flow NaN from epoch 0, SBC KS
p = 0 for all params.

### 3.7 Follow-up (2026-07-12): v17 (bta = 0) data-flow re-audit — clean; NEST/RING and ℓ_min
verified numerically

Full re-trace of the WL path from the full-sky CosmoGrid inputs to the .tfrecords, restricted to
`configs/v17/baseline.yaml`. **No bug found.** New verifications beyond §§2–3 (scripts:
scratchpad `check_pixel_file.py`, `check_ordering.py`, `lmin_fisher.py`):

- **v17 wiring**: the astro Latin hypercube is 6-dim `[Aia, n_Aia, bg1–4]`; the
  `astro_sample[:2]` / `[-4:]` unpacking has no overlap at d = 6; `ds` is never built (the
  `dg -> sn` pair's input is ignored by `in_place`); the standard-NLA branch
  `kg + tomo_Aia·ia` is taken; the label vector is 8 cosmo/bary + 6 astro = 14 (bary_Mc raw,
  known convention).
- **NEST/RING, exact check**: `data_vec_pix[corresponding_pix] == ring2nest(512, base_patch_pix)`
  holds *exactly* for all four metacal bins → patches RING, data vector NEST, both index maps
  mutually consistent. `gamma2_signs = [1, 1, −1, −1]` (last two patches mirrored). The stored
  CosmoGrid probe maps are empirically RING at nside 1024 (red Cl as-RING vs. near-flat as-NEST),
  so `ud_grade(order_in="RING")` and the reorder round-trips in `scales.py` are all correct. The
  noise-catalog pixel column is RING and footprint-consistent.
- **ℓ_min = 30 is essentially free**: Gaussian amplitude-Fisher with the §3 signal/noise Cls,
  holding Cl_s(ℓ<30) = Cl_s(30) as an upper bound: extending to ℓ ≥ 8 (or ℓ ≥ 2) gains ≤ 1.2 %
  in σ for metacal1 and ~0.3 % jointly. The CosmoGrid box-replication cut costs nothing
  measurable; the ℓ_min side of the "effective ℓ-range" residual-gap candidate (§3.6) is closed.
  The tfrecords keep everything up to the band limit 3·n_side−1 = 1535 (`l_max: Null`), so no
  small-scale information is discarded before the network-side smoothing — correct design.
- **Minor, consistent-by-construction observations (not bugs)**: (i) ~11 % of noise-catalog
  galaxies (~14 % of touched pixels) lie outside the 318,616-pixel joint analysis footprint and
  are silently dropped in `noise_gen_in_place` — the data maps are cut to the same footprint, and
  per-pixel noise inside it is unaffected (cost quantified in §3.8c); (ii) no pixel-window
  mismatch from the 1024→512 `ud_grade`: averaging the four equal-area children reproduces
  exactly the parent-pixel average, the same nside-512 construction as the data's catalog
  binning; (iii) the 4 patches per permutation are exactly
  disjoint, and the ℓ ≥ 30 cut (correlation length ≈ 6°) plus independent noise decorrelates
  them up to thin boundary strips (4 of 6 patch pairs sit ~2–3° apart) — effectively independent
  training examples; (iv) UFalcon maps carry a large monopole (mean ≈ 3.5e−2 in kg metacal3)
  which is correctly removed by the patch mean removal and the ℓ = 0, 1 zeroing of the KS
  factors.

Mode removal and the signal/shape-noise calibration are unchanged from v16 (§§3.1, 3.2, 3.5c
apply verbatim to v17). Bottom line: the v17 data path is clean; nothing in it can explain the
constraining-power gap beyond the known ≈ 9 % masked-KS penalty. The decisive open test remains
the v17 retrain (compression-level bta hypothesis, §3.4).

### 3.8 Follow-up (same day): tomo_Aia verified; masked-KS penalty common to the reference
papers; masking cost bounded

**(a) tomo_Aia / NLA redshift evolution — correct.** Exact reimplementation of
`redshift.get_tomo_amplitudes` on the v17 n(z) files (scratchpad `check_tomo_aia.py`; the
notebook `dev/notebooks/systematics/redshift_evolution.ipynb` motivated the 0.05/0.95 quantile
truncation). All four metacal n(z) have the artifact spike at the last grid point (z ≈ 3, ~0.2 %
of the mass, nz jumping 30–200× above its neighbors); the quantile truncation excises it
completely: hand-zeroing the last 10 grid points shifts the *truncated* amplitude by ≤ 0.25 %
(vs 3–4 % at η = 3 and a catastrophic ×6 at η = 6 without truncation, where the z = 3 spike
carries weight ((1+3)/(1+0.62))^6 ≈ 224). Fiducial (Aia = 0.5, η = 1.5) tomo_Aia =
[0.360, 0.442, 0.558, 0.652], monotone as expected. All three production call sites read the
config truncation settings (grid `lensing_transform`, `parameters.py` fiducial/perturbations,
`observation.py`), so grid/fiducial/obs are mutually consistent. Two footnotes: (i) the
truncation *redefines* the effective (Aia, η) → per-bin-amplitude map (bin 1 at η = 3: 0.53
truncated vs 0.68 full) — internally consistent, but the inferred η is not directly comparable
to the DES Y3 continuous-NLA convention; (ii)
`get_tomo_amplitudes_according_to_config_vectorized` hardcodes its truncation defaults instead
of reading the config (currently no production callers — dormant footgun).

**(b) Masked-KS penalty (§§3.2/3.5c) is shared by J24/G24 — not a differentiator.** The two
factors behind the §3.2 transfers: the footprint boundary scatters true E-mode signal into
pseudo-B, which the E-only convention discards (~×0.90, signal only — isotropic noise scatters
symmetrically) and the post-KS footprint re-cut discards ringing outside the mask (~×0.89,
signal and noise alike) → power-level S/N ×0.90 → the σ ≈ +8.6 % of §3.5c. The reference papers
build their κ maps with the same masked spherical-KS E-only construction (our observation
building mirrors eq. (10) of 2403.02314), so the penalty is a property of the statistic and
cancels in the comparison — it matters only against idealized (full-γ / Fisher) forecasts.
Second-order estimator differences (e.g. whether their Cl estimator re-cuts the footprint after
inversion) could move a few % at most.

**(c) Cost of the more conservative joint (metacal ∩ maglim) mask: ≈ 4–6 % in σ, upper bound.**
Our footprint keeps 318,616 of ~371k catalog-touched pixels (~4178 deg² of full pixels) and
88.6–90.3 % of the source galaxies per bin. If the references keep every covered pixel of the
lensing-only footprint, the bound on their advantage is: signal-dominated limit F ∝ covered
area → σ +6 %; noise-dominated limit (boundary pixels are ~0.8× depth, contribute ∝ coverage²)
→ σ +4 %. Real but small — a ~1.05× slice of the residual 1.26–1.33× gap, and only if their
mask is actually wider at nside 512 (unverified).

### 3.9 Follow-up (2026-07-13): full-sky input maps vs. independent CCL theory — kg/ia amplitude and scale are CORRECT; hypothesis ruled out

Every prior signal check (§§3.2, 3.3, 3.5c) divided the pipeline output by each map's *own*
full-sky Cl, so a wrong absolute amplitude or scale baked into the `kg` (lensing) or `ia`
(NLA intrinsic-alignment) input maps themselves would have passed unnoticed while still lowering
the S/N of the forward-modelled data vector. This is the one WL data-side hypothesis §§2–3.8 never
tested. Now closed. Script: `dev/scripts/wl_audit/wl_theory_comparison.py` (Perlmutter CPU +
`~/.local` pyccl 3.2.1 / camb; maps on CFS
`/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/fiducial/cosmo_fiducial`).
Results in `/pscratch/sd/a/athomsen/deep_lss/runs/wl_audit/wl_theory_comparison.{npz,png}`.

Method: measured full-sky auto/cross pseudo-Cls of the nside-1024 `kg`/`ia` maps (healpy
`anafast`, deconvolved by the nside-1024 pixel window, averaged over perms) vs. an **independent**
pyccl prediction at the exact CosmoGrid fiducial cosmology (`CosmoGridV1_metainfo.h5`:
Ω_cdm = 0.20928, Ω_b = 0.0493, h = 0.6736, σ8 = 0.84, n_s = 0.9649, Σm_ν ≈ 0.06 eV, w0 = −1;
halofit) built from the **stored** per-bin `n(z)` (`/nz/metacal*` in the map file, the same n(z)
the projection used):
- `kg` → GG (`WeakLensingTracer`, shear only)
- `ia` → II (`WeakLensingTracer`, `has_shear=False`, `ia_bias=1`, `use_A_ia=True`), i.e. the NLA
  auto at A_IA = 1, η = 0 — exactly the amplitude/convention baked into the map. **The two NLA
  normalisations are numerically identical** (verified 2026-07-13): CCL applies
  `−A_ia·5e-14·RHO_CRITICAL·Ω_m/D(a)` and UFalcon `F_NLA = −A_IA·C1·ρ_crit,0·Ω_m/D(z)` with
  C1 = 5e-14 h⁻², and both products equal **0.013876831** to 10 significant figures (the often-quoted
  0.0134 is a different ρ_crit rounding — not what either code uses). Consequently any residual II/GI
  offset can only come from the growth factor D — which is exactly what the w0 scan below finds. The
  forward model applies the per-bin `tomo_Aia = Aia·⟨((1+z)/(1+z0))^n_Aia⟩` on top, so η = 0 in the
  map is correct — no double counting.
- `kg × ia` → GI (negative cross).

Result — **the input maps are correct in both absolute amplitude and scale dependence.** Two
independent map variants were measured: dark-matter-only (`v11dmo`, the clean match to halofit) and
the baryonified maps actually used in production (`v11dmb`). Median measured/theory over ℓ bins
(bins 1–4; dmo unless noted):

| probe | ℓ∈[100,300] | [300,600] | [600,1000] | [1000,1500] |
|---|---|---|---|---|
| kg / GG (dmo) | 0.98–0.99 | 0.98 | 0.98 | 0.98–0.99 |
| ia / II (dmo) | 0.98–0.99 | 0.97–0.99 | 0.97–0.98 | 0.98 |
| kg·ia / GI (dmo) | 0.98–0.99 | 0.97–0.98 | 0.97–0.98 | 0.98–0.99 |

All twelve spectra (4 bins × {GG, II, GI}) sit within ≈1–3 % of the independent theory across the
full ℓ ∈ [30, 1500], flat in ℓ (8-perm run; the ≈0.95 dip at ℓ ∈ [30, 100] seen in an early 2-perm
run was cosmic variance over the few low-ℓ modes and vanishes at 8 perms → ≈0.99, so no finite-box
deficit above the ℓ_min = 30 cut). The only ℓ-dependent departure is in the **production (dmb) maps**: they
track dmo/theory to ℓ ≈ 300 then droop to ≈0.83–0.87 (kg), ≈0.84–0.95 (ia), ≈0.83–0.92 (cross) by
ℓ = 1000–1500 — the **expected baryonic suppression** of small-scale power (halofit is DMO), shared
by all three probes and therefore not an amplitude/scale misspecification. Consequences:

1. **kg amplitude/scale: correct.** No lensing-efficiency, prefactor, or Born-projection error.
2. **ia amplitude/scale AND the NLA normalization (A_IA=1, η=0, C1) convention: correct.** ia/II ≈ 1
   directly verifies the intrinsic-alignment map is neither under- nor over-amplified; the physical
   IA signal is `tomo_Aia · ia` with the fiducial `tomo_Aia = [0.360, 0.442, 0.558, 0.652]` (§3.8a),
   which is small vs. lensing — as it should be, not because the template is misnormalised.
3. **Relative kg↔ia amplitude: correct.** The GI cross matches theory (right sign and magnitude),
   so there is no relative mis-scaling between the two probes that could dilute the combined
   `kg + tomo_Aia·ia` S/N.

The user's hypothesis — a misspecified amplitude/scale for kg or ia reducing the map-level S/N — is
therefore **ruled out**. The WL data path is clean end to end, including the absolute normalisation
of its inputs; the constraining-power gap remains compression/inference-side (§4 items 1–2).

**Cosmology scan (2026-07-13, `dev/scripts/wl_audit/wl_theory_scan.py`): the check holds across the
prior, and it surfaced one real (benign) IA-map approximation.** Repeated the measured-vs-CCL
comparison at 3 grid cosmologies spanning S8 (0.58 / 0.79 / 0.98 at w0 ≈ −1) and 3 spanning w0
(−1.58 / −1.24 / −0.56 at S8 ≈ 0.75–0.85). Only the baryonified (v11dmb) maps exist off-fiducial, so
ℓ ∈ [30, 300] is the clean amplitude/scale band (baryons negligible). **Directory indexing gotcha
(cost one wrong run): the CosmoGrid grid directory is `cosmo_{sobol_index:06d}` = `path_par`, NOT
`cosmo_{id_param}` — sobol_index ≠ id_param (e.g. id_param 1412 → sobol 28190 → `cosmo_028190`); the
cosmology must be paired by matching `sobol_index` in the metainfo.** Findings (median over
ℓ ∈ [100, 300], 2 perms):

- **kg / GG (lensing): correct everywhere.** Ratio 0.97–1.02 at ℓ ∈ [100, 300] (0.97–1.04 over the
  full ℓ < 300 band) across the *entire* S8 = 0.58–0.98 and w0 = −1.58 to −0.56 range. The
  cosmological (lensing) signal — the part that carries the constraining power — has the right
  amplitude and scale at every cosmology tested.
- **ia / II (intrinsic alignment): a flat-in-ℓ, cosmology-dependent amplitude offset.** ia/II runs
  from 1.16 (w0 = −1.58) to 0.88 (w0 = −0.56); the GI cross tracks its square root, i.e. the offset
  is a pure multiplicative factor `f` on the IA field (II ∝ f², GI ∝ f, GG ∝ 1). `f` correlates with
  w0, not S8, and is ≈1 (0.99) at the near-fiducial points.
- **Attributed to the IA growth-factor convention.** `f` matches, to ≈1–2 %, the ratio of the NLA
  weight `∫ n(z)/D(z) dz` computed with UFalcon's *analytic* ΛCDM-form growth integral
  (`g = 5Ω_m/2·E(z)·∫₀^a da'/(a'E)³`, `probe_weights.F_NLA_model`) vs. CCL's true wCDM growth ODE
  (measured `f` vs. D-ratio: 1.080/1.081, 1.044/1.049, 1.001/1.010, 0.988/1.008, 0.961/0.988,
  0.938/0.959 for w0 = −1.58…−0.56). The analytic integral is exact at w0 = −1 but drifts up to
  ≈±8 % at the w0 prior edges. The lensing (kg) field is ray-traced from the actual sim density, so
  it does **not** inherit this approximation (hence GG stays exact); only the IA weight, which
  multiplies that same field by `F_NLA(z)`, does.
- **Impact: negligible for the constraining-power question.** (i) It is an IA-only, subdominant
  effect (physical IA is `tomo_Aia · ia` with `tomo_Aia ≈ 0.4–0.65`); (ii) it does not touch the
  correctly-normalised lensing signal or its S/N; (iii) the IA amplitude `Aia` is a marginalised
  nuisance (prior [−3, 3]), so a mostly-multiplicative IA rescaling is largely absorbed. The one
  genuine subtlety is that the offset is w0-dependent, so it introduces a mild, unphysical
  w0–Aia coupling in the IA sector — worth knowing, but a second-order effect on a nuisance
  direction, not a driver of the Om/S8 gap. (A future fix, if ever wanted, is to build the IA maps
  with the ODE growth for wCDM; not worth a regeneration for this analysis.)

## 4. Ranked hypotheses for the constraining-power gap (state as of 2026-07-13)

The data path is clean (§§2–3.8), so the live candidates are all compression/inference-side:

1. **VMIM target dimensionality** (§3.6) — *measured*, mechanism understood: the 6-param target
   costs ~25 % FoM vs the references' 3-param target, and the fac2 decoupling localizes the loss
   to the MI target diluting summary information into the IA dims (compression-side, not
   flow-side). Candidate fix: weighted/two-block GMM head.
2. **Compression-level `bta` penalty** (§§3.3–3.4) — the network was trained across a ×7.7 bin-1
   IA-power swing and may have discarded information that bta = 0 conditioning cannot recover
   (the likelihood-level cost is measured to be zero). Decisive test = the v17 (bta-free) retrain.
3. **Unexplained remainder** — after item 1, the closest like-for-like comparison leaves a
   ≈ 1.26–1.33× per-parameter gap vs J24 (§3.6). Data-side factors can only nibble at it: the
   conservative joint mask is ≤ 4–6 % in σ *if* the references keep boundary pixels (§3.8c,
   unverified); Cl-estimator conventions a few % at most (§3.8b).
4. **Constants, not differentiators:** the masked-KS E-only penalty (≈ 9 % in σ, §§3.2/3.5c) is
   shared by the J24/G24 constructions (§3.8b) and matters only vs idealized forecasts.
5. *(ruled out)* prior volume in ns/Ob/H0/baryons (§3.6; the w0-prior part is separately modest:
   w0>−1 vs wCDM FoM 426 vs 352 in the baseline); effective ℓ-range on both ends (ℓ_min = 30 is
   free, §3.7; high ℓ is shape-noise-dominated, unsmoothed ≈ smoothed expected, §3.5a);
   shape-noise miscalibration, KS/mode-removal amplitude bugs, missing Cl cross terms, NEST/RING
   errors, sim-vs-data response/weighting or pixel-window inconsistencies (§§3.1–3.2, 3.7);
   tomo_Aia / NLA redshift evolution (§3.8a); m-bias marginalization (< 1 %, §3.5b);
   **absolute amplitude/scale of the kg and ia input maps** (§3.9: measured full-sky Cls match
   independent CCL GG/II/GI theory to ≈2–3 % over ℓ ∈ [100, 1500]; the NLA A_IA=1/η=0/C1
   convention is verified directly).

## 5. Next steps

1. **v17 retrain** — the decisive compression-level bta test. Only the grid needs regeneration;
   fiducial/obs products are reusable from v16 (§3.4).
2. **Weighted/two-block GMM head** (full-weight (Om, s8, w0) block + down-weighted IA block,
   §3.6) — targets the measured ~25 % dimensionality penalty while keeping explicit IA
   posteriors. Any dim(s) > n_target design must check l-C2ST at the data point (fac2 caveat).
3. Pin down the remaining data-side few-%: check the references' nside-512 mask width and
   Cl-estimator conventions (§§3.8b–c).
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
