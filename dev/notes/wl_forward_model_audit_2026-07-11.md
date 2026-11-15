# Weak lensing forward-model audit (v16 `rot_in_place` / v17 `baseline`)

**Date:** 2026-07-11, follow-ups through 2026-07-20 (streamlined 2026-07-16; pre-streamline
version in git / `.wl_forward_model_audit_2026-07-11.md.pre-streamline`; pre-B-mode-Fisher version in
`.wl_forward_model_audit_2026-07-11.md.pre-bmode`)
**Scope:** conceptual + numerical audit of the WL part of `msfm/apps/run_grid_postprocessing.py`
with `configs/v16/rot_in_place.yaml`; §§3.5–3.6 re-audit the same path under `configs/v17/baseline.yaml`
(bta dropped); §3.6d adds a direct B-mode Cls information-loss Fisher (commit `42e3b83`); §3.8
validates the full-sky kg/ia input maps against independent CCL theory; §4 investigates the
compression/inference side; §4.7 adds a Gaussian Cl Fisher forecast validated against the SBI
likelihood-flow.
**Motivation:** the WL inference (2pt and map-level) is less constraining than expected. Suspects
going in: underestimated S/N, delta-NLA (`bta`), the KS inversion, the mode-removal procedure.
**Artifacts:**
- S/N diagnostics: `dev/scripts/wl_audit/wl_sn_diagnostics.py` (+ `.sh`), results in
  `/iopsstor/scratch/cscs/athomsen/deep_lss/runs/wl_audit/` (jobs 2735114, 2752929).
- Fisher forecast (§4.7): `msi/dev/fisher/fisher_cls.py` + `fisher_vs_chain.py`, outputs in
  `runs/v17/baseline/fisher_cls/<scale>/nb<N>/`; production `8wl,32gc` (jobs 2774744/2774810/2774811
  for nb16/8/32, SBI overlay in `8wl,32gc/nb16/chain_comparison/`), scale sweeps `lmax_1024`
  (2774871–73) and `unsmoothed` (2774874–76), each × nb8/16/32.
- Input-map theory check (§3.8): `dev/scripts/wl_audit/wl_theory_{comparison,scan}.py` +
  `README_wl_theory_check.md`; full-sky CosmoGrid maps on Clariden at
  `/iopsstor/scratch/cscs/athomsen/CosmoGrid/v11desy3/` (copied from Perlmutter CFS 2026-07-16).
- B-mode Cls Fisher (§3.6d, commit `42e3b83`): unified forecast infrastructure `msfm/utils/fisher.py`
  + `msfm/apps/run_fisher_forecast.py` driven by `configs/v17/fisher_bmode.yaml`; write-up and
  Jacobian-source reducer in `dev/bmode_fisher/{REPORT.md,reduce.py}`; data products under
  `.../v17/baseline_bmode/cls/` (E+B tree) and `.../v17/baseline/cls/` (ℓ_min=30 E-only tree). The
  dataset-producing code is gated behind `analysis.modelling.lensing.b_mode_cls` (default off → a
  normal run is byte-identical; staged/uncommitted per REPORT), with explicit E-mode-pinning configs
  `configs/v17/baseline_emode.yaml`, `configs/v16/rot_in_place_emode.yaml`.

---

## Executive summary (state as of 2026-07-16)

1. **No bug found anywhere in the WL data path.** KS inversion, mode removal, shape-noise
   calibration (§§3.1–3.2), scale cuts/smoothing/m-bias (§3.3), the full v17 re-trace incl. exact
   NEST/RING checks and ℓ_min = 30 (§3.5), and the NLA `tomo_Aia` amplitudes (§3.6a) all check out.
   Signal, noise, and real-data paths apply the same operations in the same order. The **absolute
   amplitude/scale of the kg/ia input maps is also correct** (§3.8): full-sky Cls match independent
   CCL GG/II/GI theory to ≈ 1–3 % over ℓ ∈ [30, 1500], kg/GG across the whole S8/w0 prior — the
   last untested data-side hypothesis is closed.
2. **delta-NLA `bta`: CLOSED at both levels (2026-07-18).** The `ds` template is large at low z
   (Cl(ds) ≈ 1.2 × Cl(ia) in metacal1, §3.4a), but fixing bta = 0 on the existing chains changes
   nothing at likelihood level (§3.4b), and the **v17 (bta-free) retrain confirms the
   compression-level effect is negligible too** (§3.4d): mock σ(S8) improves ~5 %, DES w0>−1
   σ(S8) unchanged — within the pre-registered ≲ 5 % bound from the §3.7 S/N budget, and an
   *upper* bound on the bta share since the v17 run also switched to the better flow+std head
   (§4.3). bta was never the gap.
3. **Wide-prior hypothesis ruled out** (§4.1): a full 11-param VMIM retrain + reference priors
   gains only +0.8 % FoM; the ns/Ob/H0/baryon information is absent from the lensing Cls.
4. **The GMM-era VMIM target-dimensionality penalty was a head-conditioning artifact** (§§4.2–4.5):
   with standardized θ + a flow head, 3p/6p/11p targets give mock FoM 490/533/488 — the penalty is
   dead on mocks. But the recovered information does **not** transfer to DES: the data posterior
   slides to the 12th–24th percentile of near-fiducial mocks as compression sharpens, while l-C2ST
   stays clean → signature of a mild **data-vs-sim mismatch in fine Cl features** (§4.3), now the
   leading candidate.
5. **Masked-KS E-only penalty ≈ 9 % in σ** (§§3.2, 3.3c) — a real, inherent, sim/data-consistent
   mask cost, but **NOT a 2-point J24 differentiator**: the direct B-mode Cls Fisher (§3.6d, commit
   `42e3b83`) measures keeping the B channel to be 2-point-lossless (FoM +0.04 %), so the mask-leaked
   E→B signal J24 was assumed to recover is unrecoverable — most plausibly a common mask cost (the
   airtight check is the J24-matched shear EE+BB Fisher, §6 item 6). Only field-level non-Gaussian
   κ_E+κ_B could still differ. The conservative joint metacal∩maglim mask costs another ≈ 4–6 %
   (§3.6c).
6. **The 2pt gap is NOT inference-side** (§4.7): the SBI likelihood-flow reproduces the Gaussian
   Cl Fisher ceiling (combined σ ratios chain/Fisher 1.07–1.14). Residual per-parameter gap vs J24
   ≈ 1.26–1.33× after the references' 3D-target choice (~25 % in FoM); dividing out only the
   mask-width bound (≤ ~1.05×) leaves **~1.2–1.27× unexplained** (revised up 2026-07-20: the E-only
   slice is no longer subtracted, since §3.6d measures keeping B to be 2-point-lossless), to be
   closed by nuisance-prior differences plus the item-4 mismatch (the Fisher decomposes the
   marginalization: IA ~5 %, full nuisance set ~30 % on lensing S8; clustering s8 ×5.6 against
   linear bias; 3×2pt breaks both).

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
| Noise calibration | full metacal response `e/(R_gamma + R_s)` applied when building the noise file; real per-pixel depth via actual galaxy positions/weights — `in_place` *fixes* the uniform-depth issue of the older `count` method |
| m-bias | sims multiply (1 + m), data maps are *not* m-corrected (only R) — consistent; per-example resampling = implicit marginalization |
| Ordering | mask support → mode removal → per-patch mean removal → mask → hard ℓ ≥ 30 → mask, identical in `lensing_transform` and `forward_model_observation_map` |
| Cls | signal and noise alms summed before squaring → signal×noise cross terms present; pseudo-Cl convention identical to the observation |
| Bookkeeping | `(Aia, n_Aia, bta)` unpacking matches the Latin-hypercube order; Sobol cross-checks; `_verify_tfrecord` round trip; noise independent across z-bins; 80 signal × 5 noise structure |

## 3. Data-path numerical diagnostics (job 2735114)

### 3.1 Shape-noise level — calibrated, no S/N bug

Empirical pseudo-Cl of `in_place` noise kappa_E maps vs the analytic
`N_ℓ = (4π/N_pix²) · Σ_patch [Σw²|e|²/(2(Σw)²)] · ℓ(ℓ+1)/((ℓ+2)(ℓ−1))`, median ℓ ∈ [300, 1500]:
**0.8910 / 0.8886 / 0.8894 / 0.8859** (bins 1–4). The uniform ≈ 11 % deficit is the footprint
re-cut after mode removal discarding ringing outside the patch — the data receives the identical
treatment, so a consistent convention, not a miscalibration. Cross-check: the high-ℓ (1200–1500)
plateau of the stored grid Cls sits at 1.07/1.01/1.02/1.01 × analytic noise (noise + small
signal/IA on top) — the tfrecords carry the right noise level end-to-end.

### 3.2 Signal transfer through KS + mode removal — no amplitude loss beyond known mechanisms

Pseudo-Cl of a fiducial kg map pushed through the exact pipeline chain, divided by f_patch ×
full-sky Cl (median ℓ ∈ [100, 1000]): **0.798 / 0.798 / 0.800 / 0.805**. Decomposition: ≈ 0.89
(footprint re-cut, same as noise) × ≈ 0.90 (E→B leakage discarded). Both apply identically to sims
and data. Net: signal keeps 80 % of its pseudo-power, noise 89 % → inherent ≈ 10 % power-level S/N
penalty of the masked-KS E-only design (σ propagation in §3.3c). Worth knowing, not a bug.

### 3.3 Scale cuts / smoothing, m-bias, and mode removal

**(a) Scale cuts and the unsmoothed ≈ smoothed puzzle — implementation correct; the similarity is
expected.** Verified end to end: tfrecords carry only the hard ℓ ≥ 30 cut (`l_max: Null` → low-pass
factor of ones); the network-side cut is θ_FWHM Gaussian smoothing + white noise added
**unconditionally** (custom `deepsphere.utils.GaussianNoiseLayer` fires at eval time too, unlike
Keras) + re-mask; `default_unsmoothed` genuinely built no smoothing layer (`get_smoothing_kwargs`
KeyError → None; no `smoothing/` cache in the run dir). The runs differ exactly as intended; the
similar constraints are S/N: per-mode `Cl_s/Cl_n` drops below 1 at ℓ ≈ 30/31/75/87 (bin 1 never
exceeds 1) — everything beyond the 8 Mpc/h l_max cuts (589/863/1159/1382) is deep in shape noise.
Amplitude-Fisher on the patch pseudo-Cls: F(≤l_max)/F(≤1535) = 0.87/0.91/0.93/0.97 per bin, 0.94
jointly → expected σ gain from unsmoothing ≈ **3 %**, within run-to-run scatter. The analysis is
shape-noise-limited at high ℓ. (Confirmed on the full data vector by the §4.7 scale sweep.)

**(b) Multiplicative shear bias — correct, negligible.** Grid: `m ~ N(mu, sigma)` per bin,
resampled per signal example, baked into the tfrecords (`run_grid_postprocessing.py:545`), applied
to kg *after* IA addition (correct — m scales the measured shear incl. IA), not to the noise
(correct — intrinsic-shape scatter is response-calibrated). Fiducial pipeline samples once per
label and reuses across perturbations (correct CRN for the delta loss). Data maps apply only R,
matching sims that multiply (1+m). Cost: σ_m ≈ 0.8 %/bin vs σ(S8)/S8 ≈ 5 % → < 1 % in quadrature.

**(c) Mode removal — throws away a known, bounded ≈ 9 % in σ.** The procedure (masked γ → pol
`map2alm` → keep E → ×`gamma2kappa` → `alm2map` → re-cut footprint) discards mask-leaked E→B signal
and out-of-footprint ringing — exactly the §3.2 transfers (signal 0.80 vs noise 0.89). Undoing both
as the no-loss counterfactual in the amplitude-Fisher: F ratio 0.85 → **σ inflation ≈ 8.6 %**.
Real, but identical for sims and data (no bias), common to 2pt and map-level, and inherent to
masked KS E-only — avoiding it means feeding γ maps or inpainting/iterative mass-mapping, not a
bug fix.

### 3.4 delta-NLA `bta`

**(a) `ds` template amplitude — the surprise.** The static audit expected the 2D
product-of-projections `ds = (ia − ⟨ia⟩) · δ_2D` to be LOS-suppressed and inert; the measurement
says otherwise at low z (δ_2D of nearby structure is order unity):

| bin | Cl(ds)/Cl(ia) (median, ℓ ∈ [100, 1000]) | IA-term power change at bta = 2 |
|---|---|---|
| metacal1 | **1.23** | **× 7.7** |
| metacal2 | 0.22 | × 1.9 |
| metacal3 | 0.077 | × 0.83 |
| metacal4 | 0.033 | × 0.41 |

(The power-change column includes the 2·bta·Cl(ia, ds) cross term; ia and ds are strongly
correlated.) The grid marginalizes an effective `Aia·bta ∈ [−6, 6]` in front of a template tracing
the matter field squared — a large, cosmology-degenerate freedom in the low-z bins. Coherence
check: the stored grid Cl plateau excess over analytic noise is 7 % in metacal1 vs 1–2 % elsewhere,
exactly where ds lives. Independently, the 2D product has the wrong scale dependence vs the true
projected 3D δ·s TATT term (2105.13544 eq. 19), so `bta` is not a faithful TATT extension either; a
per-shell δ-weighted UFalcon benchmark
(`data/v16/rot_in_place/obs/fiducial_bench_nla_per_shell_obs_maps.h5`) shifts contours only
minimally, so swapping implementations is not a fix — v17 drops bta instead.

**(b) bta = 0 conditioning on the existing DES chains — no likelihood-level cost.** Compared
`chain_{obs}_w0gt-1.npy` vs `chain_{obs}_w0gt-1_nla.npy` (bta dropped and fixed to 0; exact
conditioning since bta is a flow parameter) on the 2pt lensing run
(`runs/v16/rot_in_place/cls/lensing/m1_default/ensemble_flow_1000000/`), 1.02M samples each
(scratchpad `compare_nla_chains.py`):

| | DESy3 | DESy3_no_psi_rot |
|---|---|---|
| σ(S8) free / bta=0 | 0.0380 / 0.0395 | 0.0385 / 0.0390 |
| σ(Om) free / bta=0 | 0.0671 / 0.0653 | 0.0654 / 0.0637 |
| FoM(Om,S8) ratio free/bta=0 | 1.004 | 0.975 |
| S8 mean shift | +0.0000 | −0.0006 |
| bta marginal | mean 0.79, std 0.567 (≈ prior U[0,2]) | mean 0.78, std 0.555 |
| corr(bta, S8) | −0.01 | −0.00 |
| corr(Aia·bta, S8) | +0.44 | +0.38 |

Fixing bta = 0 changes nothing: widths equal within MCMC noise, zero S8 shift, bta marginal ≈ prior
(near-unidentifiable). The corr(Aia·bta, S8) ≈ 0.4 is inherited from the standard Aia–S8
degeneracy. The bta hypothesis therefore survives **only in compression-level form**: the summary
network, trained across the ×7.7 bin-1 power swing, may have discarded cosmological information to
cope with the ds variance — unrecoverable by conditioning. Decisive test: retrain on bta = 0 data
(done — result in §3.4d).

**Ceiling on the compression-level effect (added 2026-07-16):** the ds contamination is only large
in bins 1–2 (×7.7, ×1.9; bins 3–4 ≤ ×0.83), but by the §3.7 S/N budget bins 1–2 carry just
(4.39² + 10.28²)/32.7² ≈ 12 % of the total (S/N)². Even the worst case — the network learned to
ignore bins 1–2 entirely — costs only 1/√(1 − 0.12) ≈ **6 % in σ**. Pre-registered expectation:
**≲ 5 % σ improvement** from the v17 retrain, so the result is interpretable either way (a much
larger gain would mean the ds variance was corrupting the compression *across* bins, not just
within the contaminated ones). **Outcome (2026-07-18, §3.4d): the retrain landed exactly inside
the bound — no cross-bin corruption, hypothesis closed.**

**(c) v17 config** (`configs/v17/baseline.yaml`): identical to v16/rot_in_place except
`extended_nla: False`, the `dg -> ds` pair dropped from metacal `map_types` (ds never built; the
`dg -> sn` pair stays, its input ignored by `in_place`), and bta removed entirely (`params.ia.tatt`,
grid prior, fiducial). Every code consumer of tatt/bta is behind an `extended_nla`/`"bta" in params`
guard except one, fixed: `msi/utils/plotting.py` now uses `.get("tatt", [])`. The y3-deep-lss
probes configs list training params explicitly, so bta-free variants were added:
`configs/probes/lensing_nla.yaml`, `combined_nla.yaml` (msi plotting fix + `_nla` configs
uncommitted). The bta-containing configs fail fast on v17 data (`all_params.index('bta')` →
ValueError) and vice versa. Verified: all_params derivation, prior/fiducial lookups, probes column
gathers, config-checker invariants. Rerun notes:

- Only the **grid** needs regeneration; fiducial/obs products are bit-identical to v16 at the
  fiducial (bta = 0 ⇒ standard NLA) — reuse/symlink from `data/v16/rot_in_place/`.
- The Latin hypercube drops 7 → 6 dims, so **all** astro draws re-randomize per cosmology —
  v16/v17 comparisons are ensemble-level, not example-level.
- `normalization.lensing` stds are fiducial-based (bta = 0), stay valid.

**(d) v17 retrain result (2026-07-18): difference to v16 negligible — compression-level bta
hypothesis CLOSED.** Lensing map-level comparison, v17
`runs/v17/baseline/maps/lensing/t0_default` vs v16 `runs/v16/rot_in_place/maps/lensing/t6_default`
(same network; `ensemble_flow_150000`, 1.024M samples each):

| chain | σ(S8) v16 → v17 | FoM(Om,S8) v16 → v17 | S8 mean shift |
|---|---|---|---|
| fiducial_bench_mean (mock) | 0.0277 → 0.0263 (−5.1 %) | 1329 → 1446 (+8.8 %) | −0.001 |
| DESy3_w0gt-1 | 0.0235 → 0.0235 (0 %) | 994 → 1088 (+9.5 %) | +0.001 |
| DESy3 (wCDM) | 0.0275 → 0.0254 (−7.6 %) | 896 → 1090 (+21.7 %) | +0.002 |

Exactly the pre-registered ≲ 5 % outcome (§3.4b ceiling): mock σ(S8) −5 %, DES w0>−1 unchanged;
the larger DES-wCDM FoM swing is within the §4.4 DES-point seed scatter (±10 %+ across
identical-config variants; the comparison is also ensemble-level, all astro draws re-randomized,
§3.4c). Two reasons the true bta share is even *smaller* than these numbers: (i) the v17 run also
switched the VMIM head from GMM to flow+standardized (only config difference besides the data;
worth up to ~+30 % mock FoM on the Cls path, §4.3), so part of the small gain is the head, not
bta; (ii) no cross-bin corruption signature — the gain sits at, not above, the bins-1–2 S/N
budget bound. **bta is cleared at likelihood level (§3.4b) and compression level (this test);
the residual gap vs J24 is untouched and hypothesis 3 (§5) remains the leading candidate.**

### 3.5 v17 (bta = 0) data-flow re-audit — clean; NEST/RING and ℓ_min verified numerically

Full re-trace from the full-sky CosmoGrid inputs to the tfrecords under `configs/v17/baseline.yaml`.
**No bug found.** New verifications beyond §§2–3.4 (scratchpad `check_pixel_file.py`,
`check_ordering.py`, `lmin_fisher.py`):

- **v17 wiring**: 6-dim astro hypercube `[Aia, n_Aia, bg1–4]`; `astro_sample[:2]`/`[-4:]` unpacking
  has no overlap at d = 6; ds never built; standard-NLA branch taken; label vector = 8 cosmo/bary +
  6 astro = 14 (bary_Mc raw, known convention, §4.6).
- **NEST/RING, exact**: `data_vec_pix[corresponding_pix] == ring2nest(512, base_patch_pix)` holds
  exactly for all four bins → patches RING, data vector NEST, mutually consistent.
  `gamma2_signs = [1, 1, −1, −1]`. CosmoGrid probe maps empirically RING at nside 1024 (red Cl
  as-RING vs near-flat as-NEST), so `ud_grade(order_in="RING")` and the `scales.py` reorder
  round-trips are correct. Noise-catalog pixel column RING, footprint-consistent.
- **ℓ_min = 30 essentially free**: amplitude-Fisher with Cl_s(ℓ<30) held at Cl_s(30) as an upper
  bound: extending to ℓ ≥ 8 (or 2) gains ≤ 1.2 % in σ (metacal1), ~0.3 % jointly. The tfrecords
  keep everything to the band limit 1535 (`l_max: Null`) — no small-scale loss before the
  network-side smoothing. (Corroborated 2026-07-20 by the full-data-vector multiprobe Fisher:
  dropping the ℓ_min = 30 high-pass entirely is +0.30 % FoM, §3.6d.)
- **Minor, consistent-by-construction (not bugs)**: (i) ~11 % of noise-catalog galaxies (~14 % of
  touched pixels) fall outside the 318,616-pixel joint footprint and are dropped in
  `noise_gen_in_place` — data maps are cut identically (cost bounded in §3.6c); (ii) no
  pixel-window mismatch in the 1024→512 `ud_grade` (children average = parent average, same
  construction as the data's catalog binning); (iii) the 4 patches per permutation are disjoint and
  effectively independent (ℓ ≥ 30 correlation length ≈ 6°, independent noise, ~2–3° gaps for 4 of 6
  pairs); (iv) the large UFalcon monopole (≈ 3.5e−2 in kg metacal3) is removed by patch mean
  removal + the ℓ = 0, 1 KS zeroing.

§§3.1–3.3c apply verbatim to v17. Bottom line: the v17 data path is clean; nothing in it explains
the gap beyond the ≈ 9 % masked-KS penalty. The v17 retrain has since confirmed this (§3.4d).

### 3.6 tomo_Aia; masked-KS penalty vs the reference papers; masking cost

**(a) tomo_Aia / NLA redshift evolution — correct.** Exact reimplementation of
`redshift.get_tomo_amplitudes` on the v17 n(z) (scratchpad `check_tomo_aia.py`). All four metacal
n(z) carry an artifact spike at the last grid point (z ≈ 3, ~0.2 % of mass, 30–200× above
neighbors); the 0.05/0.95 quantile truncation excises it completely (hand-zeroing the last 10 grid
points shifts the truncated amplitude ≤ 0.25 %, vs 3–4 % at η = 3 and ×6 at η = 6 untruncated).
Fiducial (Aia = 0.5, η = 1.5): tomo_Aia = [0.360, 0.442, 0.558, 0.652], monotone. All three
production call sites read the config truncation settings — grid/fiducial/obs mutually consistent.
Footnotes: (i) the truncation redefines the effective (Aia, η) map (bin 1 at η = 3: 0.53 truncated
vs 0.68 full) — internally consistent, but inferred η is not directly comparable to the DES Y3
continuous-NLA convention; (ii) `get_tomo_amplitudes_according_to_config_vectorized` hardcodes its
truncation defaults (no production callers — dormant footgun).

**(b) Masked-KS E-only penalty: the "J24 differentiator" reading is refuted at 2-point by the direct
B-mode Fisher (§3.6d).** The two §3.2 factors: the mask scatters true E signal into pseudo-B, which
E-only discards (~×0.90, signal only — isotropic noise scatters symmetrically), and the post-KS re-cut discards
ringing (~×0.89, signal and noise alike, ~S/N-neutral) → the σ ≈ +8.6 % of §3.3c, driven by the
E-only signal loss. Checked against the papers (2026-07-16):

- **G24 (2405.10881) shares it**: full-sky-extended KS, all statistics on E-mode κ, B-modes only a
  null test → the penalty cancels vs G24.
- **J24 (2403.02314) keeps B**: their 2pt baseline is a spin-2 pseudo-Cl on the *shear* maps with
  **both C_ℓ^EE and C_ℓ^BB in the inference vector** (EB is the null test); their KS keeps E and B
  κ maps, the CNN gets 8 channels (E+B × 4 bins), peaks use E, B, and cross maps. The audit
  previously inferred that "the mask-leaked E signal we discard survives in their B channels," making
  the ≈ 9 % a J24 differentiator.

**That inference is directly refuted at 2-point (§3.6d):** adding the 42 B-touching columns to our own
Cl vector is lossless (B shape-noise-dominated), so keeping B — as J24's EE+BB 2pt vector does —
recovers essentially none of the leaked signal. The ≈ 9 % is thus a real mask cost but not a 2-point
J24 differentiator (most plausibly common, pending the §6-item-6 like-for-like check); only field-level
non-Gaussian κ_E+κ_B could still differ (§3.6d). Any J24-matched Fisher must nonetheless use a matched
spin-2 shear EE(+BB) vector, not masked-KS E-only κ Cls; second-order estimator differences (mask
decoupling conventions etc.) are few-% at most — J24 note their mask correction "is not actually
necessary to give unbiased results," same convention-consistency logic as ours.

**(c) Cost of the conservative joint (metacal ∩ maglim) mask: ≈ 4–6 % in σ, upper bound.** Our
footprint keeps 318,616 of ~371k catalog-touched pixels (~4178 deg²) and 88.6–90.3 % of source
galaxies per bin. If the references keep every covered lensing-only pixel: signal-dominated limit
(F ∝ area) → σ +6 %; noise-dominated limit (boundary pixels ~0.8× depth, ∝ coverage²) → σ +4 %.
A ~1.05× slice of the residual gap, and only if their mask is actually wider at nside 512
(unverified).

### 3.6d B-mode information-loss Fisher (commit `42e3b83`) — keeping B is 2-point-lossless

Direct test of the §3.6b hypothesis, replacing inference with measurement. `lensing.mode_removal`
reconstructs κ from masked shear via Kaiser–Squires and keeps only the E-mode alm; masking a pure-E
shear field leaks E→B, so the question is whether discarding that B-mode inflates the posterior. The
commit adds `mode_removal(keep_b_mode=True)` (projects the B-mode alm from the *same* spin-2
transform, no extra `map2alm`; gated behind `analysis.modelling.lensing.b_mode_cls`, default off, so
a normal run is byte-identical), threads a `kg_b/ia_b/sn_b` B-channel through the same
IA+m-bias+mask+smoothing chain, and forecasts the augmented data vector `concat(cl_E, cl_B)` against
the E-only baseline.

**Method.** Unified `msfm.utils.fisher` (`run_fisher_forecast.py` + `configs/v17/fisher_bmode.yaml`):
sample covariance over the 39960 fiducial realizations + central finite-difference Jacobian from the
± perturbations, Hartlap-debiased `F = JᵀC⁻¹J`, marginalized over all 14 params. E = 36-col
E/clustering (standard multiprobe data vector), B = 42 B-touching cols (BB auto+cross, B×E,
B×maglim). The raw sample covariance cond ~1e15 is pure *dynamic range* (Cls span 1e-7…1e-12), not
rank deficiency: solving on the correlation matrix (cond ~640) reproduces σ(S8)/FoM to all printed
digits, so the null result is physics, not a numerically washed-out signal.

| Data vector | ℓ_min (lensing) | cols | σ(Ωm) | σ(S8) | FoM(Ωm,S8) |
|---|---|---|---|---|---|
| **Baseline** (current, published) | 30 | 36 E | 0.01190 | 0.01145 | 7714 |
| **+ low-ℓ** (drop ℓ_min) | None | 36 E | 0.01190 | 0.01144 | 7737 |
| **+ B-mode** | None | 78 E+B | 0.01190 | 0.01144 | 7740 |

- **Dropping the lensing ℓ_min = 30 high-pass:** FoM **+0.30 %**, σ(S8) −0.10 % — a full-data-vector
  multiprobe confirmation of the §3.5 amplitude-Fisher "ℓ_min = 30 essentially free."
- **Adding the 42 B-mode columns on top:** FoM **+0.04 %**, σ(S8) −0.00 %; every per-parameter σ
  moves ≤ 0.1 %. **B is empty at 2-point:** per-column S/N (|mean|/std over realizations) median
  **0.005** for B vs **6.5** for E; only **9.5 %** of B columns clear S/N > 1 vs **84 %** for E.

**Interpretation (evidence discipline).** *Directly established, strong:* in our forward model the
mask-leaked E→B signal is noise-swamped and carries no usable 2-point cosmological information — so
the §3.6b recovery mechanism ("survives in J24's B channels") is refuted at 2-point. *Extrapolated,
not measured:* that J24's own analysis pays the same cost — J24 use a spin-2 *shear* EE+BB vector, not
masked-KS κ; the "common mask cost" reading rests on WL B being shape-noise-dominated generally (it is
a null test, for J24 too), not on a measurement of their pipeline. The airtight like-for-like check is
the J24-matched shear EE+BB Fisher (§6 item 6). **Caveat:** this is a Gaussian 2-point statement; if
the E→B leakage carries non-Gaussian information the Cls cannot see it — a field-level κ_E+κ_B
two-channel network (as in J24's CNN) is the only surviving way it could differ, and the power spectra
give no 2-point motivation to build it.

### 3.7 kg-vs-shape-noise S/N and IA-amplitude diagnostics (job 2752929)

Diagnostics 4 and 5 in `wl_sn_diagnostics.py`, on the same v16 pseudo-Cls (figures `sn_ratio.png`,
`ia_scale.png`).

**(a) Signal-to-noise — healthy, not suppressed.** Per-mode `Cl_kg/Cl_sn` (identical masked-KS
footing, so transfers cancel) peaks at 0.37/0.99/2.14/2.08 and crosses 1 at ℓ ≈ 30/30/35/67 —
shape-noise-dominated per mode, as expected. Integrated,
`(S/N)² = Σ (2ℓ+1)·(f_patch/2)·[Cl_s/(Cl_s+Cl_n)]²` per single DES-area patch (ℓ ≥ 30, 8 Mpc/h
cuts) = **4.39 / 10.28 / 21.33 / 22.16** per bin → **32.7 combined** (33.7 without l_max) — at or
above the published DES Y3 cosmic-shear detection S/N (~27). The data path is not swamping the WL
signal. The cumulative-S/N-vs-ℓ_max panel confirms the 8 Mpc/h cut sits on the shallow part of the
curve (consistent with §3.3a).

**(b) IA amplitude scale — physically reasonable NLA, GI-dominated.** At fiducial,
`sqrt(Cl_ia/Cl_kg)` = 0.389/0.166/0.100/0.067 per unit A_IA — largest at low z, the textbook NLA
signature. GI dominates II: II/GG = 0.020/0.005/0.003/0.002 vs GI/GG = −0.246/−0.093/−0.036/−0.034
(r = −0.88/−0.63/−0.32/−0.40); GI negative (radial alignment, standard NLA) → net fiducial
suppression (Cl_obs − Cl_kg)/Cl_kg = −0.227/−0.088/−0.033/−0.033. Even at |Aia| = 3 the IA/kg
amplitude fraction stays ≤ 0.84 (low-z) → 0.26 (high-z). All within the expected DES Y3 NLA range.

**Upshot:** neither the WL S/N nor the IA amplitude is anomalous at the data level — the gap is
compression/inference-side (§§4–5), not a signal lost or drowned in the forward model.

### 3.8 Full-sky input maps vs independent CCL theory — kg/ia amplitude and scale are CORRECT

Every prior signal check (§§3.2, 3.3c, 3.4a) was a *transfer ratio* against each map's own full-sky
Cl, so a wrong absolute amplitude/scale in the kg/ia inputs would have passed unnoticed. Now tested
externally and closed. Scripts: `wl_theory_comparison.py` (fiducial) + `wl_theory_scan.py` (S8/w0
scan); full write-up in `README_wl_theory_check.md`. Originally run on Perlmutter
(`/pscratch/sd/a/athomsen/deep_lss/runs/wl_audit/`); maps + repointed script defaults now on
Clariden (rerunnable there once pyccl + camb are installed — currently in no Clariden env).

**Method.** Full-sky auto/cross pseudo-Cls of the nside-1024 kg/ia maps (anafast, pixel-window
deconvolved, perm-averaged) vs pyccl at the exact CosmoGrid cosmology (halofit), from the stored
per-bin n(z): kg → GG, ia → II via `WeakLensingTracer(has_shear=False, ia_bias=1, use_A_ia=True)`
(NLA at A_IA = 1, η = 0 — exactly the maps' baked-in convention; `tomo_Aia` is applied on top,
§3.6a, no double counting), kg × ia → GI. The CCL and UFalcon NLA prefactors are numerically
identical (0.013876831 to 10 s.f.), so residual II/GI offsets can only come from the growth
factor D — which is exactly what the scan finds.

**Fiducial — inputs correct in amplitude and scale.** On the DMO (`v11dmo`) maps, all twelve
spectra (4 bins × {GG, II, GI}) sit within ≈ 1–3 % of theory, flat over ℓ ∈ [30, 1500] (8 perms; an
early 2-perm ≈ 0.95 dip at ℓ < 100 was cosmic variance). The production baryonified (`v11dmb`) maps
track to ℓ ≈ 300 then droop to ≈ 0.83–0.95 by ℓ = 1000–1500 — the expected baryonic suppression
(halofit is DMO). Consequences: kg correct (no lensing-efficiency/prefactor/Born error); ia and the
NLA A_IA=1/η=0/C1 convention correct (the physical IA signal is small because tomo_Aia ≈ 0.36–0.65,
not because the template is misnormalised); relative kg↔ia correct (GI matches in sign and
magnitude). The misspecified-input hypothesis is **ruled out**.

**Cosmology scan — holds across the prior; one benign IA approximation.** 3 grid points each in S8
(0.58/0.79/0.98) and w0 (−1.58/−1.24/−0.56); only dmb exists off-fiducial, so ℓ ∈ [30, 300] is the
clean band. (Gotcha, cost one run: grid dirs are `cosmo_{sobol_index:06d}` = `path_par`, NOT
`cosmo_{id_param}` — pair cosmologies via `sobol_index` in the metainfo.) **kg/GG correct
everywhere** (0.97–1.02 at ℓ ∈ [100, 300] across the whole range). **ia/II shows a flat-in-ℓ
multiplicative offset** f = 1.16 → 0.88 correlated with w0 (II ∝ f², GI ∝ f, GG ∝ 1), attributed to
≈ 1–2 % precision to UFalcon's analytic ΛCDM-form growth integral in the NLA weight vs the true
wCDM growth ODE — exact at w0 = −1, ±8 % at the prior edges; kg is ray-traced and does not inherit
it. Impact: IA-only, subdominant, largely absorbed by the marginalised Aia; leaves a mild
unphysical w0–Aia coupling (second-order, nuisance-direction). An ODE-growth fix exists but does
not justify map regeneration.

## 4. Compression / inference-side investigation

Tested on the 2pt lensing pipeline (`runs/v16/rot_in_place/cls/lensing/`, lmax_1024 scale cuts).
All numbers blinded (posterior sizes only); FoM = det(Cov[Om, S8])^(−1/2), S8 = s8·√(Om/0.3).

### 4.1 Wide-prior hypothesis ruled out for the lensing Cls

Tested twice, with escalating strength; both null. **(a)** Frozen-summary extended conditioning
(`run_inference.py --extend_params`: refit the flow on the existing 6-param summaries conditioned
on all 11 grid params, rerun with Gower-Street reference priors): +2 % (w0>−1+NLA), +5 % (ΛCDM+NLA)
— ambiguous, compression could have discarded the information first. **(b)** Full VMIM retrain with
the MI target extended to all 11 params (`lmax_1024_ext`, job 2735233): **refpriors gain +0.8 %**
(w0>−1+NLA, FoM 155.7 → 157.0). Direct evidence the information is absent: on 1000 grid mocks the
posterior/prior width ratios of the extension params are ns 0.995, Ob 0.992, H0 0.977, bary_Mc
0.986, bary_nu 0.992 — pure prior — while Om/s8/Aia come out constrained (0.70/0.38/0.27).
**Hypothesis 2 is dead for the 2pt lensing analysis** (map-level inherits the physics argument,
untested).

### 4.2 VMIM target dimensionality and the fac2 decoupling test

*(Interpretation superseded by §4.3: the penalty was head-optimization conditioning, not MI-target
dilution. Kept for the measurements and the fac2 caveat.)*

With the GMM head, target dimension cost FoM monotonically: 3p/6p/11p targets
(`lmax_1024_min`/baseline/`lmax_1024_ext`) → grid-mock median FoM 491/404/204, DES wCDM
452/352/148 — while every coverage test stayed acceptable. Coverage tests *cannot* localize such a
loss even in principle: exact Bayes on a lossy statistic passes SBC/HPD/TARP identically —
coverage certifies self-consistency, not information content. (`dim_summary_fac: 1` ties
dim(summary) = n_params, so the target scales both the head's posterior and the flow's random
variable s.)

Decoupling test (job 2737601, `lmax_1024_min_fac2`: 3p target, 6-dim s): grid-mock FoM2D **537**
vs 491 (3p, 3-dim) vs 404 (6p, 6-dim) — the NLE flow handles a 6-dim s fine in-distribution.
**Durable caveat:** min_fac2 is the only run where l-C2ST at the DES point rejects (p = 0.020),
and its DES FoM sits at 0.71× its own mock median (others 0.87–0.92×) — redundant summary
dimensions (dim(s) > n_target) are fragile exactly at the real-data point; any such design must
check l-C2ST at the data point. min_fac2 is a test artifact, not a production candidate.

### 4.3 Head conditioning (`standardize_theta`): the mock-level penalty was an optimization artifact

Jobs 2737882 (`lmax_1024_std`, GMM head) + 2737883 (`lmax_1024_flow_std`, RealNVP head); both
6-param target, fac = 1, `standardize_theta`. Standardizing θ inside the variational head is
affine-invariant for the MI bound, so any change is pure optimization conditioning (the physical
target spans ~30× in scale, Om vs n_Aia); newly implemented for the flow head
(`nets/estimators/normalizing_flow.py` + `utils/mutual_info_loss.py`, mirroring the GMM's
physical-units log-Jacobian convention).

- *Grid-mock median FoM2D:* flow_std **538**, std **526**, min_fac2 537, min 491, baseline 404.
  Three different interventions converge at ~535 (+30 % over baseline): the unstandardized head was
  under-training on the tight (Om, s8) directions; the scale-homogeneous 3p target had escaped the
  penalty for free, mimicking a target-content effect. Head NLLs (physical units, same 6p target):
  flow_std −2.29 vs GMM-std −1.86 — with standardization the flow trains stably (the historical
  instability was raw θ feeding the coupling MLPs) and reaches the tighter bound; mock FoM tracks
  the bound monotonically.
- *DES data point:* wCDM FoM2D flow_std 341 / std 361 / baseline 352 / min_fac2 379 / min 452 —
  **the +30 % mock gain does NOT transfer**. DES percentile within near-fiducial grid mocks
  (|ΔOm| < 0.05, |Δs8| < 0.10, n = 74): baseline 53rd, min 62nd, min_fac2 35th, std 24th, flow_std
  **12th** — a monotone slide as compression sharpens, while l-C2ST stays non-rejecting (std
  p = 0.28, flow_std p = 0.41): the flows are locally calibrated and the DES posteriors are
  *genuinely* wider than mock ones. The extra Cl features the improved summaries exploit yield
  power on sims but not on data — the expected signature of a mild **data-vs-sim mismatch in fine
  Cl features** (candidates: uniform sim noise depth, baryon/IA model residuals, data
  systematics). Consistently, the 3-dim min summary is the best on data (452) despite being the
  weakest improved variant on mocks — fewer dimensions expose less surface to off-manifold
  features. Follow-up discriminator: PPC (`run_ppc.py`) on the std/flow_std summary spaces, or
  locating the DES summary within p(s|θ̂) along the extra directions.

### 4.4 Flow-head ablation and productionization

Jobs 2738338–42 (`lmax_1024_flow_{p6h128, p4h128, np4h128, p6h64, p4h64}`): couplings {4,6} ×
permutation {on,off} × width {64,128}, all fac = 1 + standardized. Head bound ranking (vali NLL):
p6h128 −2.312 > np6h128 −2.292 > p4h128 −2.210 ≈ np4h128 −2.206 > p6h64 −2.188 > p4h64 −2.124 —
depth and width worth ~0.1 nat each, the roll-by-1 permutation a small free gain (fixes the static
half-split that kept Om/s8/w0 from conditioning on each other). Downstream, mock median FoM2D is
FLAT across the grid (526–547; l-C2ST clean, np4h128 marginal p = 0.05) and uncorrelated with the
bound — past standardization, head capacity is not binding on the 6p target. DES wCDM FoM scatters
325–412 (seed noise at the low-percentile data point; consistent with §4.3).

Productionization: `standardize_theta` now defaults to true in BOTH training apps (Cls app:
empirical label stats; maps `run_training.py`: analytic uniform-prior stats + `density_estimator`
pass-through — valid because the MI bound is affine-invariant); `configs/loss/vmim.yaml` = flow
head 4×128 `permute: true` (reduced same day from 6×128 on the flat-downstream-FoM justification;
6 couplings documented opt-in for harder targets, e.g. 11p ext); old GMM default preserved as
`vmim_gmm.yaml`; redundant loss configs (vmim_std, vmim_flow_std, vmim_flow, vmim_ext, the
`configs/loss/cls/` duplicate subdir) removed, dev submission scripts repointed. The low-Om SBC
tilt persists in all standardized runs (KS p ~1e-3–1e-4) — shared, not architecture-specific.
NOTE: maps benchmarks (`bench_t*`) trained via vmim.yaml after this date use flow + standardized
heads — head-loss values not comparable to pre-flip runs.

### 4.5 Target-dimensionality rerun with the flow+standardized head — penalty dead on mocks

Jobs 2738422 (`lmax_1024_flow_min`, 3p) / 2738423 (`lmax_1024_flow_ext`, 11p); 6p point =
`lmax_1024_flow_p4h128`. Grid-mock median FoM2D: **3p 490 / 6p 533 / 11p 488** (GMM era:
491/404/204) — the catastrophic 11p collapse was entirely head conditioning; with a
well-conditioned head the MI target's content barely matters on mocks. On DES (blinded): wCDM
FoM2D 409 / 383 / 298 (GMM era 452/352/148); λCDM 585 / 629 / 474 — the data-vs-sim penalty of
§4.3 *grows with target dimensionality* (more summary directions exposed to off-manifold
features); the 3p compression remains the most robust on data. Coverage: l-C2ST clean (min
p = 0.060, ext p = 0.530); flow_ext SBC is the only run where the low-Om tilt vanishes (p = 0.52)
but bta fails hard (p = 4e-5, the likelihood-flat parameter) with H0/Aia marginal (0.023/0.046);
flow_min keeps the Om tilt (p = 0.011). Head NLLs not comparable across targets. All
`lmax_1024_flow_*` runs are test artifacts.

**Bottom line for the gap vs J24/G24/W26:** wide implicit priors contribute ~nothing (§4.1); the
references' 3D-flow-target choice explains ~25 % in FoM; the closest like-for-like comparison (min
run, w0>−1: σ(Om) = 0.0555, σ(S8) = 0.0345 vs J24 Cl 0.044/0.026) leaves a **residual
≈ 1.26×/1.33× per-parameter gap** (down from ≈ 1.5× for the 6p baseline). Accounting (revised
2026-07-20): the E-only slice can **no longer** be divided out — the direct B-mode Fisher (§3.6d)
shows keeping B is 2-point-lossless, so it is not a 2-point differentiator vs J24's EE+BB vector.
Dividing out only the mask-width bound (≤ ~1.05×, §3.6c) leaves **~1.2–1.27× unexplained** (up from
the earlier ~1.1–1.15×), to be attributed to nuisance-prior differences (J24's IA prior is not
U[−3,3] × wide-η; cf. the §4.7 marginalization ladder, IA ~5 % / full nuisance set ~25 % on
lensing S8), the mild data-vs-sim mismatch (§4.3), and the residual masked-KS-κ-vs-shear-EE estimator
convention. The J24-matched shear EE+BB Fisher (§6 item 6) is what turns this residual into named
factors.

### 4.6 Caveat: `bary_Mc` label convention (cost one wasted run)

`lmax_1024_ext.broken_raw_mc`: the stored label tables keep `bary_Mc` RAW (1e12–1e15) while
configs/priors/inference use log10(Mc); there are TWO independent label-gather sites
(`deep_lss/utils/cls_preprocessing.py` for Cls runs, `msi/utils/preprocessing.py` for map runs).
Both now convert raw→log10 behind a `min > 1e10` assert. Regression signature: compression
vali_mse ~1e27 (or, with θ standardization, a silently absorbed +33-nat offset), flow NaN from
epoch 0, SBC KS p = 0 for all params.

### 4.7 Gaussian Cl Fisher forecast vs the SBI likelihood-flow — inference is not the 2pt bottleneck

`msi/dev/fisher/fisher_cls.py` builds F = J^T C^-1 J for (Om, s8, w0) from the **v17/baseline**
binned Cls: covariance from the 39960 signal+noise fiducial realizations in `cls/fiducial_cls.h5`
(Hartlap-corrected), central finite differences from the perturbation Cls in the fiducial
tfrecords, data vector = the exact hard_rebinned Cls the Cls MLP trains on (16 √-spaced
ℓ-bins/pair) at the production **8wl,32gc** cut (lensing ℓ_max = 589/863/1159/1382 ≈ 8 Mpc/h,
clustering 133/195/255/305 ≈ 32 Mpc/h, ℓ_min = 30). Analysis top-hat priors variance-matched to
Gaussians, coupled to the model (headline mode). ~13 min (nb16) on one CPU allocation. Binning is
not fully converged at nb32 (combined σ(S8) 0.0196/0.0191/0.0183 for nb8/16/32, ~4 % nb16→nb32;
lensing essentially converged 0.0374/0.0370/0.0369); **nb16 is the reference** (matches the v3_mlp
compression's binning).

**Unified msfm forecast infrastructure (added 2026-07-20, commit `42e3b83`).** `msfm/utils/fisher.py`
+ `msfm/apps/run_fisher_forecast.py` generalize this section's msi-side `fisher_cls.py` into the
offline numpy counterpart of the training-time `deep_lss.utils.delta_loss.get_jac_and_cov_matrix`:
the same `F = JᵀC⁻¹J` sample-cov + finite-diff-Jacobian forecast, but over arbitrary *named Cl blocks*
and marginalized over all 14 params, so any subset of Cl columns can be compared offline (its first
application is the §3.6d B-mode study). The two tools now coexist: msi `fisher_cls.py` for the (Om,
s8, w0) 3-param headline compare against SBI chains; msfm `fisher.py` for block/config comparisons at
full 14-param multiprobe marginalization.

**Headline forecast (astro_nuisance mode — the correct match to the SBI marginalization: the
network predicts cosmo + astro while H0/Ob/ns/bary_Mc/bary_nu vary in the grid unpredicted, so the
chain cosmo posterior is implicitly marginalized over both):**

| probe | σ(Om) | σ(s8) | σ(w0) | σ(S8) | FoM(Om,S8) |
|---|---|---|---|---|---|
| lensing | 0.0569 | 0.1027 | 0.3038 | 0.0370 | 476 |
| clustering | 0.0223 | 0.1298 | 0.1460 | 0.1188 | 379 |
| combined | 0.0189 | 0.0402 | 0.1197 | 0.0191 | 2833 |

(FoM(Om,S8) marginalizes w0 — comparable to the §4 map-level FoM2D.)

**Marginalization ladder (σ(S8), fixed → +astro → +nuisance).** Lensing 0.0285 → 0.0300 → 0.0370:
IA marginalization costs only ~5 %, the full nuisance set another ~25 % (mostly w0/bary) — the
lensing cosmo constraint is robust. Clustering 0.0316 → 0.1137 → 0.1188: **linear-bias
marginalization is catastrophic for s8** (fixed σ(s8) 0.0233 → 0.1295, ×5.6; Om survives at
×1.17). Combined 0.0092 → 0.0167 → 0.0191: the 3×2pt cross-correlations break both degeneracies —
combined σ(S8) = 0.019 is ~2× the fixed-nuisance ideal but far tighter than either probe alone.

**Validation against the real SBI posteriors** (`fisher_vs_chain.py`, headline `cov_astro_nuisance`
vs the v3_mlp `ensemble_flow_1000000/chain_fiducial_bench_mean.npy`, 1.024M samples, same cut).
Marginal 1σ ratio chain/Fisher:

| probe | Om | s8 | w0 | S8 |
|---|---|---|---|---|
| lensing | 1.05 | 1.18 | 0.69 | — |
| clustering | 1.16 | 1.20 | 1.44 | — |
| combined | **1.08** | **1.07** | **1.14** | **1.08** |

Combined agreement (the headline compare) within 7–14 %; lensing/clustering within ~20 % except
lensing w0 (chain *tighter*, 0.69 — expected: w0 is weakly lensing-constrained and the flow
respects the hard top-hat edges the Gaussian ellipse tails past).

**Scale-cut sweep (nb32) — lensing scale-flat; clustering/combined balloon only in an unphysical
regime.** Rerun at `lmax_1024` (ℓ_max = 1024 all bins) and `unsmoothed` (full ℓ_max = 1500), raw
Cls. **Lensing barely moves**: FoM(Om,S8) 480.6 / 481.8 / 503.6 (~5 % over the whole range; σ(S8)
0.0369 → 0.0366) — full-data-vector confirmation of §3.3a. **Clustering/combined balloon**
(clustering σ(S8) 0.118 → 0.043, FoM 395 → 2377; combined σ(S8) 0.0183 → 0.0106, FoM 3099 → 9813),
but this is an optimistic upper bound, not a forecast: beyond ℓ ≈ 300 maglim clustering needs
nonlinear bias (absent from the derivative set), approaches the shot-noise floor, and the Gaussian
covariance misses non-Gaussian terms — the ℓ ≈ 300 clustering cap is a deliberate modeling choice,
and no SBI chain exists at the extended cuts. Summary tool: scratchpad `scale_bin_summary.py`.

**What this establishes.** The SBI likelihood-flow reproduces the Gaussian Fisher ceiling →
**density estimation loses ~nothing on the 2pt path**. With §4.1 (priors) and §4.5 (dimensionality
= head conditioning) also cleared, any residual 2pt gap vs J24/G24 lives in the **data-vector
information content itself** — scale cut, mask, noise, estimator conventions, nuisance
marginalization — squarely inside hypothesis 3 (§5.3). Numerically, the v17 lensing-Cl Fisher
σ(S8) = 0.037 sits close to the best map-level compression (min run 0.0345, §4.5) and both are
~1.4× J24's 0.026 — the strongest Gaussian 2pt possible at our footprint *for the masked-KS E-only
statistic* already carries the gap. (Caveat: J24's cut/mask/priors differ and their vector is
spin-2 shear EE+BB; the E-vs-E+B part of that difference is now measured to be 2-point-negligible
— §3.6d, adding the B Cls is +0.04 % — so the residual is a consistency statement about scale
cut/mask/priors, not an E-only decomposition.) Cls-only result; the map-level compression test (v17
retrain) has since come back null as pre-registered (§3.4d).

## 5. Ranked hypotheses for the constraining-power gap (state as of 2026-07-16)

The data path is clean (§§2–3.8, incl. absolute input normalisation) and 2pt density estimation is
cleared (§4.7). Live candidates:

1. **VMIM target dimensionality** (§§4.2–4.5) — measured and largely resolved: ~25 % FoM for 6p vs
   the references' 3p target; the mock-level penalty was head conditioning, recovered by
   `standardize_theta` — but the recovered information does not transfer to DES (item 3).
2. **Compression-level `bta` penalty** (§3.4) — **CLOSED 2026-07-18 by the v17 retrain (§3.4d)**:
   mock σ(S8) −5 %, DES w0>−1 σ(S8) unchanged, exactly inside the pre-registered ≲ 5 % bound from
   the §3.7 S/N budget (bins 1–2 = ~12 % of (S/N)²) — and an upper bound on the bta share, since
   the v17 run also carries the flow+std head upgrade. bta contributes nothing to the residual
   gap at either likelihood or compression level.
3. **Data-vector / data-vs-sim mismatch — the leading candidate** (§§4.3, 4.5, 4.7). 2pt: the flow
   saturates the Fisher ceiling, so the residual ≈ 1.26–1.33× gap vs J24 must be in the data vector
   (scale cut; mask width ≤ 4–6 % if the references keep boundary pixels, §3.6c, unverified;
   estimator conventions few-%, §3.6b; noise treatment; nuisance marginalization, §4.7).
   Map-level: improved summaries gain FoM on mocks but not on DES, whose posterior slides to the
   12th–24th percentile of near-fiducial mocks while l-C2ST stays clean (§4.3) — candidates:
   uniform sim noise depth, baryon/IA model residuals, data systematics. Two supporting
   observations: (i) the 2pt Fisher ceiling (σ(S8) = 0.037) and the best map-level compression
   (0.0345) land within ~7 % of each other and are both ~1.4× J24 — two very different pipelines
   hitting the same wall points at what they *share* (the data-vector construction and the
   simulator), not at what differs (compression, inference); (ii) caution on the map-level
   evidence: DES-point FoM scatters 325–412 across identical-config variants (seed noise, §4.4),
   so the monotone *percentile trend* is the evidence — individual DES FoM comparisons between
   variants are not.
4. **Masked-KS E-only construction (≈ 9 % in σ, §§3.2/3.3c)** — a real mask cost, but **no longer a
   2-point J24 differentiator**: the direct B-mode Fisher (§3.6d) measures keeping B to be
   2-point-lossless, refuting the §3.6b assumption that J24's B channels recover the mask-leaked E
   signal. Most plausibly a common cost (pending the §6-item-6 like-for-like check); only field-level
   non-Gaussian κ_E+κ_B could still differ. Demoted below hypothesis 3.
5. *(ruled out)* prior volume in ns/Ob/H0/baryons (§4.1; the w0-prior part is separately modest:
   w0>−1 vs wCDM FoM 426 vs 352 in the baseline); effective ℓ-range on both ends (ℓ_min = 30 free,
   §3.5; high ℓ shape-noise-dominated, §3.3a, confirmed by the §4.7 sweep); shape-noise
   miscalibration, KS/mode-removal amplitude bugs, missing Cl cross terms, NEST/RING errors,
   sim-vs-data response/weighting/pixel-window inconsistencies (§§3.1–3.2, 3.5); tomo_Aia / NLA
   redshift evolution (§3.6a); m-bias (< 1 %, §3.3b); absolute kg/ia input amplitude/scale (§3.8;
   the only real finding — a w0-dependent ±8 % analytic-growth offset on the IA weight — is
   nuisance-level, absorbed by Aia). Map-level WL detection S/N ~33 per patch (≥ DES Y3's ~27) and
   the IA amplitude is in the expected NLA range (§3.7) — the signal is neither swamped nor
   mis-scaled.

## 6. Next steps (reprioritized 2026-07-16, targeting hypothesis 3)

1. **Data-space Cl discriminator (cheapest, most direct — do first).** For the 2pt path no
   compression is needed: the data vector is the 16-bin Cls and 39,960 fiducial realizations
   exist in `cls/fiducial_cls.h5`. Compute the DES Cl vector's per-bin z-scores and a whitened χ²
   against the fiducial (or best-fit θ̂) mock ensemble, per bin-pair and ℓ-bin. If the §4.3 story
   is right the mismatch must be visible *here*, localized in ℓ/bin — which immediately
   discriminates the candidates: noise-depth → high-ℓ auto plateaus; baryons/IA → low-z bins at
   mid-ℓ; additive systematics → low ℓ. Notebook-scale, no training.
2. **Attribution of the extra summary directions.** For flow_std vs min: regress the summary
   dimensions on the binned Cl features (or take the MLP Jacobian at the data point) and identify
   *which Cl features* the directions that push DES off-manifold are reading. Combined with (1),
   turns "mild data-vs-sim mismatch" from a diagnosis into a named feature.
3. **Systematics-template injection.** Forward-model the known DES Y3 survey-property maps (PSF
   residual ellipticity, depth variation, stellar density) through the same masked-KS chain and
   measure Δs against the p(s|θ̂) width per summary dimension — a sharp, cheap test of the
   "data systematics" candidate using existing pipeline machinery.
4. ~~**v17 retrain**~~ — **DONE 2026-07-18 (§3.4d)**: negligible difference to v16, inside the
   pre-registered ≲ 5 % bound; compression-level bta test formally closed. No surprise to chase.
5. **PPC / summary-space discriminator** — PPC (`run_ppc.py`) on the std/flow_std summary spaces,
   or locating the DES summary within p(s|θ̂) along the extra directions: the confirmatory
   version of (1)–(2) for the map-level path (§§4.3/4.5).
6. **J24-matched Fisher** (§4.7 follow-up, elevated 2026-07-20) — spin-2 shear EE+BB on our footprint
   with J24's scale cut and nuisance priors. Now the **primary** way to turn the residual (grown to
   ~1.2–1.27× per §4.5) into named factors *and* to directly confirm/refute the §3.6d "common mask
   cost" extrapolation on J24's actual vector — the msfm `run_fisher_forecast` block infrastructure
   (§4.7) already supports it (add an EE/BB shear block). Also use the marginalization ladder to check
   whether tighter IA/bias priors recover the clustering s8 (×5.6 bias degeneracy). Check the
   references' nside-512 mask width while at it (§3.6c).
7. **Deferred: E+B / γ-map network input** (hypothesis 4). The **2-point motivation is now measured
   to be absent** (§3.6d: adding the B Cls is lossless, B shape-noise-dominated). The only remaining
   rationale is speculative *non-Gaussian* field-level information — the definitive test being a
   κ_E+κ_B two-channel network (per the §3.6d REPORT caveat). Deprioritized: with no 2-point payoff
   and B-modes carrying a disproportionate share of un-modeled data systematics, adding them while
   hypothesis 3 is open could worsen the robustness problem, not improve it.
8. **Deferred: weighted/two-block head** — §4.5 killed the dimensionality penalty on mocks; only
   worth revisiting if the data-robustness of the 3p target (§4.3) motivates a hybrid. Any
   dim(s) > n_target design must check l-C2ST at the data point (§4.2 caveat).

## 7. Reproducing the diagnostics

```bash
sbatch dev/scripts/wl_audit/wl_sn_diagnostics.sh
```

runs on one Clariden allocation in ~1 minute (TF container + `~/dlss/tf_env`); inputs are the repo
data files (`DESY3_noise_v11.h5`, `DESY3_pixels_v11_fiducial_512.h5`), one fiducial CosmoGrid
permutation, and (optionally) the stored `grid_cls.h5`. Thresholds/paths CLI-configurable
(`--sn_lmin`, `--lmax_cuts`, `--des_ref_sn`). Five diagnostics: 1 shape-noise level
(`noise_level.png`), 2 KS/mode-removal signal transfer (`signal_level.png`), 3 delta-NLA ds
template + GI cross (`ds_amplitude.png`), 4 kg-vs-shape-noise S/N vs scale (`sn_ratio.png`),
5 IA amplitude scale incl. GI (`ia_scale.png`); arrays in `wl_sn_diagnostics.npz`, outputs in
`runs/wl_audit/` (job 2752929).
