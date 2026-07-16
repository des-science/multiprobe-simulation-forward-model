# Are the full-sky `kg` / `ia` input maps right? — input map vs. independent CCL theory

**Date:** 2026-07-13
**Author:** Arne Thomsen (with Claude)
**Code:** `wl_theory_comparison.py` (fiducial), `wl_theory_scan.py` (S8 / w0 cosmology scan)
**Results:** `/pscratch/sd/a/athomsen/deep_lss/runs/wl_audit/`
**Companion:** `dev/notes/wl_forward_model_audit_2026-07-11.md` §3.9 (this is the standalone version)

---

## TL;DR

The lensing (`kg`) and intrinsic-alignment (`ia`) full-sky input maps are **correct in absolute
amplitude and scale**, at the fiducial cosmology and across the S8 / w0 prior. The hypothesis that a
misspecified `kg` / `ia` amplitude or scale was depressing the forward-model S/N is **ruled out**.

One real but benign approximation surfaced: the `ia` maps' NLA weight uses an analytic ΛCDM-form
growth factor, giving a ≈±8 % w0-dependent error in the **IA amplitude only** at the w0 prior edges.
It does not touch the lensing signal and is largely absorbed by the marginalised `Aia`.

## 1. Why this test was needed

The 2026-07-11 audit checked the WL data path exhaustively and found no bug — but every signal check
in it (§§3.2, 3.3, 3.5c) divided the pipeline output by **each map's own full-sky Cl**. Those are
*transfer* ratios. A wrong absolute amplitude or a wrong scale dependence baked into the `kg` or `ia`
input maps themselves would cancel in such a ratio and pass unnoticed — while still lowering the S/N
of the forward-modelled data vector. This was the one WL data-side hypothesis never tested.

The missing ingredient was an **external** prediction. This test supplies one (pyccl / CAMB).

## 2. Method

Measured, per metacal bin:

- full-sky auto/cross pseudo-Cls of the nside-1024 `kg` and `ia` maps (`healpy` `map2alm`/`anafast`),
- deconvolved by the nside-1024 pixel window (`hp.pixwin`), averaged over CosmoGrid permutations.

Compared against an **independent** pyccl prediction at the exact CosmoGrid cosmology
(`data/CosmoGridV1_metainfo.h5`), built from the **stored** per-bin `n(z)` (`/nz/metacal*` inside the
map file — i.e. the same n(z) the projection used):

| map | CCL tracer | spectrum |
|---|---|---|
| `kg` | `WeakLensingTracer` (shear only) | GG |
| `ia` | `WeakLensingTracer(has_shear=False, ia_bias=1, use_A_ia=True)` | II |
| `kg × ia` | the two crossed | GI (negative) |

Fiducial cosmology (metainfo `parameters/fiducial[0]`): Ω_cdm = 0.209277, Ω_b = 0.0493, h = 0.6736,
σ8 = 0.84, n_s = 0.9649, w0 = −1, Σm_ν ≈ 0.06 eV (`m_nu` field is 0.02 **per species**; O_nu = 0.00142
⇒ 0.06 eV total), halofit.

### The IA convention — why II at A_IA = 1, η = 0 is the right comparison

The stored `ia` map is the NLA template at **unit amplitude and no redshift evolution**; the forward
model applies the per-bin `tomo_Aia = Aia·⟨((1+z)/(1+z0))^n_Aia⟩` on top (`msfm/utils/redshift.py`),
so η = 0 in the map is correct and there is no double counting.

The two NLA normalisations are **numerically identical** (verified):

- CCL (`pyccl/tracers.py`, `use_A_ia=True`): `−A_ia · 5e-14 · RHO_CRITICAL · Ω_m / D(a)`
- UFalcon (`probe_weights.F_NLA_model`): `−A_IA · C1 · ρ_crit,0 · Ω_m / D(z)`, `C1 = 5e-14 h⁻²`

Both prefactors equal **0.013876831** to 10 significant figures (`5e-14 × 2.775366e11`). The commonly
quoted 0.0134 is a different ρ_crit rounding and is *not* what either code uses. **Because the
constants match exactly, any residual II / GI offset can only come from the growth factor `D`** — which
is precisely what §4 finds.

## 3. Fiducial result — maps are correct

Two map variants: `v11dmo` (dark-matter-only, the clean match to halofit) and `v11dmb` (baryonified,
used in production). Median measured/theory, **dmo**, 8 perms:

| probe | ℓ∈[30,100] | [100,300] | [300,600] | [600,1000] | [1000,1500] |
|---|---|---|---|---|---|
| kg / GG | 0.99–1.00 | 0.99 | 0.98 | 0.98 | 0.98–0.99 |
| ia / II | 0.96–1.00 | 0.98–0.99 | 0.97–0.99 | 0.97–0.98 | 0.97–0.98 |
| kg·ia / GI | 0.97–1.00 | 0.99 | 0.98 | 0.97–0.98 | 0.98 |

All twelve spectra (4 bins × {GG, II, GI}) sit within **≈1–3 % of independent theory across
ℓ ∈ [30, 1500], flat in ℓ**. (An early 2-perm run showed a ≈0.95 dip at ℓ ∈ [30, 100]; that was
cosmic variance over the few low-ℓ modes and vanishes at 8 perms → ≈0.99. No finite-box deficit
above the ℓ_min = 30 cut.)

The **dmb** maps track dmo/theory to ℓ ≈ 300, then droop to ≈0.83–0.87 (kg) by ℓ = 1000–1500 — the
**expected baryonic suppression** (halofit is DMO), shared by all three probes, hence not an
amplitude/scale error.

Conclusions:

1. **kg amplitude/scale: correct.** No lensing-efficiency, prefactor or Born-projection error.
2. **ia amplitude/scale and the NLA (A_IA=1, η=0, C1) convention: correct.** `ia/II ≈ 1` verifies the
   IA template is neither under- nor over-amplified. The physical IA signal is small vs. lensing
   because `tomo_Aia = [0.360, 0.442, 0.558, 0.652]` at fiducial — not because of a bad template.
3. **Relative kg↔ia amplitude: correct.** GI matches in sign and magnitude, so nothing dilutes the
   combined `kg + tomo_Aia·ia` S/N.

## 4. Cosmology scan — holds across the prior, and one real IA approximation

Repeated at 3 grid cosmologies spanning **S8** (0.58 / 0.79 / 0.98 at w0 ≈ −1) and 3 spanning **w0**
(−1.58 / −1.24 / −0.56 at S8 ≈ 0.75–0.85). Only `v11dmb` exists off-fiducial, so ℓ ∈ [30, 300] is the
clean band (baryons negligible); the full ℓ range is still measured and plotted.

> **Indexing gotcha (cost one wrong run):** the CosmoGrid grid directory is
> `cosmo_{sobol_index:06d}` = `path_par`, **not** `cosmo_{id_param}` — `sobol_index ≠ id_param`
> (id_param 1412 → sobol 28190 → `cosmo_028190`). Pair the cosmology by matching `sobol_index` in the
> metainfo. Using `id_param` silently loads a *different, existing* cosmology and yields nonsense
> ratios (0.3–1.5).

Median over ℓ ∈ [100, 300], 2 perms:

- **kg / GG: correct everywhere — 0.97–1.02** at ℓ ∈ [100, 300] (0.97–1.04 over the full ℓ < 300
  band) across the entire S8 = 0.58–0.98 and w0 = −1.58…−0.56 range. The cosmological signal that
  carries the constraining power has the right amplitude and scale at every cosmology tested.
- **ia / II: a flat-in-ℓ, cosmology-dependent amplitude offset**, from 1.16 (w0 = −1.58) to 0.88
  (w0 = −0.56). GI tracks its square root ⇒ the offset is a pure multiplicative factor `f` on the IA
  field (II ∝ f², GI ∝ f, GG ∝ 1). `f` correlates with **w0, not S8**, and is ≈1 near fiducial.

### Attribution: the analytic growth factor

`f` matches, to ≈1–2 %, the ratio of the NLA weight `∫n(z)/D(z)dz` computed with UFalcon's *analytic*
ΛCDM-form growth integral (`g = 5Ω_m/2 · E(z) · ∫₀^a da'/(a'E)³`) vs. CCL's true wCDM growth ODE:

| cosmology | w0 | `f` (√ of II ratio) | D-ratio (analytic / CCL) |
|---|---|---|---|
| cosmo_000002 | −1.58 | 1.080 | 1.081 |
| cosmo_016147 | −1.24 | 1.044 | 1.049 |
| cosmo_194878 | −1.06 | 1.001 | 1.010 |
| cosmo_168548 | −1.04 | 0.988 | 1.008 |
| cosmo_082075 | −0.92 | 0.961 | 0.988 |
| cosmo_000964 | −0.56 | 0.938 | 0.959 |

The analytic integral is exact at w0 = −1 but drifts up to ≈±8 % at the w0 prior edges. The `kg`
field is ray-traced from the actual simulated density, so it does **not** inherit the approximation
(hence GG stays exact); only the IA weight, which multiplies that same field by `F_NLA(z)`, does.
This is consistent with §2: the normalisation constants are identical, so `D` is the only degree of
freedom left.

### Why it does not matter here

1. IA-only and subdominant — physical IA is `tomo_Aia · ia` with `tomo_Aia ≈ 0.4–0.65`.
2. It does not touch the correctly normalised lensing signal or its S/N.
3. `Aia` is a marginalised nuisance (prior [−3, 3]), so a multiplicative IA rescaling is largely
   absorbed.

The one genuine subtlety: the offset is **w0-dependent**, so it introduces a mild unphysical w0–Aia
coupling in the IA sector. Second-order, on a nuisance direction — not a driver of the Om/S8 gap. A
fix (build the IA maps with the ODE growth for wCDM) exists but does not justify a regeneration for
this analysis.

## 5. Bottom line

The WL data path is clean end to end, **including the absolute normalisation of its inputs**, at the
fiducial and across the prior. The constraining-power gap remains compression/inference-side (audit
§4 items 1–2: VMIM target dimensionality; the compression-level `bta` penalty → the v17 retrain).

## 6. Reproducing

Needs `pyccl` (3.2.1) + `camb`, pip-installed into `~/.local` for the `dlss15` python. Maps are on
CFS, so this runs on a Perlmutter CPU node (no allocation needed for the small perm counts below);
the fiducial 8-perm run takes ≈45 min, the scan ≈40 min.

```bash
source activate dlss15
cd ~/multiprobe-simulation-forward-model

# fiducial, both map variants (the headline test)
python dev/scripts/wl_audit/wl_theory_comparison.py --n_perms 8 --variant both

# S8 and w0 cosmology scans (dmb only; params in every panel title)
python dev/scripts/wl_audit/wl_theory_scan.py --n_perms 2
```

Outputs in `/pscratch/sd/a/athomsen/deep_lss/runs/wl_audit/`:

| file | contents |
|---|---|
| `wl_theory_ratio.png` | fiducial measured/theory, GG/II/GI, dmb vs dmo |
| `wl_theory_comparison.png` | fiducial absolute `ℓ(ℓ+1)Cℓ/2π`, maps vs CCL, 4 bins |
| `wl_theory_comparison.npz` | fiducial raw measured + theory Cls |
| `wl_theory_scan_S8_scan.png` / `.npz` | S8 scan (params in titles) |
| `wl_theory_scan_w0_scan.png` / `.npz` | w0 scan (params in titles) |

Both scripts are CLI-configurable (`--n_perms`, `--variant`, `--out_dir`); see `--help`.
