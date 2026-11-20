# Y6 v1 forward model vs the DES Y6 3x2pt key project: modelling differences

Dev-phase note (2026-11-20). Repo docs must not point here.

User decision 2026-11-15: follow the DES Y6 3x2pt key project where the map-level model allows,
and otherwise keep v18. This list records where Y6 v1 differs from it, and whether each difference
is a choice we could adopt later or comes with a map-level analysis on CosmoGridV1.

KP: arXiv:2601.14559 (results). FW: arXiv:2601.14859 (modelling and methodology). The two papers'
Table 3 hold the same priors. References checked against both papers on 2026-11-20. Ours:
`configs/desy6/v1/default.yaml`. The differences to our Y3 v18 are in `differences_to_y3.md`, and the open
work in `plan.md`.

## Same as the KP

- Catalogs: the metadetection sources (4 bins) and the MagLim lenses, the released fiducial
  selections without further cuts; the Y6 joint mask.
- Lens bins 1, 3, 4, 5, 6. The KP drops bin 2 after a failed PPD test, with one of its n(z) modes
  pushing against its prior (KP Sec. IV.8).
- Lens weights: ENet, the KP fiducial (arXiv:2601.14484).
- Shear response: R per bin from metadetection, equal to DES's `3x2pt_pipeline_public.ipynb`.
- Fiducial n(z): the 3x2pt FITS `nz_source` and `nz_lens`.
- m prior per bin: mean -3.4, 6.5, 15.9, 1.7 and width 5.8, 6.6, 5.9, 12.2 ×10⁻³, ours from the
  `M_BIN*` columns, theirs in KP Table 3.
- Lens bias: linear per bin with a [0.8, 3.0] box, the KP baseline (KP Sec. V.1.1). Nonlinear EPT
  bias (b1, b2) is their alternative (V.1.2).
- IA pivot z0 = 0.3 (FW Sec. III.4).
- Not modeled in either: reduced shear and source magnification (FW Sec. III.7).

## Choices (candidates to adopt later)

| # | aspect | KP | ours | note |
|---|---|---|---|---|
| 1 | IA model | TATT: A1, η1, A2, η2 sampled, b_TA = 1 fixed; A1(z) and A2(z) evolve with z (FW Eqs. 23-25) | δ-NLA: Aia, n_Aia, bta sampled, i.e. TATT with A2 = 0. One effective amplitude per bin, averaged over the truncated fiducial n(z) | user decision 2026-11-20. A2 needs a tidal-torquing map per source bin from the projection. KP App. C: NLA shifts S8 by +0.24σ, freeing b_TA by 0.12σ; the results "appear to be insensitive to the choice of the IA model" |
| 2 | IA priors | A1 [-1, 3], A2 [-3, 3], η1 and η2 N(0, 3) cut to [-5, 5] | Aia [-3, 3], n_Aia [-4, 6], bta [0, 2], uniform (v18's) | user decision 2026-11-20, for now. Their η range [-5, -4) lies outside our box. msi learns a likelihood, so their Gaussian could be applied at inference inside our box. Table 3's caption defines N(μ, σ²), but its m rows use the `M_BIN*` standard deviations, so whether 3 is σ or σ² is unclear |
| 3 | IA and bias fiducials | synthetic data (FW Table 3): A1 0.13, η1 = η2 = 2.0, A2 -0.2, b_TA 1; b1 = 1.54, 1.85, 1.76, 1.93, 1.90 (bins 1, 3-6) | Aia 0.5, n_Aia 1.5, bta 0; bg = 1.35, 1.50, 1.57, 1.63, 1.67 (power law bg 1.5, n_bg 0.5) | user decision 2026-11-20 (v18's). Only the fiducial set and the mocks use them |
| 4 | lens magnification | α per lens bin, Gaussian priors centered 1.58, 1.38, 2.04, 2.21, 2.45, 2.42 (bins 1-6) | not modeled (as v18) | user decision 2026-11-20. Adopting it later needs `kg` maps of the lens bins, so a re-projection |
| 5 | uncertainty of the lens weights | the ENet - ISD difference is marginalized in the w(θ) covariance (KP Sec. II.3) | the ENet weights correct the observation; no uncertainty | open (plan D11) |
| 6 | mean shear | residual additive c, the weighted mean shear, subtracted from the data (KP Sec. II.2) | logged, not subtracted (as v18) | open (plan D8). Observation only, not in the tfrecords |

## Different methods for the same uncertainty

| # | aspect | KP | ours | note |
|---|---|---|---|---|
| 7 | source n(z) | 7 modes shared by the 4 bins, u ~ N(0, 1) on [-3, 3] (FW Sec. III.6) | each realization is projected with one draw from `desy6_source_redshift.h5` (80760 realizations) | both describe DES's source n(z) uncertainty; we did not check how the modes were built. Open (Tomasz): possibly only 20 realizations grid-wide |
| 8 | shear calibration m | Gaussian per bin, with small covariances with the u modes (KP Sec. II.5) | the realization's own m (`shearm`), drawn jointly with its n(z), applied by msfm | same marginal per bin (above) |
| 9 | lens n(z) | 3 modes per bin, u ~ N(0, 1) on [-3, 3] (FW Sec. III.6) | each realization is projected with one draw from `desy6_lens_redshift.h5` (~3000 per bin) | |

## Inherent to a map-level analysis on CosmoGridV1

| # | aspect | KP | ours |
|---|---|---|---|
| 10 | statistic and likelihood | w(θ), γ_t, ξ± in 2.5'-250'; Gaussian likelihood; analytic covariance (Gaussian, cNG and SSC terms; halofit, no IA) | nside-512 maps of κ and δ_g, compressed by a network, likelihood learned by a flow; no covariance |
| 11 | matter field | HMCode2020 with log10 T_AGN = 7.7 fixed (FW Sec. III.1) | CosmoGridV1 N-body (PKDGRAV3), UFalcon Born projection |
| 12 | baryons | not modeled; removed by the scale cuts (row 15) | baryonification, bary_Mc [12, 15] and bary_nu [-2, 2] sampled |
| 13 | cosmology | Ωm [0.1, 0.6], A_s ×10⁹ [0.5, 5], h [0.58, 0.8], Ωb [0.03, 0.07], n_s [0.93, 1.00], w [-2, -1/3], Σm_ν [0.06, 0.6] eV sampled | CosmoGridV1: Om [0.1, 0.5], s8 [0.4, 1.4], H0 [64, 82], Ob [0.03, 0.06], ns [0.87, 1.07], w0 [-2, -0.333]; Σm_ν fixed at 0.06 eV |
| 14 | fiducial cosmology | synthetic data (FW Table 3): Ωm 0.31, A_s 1.831 ×10⁻⁹ (σ8 0.76), h 0.69, Ωb 0.051, n_s 0.965, w -1, Σm_ν 0.077 eV | CosmoGridV1's: Om 0.26, s8 0.84, H0 67.36, Ob 0.0493, ns 0.9649, w0 -1 |
| 15 | scale cuts | w(θ) ≥ 9 Mpc/h and γ_t ≥ 6 Mpc/h (linear bias); ξ± by Δχ² < 3.5 (NLA) or 5 (TATT), θ_min ≈ 3'; for 3x2pt Δ(Ωm, S8) < 0.5σ in ΛCDM, and 0.3σ plus Δ(Ωm, w) < 0.3σ in wCDM | l_min 30 in the tfrecords; the small scales are removed by smoothing downstream, not set yet for Y6 |
| 16 | clustering theory | non-Limber w(θ) with RSD; point-mass marginalization in γ_t (FW Sec. III.3) | maps of projected particle counts, no RSD; no γ_t, so no point mass |
| 17 | footprint | the joint mask at its native resolution | nside-512 pixels with coverage ≥ 0.73, footprint 4055 vs 4031 deg²; partial coverage absorbed by n_gal |

## Where we model more than the KP

- Source clustering: not in the KP (FW Sec. III.7). Ours: `count` shape noise Poisson-samples the
  sources from a biased density, with the source clustering bias fitted per cosmology (step
  `sc_bias`).
- Source imaging systematics: the KP applies none to the sources. Ours: the `lss_sys` map from the
  19 DES Y6 templates modulates the simulated source density (`survey_systematics: True`).
