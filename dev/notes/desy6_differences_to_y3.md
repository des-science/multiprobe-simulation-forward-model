# Y6 v1 forward model vs our Y3 v18: differences

Dev-phase note (2026-11-20). Repo docs must not point here.

`configs/desy6/v1/default.yaml` is a copy of `configs/desy3/v18/default.yaml`, and every key
without a marker keeps its v18 value on purpose. This list records each place where Y6 differs, so
that no difference is silent. A row that changes the model was a stated user decision; anything
else is a data, input or plumbing change.

Y3 side: `configs/desy3/v18/default.yaml`, `data/desy3/README.md`, `notebooks/desy3/`,
arXiv:2511.04681. Y6 side: `configs/desy6/v1/default.yaml`, `data/desy6/README.md`,
`desy6_tfrecord_plan.md`. The differences to the DES Y6 3x2pt key project are in
`desy6_differences_to_y6_kp.md`.

## Identical (the model)

- Simulations: the CosmoGridV1 grid, 2500 cosmologies × 20 permutations, baryonified (`v11dmb`),
  projected by Tomasz. Cosmology and baryon priors and the fiducial cosmology are CosmoGridV1's.
- Maps: nside 512, four patches with v18's rotation (`y_rad -0.125`, `z_rad -1.22`; step `pixels`
  asserts they do not overlap on Y6), `n_noise_per_signal: 5`.
- IA: δ-NLA, `κ += A_IA,i (ia + bta ds)`. A_IA,i weights `((1+z)/(1+z0))^n_Aia` over the fiducial
  n(z) of bin i, truncated to its 5-95% quantiles. The boxes are Aia [-3, 3], n_Aia [-4, 6] and
  bta [0, 2]; the fiducials are Aia 0.5, n_Aia 1.5 and bta 0. Only the pivot changes (row 9).
- Lens bias: linear per bin, box [0.8, 3.0]. The fiducials come from the same power law
  (bg 1.5, n_bg 0.5), evaluated on the Y6 lens n(z). No quadratic bias, no stochasticity.
- Lens systematics: the weight maps correct the observed counts (`apply_maglim_sys_map`) and stay
  out of the forward model (`maglim_survey_systematics_map: False`).
- Shape noise: `count`, `bias: fixed`, `survey_systematics: True`, `metacal_bias_arm: contam`. The
  simulated source density carries the `lss_sys` imaging systematics and the fitted source
  clustering bias.
- Scales: hard cut, l_min 30 in every bin, the smoothing applied downstream on the fly; power
  spectra l 30-1500 in 33 bins.
- Footprint method: a pixel enters if the mask covers at least `coverage_threshold` of it, chosen so
  the footprint area matches the mask area. Partial coverage is absorbed by n_gal on average.
- Catalog maps: the same e1/e2 signs and ψ rotation (confirmed on Y6 by κ_E × δ_g and the
  κ_B × δ_g null), the shear divided by R, the mean shear logged and not subtracted, raw lens
  counts.
- m stays out of the label vector.

## Model and data differences

| # | item | Y3 v18 | Y6 v1 | note |
|---|---|---|---|---|
| 1 | source catalog | metacal, 4 bins; n_gal 5498-5586 deg⁻² | metadetection `noshear`, 4 bins, 139.7M sources; n_gal 7434-8183 deg⁻² | data. 35-47% denser per bin, so less shape noise per pixel |
| 2 | shear response | `R_gamma + R_s` (with the selection response), config values | R per bin from `1p/1m/2p/2m`: 0.856, 0.869, 0.819, 0.692. Divides the cached maps in step `maps`; no config value | data. Metadetection needs no separate selection response. R equals DES's `3x2pt_pipeline_public.ipynb` to its last digit |
| 3 | lens bins | MagLim bins 1-4 (`z_lims` 0.2-0.85) | MagLim bins 1, 3, 4, 5, 6 | user decision, as the KP (bin 2 dropped there after a failed PPD test). 5 bias parameters, named by survey bin (`bg1`, `bg3`-`bg6`) |
| 4 | lens weight maps | DES Y3 ISD weight maps at nside 4096 (DES's run on the PCA of 107 SP maps, 50 kept, 2σ, 10 bins) | ENet, the KP fiducial, read off the per-galaxy `weight` (constant per nside-1024 pixel), no refit. Std 1.3-3.6% per bin | user decision. The method choice is DES's. Their rms reproduces σ_weight of arXiv:2601.14484 to the printed digit |
| 5 | lens n_gal | raw counts / footprint area | raw counts × the weight maps / footprint area | so the clean simulation matches the corrected observation in expectation. Differs by 0.08-0.28% (Y3: 0.01-0.16%) |
| 6 | footprint | DES Y3 lens joint mask (gold 2.2.1, redMaGiC + MagLim), nside 4096; threshold 0.72; 318,616 pixels; data vector 458,752 | DES Y6 joint mask (lenses and shear), nside 131072 → coverage at 4096; threshold 0.73; 309,226 pixels; data vector 457,728 | data. Mask area 4184 vs 4031 deg², footprint 4178 vs 4055 deg², mean coverage 0.92 vs 0.88 |
| 7 | redshift uncertainty | each realization projected with the fiducial n(z), shifted by Δz_s / Δz_l and, for the lenses, stretched by σ_z,l, from Gaussian priors | each realization projected with one draw from DES's ensembles: `desy6_source_redshift.h5` (80760 joint n(z) + m) and `desy6_lens_redshift.h5` (~3000 per bin) | in the projection, no shift or stretch parameter. Open (Tomasz): realization row = permutation index in the test files, so possibly only 20 realizations grid-wide |
| 8 | shear calibration m | drawn by msfm per example from per-bin Gaussians (μ -6.3 to -36.9 ×10⁻³, σ 7.6-9.1 ×10⁻³), unseeded | the realization's own m, attribute `shearm` of the projection's `nz/<bin>`, drawn jointly with its n(z); asserted within 5σ of the `M_BIN*` prior (μ -3.4, 6.5, 15.9, 1.7 ×10⁻³; σ 5.8, 6.6, 5.9, 12.2 ×10⁻³) | `multiplicative.source: projection`. One m for the four patches of a permutation. The projection does not apply it (measured), so msfm does |
| 9 | IA pivot | z0 = 0.62 | z0 = 0.3, the KP's | user decision. The box and fiducials stay, so the same (Aia, n_Aia) gives per-bin amplitudes 1.246^n_Aia times Y3's: 1.39× at the fiducial n_Aia = 1.5, 0.41-3.7× across the box |
| 10 | fiducial n(z) | txt files from outside the repo, grid from z = 0.005 | the 3x2pt FITS `nz_source` / `nz_lens`, 300 points from z = 0 | data. Mean z 0.41, 0.54, 0.85, 1.16 (sources) and 0.31-1.01 (lenses). Also feeds the IA amplitude and the bias fiducials |
| 11 | source systematics map | `lss_sys` on 34 SP maps (`STD32`) | `lss_sys` on the 19 DES Y6 templates. Not done yet | the `lss_sys` side has its own list (`lss_sys/claude/desy6/differences_to_y3_sources.md`) |
| 12 | source clustering bias | `sc_bias_fit_count.ipynb` | step `sc_bias`. Not done yet | the count histogram runs to 400 galaxies per pixel instead of 200 (`shape_noise.bias_fit_max_count`): Y6 has ~100 sources per pixel vs Y3's 72 and a largest count of 359, which 200 would drop |
| 13 | projection resolution | nside 1024, `ud_grade`d to 512 by msfm | nside 512 (test files) | equivalent: the maps are linear in the particle counts, so averaging four nside-1024 pixels equals projecting at 512 |
| 14 | shape-noise catalog | metacal shapes (`noise_file.ipynb`) | metadetection `noshear` (e1/R, e2/R, w, pixel), step `noise` | data, same layout |

## Not there yet

- Fiducial set: v18 has 17 fiducial cosmologies (the fiducial and ± perturbations of the 8
  CosmoGrid parameters) × 1000 permutations. Y6 has no fiducial projection, and its config drops
  `fiducial.n_cosmos`, `n_perms_per_cosmo`, `n_noise_per_signal` and `perturbations`.
- Scale cuts and smoothing scales downstream (y3-deep-lss) are not set for Y6.
- The production projection. Everything above that depends on it was checked on the two test files
  only.

## Plumbing (no effect on the model)

- Layout: Y3 under `configs/desy3/`, `data/desy3/`, `notebooks/desy3/`, `pipelines/desy3/`; Y6
  next to it under `desy6/`. Configs are versioned per survey (`desy6/v1`).
- Products: one script with checks, `msfm.apps.products.desy6` (steps `nz`, `pixels`, `maps`,
  `values`, `noise`, `sc_bias`); `values` writes the marked config lines. In Y3, notebooks and
  hand-copied numbers. `--provenance` records inputs, outputs and the msfm commit.
- Survey guards: `files:`/`dirs:` repo paths must lie in `data/` or `data/<survey>/`; a projection
  must store `shearm` exactly when the config takes m from it.
- n(z) files load by bin label and survey name; the catalog cache is per survey (`dirs.cache`);
  galaxy biases follow the lens bin labels.
- v18 keys that Y6 does not read, dropped: `analysis.n_pix`, `params.bg.stochasticity` (`rg`, off in
  v18 too), `maglim.z_lims` (the Y6 catalog is already binned), `R_gamma`, `R_s` (row 2).

## Environment

- Y3 tfrecords: Perlmutter (NERSC) CPUs.
- Y6: daint, account `lp184`. Forward model on the CPUs of the GH200 nodes in the `tensorflow`
  container with `tf_env`. Products in `torch_env` (plus healsparse), login node or one CPU node.
