# DES Y6 port: inventory of Y3-specific items

> **Partly superseded by `desy6_tfrecord_plan.md` (same day).** Three things in this inventory are out of date:
> - "Y6 data not available": the Y6 catalogs and n(z) sets are on disk;
> - the m-given-u design of section 1: the projection now stores m per realization (`shearm`), and n(z) comes from a realization set;
> - the projection is running.
>
> The code locations below still hold.

Written 2026-11-15, as preparation only (no jobs, no code or config changes). Goal: re-run the frozen
Y3 v18 analysis (Thomsen et al. 2025, arXiv:2511.04681) on DES Y6, with its own config version and
`desy6_<product>` files next to the Y3 ones. v18, its data products and the production runs stay frozen.

**Type legend**

- **(a)** swap a file or value when the Y6 data arrives, no code change
- **(b)** code change we can make now so that Y6 fits (a generalization, testable on Y3 with unchanged output)
- **(c)** unknown until the Y6 catalogs, data products or papers are in hand

**Open:** marks anything not verified. Everything else was read in the code, the configs, or the
files on disk named next to it.

Line numbers are msfm paths unless a repo is named. y3-deep-lss and msi rows come from a read-only
sweep of both repos; the five most important ones were re-checked by hand.

---

## 0. Y6 facts this inventory rests on

### Verified from the released Y6 3x2pt chain

Source: the header and the embedded params/values ini of
`multiprobe-simulation-inference/data/DESy6_key_project_chains/wCDM/tatt/3x2pt.txt`. The release is
arXiv:2601.14559, with cosmic shear in arXiv:2602.10065.

| | Y3 (v18) | Y6 3x2pt |
|---|---|---|
| Source bins | 4 (metacal) | 4 (`m1..m4`) |
| Lens bins | MagLim, 6 bins, first 4 kept | MagLim, 6 bins, **bin 2 dropped**, so 1, 3, 4, 5, 6 kept (`wolensbin2`, `u_1_* = 0`, `b1e_bin2` fixed) |
| Source photo-z | per-bin shift Δz_s, independent Gaussians | **7 modes shared across bins** (`photoz_uz`, `basis_file = run_ini/U_source_v3d1.npz`) |
| Shear calibration m | per-bin, independent Gaussians | **correlated with the source modes**: `correlated_priors` maps `m1..4_uncorr` and `u_0..6_uncorr` through one `covmat_multigauss.asc` + `mean_multigauss.asc` |
| Lens photo-z | per-bin shift + stretch (DNF) | **3 modes per bin** (`basis_file = run_ini/U_lens_sompzwz.npz`, i.e. SOMPZ + WZ) |
| Lens magnification | not modeled | **α varied per lens bin** (flat [-2, 6], centers 1.58 to 2.45) |
| Lens bias | linear per bin (our model) | fiducial **nonlinear** (`b1e`, `b2e`), linear as a variant; `b1e` prior [0.8, 3.0], same range as our `bg` |
| IA | δ-NLA, pivot `z0 = 0.62` | TATT (`a1, a2, alpha1, alpha2`), `z_piv = 0.3` |
| Σmν | fixed by CosmoGrid | varied, [0.06, 0.6] |

**Open.** The `run_ini/` files (`U_source_v3d1.npz`, `U_lens_sompzwz.npz`, `covmat_multigauss.asc`,
`mean_multigauss.asc`, `priors_v1.ini`) are **not in the chain release**. Two things are unknown:

- whether DES publishes them, for example alongside the 2pt FITS file `3x2pt_..._UNBLINDED_wmask_nob2.fits`, which also carries the mean n(z);
- how `photoz_uz.py` builds an n(z) from the modes. I have not read it.

### Environment state, checked 2026-11-15

These matter for the order of work.

- **The Y3 catalog dir `dirs.catalog` is empty.** `/iopsstor/scratch/cscs/athomsen/deep_lss/data/y3-cats/` exists with no files. The only Y3 catalog products left are the four `data/cache/desy3_*.npy` maps (Jun 15) and the untracked `data/desy3_noise_512.h5`.
- **The scratch CosmoGrid grid maps are gone again.** `CosmoGrid/v11desy3/bary/grid/cosmo_*/perm_*` dirs exist but hold 0 files (checked 4 cosmologies). The fiducial `perm_0000` survives on the **a0158 store** (`deep_lss/data/fiducial/cosmo_fiducial/`).
- **Sizes.** A projected permutation file is **~0.6 GB** (`projected_probes_maps_v11dmb.h5`, nside 1024, 606 MB). The 2500 × 20 grid permutations of one projection are therefore ~30 TB before the fiducial set and the dmo variant. The lp184 store has 40 TB at 14% used.
- **lss_sys environment.** `prgenv-gnu/25.6:v2` is not pulled on daint (memory `reference_daint_environment`).
- **a0158 store.** The Y6 key-project chains themselves sit on it (the `data/DESy6_key_project_chains` symlink), so they move with the Globus transfer.

---

## 1. The largest item: the CosmoGrid projection is survey-specific and lives outside msfm

Every input map msfm reads is `map/{kg,ia,dg}/{z_bin}` from `projected_probes_maps_v11dmb.h5`:

- read at `postprocessing.py:727-735`, `observation.py:284-294,555` and `source_clustering_bias.py:273`;
- filename fixed by `filenames.py:22-34` through `--cosmogrid_version` (`run_grid_postprocessing.py:123-129`).

These maps are the CosmoGridV1 shells summed by UFalcon with the **Y3 n(z)**: lensing kernel, NLA kernel, source δ and lens δ. Each permutation file carries its own shifted n(z) in `nz/<z_bin>` next to the kernels in `kernel/{kg,ia,dg,dh}/<z_bin>`, which is how the Y3 photo-z marginalization works (paper `data.tex`, "projected with its own redshift distribution, shifted by Δz"). msfm has no photo-z code at all.

For Y6 this means:

1. **Reproject everything** with the Y6 n(z): grid, fiducial and benchmarks, both dmb and dmo if both are kept, under new z-bin labels. Type **(c)**: it needs the Y6 mean n(z) and the mode basis.
   - **Open:** who runs UFalcon (the v11desy3 projection was done upstream, at ETH and NERSC).
   - **Open:** where the CosmoGridV1 shells are; only projected maps were ever copied here.
   - **Open:** the compute and storage cost (~30 TB per projection, above).
2. **The per-realization n(z) has to come from the mode prior, not from per-bin shifts.** For the lenses this changes only the projection input.
3. **m has to be drawn jointly with the source modes.** Today m is drawn in msfm, independently of the n(z) shift baked into the map:
   - `lensing.get_m_bias_distribution` is a `MultivariateNormalDiag` (`lensing.py:52-55`);
   - it is sampled per signal at `run_grid_postprocessing.py:560`, `run_fiducial_postprocessing.py:426` and `observation.py:325-328`.

   With the Y6 joint m–u covariance, the projection must **record the u it drew per permutation**, and msfm must draw m from p(m | u). Type **(b)** for the msfm half, buildable now against a toy covariance. Type **(c)** for the projection half.
4. **Bin labels.** The projected-file keys are the config `z_bins` labels, so the projection and the config must agree on them (see decision D2).

---

## 2. Config: `configs/v18/default.yaml`

The four `configs/v18/obs/*.yaml` are full copies that differ only in modelling switches and the
`metacal_bias*` / `sc_calibration` keys (diffed). A Y6 version therefore needs five files carrying
the same survey block. **(b), optional:** give msfm configs an `extends:` composition like
y3-deep-lss has, so the survey block exists once.

### `dirs` and `files`

| key (line) | what it is / read at | type | Y6 needs |
|---|---|---|---|
| `dirs.catalog` (11) | DES Y3 catalog dir. `catalog.py:189,309,383,434`. The path goes through the `/users/athomsen/scratch` symlink, which compute-node containers do not see; the dir is empty | (a) + (b) | Y6 catalog location, canonical `/iopsstor` or lp184 path |
| `dirs.redshift_distributions` (13) | n(z) dir. `files.py:452` | (a) | `desy6_nz_*` files (rows below) |
| `files.pixels` (17) | `desy3_pixels_fiducial_512.h5`. `files.load_pixel_file` (`files.py:82`), `get_clustering_systematics` | (a) + (c) | new pixel file from the Y6 mask (section 3) |
| `files.noise` (18) | `desy3_noise_512.h5`, `(e1, e2, w, pix)` per source bin. `files.py:419-426` | (a) + (b) | Y6 noise file through a metadetection reader (section 5) |
| `files.meta_info` (19) | `CosmoGridV1_metainfo.h5` | none | survey-agnostic, unless the Y6 projection tree changes `path_par` |
| `files.healpy_data` (20) | pixel weights | none | survey-agnostic |
| `files.metacal_systematics` (22) | `desy3_metacal_systematics_STD32_512.h5`, lss_sys export. `files.get_metacal_systematics` (`files.py:186-318`), label guard `configuration.py:9-50` | (c) | an lss_sys run on the Y6 source sample and Y6 survey-property maps |
| `files.metacal_bias` (24), `metacal_bias_arm` (25) | `desy3_metacal_bias.h5`, per-cosmology source-clustering bias. `files.read_metacal_bias` (`files.py:468-480`) | (c) | refit on Y6, only after the reprojection exists (section 4) |
| `files.sc_calibration` (obs/sc_gatti only) | `desy3_sc_calibration_gatti.npy` | (a), low priority | only if the Gatti variant is kept for Y6 |

### `survey`

| key (line) | what it is / read at | type | Y6 needs |
|---|---|---|---|
| `survey.name: DESy3` (28) | tfrecord tag at `run_grid_postprocessing.py:297,776` and `run_fiducial_postprocessing.py:319,1015`. Downstream the string is also a data-selecting label (section 7) | (a) | `DESy6` |
| `survey.metacal.z_bins` (33) | source bin labels; keys into the pixel file, noise file and projected maps | (a) | Y6 labels (decision D1) |
| `survey.metacal.n_gal` (35) | raw source density over the masked footprint, from `notebooks/config.ipynb`. Sets n_bar in the count shape noise (`postprocessing.py:474`, `observation.py:366`) and the bias-fit check (`source_clustering_bias.py:173-174`) | (a) | recompute on the Y6 catalog and mask. **Open:** if it is much higher than Y3's ~72 per pixel, `RANGE_1PT` needs widening (section 4) |
| `survey.metacal.shear_bias.multiplicative.{mu,sigma}` (37-40) | m prior. `lensing.py:52-55` | (b) + (c) | a **joint** m–u covariance in place of per-bin `mu`, `sigma` (section 1, item 3) |
| `survey.metacal.z0: 0.62` (42) | IA power-law pivot. `redshift.py:64,166` | decision | an analysis choice. Y6 KP uses `z_piv = 0.3`; keeping 0.62 keeps the Y3 parametrization (decision D5) |
| `survey.metacal.R_gamma`, `R_s` (45-46) | metacal shear and selection response. Map making divides by `R_gamma + R_s` (`catalog.py:223-224,336-338,466-467`); a debug assert compares with the catalog's `R11/R22` (`catalog.py:259-264`, on by default) | (b) + (c) | metadetection has one response per bin and no per-object R (section 5). Replace with a single `response` key, keeping Y3 back-compat |
| `survey.maglim.z_bins` (51) | lens bin labels | (a) | Y6 lens labels for bins 1, 3, 4, 5, 6 if following DES (decision D2) |
| `survey.maglim.n_gal` (53) | lens density over the mask, from `config.ipynb`. `run_grid_postprocessing.py:591`, `base_pipeline.py:124`, `observation.py:544`, `run_fiducial_postprocessing.py:806` | (a) | recompute |
| `survey.maglim.z_lims` (55) | Y3 binning on DNF `zmean_sof` (`catalog.py:392-396`) | (c) | **Open:** how Y6 MagLim++ assigns lens bins (a DNF cut as in Y3, or a catalog column) |
| `survey.Aeff: 4143` (57) | debug printouts only (`catalog.py:239,412`) | (a) | Y6 effective area |

### `analysis`

| key (line) | what it is / read at | type | Y6 needs |
|---|---|---|---|
| `n_side`, `n_pix`, `n_side_down` (60-63) | resolution and padding | none | unchanged. The padding is recomputed by `pixel_file.ipynb` from the new mask |
| `n_patches: 4` (65) | four non-overlapping footprint copies per lightcone | (c) | holds only if four rotated copies of the Y6 mask still do not overlap. `pixel_file.ipynb` asserts it (section 3) |
| `params.bg.{linear,quadratic}` (77-78) | `[bg1..bg4]`. Parameter list (`parameters.py:84`) and Latin hypercube width (`run_grid_postprocessing.py:248,323`) | (a) + (b) | `bg1..bg5` for 5 lens bins. **The grid unpack is literal** (code table, section 6) |
| `footprint.rotation.{y_rad,z_rad}` (82-84) | "found by trial and error" for Y3. `catalog.py:59-60`, `files.py:236-238`, `pixel_file.ipynb` | (c) | re-verify against the Y6 mask. **Open:** whether the Y3 angles still tile; the DES wide outline is the same, the holes are not |
| `scale_cuts.{lensing,clustering}.l_min` (91, 97) | length-4 lists, value 30 | (a) | length = n bins |
| `modelling.lensing.nla.*` (118-121) | n(z) truncation for the IA amplitude | none | value-agnostic, applied to the Y6 n(z) |
| `modelling.clustering.maglim_survey_systematics_map` (132) | off in v18 | none | the lens weights enter the observation instead (pixel file) |
| `normalization.{lensing,clustering}` (134-135) | length-4 lists | (a) | length = n bins |
| `grid.priors.bg1..bg4` (162-165) | [0.8, 3.0] | (a) | extend to the Y6 lens bins. The Y6 `b1e` prior has the same range |
| `fiducial.bg1..bg4` (197-200) | from a power law bg = 1.5, n_bg = 0.5 over the **Y3** lens n(z) | (a) | recompute over the Y6 lens n(z), or take the Y6 fiducials |
| `fiducial.perturbations.bg*` (214-217) | | (a) | extend |
| everything cosmological (`params.cosmo`, `bary`, `grid.priors` for cosmo and bary, `Om_s8_border_points`, `fiducial` cosmology) | CosmoGrid properties | none | survey-agnostic |

---

## 3. Data products in `data/` and the notebooks that make them

| product | made by | inputs (Y3) | type | Y6 needs |
|---|---|---|---|---|
| `desy3_pixels_fiducial_512.h5` | `notebooks/pixel_file.ipynb` | Y3 joint maglim mask (`y3_gold_2.2.1_..._maglim_v2.2_mask.fits.gz`, nside 4096, laptop path), threshold 0.72, footprint rotation, four symmetry copies (`gamma_2_sign = [1, 1, -1, -1]`), maglim weight maps `w_map_bin{i}` (Rodríguez-Monroy, laptop path) | (a) + (c) | Y6 3x2pt mask, Y6 lens weight maps. **Open:** the Y6 lens systematics weights method and file format. The notebook loads `configs/v15/extended.yaml` and has `desy3_` in its output name (cell 3) |
| `desy3_noise_512.h5` (untracked, 1.5 GB) | `notebooks/noise_file.ipynb` → `catalog.get_shapes_from_cat` | metacal e1, e2, weight, pixel per bin, e already divided by R | (b) + (a) | metadetection reader (section 5). The notebook loads `configs/v16/default.yaml` |
| `desy3_metacal_bias.h5` (`clean`/`contam`) | `notebooks/sc_bias_fit_count.ipynb` | projected grid `map/dg/metacal{i}`, the DES count map (`data/cache`), the lss_sys contamination | (c) | needs the Y6 reprojection first. The notebook hardcodes the v11desy3 grid path, the fiducial copy on the **a0158 store** and the output name `data/desy3_metacal_bias.h5` |
| `desy3_metacal_systematics_STD32_512.h5` | the lss_sys repo (`output/README.md` is the contract) | Y3 survey-property maps + metacal | (c) | a Y6 lss_sys run on the Y6 source sample; label, attribute and guard must agree (`configuration.py:9-50`) |
| `desy3_sc_calibration_gatti.npy` | `notebooks/sc_calibration_gatti.ipynb` | | (a), low priority | variant only. The notebook loads `configs/v16/gatti_sc.yaml`, **which does not exist** (v16 has `sc_gatti.yaml`), so it is stale |
| `redshift_distributions/desy3_nz_{metacal,maglim}_bin{i}.txt` | external (DES Y3 release) | | (a) | Y6 mean n(z). **Open:** taken from the 2pt FITS file? Used for the IA amplitude (`redshift.py`) and must match what the projection used |
| `cache/desy3_metacal_gamma_*.npy`, `desy3_metacal_count.npy`, `desy3_maglim_count.npy` | `notebooks/build_des.ipynb` → `catalog.py` | Y3 catalogs | (b) + (a) | names are literal (`catalog.py:168-169,290-291,365`) and loaded whenever present, unchecked against the config (section 6). The notebook loads `configs/v16/default.yaml` |
| `CosmoGridV1_metainfo.h5` | CosmoGridV1 + `notebooks/permutations_list.ipynb` | | none | survey-agnostic |
| `healpy_data/` | healpy | | none | |
| `obs/buzzard_flock*_obs_maps.h5` | `notebooks/build_buzzard_flock.ipynb` + `msfm/utils/buzzard.py` | Buzzard Y3 mocks | (c) | a Y6 validation-mock equivalent. **Open:** which simulation DES Y6 uses for this (Cardinal is my guess, not checked) and in what format. `buzzard.py:69` also has a literal `range(1, 5)` |

### Notebooks that fill in config values

| notebook | fills | Y6 |
|---|---|---|
| `config.ipynb` | `survey.{metacal,maglim}.n_gal` over the mask | rerun. It loads an **absolute Perlmutter** path to `configs/v16/default.yaml` |
| `patches.ipynb` | `footprint.rotation` | predates the config schema (`configs/config.yaml`, `nside` key) and reads Y3 positions from a laptop path. It will not run as is. Rewrite against the Y6 mask only if the Y3 angles fail the non-overlap assert |
| `scale_cuts.ipynb` | the scales-config values (section 7) | rerun on the Y6 n(z) and Y6 fiducial maps. It loads `configs/v16/default.yaml` and a fiducial dir that no longer holds maps |

**(b)** Every production notebook points at a different config version (v15, v16, v18) or at a laptop/Perlmutter path. Before Y6, each needs one `conf_path` variable at the top.

---

## 4. Source clustering bias fit

`msfm/utils/source_clustering_bias.py` plus `sc_bias_fit_count.ipynb`.

- **`RANGE_1PT = (0.5, 200.5)`, `N_BINS_1PT = 200`** (`:53-54`) assume Y3's ~72 sources per pixel. **(b):** derive the range from n_bar. **Open:** whether Y6 needs it, depending on the Y6 per-bin density.
- **The target count map** comes from the `desy3_metacal_count.npy` cache (`:161`). For Y6 it is the metadetection noshear count map. **Open:** whether detection-level blending in metadetection changes the one-point function enough to matter. The fit absorbs it either way, which is the point of fitting.
- **The fit needs the Y6-projected `dg/<source bin>` maps** of all 2501 cosmologies, so it waits on section 1.
- **The rules carry over unchanged** (contaminate the sim, Poisson everywhere, base-patch frame); see memory `project_isd_systematics_bias_fit`.

---

## 5. Metadetection: what changes in shear, response and noise

The Y6 shapes are metadetection (mdet), not metacal. Points 1-3 are what I expect from the method. They are **not verified against the Y6 shape catalog paper or the catalog**, so treat each as an open question until checked.

1. **Response.**
   - Expected: mdet measures the response from the `1p/1m` (and `2p/2m`) sheared catalogs, redoing the selection on each, so R already contains the selection response. There is then no per-object R and no separate `R_s`.
   - Consequences:
     - the map estimator Σ w e / (Σ w · R) keeps its form, with one R per bin;
     - `R_gamma + R_s` becomes one `response` value;
     - the debug assert against per-object `R11/R22` (`catalog.py:259-264`) has to go or be re-implemented from the sheared catalogs;
     - `sigma_e_H12` (`catalog.py:247-253`) uses `R_gamma` and needs the same change.
   - **Open:** whether DES applies a scalar per bin ((R11 + R22)/2) or the 2×2 matrix, and where the per-bin values are published.
2. **Shapes, weights, bins.**
   - Expected: the shapes come from the `noshear` catalog.
   - **Open:** the column names (the fitter's `g_1`/`g_2` equivalent).
   - **Open:** whether the inverse-variance weight is a column or has to be looked up on an (S/N, size-ratio) grid.
   - **Open:** the name of the tomographic-bin column (SOMPZ assignment). In Y3 it was `index/select_bin{i}`.
3. **Shape noise.** The rotate-and-resample model (`lensing.noise_gen`, `noise_gen_in_place`) needs only `(e/R, w, pix)` per galaxy and carries over. Only the reader changes.
4. **m prior.** Verified: correlated with the source n(z) modes. Covered in section 1, item 3.
5. **Sign conventions and frame.** Y3 uses `sign_e1 = +1, sign_e2 = -1` plus the ψ rotation into the rotated footprint (`catalog.py:117-158`).
   - **Open:** the mdet catalog's e1/e2 convention.
   - Before trusting any Y6 map, rerun the **κ_E × δ_g** and **κ_B × δ_g** checks (memory `project_desy3_shear_convention_validated`, scripts in `claude/tmp/meansub_des/`). The E/B power scan does not catch a global flip.
6. **Mean shear.** Y3 decision: not subtracted, ≤ 0.05σ, with no additive term in the forward model. **Open:** redo the measurement on Y6, and check whether the Y6 cosmic-shear analysis models PSF leakage or other additive terms we would have to forward model.

---

## 6. msfm code with Y3 assumptions baked in

Everything not listed reads its bin counts from `len(conf["survey"][...]["z_bins"])`. This holds for `postprocessing.py`, `observation.py`, `tfrecords.py`, `base_pipeline.py`, both pipelines and `parameters.py`. Every `cross_statistics.get_cross_bin_indices` call passes `n_z` explicitly, so its default of 4 is harmless.

| where | what | type | change |
|---|---|---|---|
| `catalog.py:191-194, 311-313, 385-387, 436-438` | catalog filenames `DESY3_indexcat.h5`, `DESY3_GOLD_2_2.1.h5`, `DESY3_metacal_v03-004.h5`, `DESY3_GOLD_2_2.1_DNF.h5`, `DESY3_maglim_redmagic_v0.5.1.h5` | (b) | survey-keyed reader: Y3 reader unchanged, Y6 reader added once the schema is known |
| `catalog.py:199-211, 259-267, 321-329, 389-392, 446-458` | HDF5 schema: `/index/select_bin{i+1}`, `/catalog/gold/{ra,dec}`, `/catalog/unsheared/{e_1,e_2,weight,R11,R22}`, DNF `zmc_sof`/`zmean_sof`, `index/maglim/select` | (b) + (c) | same reader split; Y6 schema **open** |
| `catalog.py:223-224, 336-338, 466-467` | divide by `R_gamma[i] + R_s[i]` | (b) | one per-bin response (section 5) |
| `catalog.py:168-169, 290-291, 365` | cache names `desy3_*`, loaded whenever present (`force_recompute=False`) | (b) | key the names by `survey.name`. Without this a Y6 config silently reads the Y3 maps, here and from downstream (section 7) |
| `filenames.py:52` | n(z) filename prefix `desy3_nz_` | (b) | derive from the survey or a config key |
| `files.py:458-459` | n(z) files loaded by **index** `bin1..binN`, not by label | (b) | load by label. With lens bins 1, 3, 4, 5, 6, an index loop silently reads bins 1-5 |
| `files.py:244, 253` | lss_sys file `bin{i_z}`, asserted `n_bins == n_z` | (b) + (c) | map config bins to file bins explicitly if the Y6 export keeps all 6 lens or 4 source bins under other indices |
| `observation.py:577, 591` | fiducial `bg{i}`, `qbg{i}` for i = 1..n | note | fine as long as the bias parameters are named by position (decision D2) |
| `run_grid_postprocessing.py:400-407` | `bg1, bg2, bg3, bg4 = astro_sample[-4:]` (and the 8-tuple quadratic case) | (b) | slice by `len(params.bg.linear)`. With 5 lens bins it would pass 4 biases for 5 maps |
| `lensing.py:52-55` | `MultivariateNormalDiag` m prior | (b) | full covariance, plus a conditional draw given the projection's u (section 1) |
| `run_grid_postprocessing.py:560`, `run_fiducial_postprocessing.py:426`, `observation.py:325-328` | m drawn independently of the n(z) realization | (b) + (c) | read the per-permutation u from the projected file, draw m given u |
| `filenames.py:22-34`, `run_grid_postprocessing.py:123-129` | projected filename by version string (`v11dmb`/`v11dmo`) | (b) | a config key, or a new version choice, for the Y6 projection |
| `observation.py:603-613` | experimental magnification path: literal Perlmutter file `/pscratch/.../laura/DES_maps_260128/mag_map_bin_{i+1}_run_{perm}.npy`, perm parsed from `map_dir[-4:]` | (c) | only if Y6 needs magnification (decision D4) |
| `source_clustering_bias.py:53-54` | `RANGE_1PT` | (b) | section 4 |
| `buzzard.py:69` | `range(1, 5)` | (c) | Y3 Buzzard only |
| `run_single_postprocessing.py:12-13, 74-82` | NERSC v11desy3 paths in the usage docstring | cosmetic | |
| docstrings in `clustering.py`, `files.py`, `source_clustering_bias.py`, `configuration.py`, `observation.py`, `postprocessing.py` | "DES Y3 imaging systematics" | cosmetic | |

### Pipelines

`pipelines/v18/packed/submit.sh:31-40` hardcodes Perlmutter: `REPO=/global/homes/...`, `DIR_IN=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary`, `DATA_ROOT=/pscratch/...`, regular QOS packing.

- Y6 needs a `pipelines/<new version>/`, mirroring the config version.
- **Open:** whether Y6 postprocessing runs on Perlmutter (where the projections would live) or on daint. That depends on where the reprojection lands.

### Data audit gate

`dev/scripts/data_audit/check_config_paths.py --strict` resolves every config path. A Y6 config pointing at not-yet-delivered `desy6_*` files will fail it. Add them to `EXPECTED_MISSING` with a note until they exist.

---

## 7. Downstream real dependencies (y3-deep-lss, msi)

Checked lines are marked ✓. The rest come from the sweep.

| where | what | type | change |
|---|---|---|---|
| y3-deep-lss `configs/scales/8wl,32gc.yaml` ✓ | `theta_fwhm` (8 and 32 Mpc/h at the **Y3** bin mean redshifts), `white_noise_sigma` (1% of the **Y3** fiducial signal, clustering also set by Y3 `n_gal`), `l_max` (follows), `theta_fwhm_base: 40.4`; all length 4. Same for `8wl,40gc`; `lmax_1024` `white_noise_sigma` too | (a) | new Y6 scales file from `scale_cuts.ipynb` + `dev/scripts/noise_models/derive_scale_cut_noise.py` on Y6 fiducial maps |
| y3-deep-lss `deep_lss/utils/configuration.py:508-516` ✓ | per-pair ℓ cuts fall back to a default when a bin index exceeds the list, **silently** | (b) | length assert against n_z |
| y3-deep-lss `deep_lss/utils/configuration.py:296, 380-386` | smoothing channel count taken from the scales lists | (b) | same assert |
| y3-deep-lss `deep_lss/utils/evaluation.py:566-568` ✓, `cls_evaluation.py:205,207` | write observation keys `"DESy3"`, `"DESy3_no_sys"` | (b) | derive from `survey.name` |
| msi `msi/utils/observations.py:183` ✓ | selects only `DESy3*` keys; a `DESy6` key gets no chain and **no error** | (b) | same |
| msi defaults: `run_inference.py:199`, `coverage.py:618`, `run_fom_percentile.py:27`, `run_ppc.py:263`, `ppc.py:1371`, `observations.py:418`; y3-deep-lss `run_diagnostics.py:937,1113`; every runs config `obs_labels` | `"DESy3"` | (b) + (a) | same |
| y3-deep-lss `deep_lss/utils/evaluation.py:558-563`, `cls_evaluation.py:195-201`; msi `ppc.py:1380-1381` | reach the msfm `desy3_*` caches | (b) | fixed by the msfm cache keying |
| y3-deep-lss `submissions/clariden/common.sh:29` ✓ | `DESy3_grid_dmb_????.tfrecord` glob | (b) | from `survey.name` |
| y3-deep-lss `configs/probes/{clustering,combined,...}.yaml` | `params: [..., bg1, bg2, bg3, bg4]`. A subset is accepted, so `bg5` would be marginalized **silently** | (b) + (a) | derive from msfm `params.bg.linear`, or Y6 probe files |
| msi `configs/runs/v18/...` `params`, `coverage_round.py:47-50`, `plotting.py:55-62` (KeyError for `bg5`, which aborts the remaining chains), `shifted_posterior.py:29-35` | bg1..bg4 | (b) | same |
| msi `msi/utils/preprocessing.py:73-74` ✓, `apps/coverage_round.py:67` ✓ | `n_patches = 4`, `n_noise = 3`; `(2500, 80, ...)` | (b) | read the msfm `analysis` block |
| both `submissions/clariden/common.sh:8` | `STORE=/capstor/store/cscs/swissai/a0158/athomsen` | (a) | repoint after the Globus move (memory `project_allocation_lp184`). No new Y6 output goes there |
| msi runs configs `pred_dir`, `scales_config`; `configs/ppc/*.yaml` `base_dir` | `runs/v18`, `8wl,32gc.yaml`, `data/v18` | (a) | Y6 runs configs |
| msi `msi/utils/chains.py:62-76, 141-245`, `configs/config.yaml:3-22` | Y3 key-project chain loaders and paths | (a), comparison only | `DESy6_key_project_chains` is wired only in `dlss_plot.des_kp` |

Not a dependency: no Python code in `deep_lss` or `msi` hardcodes a bin count or a pixel count. Both come from the msfm config and pixel file. The architecture configs (`token_nside: 16`, `smooth_nside`) are tied to `n_side`, not to the survey.

---

## 8. Decisions that are yours

- **D1. Config version and names.**
  - Proposal: `configs/v19/` with `survey.name: DESy6` and `pipelines/v19/` (the README keeps the two 1:1).
  - The sample keys `survey.metacal` / `survey.maglim`, pixel-file groups and tfrecord features would stay as **role names** (source / lens) for a Y6 metadetection / MagLim++ sample. A rename sweeps msfm, deep_lss, msi and every tfrecord for no physics gain.
- **D2. Which lens bins, and how they are named.**
  - Follow DES (1, 3, 4, 5, 6)? Labels by survey bin (`maglim1, maglim3, ...`) with parameters by position (`bg1..bg5`) is the combination the code handles once `files.py:458` loads by label.
- **D3. The projection.** Who runs it, where, with what n(z) prior representation. Store u per permutation so msfm can draw m | u (section 1).
- **D4. Lens magnification.** Y6 KP varies α. v18 has none. Model it (the experimental path exists), or state it as a limitation?
- **D5. Model choices that stay or follow Y6.** IA δ-NLA vs TATT, pivot 0.62 vs 0.3, linear vs nonlinear bias, Σmν fixed. These are analysis choices, not survey dependencies, but they decide what a Y6 comparison means.

---

## 9. Proposed order of work

1. **Decisions D1-D3.** D3 gates the largest, slowest item, so start that conversation first.
2. **(b) generalizations in msfm, each verified as a no-op on Y3.** The acceptance test is a bit-identical v18 output: the observation from the cached Y3 maps, plus one fiducial permutation through the postprocessing in debug mode on the store's `perm_0000`. The grid maps are gone from scratch.
   1. Cache and n(z) names keyed by survey; n(z) and lss_sys bins by label.
   2. Response key: one `response`, `R_gamma`/`R_s` still accepted.
   3. Catalog reader split: Y3 path unchanged, Y6 stub.
   4. `bg` unpack by length.
   5. `RANGE_1PT` from n_bar.
   6. m prior with full covariance and a conditional-draw hook.
   7. Projected-map filename as a config key.
   8. One `conf_path` per production notebook.
3. **The same kind of change downstream.** The `"DESy3"` label derived from `survey.name`, bg lists from the msfm config, msi grid literals from the `analysis` block, length asserts on the scales lists. y3-deep-lss is a shared org repo, so this lands as commits you push.
4. **`configs/v19/` skeleton** with placeholders and `TODO(y6)` markers, the gate's `EXPECTED_MISSING` updated, nothing runnable yet.
5. **When the Y6 catalogs arrive:**
   1. reader;
   2. caches and count maps;
   3. `n_gal`;
   4. mask and pixel file, with the four-patch non-overlap check;
   5. noise file;
   6. the κ × δ_g sign checks and the mean shear;
   7. lss_sys systematics (needs the env on daint).
6. **When the Y6 reprojection exists:**
   1. bias-table fit;
   2. scales config;
   3. mock observations;
   4. tfrecords and training under the v18 recipes.

**Storage, for every step:** no new output on the a0158 store. Tests go under `deep_lss/claude/` on scratch. Products go on the lp184 store once the Globus move is complete.
