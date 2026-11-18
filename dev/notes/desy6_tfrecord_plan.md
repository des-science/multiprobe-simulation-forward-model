# DES Y6 training set: plan for msfm

Written 2026-11-15. Updated 2026-11-18: the Y3/Y6 split on disk, the config skeleton, and section 5
(products and config values from one script, written but not run yet, see 5.5). It builds on
`desy6_inventory.md` (2026-11-15) and supersedes it where the two disagree. Since the inventory was
written:

- the Y6 projection is running;
- n(z) comes from a realization set, with m attached to each realization;
- the Y6 catalogs are already on disk.

**Goal.** A Y6 `.tfrecord` grid training set, analogous to the frozen Y3 v18 one but built from a
new Y6 config:

- forward modeling runs on the CPUs of daint GPU nodes;
- nothing from Y3 (mask, n(z), catalogs, bias tables, ...) enters unless this plan or the config
  says so explicitly;
- y3-deep-lss and msi are out of scope here.

**Choices** follow DES Y6 3x2pt, arXiv:2601.14559 (call it KP) and its analysis framework,
arXiv:2601.14859 (call it FW), wherever the map-level model allows. Paper numbers below come from
an automated read of the arXiv HTML and are marked *(paper)*. Verify each one before it goes into
the config.

---

## 0. Is Y3 safe, and is Y6 kept distinct, today?

**Y3 is intact.** No Y3 config has changed, and the Y6 code is in new files. The one edit to shared
msfm code is a `prefix` keyword of `filenames.get_filename_z_distribution`; it defaults to `desy3`,
so every Y3 caller builds the same names as before.

**Distinctness is not guaranteed yet.** With today's code, a Y6 config would silently pick up Y3
material in six places:

1. **Catalog caches.** `catalog.py:168-169,365` load `data/desy3/cache/desy3_*.npy` whenever they exist,
   without checking them against the config. y3-deep-lss and msi call the same functions.
2. **n(z) files.** `filenames.py:52` always builds `desy3_nz_<sample>_bin<i>.txt`.
3. **m prior.** m is drawn from the config prior (`lensing.py:52-55`). For Y6 it has to come from
   the projection.
4. **Notebooks overwrite frozen Y3 products.** `pixel_file.ipynb` writes the literal
   `desy3_pixels_fiducial_512.h5`, and `sc_bias_fit_count.ipynb` writes `data/desy3/desy3_metacal_bias.h5`.
5. **Y3 and Y6 projected files look alike.** The Y6 test projection uses the same filename and the
   same bin labels as Y3 (`projected_probes_maps_v11dmb.h5`, `metacal1-4`, `maglim1-4`). A
   `--dir_in` pointing at the wrong tree would raise nothing.
6. **Footprint rotation.** `footprint.rotation` holds the Y3 angles. Reusing them for Y6 is only
   valid after a check against the Y6 mask.

Section 2 closes each of these with a guard, and section 4 lists the regression test that proves
Y3 is unchanged.

---

## 1. What is on disk (checked 2026-11-15, read-only)

### 1.1 Y6 projection (Tomasz)

The grid is at
`/capstor/store/cscs/userlab/lp184/tomaszk/261109_cosmogridv1_desy6/000_test/cosmogrid/processed/desy6/CosmoGrid/bary/grid`.

**Progress and layout**

- 2641 of 50,000 permutation files so far: 2500 `cosmo_*` dirs, about 10 `perm_*` each.
- There are no `fiducial/` or `benchmarks/` trees.
- The layout is the Y3 `v11desy3` one: `map/{kg,ia,dg}/<bin>`, `kernel/*`, `nz/<bin>`,
  `perms_info`, `shell_groups`. `w_shell` is new and `cell` is gone.
- The filename is the Y3 one, `projected_probes_maps_v11dmb.h5`.

**Five things to confirm with Tomasz before any production run (questions in section 9):**

- **nside is 1024** (12,582,912 pixels), not 512, in every file I opened. msfm downgrades on read
  (`postprocessing.py:727-735`), so this works but reads 4x more.
- **Only 4 lens bins**, and the labels are Y3's: `maglim1-4`, `metacal1-4`. Tomasz is fixing the
  files to carry all six lens bins (2026-11-18).
- **m is stored** as an attribute `shearm` on `nz/metacal{i}`, and as 0 on `nz/maglim{i}`.
- **Every sampled file carries the same n(z) and m.** I checked 12 files across cosmologies and
  permutations. The n(z) md5s and the `shearm` values are identical, and the m values equal **row 0**
  of `desy6_source_redshift.h5` exactly. So realization sampling is not active in these files.
- **I did not check which Y6 lens bins `maglim1-4` are.** That check was the run you declined.

### 1.2 Y6 data products (Tomasz, `.../261109_cosmogridv1_desy6/data_desy6a2/`)

From the public release `y6a2_files/y6_psf/final_version`. Only keys were listed, no data was read.

| file | content relevant here |
|---|---|
| `metadetect_2024-11-07.hdf5` (149 GB) | metadetection steps `1m 1p 2m 2p noshear` (5 × 4 bins × 44 columns = the 905 objects listed). Per `<step>/tomo_bin_{0..3}`: `ra`, `dec`, `gauss_g_1`, `gauss_g_2`, `w`, `gauss_s2n`, `gauss_T_ratio`, `mdet_flags`, ... Bins are assigned already. Provenance `Y6A2_METADETECT_V6_UNBLINDED/metadetect_cutsv6_all.h5` (cuts applied) |
| `maglim_2024-11-07.hdf5` (2.8 GB) | `tomo_bin_{0..5}`: `RA`, `DEC`, `weight` (ENET systematics weight per galaxy), `Z_MEAN` (DNF), `PHOTOZBIN`, mask flags `totmaskgood`, `totmaskgood_v3`, `shear_v2_extramask_good`. Bin 0 has 1,852,538 objects |
| `jointmask_DESY6_2024-06-04.hsp` (730 MB) | the joint 3x2pt mask, HealSparse format. `healsparse` is in `torch_env` (installed 2026-11-18), not in `tf_env` |
| `desy6_source_redshift.h5` | **80,760 joint realizations**: `NZ_BIN{1..4}` (300 z points) with `M_BIN{1..4}`. Its m mean and spread equal the KP m priors *(paper, Table 3)*: mean [-0.0034, 0.0065, 0.0159, 0.0017], std [0.0058, 0.0066, 0.0059, 0.0122] |
| `desy6_lens_redshift.h5` | `NZ_BIN{1..6}`, 2967 to 3061 realizations per bin, no m |
| `3x2pt_2025-07-28-20h_UNBLINDED_wmask_nob2.fits` | the KP data vector, with the fiducial n(z) |
| `3x2pt_pipeline_public.ipynb` | DES's own reading of these files; the reference for selections and flags |
| `desy6kp_cats_2024-11-07.hdf5` | provenance attributes only (Gold 2.2, MagLim v3d2, the ENET weight maps per bin, mask, metadetection, SOMPZ paths) |
| randoms, redMaGiC, `maglimpp_wtheta_add_debias_enet_dv5.fits` | not needed for maps. The debias file is a w(θ)-level correction |

**Missing.** The Gold catalog (not downloaded) and the **survey-property maps**. The user has the
survey-property maps in `/capstor/store/cscs/userlab/lp184/athomsen` and will run `lss_sys` on them
(D5, section 5.3).

**Used** (section 5): metadetection, MagLim, the joint mask, the 3x2pt FITS (n(z)), and the `M_BIN*`
columns of `desy6_source_redshift.h5` (m prior). Nothing else from the download.

---

## 2. How Y3 and Y6 coexist

1. **One explicit switch: `survey.name`, either `DESy3` or `DESy6`.**
   - A small registry, new `msfm/utils/surveys.py`, maps it to everything survey-specific: file
     prefix, catalog reader, response handling, m source, n(z) file template.
   - An unknown or missing name raises.
   - I prefer this over adding a second key such as `survey.release`, because two keys can disagree.
   - `survey.name` already names the tfrecords (`DESy6_grid_dmb_*.tfrecord`) and is what downstream
     will key on.
2. **No Y3 file in a Y6 config unless the config says so.**
   - A config check, called from every app's setup and from every Y6 product step, requires that every
     `files:`/`dirs:` path of a `DESy6` config lie under `data/desy6/`, or be named in an explicit
     list `files.survey_agnostic: [meta_info, healpy_data]`.
   - Since 2026-11-18 every Y3 file is under `data/desy3/` and the top level of `data/` holds only
     survey-agnostic inputs, so the guard is a directory check.
   - Anything else raises.
   - The explicit list is the only place where reuse is "stated", and the run log repeats it.
3. **Every new key defaults to today's Y3 behavior when absent.** The acceptance test is
   bit-identical Y3 output (section 4, step A0).
4. **Y3 notebooks and products stay frozen as the provenance of v18.**
   - They live in `notebooks/desy3/` and `data/desy3/` (moved 2026-11-18, filenames unchanged).
   - Y6 products go in `data/desy6/` and come from one config-driven script (section 5), not from
     notebooks.
   - Each step writes to the path its config key names, never to a literal.
   - `data/README.md` gains a Y6 table.
5. **Projection guard.** Before reading a permutation, msfm checks that:
   - the file has every configured bin key;
   - its `nz/<key>` mean redshift matches the configured survey n(z) within a tolerance;
   - `shearm` is present when the config says m comes from the projection.

   A Y3 tree given to a Y6 config then fails on the first file.

---

## 3. The Y6 config: `configs/desy6/v1/default.yaml` (D1, settled 2026-11-18)

**Why this name.** Configs are grouped by survey, `configs/<survey>/<version>/<subversion>.yaml`,
and versions restart per survey. Y3 moved to `configs/desy3/{v16,v17,v18,deprecated}/` and its
pipelines to `pipelines/desy3/` on 2026-11-18. y3-deep-lss already builds
`configs/$VERSION/$SUBVERSION.yaml`, `data/$VERSION/$SUBVERSION` and `runs/$VERSION/$SUBVERSION`,
so `VERSION=desy6/v1 SUBVERSION=default` puts the survey into every downstream path. Pipelines go
in `pipelines/desy6/v1/`.

**How it is built.** It starts as a copy of `configs/desy3/v18/default.yaml`, and its header names
Y3 v18 as the base. The table lists only what changes.

**Baked in or not.** The tfrecords bake in everything except the small-scale smoothing, which the
networks apply on the fly. That means the IA pivot and priors, the bias priors, the m handling, the
l_min = 30 cut and the shape-noise model must all be final before production. Scale cuts can wait.

| key | Y6 value | source / status |
|---|---|---|
| `survey.name` | `DESy6` | switch |
| `files.*` | `data/desy6/desy6_*`, plus `survey_agnostic: [meta_info, healpy_data]` | section 5 |
| `dirs.catalog` | `.../data_desy6a2` (canonical path, not a `/users/.../scratch` symlink) | settled 2026-11-18: Tomasz's copy, read in place. Five files used (section 1.2) |
| `survey.metacal.z_bins` (source role) | 4 bins with Y6 labels | D2 |
| `survey.metacal.n_gal` | raw noshear counts per bin / masked area | step `values`, written by `config` (5.4) |
| `survey.metacal.response` (new) | per-bin R from `1p/1m/2p/2m` | step `values`; cross-check KP Table 2: 0.856, 0.869, 0.819, 0.692 *(paper)* |
| `survey.metacal.shear_bias` | `mu`/`sigma` from the `M_BIN*` columns (step `values`) until WP5 is in; then `source: projection` | WP5 |
| `survey.metacal.z0` | 0.3 (KP pivot; v18 used 0.62) | D4. Baked in through the per-bin IA amplitude |
| `survey.maglim.z_bins` (lens role) | 5 bins, survey bins 1, 3, 4, 5, 6 | settled 2026-11-18: all six are projected and produced, and the config excludes bin 2 as the KP does *(paper, Sec. IV.8)* |
| `survey.maglim.n_gal` | weighted counts per bin / masked area | step `values` (check that the ENET weights average ≈ 1) |
| `survey.maglim.z_lims` | dropped | the catalog is already binned (`tomo_bin_k`); the Y6 reader does not need it |
| `survey.Aeff` | from the mask | step `pixels`, against DES's 4031.036 deg² |
| `analysis.n_patches`, `footprint.rotation` | 4 and the Y3 angles, a choice stated in the config | step `pixels` asserts no overlap. The only Y3 reuse |
| `params.ia` | `nla: [Aia, n_Aia]`, `tatt: [bta]` or bta fixed | D4. KP uses TATT with b_TA = 1 fixed; A2/η2 have no map-level implementation |
| `grid.priors.Aia`, `n_Aia` | KP: A1 [-1, 3], η1 ~ N(0, 3) on [-5, 5] *(paper)*. The Latin hypercube is uniform, so [-5, 5] is the analogue | D4 |
| `params.bg.linear`, priors, fiducials | 5 bins, [0.8, 3.0] (= KP). Fiducials from FW for bins 1, 3, 4, 5, 6: 1.54, 1.85, 1.76, 1.93, 1.90 *(paper)* | D3 (names) |
| `scale_cuts.*.l_min`, `normalization.*` | lists of length 4 (source) and 5 (lens) | |
| `modelling.lensing.shape_noise` | `count`, `fixed`, and `survey_systematics` true or false | D5 |
| `modelling.clustering` | linear per-bin bias as in v18; KP's fiducial is nonlinear | D7 |
| cosmology, baryons, `grid.*`, `fiducial` cosmology | unchanged | fixed by CosmoGridV1. The KP priors (Ωm [0.1, 0.6], h [0.58, 0.8], ...) cannot be followed |

---

## 4. msfm code changes

Each work package is a no-op on Y3.

**A0, first: a Y3 regression baseline, before touching anything.** Deferred by the user on
2026-11-18: Y3 on branch `desy6` is compared against `main` later.

- **Observation path.** `forward_model_observation_map` on the cached Y3 maps with
  `configs/desy3/v18/default.yaml`.
- **Fiducial postprocessing.** `run_fiducial_postprocessing` in debug mode on the store's fiducial
  `perm_0000`.
- **Grid postprocessing.** `run_grid_postprocessing`, debug, on one cosmology. The Y3 grid maps are
  gone from scratch, so this needs a scratch symlink tree under `claude/tmp/` pointing every
  `perm_*` at that one file. That is physically meaningless but valid for a regression test.
- **Determinism.** Run each twice on unchanged code first. The m draw
  (`run_grid_postprocessing.py:560`) passes no seed, so I expect the tfrecords to be not bit-stable.
  If so, the baseline compares everything except the m factor, or seeds it in a separate, explicit
  change.
- Every WP below must reproduce this baseline.

| WP | change | files |
|---|---|---|
| WP1 | survey registry, `survey.name` validation, the Y6 path guard and the `survey_agnostic` list (section 2, items 1-2) | new `utils/surveys.py`, `utils/configuration.py`, app `setup()`s |
| WP2 | survey-keyed names. The cache directory becomes `data/<survey>/cache/` (hardcoded to `data/desy3/cache/` today) and cache files take the registry prefix. n(z) files load by **bin label**, since the Y6 lens bins 1, 3, 4, 5, 6 are not contiguous; Y3's index naming stays as the fallback. Product outputs come from config keys | `catalog.py:168-169,290-291,365`, `filenames.py:52`, `files.py:452-465` |
| WP3 | Y6 catalog reader, new `utils/catalog_desy6.py` (detail below) | `catalog_desy6.py`, `catalog.py` |
| WP4 | `survey.metacal.response` replaces `R_gamma + R_s` when present. The catalog assert against `R11/R22` becomes Y3-only | `catalog.py:223-224,259-264,336-338,466-467` |
| WP5 | m from the projection (detail below) | `postprocessing.py`, `run_grid_postprocessing.py:516-580`, `run_fiducial_postprocessing.py:426`, `observation.py:325-328`, `lensing.py` |
| WP6 | projection guard (section 2, item 5). An optional `survey.<sample>.projection_keys` maps config labels to file keys, only if D2 ends in a mapping | `postprocessing.py:674-735`, `observation.py`, `source_clustering_bias.py:245-283` |
| WP7 | bin-count generality (detail below) | see detail |
| WP8 | data-audit gate: the Y6 config resolves, with not-yet-produced `desy6_*` files in `EXPECTED_MISSING` until they exist | `dev/scripts/data_audit/check_config_paths.py` |

**WP3: the Y6 catalog reader**

- Source:
  - shear map Σ w g / (Σ w · R), from the `noshear` step;
  - R per bin from the `1p/1m/2p/2m` steps;
  - count map;
  - shapes for the noise file.
- Lens: count map weighted with the per-galaxy ENET `weight`, plus the raw count map.
- Mask: read from the `.hsp`.
- The public `catalog.build_*_from_cat` names become dispatchers on `survey.name`, so the
  y3-deep-lss and msi callers get Y6 maps for a Y6 config with no change on their side.
- **Open, to be decided by measurement:**
  - the R convention, scalar or 2×2 (KP lists a scalar, *paper*);
  - the shear step size (0.01 is the metadetection default; I have not verified it for Y6);
  - the e1/e2 sign convention and the ψ rotation;
  - which mask flags define the sample, read from `3x2pt_pipeline_public.ipynb`.

**WP5: m from the projection**

- Read `nz/<source bin>.attrs["shearm"]` per permutation and apply `kg *= 1 + m` per bin. The same
  m holds for all four patches of a permutation, which is what a joint m–n(z) realization means.
- No m draw for Y6. The Y3 path (`source: prior` or absent) is unchanged.
- Assert that the lens `shearm` values are 0 and that the source values lie within ±5σ of the set's
  prior.
- m stays out of the label vector, as in Y3.

**WP7: bin-count generality**

- Grid bias unpack by the length of `params.bg.linear` (`run_grid_postprocessing.py:400-407`).
- Fiducial bias lookups by name rather than position (`observation.py:577,591`).
- If D4 fixes bta, support an empty `params.ia.tatt`. Otherwise `astro_sample[:3]` would read `bg1`
  as bta (`run_grid_postprocessing.py:385-389`).
- `RANGE_1PT` derived from n_bar (`source_clustering_bias.py:53-54`). Y6 has about 33.7M sources per
  bin, roughly 100 per pixel against Y3's 72, so the fixed range [0.5, 200.5] would clip the tail.

---

## 5. Y6 data products and config values: one config-driven script (settled 2026-11-18)

### 5.1 Why a script and not notebooks

Y3 made its products with the notebooks in `notebooks/desy3/`. They stay frozen as the provenance
of v18. Y6 uses one command-line script instead, for three reasons:

- the Y3 notebooks hardcode paths that no longer exist (`configs/v15`, `/Users/arne`,
  `/global/homes`);
- they carry hidden state;
- the steps that read the 149 GB metadetection file need batch jobs.

The steps serve the same purposes as the Y3 production notebooks (table in 5.3). Most of those
notebooks were already thin wrappers around msfm (`catalog.py`, `source_clustering_bias.py`). The
script is a thin driver in the same way: the logic lives in msfm, where the observation path needs
the Y6 reader (WP3) anyway. `pixel_file.ipynb` is the exception, because its logic is inline. It
becomes a function.

There is no `notebooks/desy6/`.

### 5.2 Interface

- **Command.** `python -m msfm.apps.products.desy6 <step> --config configs/desy6/v1/default.yaml`.
- **Inputs.** From `dirs.catalog`, which points at Tomasz's download, read in place.
- **Outputs.** Every output goes to the path its config key names. Nothing is written to a literal
  path.
- **Submission.** One wrapper, `pipelines/desy6/products/run.sh <step>`, submits the compute-node
  steps. Every `sbatch` waits for the user's OK.
- **Checks.** Each step asserts its checks (5.3) and stops on a failure. Nothing depends on someone
  looking at a plot.
- **Provenance.** Each step writes its record to the tracked `data/desy6/provenance.yaml`
  (`files.provenance`). There is one record per step, replaced on a rerun, and git keeps the
  history. A record holds:
  - its inputs, with size and mtime;
  - the msfm commit. A step that writes refuses to run while `msfm/` or `configs/` have
    uncommitted changes, so the commit is the code that ran;
  - the config, the command and the date;
  - its outputs, with sha256.
- **Review.** One read-only notebook, `dev/notebooks/y6/products.ipynb`, plots every product,
  including the κ_E × δ_g sign test and the κ_B × δ_g null test (memory
  `project_desy3_shear_convention_validated`). It produces nothing.
- **Environment** (settled on daint 2026-11-18). The steps run in `torch_env` in the
  `pytorch/v2.9.1:v2` uenv. The only addition was healsparse 1.16.0 (with hpgeom 1.7.1), which
  only `pixels` needs. Since the maintenance, uenv finds its images through
  `~/.config/uenv/config.toml`, which names `/iopsstor/scratch/cscs/athomsen/.uenv-images` as the
  `default` repository. Debug job 5385066 confirmed `sbatch --uenv`, healsparse and msfm from this
  checkout on a compute node.

### 5.3 Steps

They run in this order: `nz` → `pixels` → `maps` → `values` → `config` → `noise` → `sc_bias`.
`sc_bias` comes after `config` because the count model reads `n_gal` from the config.

**Lens bins.** All six survey bins are produced: n(z), maps and `n_gal`. The config then selects
bins 1, 3, 4, 5 and 6 through `z_bins`, because the KP excludes bin 2. Tomasz is fixing the
projection to carry all six (2026-11-18).

| step | output (`data/desy6/`) | inputs | Y3 counterpart | checks | runs on |
|---|---|---|---|---|---|
| `nz` | `redshift_distributions/desy6_nz_metacal_bin{1-4}.txt`, `desy6_nz_maglim_bin{1-6}.txt` | 3x2pt FITS, HDUs `nz_source` and `nz_lens` | none; the Y3 n(z) came from outside the repo | the values equal the FITS columns; the shared z grid | login node |
| `pixels` | `desy6_pixels_fiducial_512.h5`: the rotated mask at nside 512 only, with patches for every survey bin (stored once, hard-linked under each label), the rotated coverage (`footprint/coverage`) and the full-resolution area (attribute `area_deg2`, read by `values`) | `.hsp` at full resolution: the area counted 16 coverage pixels at a time (`valid_pixels` of the whole mask needs ~450 GB, DES); the coverage sampled at 256 points per pixel (nside 8192), rotated back with `catalog.survey_pix_to_angles` | `pixel_file` + `patches` | no pixel appears twice in the four patches. The area against DES's 4031.036 deg²; the rotated coverage sums to it within 1%. The layout is the one `load_pixel_file` reads, with no `maglim/systematics` group, since the Y6 lens weights are per galaxy. **Coverage threshold** `analysis.footprint.coverage_threshold`, Y3's 0.72 as a start (D9); FW says the Y6 mask has more small-scale structure *(paper)* | compute node |
| `maps` | in `dirs.cache`: `desy6_metacal_{gamma_e1p_e2m,count,response}.npy`, `desy6_maglim_{count,weighted_count}.npy`, all survey bins | metadetection (all five steps), MagLim. No cuts: the release is the fiducial selection (`totmaskgood`, `shear_v2_extramask_good` all True) | `build_des` | R from `1p/1m/2p/2m` (Δg = 0.01) against KP Table 2 (0.856, 0.869, 0.819, 0.692 *(paper)*) within 5e-3; the MagLim weights average 1; the mean shear is logged, not subtracted (D8) | compute node |
| `values` | `desy6_survey_values.yaml`, keyed by bin label | maps (including the response the shear maps were calibrated with), pixel file, `desy6_source_redshift.h5` (`M_BIN*` only) | `config` | n_gal = count in the footprint / footprint area, raw counts for the sources, weighted for MagLim; over 95% of the galaxies in the footprint; the pixel file's rotation and threshold equal the config's; the m means against DES's notebook ([-3.4, 6.46, 15.94, 1.7]e-3) within 1e-5; `Aeff` from the pixel file | login or compute node |
| `config` | the marked lines of `configs/desy6/v1/default.yaml` (5.4) | the values file | none; done by hand for Y3 | reparse, then every marked key equals the values file | login node |
| `noise` | `desy6_noise_512.h5`, untracked, ~2.2 GB: (e1/R, e2/R, w, pix) per survey bin, float32, at `metacal<k>/cat` | metadetection `noshear`, the response and count map of `maps` | `noise_file` | per bin, the galaxies' pixels reproduce the count map of `maps` exactly; `load_noise_file` reads the configured bins back with the right row count | compute node |
| `sc_bias` | `desy6_metacal_bias.h5`, `clean` and `contam` arms | `dg/<source>` of all 2500 grid cosmologies at `perm_0000` from the Y6 projection, the count map, the systematics file | `sc_bias_fit_count` (the fit is `source_clustering_bias.py`) | `b ≥ 0`, multi-start; `systematics_label` agrees with the systematics file (`configuration.py:9-50`) | compute node, after the production projection |
| (external) | `desy6_metacal_systematics_<label>_512.h5` | an `lss_sys` run on the Y6 survey-property maps in `/capstor/store/cscs/userlab/lp184/athomsen` | as for Y3, external | label, attribute and guard agree | **TODO, the user's run** |

### 5.4 The config values

The config holds three kinds of value:

| kind | keys | how it is set |
|---|---|---|
| derived from the data | `survey.metacal.n_gal`, `survey.maglim.n_gal`, `survey.metacal.response`, `survey.Aeff`, `survey.metacal.shear_bias.multiplicative.{mu,sigma}` | generated by `config` |
| modelling choices | `z_bins`, `z0`, IA, priors, shape noise, scale cuts, `footprint.rotation` | by hand, with a comment saying why |
| paths | `files.*`, `dirs.*` | fixed by the naming convention |

**Marker.** A generated line reads `<key>: <value>  # <- desy6 values: <dotted.key>`.

**What `config` does:**
- it rewrites only the marked lines, so comments and layout stay untouched and the step is
  idempotent;
- per-bin values are picked from the values file by the labels in `z_bins`;
- it then reparses the YAML and asserts that every marked key equals the values file;
- `--check` changes nothing and exits non-zero if the two disagree. It runs before any production
  submission.

A number can then be traced from its config line to the values file, and from there to
`provenance.yaml`.

**Rejected: merging the values file in at load time.** Run dirs embed the config, the numbers would
no longer be visible in it, and it would change the loader Y3 uses.

**`footprint.rotation` is a choice.** The config takes the Y3 angles and says in a comment that they
are reused. `pixels` fails if the four rotated Y6 patches overlap; new angles are chosen only then.

### 5.5 Status (2026-11-18, during the daint maintenance)

Code lives in `msfm/apps/products/desy6.py` (the script) and `msfm/utils/catalog_desy6.py` (the
reader: release file names, response, maps). **Nothing has run in a real environment yet.** The
code was written while the cluster was down, and per the user it is only syntax-checked and read,
with no fake-library test harnesses.

| step | state |
|---|---|
| housekeeping (5.6) | done |
| `nz` | written. Its checks are based on the real FITS tables, read with h5py/numpy |
| `maps` | written: R from the sheared catalogs, saved as `desy6_metacal_response.npy`; no cuts; maps of all survey bins in `dirs.cache` |
| `config` | written: marker rewrite, reparse checks, `--check` |
| `pixels` | written: area at full resolution, rotated coverage by sub-pixel sampling, Y3 patch symmetries, padded data vector. Needs an environment with `healsparse` (≥ 1.10.1, as DES) |
| `values` | written: n_gal, R, the m prior (its check runs on the real `M_BIN` samples: max deviation 4.8e-6) and `Aeff`. `pixels` now reads its file back through `files.load_pixel_file` |
| `noise` | written: the shapes with the maps' signs, ψ rotation and R; shares the unsheared-catalog reader with `maps` |
| `sc_bias` | later: needs the production projection and the user's `lss_sys` run |
| `pipelines/desy6/products/run.sh` | written: submits `pixels`, `maps` or `noise` (1 CPU node, `torch_env` in the pytorch uenv, `lp184`) after their `--dry-run` on the login node; refuses uncommitted changes in `msfm/` or `configs/`; `DRYRUN=1` prints the `sbatch` line |
| `dev/notebooks/y6/products.ipynb` | written, read-only: provenance (sha256 of every output against the record), n(z), the coverage threshold (histogram, area and interior holes vs threshold, for D9), patches and padding, values vs published numbers plus `config --check`, maps and mean shear (D8), noise file, and the κ_E × δ_g sign / κ_B × δ_g null test (Knox errors, 7 bands in ℓ 30–1000) |

Flags: `--dry-run` (lists the inputs and outputs) on every step but `config`, whose dry run is
`--check`; `--max-objects N` (a catalog slice, writes nothing) on `maps` and `noise`. Config keys added for
the steps: `dirs.cache`, `files.provenance`, `files.survey_values`,
`analysis.footprint.coverage_threshold`. Every not-yet-produced path is
in `EXPECTED_MISSING` of `check_config_paths.py`; prune each entry once its step has run.

First commands after the maintenance, on the login node in `torch_env` (from the repo root):

```bash
uenv run --view=default pytorch/v2.9.1:v2 -- ~/dlss/torch_env/bin/python -m msfm.apps.products.desy6 nz --config configs/desy6/v1/default.yaml
uenv run --view=default pytorch/v2.9.1:v2 -- ~/dlss/torch_env/bin/python -m msfm.apps.products.desy6 maps --config configs/desy6/v1/default.yaml --max-objects 100000
```

Then, after a commit (so `provenance.yaml` records a clean one), the compute-node steps, each
`sbatch` with the user's OK, and the login-node steps in between:

```bash
pipelines/desy6/products/run.sh pixels   # needs healsparse in torch_env
pipelines/desy6/products/run.sh maps
uenv run --view=default pytorch/v2.9.1:v2 -- ~/dlss/torch_env/bin/python -m msfm.apps.products.desy6 values --config configs/desy6/v1/default.yaml
uenv run --view=default pytorch/v2.9.1:v2 -- ~/dlss/torch_env/bin/python -m msfm.apps.products.desy6 config --config configs/desy6/v1/default.yaml
pipelines/desy6/products/run.sh noise
```

Open points to verify on the first real runs:
- the KP Table 2 response values come from an automated paper read. The `maps` check stops if R
  differs by more than 5e-3;
- the e1/e2 sign and ψ convention is Y3's until the κ_E × δ_g sign test;
- the Y6 n(z) grid starts at z = 0.00 (n = 0 there); check it once the n(z) loader reads Y6 labels
  (WP2);
- `pixels` uses only `HealSparseCoverage.read`, `HealSparseMap.read` (whole, and with `pixels=`),
  `valid_pixels` and `get_values_pos`. Its memory assumes the bit-packed mask stays packed when read
  (a few GB; ~28 GB unpacked).

### 5.6 Housekeeping before the first step (done 2026-11-18)

The skeleton and `data/README.md` described Y6 notebooks at first. Since then:

- the `# <- <notebook>.ipynb` comments became step markers;
- `footprint.rotation` got the Y3 angles with a reuse comment;
- `data/README.md` names the step as each product's source, and lists `desy6_source_redshift.h5` as
  a fifth input (its m columns only).

---

## 6. Running the forward model on daint CPUs

**Environment**

- `--environment=tensorflow` with `tf_env`, which has `esub`, `copy_guardian` and `sobol_seq`.
- `CUDA_VISIBLE_DEVICES=` so that TensorFlow never initializes the GPUs. The noise code already pins
  `/CPU:0`.
- Account `lp184`, partition `normal`; `debug` only for single-node smoke tests, which run one at a
  time.

**Container mount (done 2026-11-18).** The tensorflow EDF mounts
`/capstor/store/cscs/userlab/lp184`, where the projection and the Y6 release live. Debug job
5385072 confirmed it: CPU-only TF 2.17, msfm from this checkout, and one projected file
(`cosmo_000001/perm_0000`) opened inside the container.

**Executor**

- Reuse `pipelines/common/packed/run_packed.py` unchanged. It already calls the apps' `main`/`merge`
  without esub.
- Add one generic `pipelines/common/daint/packed_node.slurm`: a single srun step with
  `4 × k` tasks bound to the four Grace NUMA domains, `OMP_NUM_THREADS = TF_NUM_INTRAOP_THREADS =
  72 / k`, each task taking its index slice from `SLURM_PROCID`.
- `pipelines/desy6/v1/daint/submit.sh grid|fiducial` keeps the v18 packed interface: `DRY_RUN`, `ARRAY`,
  `--rerun`, `SKIP_MERGE`.
- Submissions go through the `submit` skill, and every `sbatch` waits for your OK.

**A known trap.** Memory `reference_srun_cpu_bind_none` says sub-allocating srun steps need
`--cpu-bind=none` or the log comes back empty. The first job therefore prints each task's
`os.sched_getaffinity` and log. If SLURM binding fails, fall back to `--cpu-bind=none` plus
`numactl`/`taskset` inside the task (**Open:** whether the container has `numactl`). Perlmutter
measured unpinned packing as ~4x slower, so pinning is not optional.

**Sizing (the benchmark you want later, as one deliberate step)**

- A GH200 node has 4 × 72 Grace cores and ~476 GB host DRAM (~119 GB per domain; `free` also counts
  the HBM).
- The Perlmutter bandwidth knee (3 tasks per 16-core EPYC domain, 24 per node) does not transfer to
  Grace's memory system.
- Benchmark: one node, one grid cosmology per task, k ∈ {2, 3, 4, 6} tasks per domain.
- Record per task: wall time, peak RSS, tfrecord size and read rate from capstor.
- Result: node-hours per cosmology, extrapolated to 2500. The Perlmutter reference is ~150
  node-hours for the v17 grid.

**IO and storage**

- Inputs: ~30 TB read at nside 1024, or ~7.5 TB at 512, straight from Tomasz's store copy.
- Production outputs: `/iopsstor/scratch/cscs/athomsen/deep_lss/data/desy6/v1/default/tfrecords/grid/`
  (the y3-deep-lss `INPUT` convention, canonical path).
- Tests and benchmarks: `deep_lss/claude/bench/`.
- Nothing goes on the a0158 store. Finished tfrecords go to the lp184 store, given the scratch
  purges.

---

## 7. Validating the Y6 tfrecords

- **Schema**:
  - `n_z_metacal = 4`, `n_z_maglim = 5`;
  - `n_pix` equal to the Y6 data-vector length;
  - label width and order equal to `parameters.get_parameters`;
  - 2500 files with 80 signals × 5 noise realizations per cosmology.
- **Nuisances.** The m values differ across permutations and match the file's `shearm`; the n(z)
  realization varies. Both depend on Tomasz's fix (Q2).
- **Noise and counts.** Per-pixel shape-noise rms and the source-count one-point function against
  the Y6 data maps.
- **Signal.** Mean κ power at the fiducial against theory with the Y6 n(z), using
  `dev/scripts/wl_audit`. Optional.
- **Data vs mock.** `forward_model_observation_map` on the Y6 catalog maps next to one mock, as in
  the paper's figure. This also exercises the Y6 observation path.
- **Y3 again.** Rerun the A0 regression at the end.

---

## 8. Order of work

1. **Phase A, now, no Y6 inputs needed:**
   1. A0 baseline (deferred, see section 4);
   2. WP1, WP2, WP4, WP6, WP7, WP8;
   3. WP5 against the test projection, which already carries `shearm`;
   4. the `configs/desy6/v1/default.yaml` skeleton (written 2026-11-18; `null` marks the values
      that the product steps fill in).
2. **Phase B, the catalogs are already here.** These are the product steps of section 5, in order:
   1. housekeeping (5.6, done);
   2. `nz`, which needs no new reader code;
   3. `pixels`, once an environment with `healsparse` exists;
   4. `maps`, `values`, `config` and `noise`, after WP3 and WP4;
   5. the review notebook, with the sign and mean-shear checks.
3. **Phase C, the test projection is usable now:**
   1. the daint executor and the mount check;
   2. a smoke test on 1-2 grid cosmologies;
   3. the CPU sizing benchmark;
   4. `sc_bias` at `perm_0000`, once the user's `lss_sys` run exists and if Tomasz confirms that the
      dg/source maps in the test files are final apart from the realization sampling.
4. **Phase D, after the production projection:** `sc_bias` (if it was not done in C), the full grid
   tfrecords, validation. Fiducial tfrecords come later: production y3-deep-lss reads only the grid
   set, and there is no fiducial projection yet.

---

## 9. Decisions (yours) and questions (Tomasz)

**Decisions**

- **D1.** Settled 2026-11-18: `configs/desy6/v1/default.yaml` and `pipelines/desy6/v1/`, with Y3
  under `configs/desy3/` and `pipelines/desy3/`.
- **D2.** Bin labels.
  - Preferred: Tomasz labels the Y6 bins distinctly in the projection, for example `desy6_source1-4`
    and `desy6_lens1-6`. msfm uses the same labels, so a Y3 tree cannot satisfy a Y6 config.
  - Fallback: the `projection_keys` mapping (WP6).
- **D3.** Bias parameter names: by survey bin (`bg1, bg3, bg4, bg5, bg6`, matching the DES chains'
  `b1e_bin*`) or by position (`bg1..bg5`). I recommend by survey bin.
- **D4.** IA: pivot 0.3, the KP prior boxes, and bta fixed at 1 or sampled on [0, 2] as in v18. All
  of it is baked into the tfrecords.
- **D5.** Settled 2026-11-18: as v18, `count` with `survey_systematics: true`. So both `sc_bias` and
  the systematics file are needed. The user runs `lss_sys` on the Y6 survey-property maps; until
  then the file is a TODO input, as it was for Y3.
- **D6.** Lens magnification (KP samples α). Not modeled in v18. Ask Tomasz for `kg` maps of the
  lens bins now anyway (Q5), since adding them later means reprojecting.
- **D7.** Lens bias stays linear (v18), not KP's nonlinear fiducial.
- **D8.** Mean shear. KP subtracts it from the data *(paper)*; v18 does not subtract it anywhere.
  This affects only the observation, not the tfrecords.
- **D9.** Mask coverage threshold, `analysis.footprint.coverage_threshold`. Starts at Y3's 0.72;
  re-derive it from `footprint/coverage` in the pixel file after the first `pixels` run, which also
  logs the footprint area at 0.5–0.9 and the mean coverage inside it.
- **D10.** IA amplitude n(z): the mean n(z), as in v18, or each permutation's own `nz/<bin>`.

**Questions for Tomasz**

- **Q1.** Where does the production projection write, and at which nside? The files in `000_test`
  are nside 1024.
- **Q2.** n(z) sampling.
  - All 12 files I sampled carry realization 0.
  - How will source and lens realizations be assigned: per permutation, per (cosmology,
    permutation), with which seed?
  - Can the realization indices be stored as attributes next to `shearm`?
  - For fiducial derivative pairs, both signs need the same realization per permutation.
- **Q3.** Will all six lens bins be projected, and under which labels? The user is arranging all six
  with Tomasz (2026-11-18); the labels are still open (D2).
- **Q4.** Which lens n(z) set: `desy6_lens_redshift.h5`?
- **Q5.** Can `map/kg` be added for the lens bins, to keep magnification possible?
- **Q6.** Are fiducial and benchmark projections planned? They are needed for fiducial tfrecords,
  mocks and the fiducial key of the bias table.

---

## 10. Risks

- **Scratch purges.** The Y3 CosmoGrid tree vanished from scratch twice. Inputs stay on the store;
  outputs get copied to the lp184 store promptly.
- **Unknown cost.** aarch64 healpy/libsharp throughput is unmeasured until the benchmark, which is
  the point of doing it first.
- **The Y3 baseline may not be bit-stable** (unseeded m draw). Establish that before refactoring,
  or a real regression will hide in the noise.
