# DES Y6 training set: plan for msfm

Written 2026-11-15. Updated 2026-11-18: the Y3/Y6 split on disk, the config skeleton, and section 5
(products and config values from one script; status in `data/desy6/README.md`). It builds on
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
(D5, `data/desy6/README.md`).

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
   - `data/desy6/README.md` describes the Y6 products.
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
| `survey.metacal.n_gal` | raw noshear counts per bin / masked area | step `values`, which writes the config line (5.4) |
| `survey.metacal.response` | dropped (2026-11-20): R from `1p/1m/2p/2m` divides the cached shear in step `maps` and nothing reads a config value; KP Table 2: 0.856, 0.869, 0.819, 0.692 *(paper)* | step `maps` |
| `survey.metacal.shear_bias` | `mu`/`sigma` from the `M_BIN*` columns (step `values`) until WP5 is in; then `source: projection` | WP5 |
| `survey.metacal.z0` | 0.3 (KP pivot; v18 used 0.62) | D4. Baked in through the per-bin IA amplitude |
| `survey.maglim.z_bins` (lens role) | 5 bins, survey bins 1, 3, 4, 5, 6 | settled 2026-11-18: all six are projected and produced, and the config excludes bin 2 as the KP does *(paper, Sec. IV.8)* |
| `survey.maglim.n_gal` | raw counts × the lens weight maps per bin / masked area (the corrected observation) | step `values` |
| `survey.maglim.z_lims` | dropped | the catalog is already binned (`tomo_bin_k`); the Y6 reader does not need it |
| `survey.Aeff` | from the mask | step `pixels`, against DES's 4031.036 deg² |
| `analysis.n_patches`, `footprint.rotation` | 4 and the Y3 angles, a choice stated in the config | step `pixels` asserts no overlap. The only Y3 reuse |
| `params.ia` | `nla: [Aia, n_Aia]`, `tatt: [bta]` or bta fixed | D4. KP uses TATT with b_TA = 1 fixed; A2/η2 have no map-level implementation |
| `grid.priors.Aia`, `n_Aia` | KP: A1 [-1, 3], η1 ~ N(0, 3) on [-5, 5] *(paper)*. The Latin hypercube is uniform, so [-5, 5] is the analogue | D4 |
| `params.bg.linear`, priors, fiducials | 5 bins, [0.8, 3.0] (= KP). Fiducials from FW for bins 1, 3, 4, 5, 6: 1.54, 1.85, 1.76, 1.93, 1.90 *(paper)* | D3 (names) |
| `scale_cuts.*.l_min`, `normalization.*` | lists of length 4 (source) and 5 (lens) | |
| `modelling.lensing.shape_noise` | `count`, `fixed`, and `survey_systematics` true or false | D5 |
| `modelling.clustering` | linear per-bin bias as in v18, which is also the KP's fiducial (Sec. IV.1, V.1.1; nonlinear b2 is an extension) *(paper, checked 2026-11-19)* | D7, settled by the paper |
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
- Lens: the raw count map, corrected on the observation path by the ENet weight maps of the pixel file, as in Y3.
- Mask: read from the `.hsp`.
- The public `catalog.build_*_from_cat` load `<dirs.cache>/<survey>_*` for either survey and pick
  the configured bins by label, so the y3-deep-lss and msi callers get Y6 maps for a Y6 config with
  no change on their side. Only Y3 is rebuilt from the catalog there; Y6 comes from step `maps`
  (done 2026-11-20).
- **Open, to be decided by measurement:**
  - the R convention, scalar or 2×2 (KP lists a scalar, *paper*);
  - the shear step size (0.01 is the metadetection default; I have not verified it for Y6);
  - the e1/e2 sign convention and the ψ rotation;
  - which mask flags define the sample, read from `3x2pt_pipeline_public.ipynb`.

**WP5: m from the projection**

- **Done 2026-11-20** for the grid: `survey.metacal.shear_bias.multiplicative.source: projection`
  (absent = `prior`, Y3) makes `postprocess_grid_permutations` read `shearm` per permutation, bounded
  to 5σ of the config prior, and the grid loop passes it to `lensing_transform`; the fiducial app
  raises for it. Y3 regression IDENTICAL (job 5405417).
- The projection does not apply m (measured 2026-11-20 on the two fixed test files): `kernel/kg`
  equals a recomputed Born kernel of the stored n(z) to 0.1% after a per-shell calibration on the
  Y3 reference files (no m), with no trace of 1 + m (up to 2.3% between the permutations in bin 4),
  and the `map/kg` monopole equals sum(kernel x mean particle count) to 0.1%, as in Y3.
- Read `nz/<source bin>.attrs["shearm"]` per permutation and apply `kg *= 1 + m` per bin. The same
  m holds for all four patches of a permutation, which is what a joint m–n(z) realization means.
- No m draw for Y6. The Y3 path (`source: prior` or absent) is unchanged.
- Assert that the lens `shearm` values are 0 and that the source values lie within ±5σ of the set's
  prior.
- m stays out of the label vector, as in Y3.

**WP1 path guard and WP6 projection guard: done 2026-11-20, simpler than items 2 and 5 of section 2**

- Paths: `configuration.check_survey_paths` requires every `data/` path in `files:`/`dirs:` to lie
  directly in `data/` (shared) or under `data/<survey.name.lower()>/`. Same rule for every survey, no
  `survey_agnostic` list; all 16 live configs pass. Called by `print_and_check_modeling_in_config`
  (the three apps) and `desy6.main`.
- Projection: `lensing.check_projection` requires `shearm` on `nz/<source bin>` exactly when
  `multiplicative.source` is `projection`. Y3 projections have none, so it catches a Y6 config on a
  Y3 tree and a Y3 config on a Y6 tree. Called wherever a projection is opened: `_get_full_sky_perm`
  (grid and fiducial), `observation.py`, `source_clustering_bias.py`.
- The mean-z check was dropped after measuring it: the Y6 ensembles spread up to ±0.087 in mean z
  (metacal4), Y3 and Y6 differ by 0.075, 0.017, 0.10, 0.22 in the four source bins and by < 0.05 in
  the lens bins, and Y3's own projection is 0.035 off `desy3_nz_maglim_bin2.txt`. No fixed tolerance
  separates more than metacal3–4. A missing bin key already raises, so it needs no check either.
- Side finding: the two test perms' source n(z) sit about 2σ below the ensemble mean in all four
  bins (metacal1 0.391 vs 0.414 ± 0.012). Re-check on the production re-projection.

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

The steps serve the same purposes as the Y3 production notebooks (see `data/desy6/README.md`). Most of those
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
- **Checks.** Each step asserts its checks (listed in `data/desy6/README.md`) and stops on a failure. Nothing depends on someone
  looking at a plot.
- **Provenance** (opt-in, `--provenance`, for production runs). Without the flag a step only
  writes its outputs, so the code can keep changing during development. With it, the step refuses
  uncommitted `msfm/` or `configs/` and writes its record to the tracked
  `data/desy6/provenance.yaml` (`files.provenance`). There is one record per step, replaced on a
  rerun, and git keeps the history. A record holds:
  - its inputs, with size and mtime;
  - the msfm commit, which is the code that ran;
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

### 5.3 Steps and status

What each step does, its checks, its results and its status are in `data/desy6/README.md`, the
reference while the steps run. `sc_bias` comes after `values` because the count model reads `n_gal`
from the config.

### 5.4 The config values

The config holds three kinds of value:

| kind | keys | how it is set |
|---|---|---|
| derived from the data | `survey.metacal.n_gal`, `survey.maglim.n_gal`, `survey.Aeff`, `survey.metacal.shear_bias.multiplicative.{mu,sigma}` | written by `values` |
| modelling choices | `z_bins`, `z0`, IA, priors, shape noise, scale cuts, `footprint.rotation` | by hand, with a comment saying why |
| paths | `files.*`, `dirs.*` | fixed by the naming convention |

**Marker.** A generated line reads `<key>: <value>  # <- step values: <dotted.key>`.

**What `values` does with them** (one step since 2026-11-20; the intermediate values file and the
separate `config` step were dropped):
- it rewrites only the marked lines, so comments and layout stay untouched and the step is
  idempotent;
- per-bin values are picked by the labels in `z_bins`;
- it then reparses the YAML and asserts that only the marked values changed, and that every marker
  has a value and every value a marker;
- `--check` changes nothing and exits non-zero if a marked value is stale. It runs before any production
  submission.

A number can then be traced from its config line, after a run with `--provenance`, to the inputs
recorded in `provenance.yaml`.

**Rejected: merging the values file in at load time.** Run dirs embed the config, the numbers would
no longer be visible in it, and it would change the loader Y3 uses.

**`footprint.rotation` is a choice.** The config takes the Y3 angles and says in a comment that they
are reused. `pixels` fails if the four rotated Y6 patches overlap; new angles are chosen only then.

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
   1. housekeeping (done);
   2. `nz`, which needs no new reader code;
   3. `pixels`, once an environment with `healsparse` exists;
   4. `maps`, `values` and `noise`, after WP3 and WP4;
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
- **D7.** Lens bias stays linear (v18). Checked 2026-11-19: linear is also the KP's fiducial (Sec. IV.1, V.1.1);
  the earlier note that the KP's fiducial is nonlinear was wrong.
- **D8.** Mean shear. KP subtracts it from the data *(paper)*; v18 does not subtract it anywhere.
  This affects only the observation, not the tfrecords.
- **D9 (decided 2026-11-19).** Mask coverage threshold, `analysis.footprint.coverage_threshold`:
  **0.73**, chosen as in Y3 so the footprint area matches the mask area. Y3's 0.72 gives 4178.3 deg²
  against a 4183.5 deg² mask (−0.1%). For Y6 the exact value is 0.735, and 0.73 gives 4055.2 against
  4031.0 deg² (+0.6%). Partial coverage is handled as in Y3: footprint pixels count as fully
  observed, and n_gal (galaxies in the footprint ÷ footprint area) absorbs the missing coverage on
  average. The per-pixel coverage pattern is not modeled. Mean coverage in the footprint: 0.92 for
  Y3, 0.88 for Y6.
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
