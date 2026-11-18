# `data/`

Inputs to the forward model. Everything here is reached through an explicit `files:` or `dirs:`
key in a config, resolved as `os.path.join(<repo root>, value)` — nothing globs or walks this
directory. **The configs are therefore the single source of truth for the layout**, and a file
moved without updating them fails silently. The one hardcoded exception is the catalog cache
directory, `data/desy3/cache/`, in `msfm/utils/catalog.py`.

Run `dev/scripts/data_audit/check_config_paths.py --strict` after touching anything here. It
resolves every path in every config and exits non-zero on a regression.

## Layout

Each survey has its own subdirectory. The top level holds only what is not specific to a survey.

| | |
|---|---|
| `data/*` (flat) | survey-agnostic inputs: `CosmoGridV1_metainfo.h5`, `healpy_data/`. A DES Y6 config may use them only if it lists them explicitly |
| `data/desy3/` | DES Y3. Production inputs flat, plus `redshift_distributions/`, `cache/`, `obs/` and `deprecated/`. **Frozen** as the provenance of the v18 analysis |
| `data/desy6/` | DES Y6, being built (see "DES Y6" below). Same internal layout as `desy3/` |
| `data/deprecated/` | superseded survey-agnostic inputs (`CosmoGridV1_original_metainfo.h5`) |
| `data/figures/` | figures embedded in the repository `README.md` |

Within `desy3/`:

| | |
|---|---|
| `desy3/*` (flat) | production inputs: the set the v18 model reads |
| `desy3/cache/` | catalog-derived maps, gitignored, rebuilt on demand by `msfm/utils/catalog.py` |
| `desy3/obs/` | Buzzard flock mock observations, gitignored |
| `desy3/deprecated/` | superseded inputs, kept only so `configs/desy3/deprecated/` and v16/v17 still resolve. **Frozen**: they keep the old uppercase names |

## Production inputs

These are exactly the `files:` / `dirs:` values of `configs/desy3/v18/default.yaml`, the current Y3
production config:

```yaml
dirs:
  redshift_distributions: data/desy3/redshift_distributions
files:
  pixels:              data/desy3/desy3_pixels_fiducial_512.h5
  noise:               data/desy3/desy3_noise_512.h5
  meta_info:           data/CosmoGridV1_metainfo.h5
  healpy_data:         data/healpy_data
  metacal_systematics: data/desy3/desy3_metacal_systematics_STD32_512.h5
  metacal_bias:        data/desy3/desy3_metacal_bias.h5
  metacal_bias_arm:    contam
```

`configs/desy3/v18/obs/*.yaml` vary only the shape-noise model, which swaps `metacal_bias_arm` for
`clean` (`sc_no_sys`) or replaces the two `metacal_bias*` keys with
`sc_calibration: data/desy3/desy3_sc_calibration_gatti.npy` (`sc_gatti`).

## Where every file comes from

For DES Y3, `notebooks/` is **production**: it holds exactly the notebooks that generate the
contents of this directory, one slot per product, under the same survey subdirectory
(`notebooks/desy3/` for `data/desy3/`). DES Y6 uses a script instead (see "DES Y6" below).
`dev/notebooks/` is **debugging**: analysis and comparison notebooks that consume these products
rather than make them. That split is what decides where a notebook belongs.

### Produced in this repository

Paths are relative to `data/`.

| file | size | produced by | tracked | read by |
|---|---|---|---|---|
| `desy3/desy3_pixels_fiducial_512.h5` | 63 M | `notebooks/desy3/pixel_file.ipynb` | yes | `files.pixels` |
| `desy3/desy3_noise_512.h5` | 1.5 G | `notebooks/desy3/noise_file.ipynb` | **no** — too large for git, regenerate it | `files.noise` |
| `desy3/desy3_metacal_bias.h5` | 8.5 M | `notebooks/desy3/sc_bias_fit_count.ipynb` (`clean`/`contam` HDF5 groups) | yes | `files.metacal_bias` + `files.metacal_bias_arm` |
| `desy3/desy3_sc_calibration_gatti.npy` | 1 K | `notebooks/desy3/sc_calibration_gatti.ipynb` | yes | `files.sc_calibration` |
| `desy3/cache/desy3_metacal_gamma_e1p_e2m.npy` | 193 M | `notebooks/desy3/build_des.ipynb` → `catalog.build_metacal_map_from_cat` | no | `msi/utils/ppc.py` |
| `desy3/cache/desy3_metacal_count.npy` | 49 M | as above | no | `source_clustering_bias.py`, `msi/utils/ppc.py` |
| `desy3/cache/desy3_maglim_count.npy` | 97 M | `notebooks/desy3/build_des.ipynb` → `catalog.build_maglim_map_from_cat` | no | `msi/utils/ppc.py` |
| `desy3/cache/desy3_metacal_gamma_no_psi_rot_e1p_e2m.npy` | 193 M | `notebooks/desy3/build_des.ipynb` | no | diagnostic only, no production consumer |
| `desy3/obs/buzzard_flock*_obs_maps.h5` | 640 M | `notebooks/desy3/build_buzzard_flock.ipynb` | no | nothing here: copied into the deep_lss data dir, which `run_evaluation --include_mocks` reads |

The `cache/` maps are rebuilt from `dirs.catalog` (the DES Y3 catalogs on scratch) whenever they
are absent. That takes hours and needs a node that has the catalogs, so do not delete them
casually — but they are reproducible, which is why they are not in git.

### Not produced here

| file | origin |
|---|---|
| `CosmoGridV1_metainfo.h5` | CosmoGridV1, then **mutated in place** by `notebooks/permutations_list.ipynb` (opened `"r+"`) to rewrite the parameter paths |
| `healpy_data/` | healpy ancillary data (full-sky quadrature weights), see its own `README.md` |
| `desy3/redshift_distributions/*.txt` | DES Y3 n(z). Filenames are synthesized by `msfm/utils/filenames.py` from `survey.<sample>.z_bins`, so they cannot be renamed freely. `maglim_bin5/6` are unused — every config runs 4 maglim bins |
| `desy3/desy3_metacal_systematics_STD32_512.h5` | exported by the `lss_sys` repository. `STD32` is that export's run label; `msfm/utils/configuration.py` asserts the file's `label` attribute against the bias table's `systematics_label`, so **the name, the attribute and the assert must stay in agreement** |
| `figures/combined_moll+gnom.png` | plotted in `deep_lss_paper/paper_1/map_smoothing.ipynb`, embedded in the repository `README.md` |

### Notebooks that derive config *values* rather than files

| notebook | fills in |
|---|---|
| `notebooks/desy3/config.ipynb` | `survey.{metacal,maglim}.n_gal`, `survey.Aeff` |
| `notebooks/desy3/patches.ipynb` | `analysis.footprint.rotation.{y_rad,z_rad}` |
| `notebooks/desy3/scale_cuts.ipynb` | `analysis.scale_cuts` |

## DES Y6

`configs/desy6/v1/default.yaml` names these files. Each one not produced yet is listed in
`EXPECTED_MISSING` in the path gate, and comes off that list once its step has run.

**Raw inputs stay outside the repository.** `dirs.catalog` points at the public Y6 release as
downloaded to `/capstor/store/cscs/userlab/lp184/tomaszk/261109_cosmogridv1_desy6/data_desy6a2`
(from `desdr-server.ncsa.illinois.edu/despublic/y6a2_files/y6_psf/final_version/`). Only five of
its files are read:

| file | size | used for |
|---|---|---|
| `metadetect_2024-11-07.hdf5` | 140 G | source shear and count maps, response, `n_gal`, noise file |
| `maglim_2024-11-07.hdf5` | 2.7 G | lens count maps (per-galaxy `weight`), `n_gal` |
| `jointmask_DESY6_2024-06-04.hsp` | 696 M | footprint, pixel file, `Aeff` |
| `3x2pt_2025-07-28-20h_UNBLINDED_wmask_nob2.fits` | 22 M | fiducial n(z), HDUs `nz_source` (4 bins) and `nz_lens` (6 bins) |
| `desy6_source_redshift.h5` | 373 M | m prior: mean and std of the `M_BIN*` columns only |

The rest of the download is not needed:
- the non-tomographic metadetection file;
- the randoms, the w(θ) debias file and redMaGiC;
- `maglim_addl_cols_v1.hdf5`;
- the lens realization set `desy6_lens_redshift.h5`, which feeds the projection;
- the `desy6kp_cats` index.

DES's `3x2pt_pipeline_public.ipynb`, in the same directory, is the reference for the response, the
weights and the mean-shear subtraction.

**Derived products** come from one config-driven script, not from notebooks:

```bash
python -m msfm.apps.products.desy6 <step> --config configs/desy6/v1/default.yaml
pipelines/desy6/products/run.sh <pixels|maps|noise>    # the compute-node steps, as a job
```

Every step writes to the path its config key names, stops on a failed check, and records its
inputs, outputs and the msfm commit in `desy6/provenance.yaml` (`files.provenance`). The steps run
in this order:

| step | file | from | tracked |
|---|---|---|---|
| `nz` | `desy6/redshift_distributions/desy6_nz_metacal_bin{1-4}.txt`, `desy6_nz_maglim_bin{1-6}.txt` | the FITS | yes |
| `pixels` | `desy6/desy6_pixels_fiducial_512.h5` | the mask | yes |
| `maps` | `desy6/cache/desy6_metacal_{gamma_e1p_e2m,count,response}.npy`, `desy6_maglim_{count,weighted_count}.npy` (`dirs.cache`), every survey bin | metadetect and MagLim | no |
| `values` | `desy6/desy6_survey_values.yaml` | the maps, the pixel file, metadetect, the m columns | yes |
| `config` | the lines of `configs/desy6/v1/default.yaml` marked `# <- desy6 values: <key>` | the values file | yes |
| `noise` | `desy6/desy6_noise_512.h5` | metadetect | no |
| `sc_bias` | `desy6/desy6_metacal_bias.h5` | the Y6 projection | yes |
| (external) | `desy6/desy6_metacal_systematics_<label>_512.h5` | `lss_sys` on the Y6 survey-property maps (TODO) | yes |

All steps but `sc_bias` exist so far; the release reader behind them is
`msfm/utils/catalog_desy6.py`. The steps serve the purposes of the Y3 production notebooks: `pixels` is
`pixel_file` + `patches`, `maps` is `build_des`, `values` is `config`, and `noise` and `sc_bias`
keep their names. The spec is section 5 of `dev/notes/desy6_tfrecord_plan.md`.

All six lens bins are produced; the config excludes bin 2 through `survey.maglim.z_bins`. The token
in the n(z) filenames stays `metacal`/`maglim` because it echoes the config key, even though the Y6
sources are metadetection.

## Naming

`<survey>/<survey>_<product>[_<variant>][_<nside>].<ext>`, for example
`desy3/desy3_metacal_bias.h5`. This extends the convention already used by
`desy3/redshift_distributions/desy3_nz_metacal_bin1.txt`.

The survey prefix repeats the directory on purpose. A file copied out of the tree still says which
survey it belongs to. The n(z) template in `filenames.py` and the cache names in `catalog.py` are
built with that prefix too.

The product token echoes the **config key** that loads the file (`files.metacal_bias` →
`desy3_metacal_bias_*`), so a key and a filename are greppable from either direction. Where a file
is specific to one shape-noise model, the trailing token is the literal value of
`analysis.modelling.lensing.shape_noise.method` (`count`, `gatti`) — which is also how the two
producing notebooks are named, `sc_bias_fit_count.ipynb` and `sc_calibration_gatti.ipynb`.

Files under `desy3/deprecated/` keep their original names. The uppercase style is the signal that
they are frozen.

### Moved into `desy3/` on 2026-11-18

On branch `desy6`, every DES Y3 file and notebook moved one level down. Filenames did not change;
`main` still has the old flat layout.

| old | new |
|---|---|
| `data/desy3_*` | `data/desy3/desy3_*` |
| `data/redshift_distributions/` | `data/desy3/redshift_distributions/` |
| `data/cache/` | `data/desy3/cache/` |
| `data/obs/` | `data/desy3/obs/` |
| `data/deprecated/*`, except `CosmoGridV1_original_metainfo.h5` | `data/desy3/deprecated/*` |
| `notebooks/*`, except `permutations_list.ipynb` | `notebooks/desy3/*` |
| `configs/{v16,v17,v18,deprecated}/` | `configs/desy3/{v16,v17,v18,deprecated}/` |
| `pipelines/{v17,v18,deprecated}/` | `pipelines/desy3/{v17,v18,deprecated}/` |

Every config, `.gitignore`, `catalog.py` and the `dev/` scripts were updated in the same change.

### Renamed on 2026-08-07

Old run logs and `dev/notes/wl_forward_model_audit_2026-07-11.md` predate this and use the old
names. The `v11` and `v2` tags were opaque version markers; they are replaced by the thing that
actually distinguishes the files. Paths are relative to `data/` as they were on that date, before
the move into `desy3/`.

| old | new |
|---|---|
| `DESY3_pixels_v11_fiducial_512.h5` | `desy3_pixels_fiducial_512.h5` |
| `DESY3_noise_v11.h5` | `desy3_noise_512.h5` |
| `metacal_biases_desy3_v2_sys.h5` | `desy3_metacal_bias_sys.h5` |
| `metacal_biases_desy3_v2.h5` | `desy3_metacal_bias_clean.h5` |
| `metacal_sources_STD32_512.h5` | `desy3_metacal_systematics_STD32_512.h5` |
| `sc_calibration_desy3.npy` | `desy3_sc_calibration_gatti.npy` |
| `metacal_wl_gamma_map_e1p_e2m.npy` | `cache/desy3_metacal_gamma_e1p_e2m.npy` |
| `metacal_wl_count_map.npy` | `cache/desy3_metacal_count.npy` |
| `maglim_gc_count_map.npy` | `cache/desy3_maglim_count.npy` |
| `metacal_biases_desy3.h5` | `deprecated/metacal_biases_desy3.h5` |
| `CosmoGridV1_original_metainfo.h5`, `DESY3_pixels_512.h5`, `DESY3_pixels_v11_512.h5`, `DESY3_pixels_v11_octant_512.h5`, `metacal_biases_buzzard_0.{h5,pkl}`, `y3_gold_*_mask.fits.gz`, `nside256/`, `mock_observations/` | moved to `deprecated/`, names unchanged |

`clean` vs `contam` is the forward model the bias table was fit against: `contam` against the model
that imprints the DES Y3 imaging systematics on the source density, `clean` against the one that
does not. Since 2026-09-01 both arms live in `desy3_metacal_bias.h5` as top-level HDF5 groups of
that name (`files.metacal_bias_arm` selects one); before that they were two separate files,
`desy3_metacal_bias_sys.h5` (`contam`) and `desy3_metacal_bias_clean.h5` (`clean`). Mixing the arms
up double counts the systematics or drops them entirely, and shows up as an error nowhere
downstream — which is why `configuration.py` asserts on it.

## Traps

- **Frozen v18 run dirs embed the old flat paths.** Each run's `configs.yaml` carries the msfm
  config it was trained with, for example `pixels: data/desy3_pixels_fiducial_512.h5`.
  `deep_lss.apps.run_evaluation` reads `files.pixels` from it, so evaluating a frozen run against
  this layout fails. Use a checkout of `main` for that.
- **Switching branches does not move ignored files.** `git checkout main` brings back the tracked
  Y3 files at the top level, but not `desy3/desy3_noise_512.h5`, `desy3/cache/` or `desy3/obs/`.
- **`data/desy3/peaks/binning/*.h5` does not exist**, yet 21 live v16/v17 configs still set
  `files.peak_binning`. Only `msfm/apps/deprecated/run_peaks.py` reads that key and v18 dropped it.
  Do not "fix" the gate by creating `data/desy3/peaks/`; it is a dangling key, not a missing file.
- **`deprecated/` is not the same as deleted.** The v16 and v17 configs are still live and still
  read `desy3/deprecated/metacal_biases_desy3.h5` and `desy3/deprecated/nside256/`.
- **Regenerating a product writes wherever its notebook says.** If you rename a file here, update
  the producing notebook in the same change, or the next regeneration silently recreates the old
  name while every config keeps resolving to a now-stale file.
- **`dev/scripts/noise_models/generate_noise.py` writes to a CWD-relative `data/`.** Run it from
  the repo root and it drops a `.npz` here that `!data/**/*.npz` will happily stage.
- **Deleting files here does not shrink a clone.** They are in published history and `.git` is
  ~850 MB. To avoid the download, use `git clone --filter=blob:none`.
