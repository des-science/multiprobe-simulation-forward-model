# `data/`

Inputs to the forward model. Everything here is reached through an explicit `files:` or `dirs:`
key in a config, resolved as `os.path.join(<repo root>, value)` — nothing globs or walks this
directory. **The configs are therefore the single source of truth for the layout**, and a file
moved without updating them fails silently.

Run `dev/scripts/data_audit/check_config_paths.py --strict` after touching anything here. It
resolves every path in every config and exits non-zero on a regression.

## Layout

| | |
|---|---|
| `data/*` (flat) | production inputs — the top level **is** the set the current model reads |
| `data/cache/` | catalog-derived maps, gitignored, rebuilt on demand by `msfm/utils/catalog.py` |
| `data/deprecated/` | superseded inputs, kept only so `configs/deprecated/` and v16/v17 still resolve. **Frozen** — retain the old uppercase names |
| `data/figures/` | figures embedded in the repository `README.md` |

## Production inputs

These are exactly the `files:` / `dirs:` values of `configs/v18/default.yaml`, the current
production config:

```yaml
dirs:
  redshift_distributions: data/redshift_distributions
files:
  pixels:              data/desy3_pixels_fiducial_512.h5
  noise:               data/desy3_noise_512.h5
  meta_info:           data/CosmoGridV1_metainfo.h5
  healpy_data:         data/healpy_data
  metacal_systematics: data/desy3_metacal_systematics_STD32_512.h5
  metacal_bias:        data/desy3_metacal_bias_sys.h5
```

`configs/v18/obs/*.yaml` vary only the shape-noise model, which swaps `metacal_bias` for
`desy3_metacal_bias_clean.h5` (`sc_no_sys`) or replaces it with
`sc_calibration: data/desy3_sc_calibration_gatti.npy` (`sc_gatti`).

## Where every file comes from

`notebooks/` is **production**: it holds exactly the notebooks that generate the contents of this
directory, one slot per product. `dev/notebooks/` is **debugging**: analysis and comparison
notebooks that consume these products rather than make them. That split is what decides where a
notebook belongs.

### Produced in this repository

| file | size | produced by | tracked | read by |
|---|---|---|---|---|
| `desy3_pixels_fiducial_512.h5` | 63 M | `notebooks/pixel_file.ipynb` | yes | `files.pixels` |
| `desy3_noise_512.h5` | 1.5 G | `notebooks/noise_file.ipynb` | **no** — too large for git, regenerate it | `files.noise` |
| `desy3_metacal_bias_sys.h5` | 4.3 M | `notebooks/sc_bias_fit_count.ipynb` (`contam` arm) | yes | `files.metacal_bias` |
| `desy3_metacal_bias_clean.h5` | 4.3 M | `notebooks/sc_bias_fit_count.ipynb` (`clean` arm) | yes | `files.metacal_bias` |
| `desy3_sc_calibration_gatti.npy` | 1 K | `notebooks/sc_calibration_gatti.ipynb` | yes | `files.sc_calibration` |
| `cache/desy3_metacal_gamma_e1p_e2m.npy` | 193 M | `notebooks/build_des.ipynb` → `catalog.build_metacal_map_from_cat` | no | `msi/utils/ppc.py` |
| `cache/desy3_metacal_count.npy` | 49 M | as above | no | `source_clustering_bias.py`, `msi/utils/ppc.py` |
| `cache/desy3_maglim_count.npy` | 97 M | `notebooks/build_des.ipynb` → `catalog.build_maglim_map_from_cat` | no | `msi/utils/ppc.py` |
| `cache/desy3_metacal_gamma_no_psi_rot_e1p_e2m.npy` | 193 M | `notebooks/build_des.ipynb` | no | diagnostic only, no production consumer |

The `cache/` maps are rebuilt from `dirs.catalog` (the DES Y3 catalogs on scratch) whenever they
are absent. That takes hours and needs a node that has the catalogs, so do not delete them
casually — but they are reproducible, which is why they are not in git.

### Not produced here

| file | origin |
|---|---|
| `CosmoGridV1_metainfo.h5` | CosmoGridV1, then **mutated in place** by `notebooks/permutations_list.ipynb` (opened `"r+"`) to rewrite the parameter paths |
| `healpy_data/` | healpy ancillary data (full-sky quadrature weights), see its own `README.md` |
| `redshift_distributions/*.txt` | DES Y3 n(z). Filenames are synthesized by `msfm/utils/filenames.py` from `survey.<sample>.z_bins`, so they cannot be renamed freely. `maglim_bin5/6` are unused — every config runs 4 maglim bins |
| `desy3_metacal_systematics_STD32_512.h5` | exported by the `lss_sys` repository. `STD32` is that export's run label; `msfm/utils/configuration.py` asserts the file's `label` attribute against the bias table's `systematics_label`, so **the name, the attribute and the assert must stay in agreement** |
| `figures/combined_moll+gnom.png` | plotted in `deep_lss_paper/paper_1/map_smoothing.ipynb`, embedded in the repository `README.md` |

### Notebooks that derive config *values* rather than files

| notebook | fills in |
|---|---|
| `notebooks/config.ipynb` | `survey.{metacal,maglim}.n_gal`, `survey.Aeff` |
| `notebooks/patches.ipynb` | `analysis.footprint.rotation.{y_rad,z_rad}` |
| `notebooks/scale_cuts.ipynb` | `analysis.scale_cuts` |

## Naming

`desy3_<product>[_<variant>][_<nside>].<ext>`, extending the convention already used by
`redshift_distributions/desy3_nz_metacal_bin1.txt`.

The leading product token echoes the **config key** that loads the file (`files.metacal_bias` →
`desy3_metacal_bias_*`), so a key and a filename are greppable from either direction. Where a file
is specific to one shape-noise model, the trailing token is the literal value of
`analysis.modelling.lensing.shape_noise.method` (`count`, `gatti`) — which is also how the two
producing notebooks are named, `sc_bias_fit_count.ipynb` and `sc_calibration_gatti.ipynb`.

Files under `deprecated/` keep their original names. The uppercase style is the signal that they
are frozen.

### Renamed on 2026-08-07

Old run logs and `dev/notes/wl_forward_model_audit_2026-07-11.md` predate this and use the old
names. The `v11` and `v2` tags were opaque version markers; they are replaced by the thing that
actually distinguishes the files.

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

`clean` vs `sys` is the forward model the bias table was fit against: `sys` against the model that
imprints the DES Y3 imaging systematics on the source density, `clean` against the one that does
not. Mixing them up double counts the systematics or drops them entirely, and shows up as an error
nowhere downstream — which is why `configuration.py` asserts on it.

## Traps

- **`data/peaks/binning/*.h5` does not exist**, yet 21 live v16/v17 configs still set
  `files.peak_binning`. Only `msfm/apps/deprecated/run_peaks.py` reads that key and v18 dropped it.
  Do not "fix" the gate by creating `data/peaks/` — it is a dangling key, not a missing file.
- **`deprecated/` is not the same as deleted.** The v16 and v17 configs are still live and still
  read `deprecated/metacal_biases_desy3.h5` and `deprecated/nside256/`.
- **Regenerating a product writes wherever its notebook says.** If you rename a file here, update
  the producing notebook in the same change, or the next regeneration silently recreates the old
  name while every config keeps resolving to a now-stale file.
- **`dev/scripts/noise_models/generate_noise.py` writes to a CWD-relative `data/`.** Run it from
  the repo root and it drops a `.npz` here that `!data/**/*.npz` will happily stage.
- **Deleting files here does not shrink a clone.** They are in published history and `.git` is
  ~850 MB. To avoid the download, use `git clone --filter=blob:none`.
