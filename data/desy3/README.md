# `data/desy3/`

The DES Y3 inputs of the v18 analysis. **Frozen**: they are the provenance of the published
results. Paths below are relative to this directory.

| | |
|---|---|
| `*` (flat) | the production inputs, exactly what v18 reads |
| `redshift_distributions/` | n(z) |
| `cache/` | catalog-derived maps, gitignored, rebuilt on demand by `msfm/utils/catalog.py` |
| `obs/` | Buzzard flock mock observations, gitignored |
| `deprecated/` | superseded inputs, kept so `configs/desy3/deprecated/` and v16/v17 still resolve |

## Production inputs

The `files:` and `dirs:` of `configs/desy3/v18/default.yaml`:

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

`configs/desy3/v18/obs/*.yaml` vary only the shape-noise model: `metacal_bias_arm: clean`
(`sc_no_sys`), or `sc_calibration: data/desy3/desy3_sc_calibration_gatti.npy` in place of the two
`metacal_bias*` keys (`sc_gatti`).

## Where every file comes from

Each product has one notebook in `notebooks/desy3/`. `products.ipynb` there makes nothing: it reviews
the products, as `notebooks/desy6/products.ipynb` does for Y6.

| file | size | made by | in git | read by |
|---|---|---|---|---|
| `desy3_pixels_fiducial_512.h5` | 63 M | `pixel_file.ipynb` | yes | `files.pixels` |
| `desy3_noise_512.h5` | 1.5 G | `noise_file.ipynb` | no, too large | `files.noise` |
| `desy3_metacal_bias.h5` | 8.5 M | `sc_bias_fit_count.ipynb` | yes | `files.metacal_bias` + `files.metacal_bias_arm` |
| `desy3_sc_calibration_gatti.npy` | 1 K | `sc_calibration_gatti.ipynb` | yes | `files.sc_calibration` |
| `cache/desy3_metacal_gamma_e1p_e2m.npy` | 193 M | `build_des.ipynb` → `catalog.build_metacal_map_from_cat` | no | `msi/utils/ppc.py` |
| `cache/desy3_metacal_count.npy` | 49 M | as above | no | `source_clustering_bias.py`, `msi/utils/ppc.py` |
| `cache/desy3_maglim_count.npy` | 97 M | `build_des.ipynb` → `catalog.build_maglim_map_from_cat` | no | `msi/utils/ppc.py` |
| `cache/desy3_metacal_gamma_no_psi_rot_e1p_e2m.npy` | 193 M | `build_des.ipynb` | no | diagnostic only |
| `obs/buzzard_flock*_obs_maps.h5` | 640 M | `build_buzzard_flock.ipynb` | no | copied into the deep_lss data dir, read by `run_evaluation --include_mocks` |

The `cache/` maps are rebuilt from the Y3 catalogs (`dirs.catalog`) when absent. That takes hours on
a node that has the catalogs, so don't delete them casually.

Not made here:

| file | origin |
|---|---|
| `redshift_distributions/*.txt` | the DES Y3 n(z). `filenames.py` builds the names from `survey.<sample>.z_bins`, so they can't be renamed freely. `maglim_bin5/6` are unused |
| `desy3_metacal_systematics_STD32_512.h5` | exported by the `lss_sys` repository. `STD32` is that run's label; `configuration.py` asserts the file's `label` attribute against the bias table's `systematics_label`, so the name, the attribute and the assert must agree |

Notebooks that fill in config values rather than files:

| notebook | fills in |
|---|---|
| `config.ipynb` | `survey.{metacal,maglim}.n_gal`, `survey.Aeff` |
| `patches.ipynb` | `analysis.footprint.rotation.{y_rad,z_rad}` |
| `scale_cuts.ipynb` | `analysis.scale_cuts` |

## `clean` and `contam`

The two arms of `desy3_metacal_bias.h5` (top-level HDF5 groups, selected by
`files.metacal_bias_arm`) are the forward model the bias was fit against: `contam` imprints the Y3
imaging systematics on the source density, `clean` does not. Mixing them up double counts the
systematics or drops them, and no error shows downstream, which is why `configuration.py` asserts on
it. Before 2026-09-01 the arms were two files, `desy3_metacal_bias_sys.h5` (`contam`) and
`desy3_metacal_bias_clean.h5` (`clean`).

## History

**Moved into `desy3/` on 2026-11-18** (branch `desy6`; `main` keeps the flat layout). Filenames did
not change.

| old | new |
|---|---|
| `data/desy3_*` | `data/desy3/desy3_*` |
| `data/redshift_distributions/`, `data/cache/`, `data/obs/` | `data/desy3/...` |
| `data/deprecated/*`, except `CosmoGridV1_original_metainfo.h5` | `data/desy3/deprecated/*` |
| `notebooks/*`, except `permutations_list.ipynb` | `notebooks/desy3/*` |
| `configs/{v16,v17,v18,deprecated}/` | `configs/desy3/...` |
| `pipelines/{v17,v18,deprecated}/` | `pipelines/desy3/...` |

**Renamed on 2026-08-07.** Old run logs and `dev/notes/wl_forward_model_audit_2026-07-11.md` use the
old names (paths relative to `data/` as it was then).

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

## Traps

- **Frozen v18 run dirs embed the old flat paths.** Each run's `configs.yaml` carries its msfm
  config (for example `pixels: data/desy3_pixels_fiducial_512.h5`), and
  `deep_lss.apps.run_evaluation` reads `files.pixels` from it. Evaluate a frozen run from a checkout
  of `main`.
- **`peaks/binning/*.h5` does not exist**, yet 21 live v16/v17 configs set `files.peak_binning`.
  Only `msfm/apps/deprecated/run_peaks.py` reads it. It is a dangling key, not a missing file.
- **`deprecated/` is not deleted.** v16 and v17 still read `deprecated/metacal_biases_desy3.h5` and
  `deprecated/nside256/`.
- **A notebook writes wherever it says.** If you rename a file here, update its notebook in the same
  change, or the next regeneration silently recreates the old name.
