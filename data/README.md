# `data/`

Inputs to the forward model, one subdirectory per survey.

| | |
|---|---|
| [`desy3/`](desy3/README.md) | DES Y3, frozen as the inputs of the v18 analysis |
| [`desy6/`](desy6/README.md) | DES Y6, being built by `msfm.apps.products.desy6` |
| `CosmoGridV1_metainfo.h5` | CosmoGridV1 metadata, mutated in place by `notebooks/permutations_list.ipynb` (opened `"r+"`) to rewrite the parameter paths |
| `healpy_data/` | healpy ancillary data (full-sky quadrature weights), see its own `README.md` |
| `deprecated/` | `CosmoGridV1_original_metainfo.h5` |
| `figures/` | figures in the repository `README.md`, plotted in `deep_lss_paper/paper_1/map_smoothing.ipynb` |

A survey config uses the top-level files only if it names them.

## The configs are the source of truth

Every file is reached through a `files:` or `dirs:` key in a config, resolved against the repository
root. Nothing globs or walks this directory, so a file moved without updating the configs fails
silently. The one hardcoded path is the Y3 catalog cache, `data/desy3/cache/`, in
`msfm/utils/catalog.py`.

After touching anything here, run

```bash
python dev/scripts/data_audit/check_config_paths.py --strict
```

It resolves every path in every config and exits non-zero on a regression.

## Naming

`<survey>/<survey>_<product>[_<variant>][_<nside>].<ext>`, for example `desy3/desy3_metacal_bias.h5`.

- The survey prefix repeats the directory on purpose, so a copied file still says which survey it
  belongs to. The n(z) template in `filenames.py` and the cache names use it too.
- The product token echoes the config key that loads the file (`files.metacal_bias` →
  `desy3_metacal_bias.h5`), so either can be grepped from the other.
- A file specific to one shape-noise model ends in the value of
  `analysis.modelling.lensing.shape_noise.method` (`count`, `gatti`).
- Files under `desy3/deprecated/` keep their old uppercase names, which marks them as frozen.

## Notebooks and scripts

The Y3 products were made by the notebooks in `notebooks/desy3/`, one per product. The Y6 products
are made by a script, `msfm/apps/products/desy6.py`. Notebooks that read the products rather than
make them live in `dev/notebooks/`.

## Traps

- **Switching branches does not move ignored files.** `main` has the old flat Y3 layout.
  `git checkout main` brings back the tracked files, but not the noise files, `cache/` or `obs/`.
- **`dev/scripts/noise_models/generate_noise.py` writes to a CWD-relative `data/`.** Run from the
  repository root, it drops a `.npz` here that `!data/**/*.npz` will stage.
- **Deleting files here does not shrink a clone.** They are in the history and `.git` is ~850 MB.
  Use `git clone --filter=blob:none` to skip the download.
