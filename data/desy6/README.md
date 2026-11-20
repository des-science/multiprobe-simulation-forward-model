# `data/desy6/`

The inputs to the DES Y6 forward model that come straight from the survey data: n(z), the footprint,
the catalog maps, the lens weight maps, the survey numbers in the config, and the shape-noise catalog. One script makes
all of them from the public Y6 release, `msfm/apps/products/desy6.py`, for
`configs/desy6/v1/default.yaml`.

## Status

| step | output (in this directory) | in git | status |
|---|---|---|---|
| `nz` | `redshift_distributions/` | yes | done 2026-11-18 |
| `pixels` | `desy6_pixels_fiducial_512.h5` | yes | done 2026-11-20, threshold 0.73, with the lens weight maps |
| `maps` | `cache/` | no | done 2026-11-19 |
| `values` | `desy6_survey_values.yaml` | yes | done 2026-11-19 |
| `config` | the marked lines of the config | yes | done 2026-11-19 |
| `noise` | `desy6_noise_512.h5` | no | done 2026-11-19 |
| `sc_bias` | `desy6_metacal_bias.h5` | yes | not written, blocked (see below) |
| external | `desy6_metacal_systematics_<label>_512.h5` | yes | TODO: an `lss_sys` run |

Paths not produced yet are in `EXPECTED_MISSING` of `dev/scripts/data_audit/check_config_paths.py`.
Remove each once its step has run.

## Inputs

The public release (`desdr-server.ncsa.illinois.edu/despublic/y6a2_files/y6_psf/final_version/`),
read in place from `dirs.catalog`:
`/capstor/store/cscs/userlab/lp184/tomaszk/261109_cosmogridv1_desy6/data_desy6a2`. Five of its 22
files are used:

| file | size | for |
|---|---|---|
| `metadetect_2024-11-07.hdf5` | 140 G | source shear, counts, response, noise catalog |
| `maglim_2024-11-07.hdf5` | 2.7 G | lens counts, and the weight maps from the per-galaxy `weight` |
| `jointmask_DESY6_2024-06-04.hsp` | 696 M | footprint |
| `3x2pt_2025-07-28-20h_UNBLINDED_wmask_nob2.fits` | 22 M | fiducial n(z) |
| `desy6_source_redshift.h5` | 373 M | m prior (the `M_BIN*` columns only) |

Not needed: the non-tomographic metadetection file, the randoms, the w(θ) debias file, redMaGiC,
`maglim_addl_cols_v1.hdf5`, the lens realizations (they feed the projection) and the `desy6kp_cats`
index. DES's `3x2pt_pipeline_public.ipynb`, in the same directory, is the reference for the
response, the weights and the mean shear.

## Running

From the repository root, in `torch_env` (`pytorch/v2.9.1:v2` uenv, plus healsparse for `pixels`):

```bash
# login node: nz, values, config
uenv run --view=default pytorch/v2.9.1:v2 -- ~/dlss/torch_env/bin/python \
    -m msfm.apps.products.desy6 <step> --config configs/desy6/v1/default.yaml

# one CPU node: pixels, maps, noise (dry run on the login node first, then sbatch)
pipelines/desy6/products/run.sh <step>    # PARTITION, TIME, LOGDIR; DRYRUN=1 prints the sbatch line
```

- `--dry-run` lists what a step reads and writes, and writes nothing. For `config` it is `--check`.
- `--max-objects N` (`maps`, `noise`) reads the first N rows of each catalog and writes nothing. A
  smoke test.
- `--provenance` is for production runs. The step then refuses uncommitted `msfm/` or `configs/`,
  and records its inputs, outputs (sha256) and the msfm commit in `provenance.yaml`.

Every step stops on a failed check. The order is `nz` → `pixels` → `maps` → `values` → `config` →
`noise` → `sc_bias`.

All survey bins are produced: 4 source and 6 lens bins. The config picks lens bins 1 and 3–6
through `survey.maglim.z_bins`, since the KP drops bin 2.

## Steps

### `nz`: redshift distributions

- **Writes** `redshift_distributions/desy6_nz_metacal_bin{1-4}.txt` and `desy6_nz_maglim_bin{1-6}.txt`:
  z and n(z) on 300 points, in the Y3 format.
- **From** the fiducial n(z) in the 3x2pt FITS (HDUs `nz_source`, `nz_lens`). In Y3 the n(z) came
  from outside the repository.
- **Result:** mean z 0.41, 0.54, 0.85, 1.16 for the sources, 0.31 to 1.01 for the lenses.
- The grid starts at z = 0 (Y3: 0.005). The token stays `metacal` because it echoes the config key,
  although the Y6 sources are metadetection.

### `pixels`: footprint, patches and lens weight maps

**Writes** `desy6_pixels_fiducial_512.h5`, in the layout `files.load_pixel_file` reads. As in Y3's
`pixel_file.ipynb`:

1. The mask comes at nside 131072 (1.6″ pixels). It is degraded to nside 4096, keeping the covered
   fraction of each pixel. The Y3 mask came at nside 4096.
2. It is rotated into the footprint frame at nside 4096 (`hp.Rotator.rotate_map_pixel`, the same
   rotation as the catalog maps), then `ud_grade`d to nside 512.
3. A pixel enters the footprint if the mask covers at least `coverage_threshold` of it.
4. Four non-overlapping patches: the footprint, rotated by 180°, mirrored, and mirrored and rotated.
   γ₂ flips sign in the mirrored ones.
5. The padded data vector: every nside-512 pixel of each nside-16 pixel the footprint touches.

**Threshold 0.73**, chosen as in Y3 so that the footprint area matches the mask area. Y3's 0.72
gives 4178 deg² against a 4184 deg² mask. Y6's 0.73 gives 4055 against 4031 deg² (+0.6%). The step
logs the exact area-preserving value (0.735).

**Partial coverage** is handled as in Y3. Footprint pixels count as fully observed, and n_gal
(step `values`) absorbs the missing coverage on average. The coverage pattern inside the footprint
is not modeled. The mean coverage in the footprint is 0.88 (Y3: 0.92), and the coverage map is
stored as `footprint/coverage`.

**Checks:** the mask area equals DES's 4031.036 deg², the rotated map keeps it within 1%, the patches
don't overlap, and the file reads back. The area is only a check: `Aeff` feeds no forward-model code.

**Result:** 309,226 footprint pixels and a 457,728-pixel data vector (Y3: 318,616 and 458,752).
About a minute on one node.

**MagLim weight maps** go into `maglim/systematics/{map,data_vector}/maglim1-6`, the group
`files.get_clustering_systematics` reads, with `filled/` alongside. DES's fiducial lens weights are
ENet (arXiv:2601.14484), an elastic-net regression of each bin's density on the 19 survey-property
maps, fit at nside 256 and predicted at nside 1024. The release has no weight map, only the
per-galaxy `weight`, which is constant inside each nside-1024 pixel. So the map is read off the
catalog, without a refit (`catalog_desy6.maglim_weight_maps`):

1. Every nside-1024 pixel holding a galaxy gives the weight there, 54-67% of them per bin.
2. Each footprint pixel is split into its 64 nside-4096 children, whose centres are rotated back
   to the sky and looked up in that map. The observed count of a pixel is the clean count times
   the mean contamination F = 1/w, so its weight is w = 1/⟨F⟩, averaged over the children with a
   value.
3. Pixels without a galaxy in any child, 0.14-0.61% per bin, take the mean ⟨F⟩ of their
   neighbours. These may lie up to two rings outside the footprint, since it has isolated pixels.

As in v18, w multiplies the observed counts (`apply_maglim_sys_map` of
`observation.forward_model_observation_map`) and stays out of the forward model
(`maglim_survey_systematics_map: False`). w spans 0.875 to 1.206, with a std of 1.3% to 3.6% per
bin.

**Checks on the weights:** one weight per nside-1024 pixel; the mean galaxy weight of each pixel
follows w (correlation 0.993 to 0.996, where a wrong frame gives 0); and the counts times w sum to
the weighted counts within 3×10⁻⁵, so the catalog's normalization and the weighted n_gal of step
`values` carry over.

### `maps`: catalog maps

- **Writes** to `cache/` (gitignored), full sky at nside 512 in the footprint frame: the shear
  (`desy6_metacal_gamma_e1p_e2m.npy`, sign convention as Y3), source counts, the response R per bin,
  and MagLim counts and weighted counts.
- **From** metadetection (the unsheared catalog, and the four sheared ones for R) and MagLim. No
  cuts: the release is already the fiducial selection. Y3: `build_des.ipynb`.
- **Checks:** per source bin N, R, n_eff and σ_e, and per lens bin N and n_eff, equal what DES's
  `3x2pt_pipeline_public.ipynb` prints for the same catalogs, to its last digit. The MagLim weights
  average 1. This tests how the catalogs are read, not the pixel positions or the shear signs (see
  Review). The mean shear is logged, not subtracted, as in Y3.
- **Result:** R = 0.856143, 0.868506, 0.819427, 0.691769, σ_e = 0.27, 0.29, 0.28, 0.35, and a mean e
  of at most 2e-4 per component. About a minute on one node.

### `values`: numbers for the config

- **Writes** `desy6_survey_values.yaml`, keyed by bin label.
- **n_gal** = galaxies in the footprint / footprint area, as in Y3's `config.ipynb`. Raw counts for
  the sources, weighted counts for MagLim.
- **R** from `maps`; **m prior** (mean and std) from the `M_BIN*` columns; **Aeff** from the pixel
  file.
- **Checks:** the footprint holds the same fraction of the galaxies as of the mask area, within
  0.02; the pixel file matches the config's rotation and threshold; and the m means match DES's
  within 1e-5. About 11% of the mask lies in pixels below the threshold, so about 10% of the
  galaxies are not in the maps (Y3: 8%).

### `config`: values into the config

Rewrites only the config lines marked `# <- desy6 values: <key>`, then reparses the config and
checks every marked key against the values file. `config --check` changes nothing and exits 1 if
a value is stale. In Y3 this was done by hand.

### `noise`: shape-noise catalog

- **Writes** `desy6_noise_512.h5` (gitignored, ~2 GB): per source bin, (e1/R, e2/R, w, pixel) for
  every galaxy, with the signs and rotation of `maps`. Y3: `noise_file.ipynb`.
- **Check:** each bin's pixels reproduce the `maps` count map exactly.
- **Result:** 139.7M galaxies, 2.2 GB. About a minute on one node.

### `sc_bias`: source-clustering bias (not written yet)

Will write `desy6_metacal_bias.h5` (`clean` and `contam` arms), as Y3's `sc_bias_fit_count.ipynb`.
It is blocked on two inputs:

- **The Y6 imaging-systematics file**, from an `lss_sys` run on the Y6 survey-property maps. v1 uses
  `count` shape noise with the systematics imprinted, like v18, so the bias must be fit against that
  model, and `configuration.py` checks that the labels agree.
- **The Y6 projection.** The fit uses `dg/<source>` of all 2500 grid cosmologies at `perm_0000`;
  only test files exist so far.

## Review

`notebooks/desy6/products.ipynb` is read-only. It plots every product, runs the κ_E × δ_g sign
test and the κ_B × δ_g null test, and checks the lens weight maps against the counts. `notebooks/desy3/products.ipynb` runs the same on the Y3 products,
for comparison.

The tests confirm Y3's e1/e2 signs and ψ rotation for Y6 (2026-11-19). κ_E × δ_g is positive for
all 20 source-lens pairs, with S/N 10 to 51 where the lens bin lies in front. κ_B × δ_g is null,
with a combined S/N of at most 2.1 in size.

The lens weight maps pass too (2026-11-20). The raw counts follow F = 1/w with slope 0.91 to 1.37
(jackknife errors 0.08 to 0.33), and the counts corrected by w are flat within 1.3σ in every bin.

## Open

- The n(z) grid starting at z = 0 needs a check once the n(z) loader reads Y6 bin labels. n(z) is
  0 there in every bin.
