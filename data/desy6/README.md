# `data/desy6/`

The inputs to the DES Y6 forward model that come straight from the survey data: n(z), the footprint,
the catalog maps, the lens weight maps, the survey numbers in the config, and the shape-noise catalog. One script makes
all of them from the public Y6 release, `msfm/apps/products/desy6.py`, for
`configs/desy6/v1/v1.yaml`.

## Status

| step | output (in this directory) | in git | status |
|---|---|---|---|
| `nz` | `redshift_distributions/` | yes | done 2026-11-18 |
| `pixels` | `desy6_pixels_fiducial_512.h5` | yes | done 2026-11-20, threshold 0.73, with the lens weight maps |
| `maps` | `cache/` | no | done 2026-11-20, with the unrotated shear |
| `values` | the marked lines of the config | yes | done 2026-11-20, lens n_gal from the weight maps |
| `noise` | `desy6_noise_512.h5` | no | done 2026-11-19 |
| `systematics` | `desy6_metacal_systematics_MAIN19_512.h5` | yes | done 2026-11-20, `lss_sys` run `metadetect_sources_MAIN19` |
| `sc_bias` | `desy6_metacal_bias.h5` | yes | done 2026-11-22, the 2500 grid cosmologies and `fiducial` from `fiducial_bench` |

Every step reran with `--provenance` on 2026-11-20/21 and reproduced its outputs byte for byte. Its
record (inputs, output sha256, msfm commit) is in `provenance.yaml`.

A path that a config names before its step has run goes into `EXPECTED_MISSING` of
`dev/scripts/data_audit/check_config_paths.py` until the step writes it. No DES Y6 path is there now.

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
# login node: nz, values
uenv run --view=default pytorch/v2.9.1:v2 -- ~/dlss/torch_env/bin/python \
    -m msfm.apps.products.desy6 <step> --config configs/desy6/v1/v1.yaml

# one CPU node: pixels, maps, noise (dry run on the login node first, then sbatch)
pipelines/daint/desy6/products/run.sh <step>    # PARTITION, TIME, LOGDIR; DRYRUN=1 prints the sbatch line
```

- `--dry-run` lists what a step reads and writes, and writes nothing.
- `values --check` writes nothing and exits 1 if a value in the config is stale.
- `--max-objects N` (`maps`, `noise`) reads the first N rows of each catalog and writes nothing. A
  smoke test.
- `--max-cosmologies N` (`sc_bias`) fits the first N grid cosmologies and the fiducial one, runs
  every check and writes nothing. The same kind of smoke test.
- `--provenance` is for production runs. The step then refuses uncommitted `msfm/` or `configs/`,
  and records its inputs, outputs (sha256) and the msfm commit in `provenance.yaml`.

Every step stops on a failed check. The order is `nz` → `pixels` → `maps` → `values` → `noise` →
`systematics` → `sc_bias`.

All survey bins are produced: 4 source and 6 lens bins. The config picks lens bins 1 and 3–6
through `survey.maglim.z_bins`, since the KP drops bin 2.

## Steps

### `nz`: redshift distributions

- **Writes** `redshift_distributions/desy6_nz_metacal_bin{1-4}.txt` and `desy6_nz_maglim_bin{1-6}.txt`:
  z and n(z) on 300 points, in the Y3 format.
- **From** the fiducial n(z) in the 3x2pt FITS (HDUs `nz_source`, `nz_lens`). In Y3 the n(z) came
  from outside the repository.
- **Result:** mean z 0.41, 0.54, 0.85, 1.16 for the sources, 0.31 to 1.01 for the lenses.
- The grid starts at z = 0 (Y3: 0.005). In msfm the n(z) only weight the per-bin IA amplitudes and
  bias fiducials, and n(z) is 0 at z = 0 in every bin, so the first point adds nothing. The token
  stays `metacal` because it echoes the config key, although the Y6 sources are metadetection.

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
the per-galaxy weights within 3×10⁻⁵, so the catalog's normalization carries over. Per pixel, the
counts times w and the sums of the per-galaxy weights differ by 0.13% to 0.38% of the mean count.

**Against the paper:** the rms of the weights reproduces σ_weight of arXiv:2601.14484 (Sec. III.2),
which publishes no weight map. Per galaxy it agrees to the printed digit in every bin:

| bin | paper | per galaxy | per nside-1024 pixel | nside-512 map |
|---|---|---|---|---|
| 1 | 2.5% | 2.46% | 2.48% | 2.31% |
| 2 | 1.9% | 1.87% | 1.89% | 1.76% |
| 3 | 1.7% | 1.66% | 1.69% | 1.52% |
| 4 | 3.0% | 2.97% | 3.01% | 2.81% |
| 5 | 1.4% | 1.37% | 1.39% | 1.28% |
| 6 | 3.9% | 3.91% | 3.98% | 3.62% |

The nside-512 map is 7-9% lower in every bin, as expected from averaging the four nside-1024
pixels it contains.

### `maps`: catalog maps

- **Writes** to `cache/` (gitignored), full sky at nside 512 in the footprint frame: the shear
  (`desy6_metacal_gamma_e1p_e2m.npy`, sign convention as Y3), the same without the spin-2 rotation
  (`desy6_metacal_gamma_no_psi_rot_e1p_e2m.npy`), source counts, the response R per bin, and MagLim
  counts. The lens counts are raw and float64, as in Y3: the weight maps of `pixels` correct them.
  The shear maps are already divided by R.
- **Read by** the builders of `msfm.utils.catalog`, which load `<dirs.cache>/<survey>_<name>` for
  either survey and pick the configured bins by label (lens survey bins 1, 3, 4, 5, 6). They build
  only the Y3 maps from the catalog, so for Y6 this step has to run first.
- **From** metadetection (the unsheared catalog, and the four sheared ones for R) and MagLim. No
  cuts: the release is already the fiducial selection. Y3: `build_des.ipynb`.
- **Checks:** per source bin N, R, n_eff and σ_e, and per lens bin N and n_eff, equal what DES's
  `3x2pt_pipeline_public.ipynb` prints for the same catalogs, to its last digit. n_eff takes DES's
  mask area, so this step does not depend on `Aeff`, which `values` writes later. The MagLim weights
  average 1. This tests how the catalogs are read, not the pixel positions or the shear signs (see
  Review). The mean shear is logged, not subtracted, as in Y3.
- **Result:** R = 0.856143, 0.868506, 0.819427, 0.691769, σ_e = 0.27, 0.29, 0.28, 0.35, and a mean e
  of at most 2e-4 per component. About 1.5 minutes on one node, with two passes over the sources.

### `values`: numbers for the config

- **Writes** the config lines marked `# <- step values: <key>`, and nothing else, for the bins of
  `z_bins`. The rewritten config must parse to the old one with only those values replaced, and
  every marker needs a value. In Y3 these numbers were copied by hand.
- **n_gal** = galaxies in the footprint / footprint area, as in Y3's `config.ipynb`. Raw counts for
  the sources. For MagLim, the raw counts times the weight maps, which is the observation that the
  clean simulations are compared to, so the two match in expectation. Y3 used the raw lens counts,
  0.01% to 0.16% more than the corrected ones; in Y6 the difference is 0.08% to 0.28%.
- **m prior** (mean and std) from the `M_BIN*` columns; **Aeff** from the pixel file. R is not a
  config value: the shear maps of `maps` are already divided by it, and `noise` reads it from the
  cache.
- **Lens bias fiducials** `analysis.fiducial.bg*`: v18's power law, bg = 1.5 and n_bg = 0.5 with
  pivot z0 = 0.62, averaged over each lens bin's n(z) from `nz`. The result is 1.35, 1.50, 1.57,
  1.63, 1.67 for bins 1 and 3-6. The pivot is a constant of the step, since only the sources have a
  `z0` key. Each value is written as the shortest decimal of the float32 amplitude msfm computes.
  The fiducial set, the mocks and the signal C_l of y3-deep-lss's scale-cut step read these values.
  A config that sets all of them by hand, unmarked, has none derived (`configs/desy6/v2` takes the
  key project's).
- **Checks:** the footprint holds the same fraction of the galaxies as of the mask area, within
  0.02; the pixel file matches the config's rotation and threshold; and the m means match DES's
  within 1e-5. About 11% of the mask lies in pixels below the threshold, so about 10% of the
  galaxies are not in the maps (Y3: 8%).

### `noise`: shape-noise catalog

- **Writes** `desy6_noise_512.h5` (gitignored, ~2 GB): per source bin, (e1/R, e2/R, w, pixel) for
  every galaxy, with the signs and rotation of `maps`. Y3: `noise_file.ipynb`.
- **Check:** each bin's pixels reproduce the `maps` count map exactly.
- **Result:** 139.7M galaxies, 2.2 GB. About a minute on one node.

### `systematics`: source imaging systematics

- **Writes** `desy6_metacal_systematics_MAIN19_512.h5`, a byte copy of the nside-512 export of an
  `lss_sys` run, passed as `--source`. `lss_sys` keeps its output out of git, so the copy and its
  provenance record are what tie the forward model to the run. Y3's file was copied by hand.
- **The run:** `metadetect_sources_MAIN19` (`lss_sys` commit `63b7b37`), Y3's method applied
  unchanged to the metadetection sources and the 19 DES Y6 survey-property templates, on this
  directory's mask. Source:
  `/users/athomsen/repos/lss_sys/output/desy6/metadetect_sources_MAIN19/metadetect_sources_MAIN19_512.h5`.
  Per bin, `contamination` = ⟨1/W⟩ imprints the survey on a clean model and `weight` = ⟨W⟩
  corrects observed counts. v1 uses only `contamination` (`shape_noise.survey_systematics`).
- **Checks:** the layout `files.get_metacal_systematics` reads (RING, nside 512, 4 bins, zero off
  the footprint, finite); its footprint is the mask of `pixels`; the file name carries the run
  label, which `configuration.py` matches against the bias table; the loader reads it, which
  asserts that the run covers every footprint pixel; and the observed source counts of `maps` rise
  with C in every bin. A failed check writes nothing.
- **Result:** all 309,226 footprint pixels covered. Over the footprint:

  | source bin | 1 | 2 | 3 | 4 |
  |---|---|---|---|---|
  | std of C, Y6 | 0.070 | 0.079 | 0.096 | 0.153 |
  | std of C, Y3 | 0.056 | 0.084 | 0.097 | 0.224 |
  | corr(counts, C), Y6 | 0.34 | 0.37 | 0.47 | 0.66 |
  | slope of counts on C, raw, Y6 | 1.12 | 1.19 | 1.18 | 1.04 |
  | slope, corrected by W, Y6 | 0.13 | 0.20 | 0.19 | 0.05 |
  | slope, raw, Y3 | 1.18 | 1.04 | 1.07 | 1.03 |
  | slope, corrected by W, Y3 | 0.19 | 0.06 | 0.08 | 0.03 |

  The counts are normalized to unit mean. W removes most of the trend but not all of it, in Y3 as in Y6.
  Seconds on a login node.

### `sc_bias`: source-clustering bias

- **Writes** `desy6_metacal_bias.h5`, as Y3's `sc_bias_fit_count.ipynb`: per cosmology and source
  bin, the b whose Poisson-sampled counts on the projection's `dg/<bin>` at `perm_0000` match the
  one-point function of the observed counts of `maps` (`source_clustering_bias.fit_bias_table`).
  One group per arm: `clean` (no imaging systematics) and `contam` (the contamination of
  `systematics` imprinted), picked by `files.metacal_bias_arm`. v1 uses `contam`, since its forward
  model imprints the same systematics. The projection root, which holds `grid/` and `benchmarks/`, is
  the `projection` of `configs/desy6/v1/locations.yaml`, unless `--source` names another. The key `fiducial`, which `run_single_postprocessing.py` reads for every benchmark
  mock, is fit on `benchmarks/fiducial_bench`: the fiducial cosmology, the fiducial set's
  resolution and the config's fiducial n(z). DES Y6 has no fiducial projection.
- **Checks:** every selected cosmology has its `perm_0000` file before anything runs; n_bar agrees
  with `survey.metacal.n_gal` to 5%; the histogram reaches the largest footprint count
  (`bias_fit_max_count`); the base patch is the sorted mask, the order the fit pairs the
  contamination with. The table is written to a temporary file first: every b finite and positive,
  every loss finite, and each arm reads back unchanged through `files.read_metacal_bias` and passes
  `configuration.print_and_check_modeling_in_config`, which matches the arm's systematics label
  against its forward model. A failed check writes nothing.
- **Status:** run on the 2500 grid cosmologies and `fiducial_bench` (2026-11-22, job 5424770),
  4 min per arm on one node (288 processes, one per core, with
  `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`). Each key is fit on its own with the same seed, so
  adding `fiducial` left every grid key and loss byte-identical. The fiducial b sits at the grid
  median (quantile 0.47-0.50 in every bin and arm).

  | | metacal1 | metacal2 | metacal3 | metacal4 |
  |---|---|---|---|---|
  | b `clean`, median (range) | 1.02 (0.44-2.57) | 1.41 (0.56-3.51) | 1.69 (0.62-4.11) | 2.76 (0.93-6.61) |
  | b `fiducial`, `clean` / `contam` | 1.012 / 0.945 | 1.394 / 1.306 | 1.677 / 1.468 | 2.728 / 1.937 |
  | b `contam` / `clean`, median | 0.93 | 0.94 | 0.89 | 0.71 |
  | largest loss, `clean` → `contam` | 5.6 → 4.8 | 5.4 → 4.5 | 4.8 → 3.8 | 4.4 → 2.3 |

  As for Y3, the imprint widens the counts, so the fit needs less bias, most in the highest bin, and
  the loss falls. The largest footprint counts are 293, 359, 280, 260 against the histogram's 400.

## Review

`notebooks/desy6/products.ipynb` is read-only. It plots every product, runs the κ_E × δ_g sign
test and the κ_B × δ_g null test, and checks the lens weight maps against the counts. `notebooks/desy3/products.ipynb` runs the same on the Y3 products,
for comparison.

The tests confirm Y3's e1/e2 signs and ψ rotation for Y6 (2026-11-19). κ_E × δ_g is positive for
all 20 source-lens pairs, with S/N 10 to 51 where the lens bin lies in front. κ_B × δ_g is null,
with a combined S/N of at most 2.1 in size.

The lens weight maps pass too (2026-11-20). The raw counts follow F = 1/w with slope 0.91 to 1.37
(jackknife errors 0.08 to 0.33), and the counts corrected by w are flat within 1.3σ in every bin.

`dev/scripts/validation/data_vs_mock.py` runs the observation path
(`observation.forward_model_observation_map` on the cached maps, the four variants y3-deep-lss
evaluates) and compares the result with the forward model. The same run on Y3 (v18) is the
reference. Results (2026-11-20):

- **Observation path:** it runs on Y6 unchanged, and every check passes in all four variants
  (shapes, zero padding, the tfrecords' Cl shape, the ψ rotation acting only on lensing and the
  weights only on clustering).
- **Noise model:** at the observed source counts, the `count` shape noise against the real
  galaxies rotated in place, as the ratio of the per-pixel κ variance:

  | source bin | 1 | 2 | 3 | 4 |
  |---|---|---|---|---|
  | count / in_place, Y6 | 0.982 | 0.985 | 0.993 | 1.003 |
  | count / in_place, Y3 | 0.990 | 0.988 | 0.993 | 0.996 |

  The Cl above ℓ = 1000 gives the same ratios. The scatter between realizations is about 0.3%, so
  the `count` model's noise is up to 2% low in Y6 bins 1 and 2, about 1% more than in Y3.
- **Data vs mock:** only indicative so far. The Y6 mock is the grid smoke test at S8 ≈ 0.9 with b_sc
  sampled from [0.5, 1.5], against Y3's fiducial mock with the fitted b_sc.

  | lensing | Y6 | Y3 |
  |---|---|---|
  | Cl ratio data / mock, ℓ < 200 | 0.73 to 0.78 | 0.98 to 1.00 |
  | Cl ratio data / mock, ℓ > 1000 | 0.96 to 0.99 | 1.00 to 1.01 |

  - **Lensing:** the low-ℓ deficit is the mock's higher S8.
  - **Lens counts:** in both surveys they depend on the galaxy bias each mock draws from its prior
    (Y3's maglim1 is at 0.74 too).

  The clean comparison needs a grid mock with the fitted bias table, or the `fiducial_bench` mock
  once it is built.
