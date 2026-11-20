# DES Y6 port of msfm: plan

Dev-phase plan, started 2026-11-15, cleaned up 2026-11-20. Repo docs must not point here.

This file holds what is still to do, the decisions and the open questions. What is done is
documented elsewhere, and this plan does not repeat it:

- `data/desy6/README.md`: every product step, its checks and its results;
- `notebooks/desy6/products.ipynb`: the read-only review of the products;
- `differences_to_y3.md` and `differences_to_y6_kp.md` (this folder): the model, against our Y3 v18
  and against the DES Y6 3x2pt key project (KP);
- `pipelines/daint/desy6/v1/README.md`: how the grid tfrecords are submitted;
- git history on branch `desy6`: how each piece was built and tested.

## Goal and rules

The goal is the Y6 grid `.tfrecord` training set, built by msfm from
`configs/desy6/v1/default.yaml` on the CPUs of daint's GH200 nodes. The model follows the KP where
the map-level model allows and otherwise keeps v18. y3-deep-lss and msi come after (section 6).

- **Y3 stays bit-identical.** Every new key defaults to the Y3 behavior when absent.
  `dev/scripts/regression/y3_regression.py` compares against
  `claude/bench/y3_regression/baseline_2026-11-17_v2`, whose two repeat runs agree on all 115
  arrays. Run it after every change to shared code.
- **Nothing from Y3 enters Y6 unless the config says so.** `configuration.check_survey_paths` keeps
  every repo path in `data/` or `data/<survey>/`, and `lensing.check_projection` requires `shearm`
  in a projection exactly when the config takes m from it. The only stated reuse is v18's footprint
  rotation.
- **Every `sbatch` waits for the user's OK.**

## 1. State (2026-11-20)

**Done.**

- msfm code:
  - survey-keyed caches and n(z) files;
  - the Y6 catalog reader;
  - m from the projection;
  - bin-count generality;
  - the two guards.

  The Y3 regression stayed identical after each.
- Product steps `nz`, `pixels`, `maps`, `values`, `noise`, `systematics` have run. `sc_bias` is
  written and was tested on `cosmo_000001` (job 5408693).
- The config holds every value except `files.metacal_bias`.
- Executor: `pipelines/daint/common/packed_node.slurm` and `pipelines/daint/desy6/v1/submit.sh`. Smoke
  test, job 5408757: 24 tasks × 12 cores ran indices 0-23 in 6:43.
- Grid smoke test of the v1 config on the test files, with `bias: prior` in place of the missing
  table, with and without the MAIN19 systematics (jobs 5408520, 5408563). Result: 5 records, 17
  labels, 4 + 5 bins, and every example carries its permutation's `shearm`.

**The projection.** Tomasz's re-projection is landing in place, under
`/capstor/store/cscs/userlab/lp184/tomaszk/261109_cosmogridv1_desy6/000_test/cosmogrid/processed/desy6/CosmoGrid/bary/grid`.
On 2026-11-20 all 50,000 files existed, and 785 still predated the re-projection. Everything below
was measured on the two fixed test files (`lp184/athomsen/y6_debug/cosmo_000001/perm_000{0,1}`) and
must be re-checked on the whole set:

- nside 512, with the labels `metacal1-4` and `maglim1-6`;
- the realization row equals the permutation index, in all 10 bins, with joint m and n(z);
- the maps are 4× the old ones (the old ones were too low), and ⟨κ⟩ matches the analytic value;
- the projection does not apply m, so msfm does;
- the source n(z) of both test permutations sit about 2σ below the ensemble mean in all four bins
  (metacal1: 0.391 vs 0.414 ± 0.012).

## 2. Remaining work, in order

1. **CPU sizing benchmark.** It can run now, on re-projected files. One node, one grid cosmology per
   task, with `SLOTS × OMP` ∈ {8 × 36, 12 × 24, 16 × 18, 24 × 12}.
   - Record per task: wall time, peak RSS, tfrecord size and read rate from capstor.
   - Result: node-hours per cosmology, extrapolated to 2500. The Perlmutter reference is ~150
     node-hours for the v17 grid.
   - Write it into "Sizing and cost" of the v1 README and set the `submit.sh` defaults.
2. **Re-scan the projection** once the last files are re-projected:
   - per file: nside, bin keys, realization row, `shearm` within 5σ;
   - m still not applied (`claude/tmp/desy6_m_kernel/`);
   - the source n(z) offset of the test permutations, on more permutations.

   If the row equals the permutation index everywhere, the grid has only 20 n(z) realizations (Q2).
3. **Step `sc_bias`** on the full grid, with `--source` the projection root, writing
   `desy6_metacal_bias.h5`. Then:
   - remove it from `EXPECTED_MISSING` in `dev/scripts/data_audit/check_config_paths.py`;
   - update its status in `data/desy6/README.md`;
   - update the `SKELETON` header and the `sc_bias` TODO of the config.
4. **Provenance rerun** (user decision: only after `sc_bias`). Every product step reruns with
   `--provenance` on committed code: `nz`, `pixels`, `maps`, `values`, `noise`, `systematics`,
   `sc_bias`. `values --check` must pass. Then drop the provenance TODO of the config.
5. **Production grid tfrecords.** Run `DRY_RUN=1 ./submit.sh grid`, then `./submit.sh grid`, as in
   the v1 README.
6. **Validation** (section 3), then the Y3 regression once more.
7. **Copy the tfrecords to the lp184 store**, since `/ritom` scratch is purged.

Later, and not needed for the grid set:

- the fiducial set (it needs a fiducial projection, Q6);
- the downstream port (section 6).

## 3. Validating the Y6 tfrecords

- **Schema.**
  - `n_z_metacal = 4` and `n_z_maglim = 5`;
  - `n_pix` equal to the Y6 data vector, 457,728;
  - label width and order equal to `parameters.get_parameters`;
  - 2500 files of 80 examples × 5 noise realizations.
- **Nuisances.** m differs across permutations and equals each file's `shearm`. The n(z)
  realization varies (Q2).
- **Noise and counts.** Per-pixel shape-noise rms and the source-count one-point function, against
  the Y6 data maps. The rms is the noise-model check of `data_vs_mock.py` (below). The one-point
  function is what step `sc_bias` fits; the table stores each cosmology's best-fit histogram (`stat`)
  next to the target.
- **Signal** (optional). Mean κ power of a grid cosmology against theory with its n(z), using
  `dev/scripts/wl_audit`.
- **Data vs mock.** `dev/scripts/observation/data_vs_mock.py`. On 2026-11-20 the observation path ran
  on Y6 with every check passing, but the comparison was only indicative (README, Review). Rerun it on
  a production mock with the fitted bias table. The same run found the `count` shape noise 1-2% below
  the in-place noise in Y6 source bins 1-2 (Y3: about 1%). Whether that matters is not decided.

## 4. Decisions

**Open.**

- **D3.** Bias parameter names. The config names them by survey bin (`bg1, bg3, bg4, bg5, bg6`,
  matching the DES chains' `b1e_bin*`); the other option is by position (`bg1..bg5`). Settle this
  in the downstream port, where the names have to be accepted (section 6).
- **D8.** Mean shear. The KP subtracts the residual additive c from the data; v18 does not
  subtract it. Only the observation is affected, not the tfrecords. Not checked yet: whether Y6
  models PSF leakage or other additive terms that we would have to forward model.
- **D11.** Uncertainty of the lens weights. The KP marginalizes the ENet - ISD difference in the
  w(θ) covariance; v1 has none. Raised by `differences_to_y6_kp.md`, not discussed yet.

**Settled.** The difference lists hold the details.

- **D1** (2026-11-18): `configs/desy6/v1/default.yaml`, `pipelines/daint/desy6/v1/`.
- **D2** (2026-11-20): bin labels Y3's, `metacal1-4` and `maglim1-6`, as the projection writes
  them. Revisit only if the production projection changes them.
- **D4** (2026-11-20): the IA pivot is the KP's, z0 = 0.3. Everything else stays v18's: δ-NLA, the
  boxes and the fiducials.
- **D5** (2026-11-18): shape noise as v18, `count` with `survey_systematics: True`, using the
  MAIN19 map.
- **D6** (2026-11-20): no lens magnification.
- **D7** (2026-11-19): linear lens bias, also the KP baseline.
- **D9** (2026-11-19): coverage threshold 0.73.
- **D10** (2026-11-20): the IA amplitude from the mean n(z) with the 5-95% truncation.

## 5. Questions for Tomasz

**Open.**

- **Q2.** n(z) realizations. In the test files the realization row equals the permutation index.
  - Is that the design, giving 20 realizations grid-wide, or a placeholder?
  - Can the realization indices be stored as attributes next to `shearm`?
  - Fiducial derivative pairs need the same realization per permutation for both signs.
- **Q6.** Are fiducial and benchmark projections planned? They are needed for the fiducial
  tfrecords, the mocks, the downstream scale cuts and the `fiducial` key of the bias table.

**Answered.**

- Q1: the production projection is written in place, at nside 512.
- Q3: all six lens bins are projected, as `maglim1-6`.
- Q4: the lens n(z) are rows of `desy6_lens_redshift.h5`.
- Q5: moot after D6.

## 6. Downstream port (y3-deep-lss, msi), later

The table comes from a read-only sweep on 2026-11-15; ✓ marks a line checked by hand. Re-check the
line numbers before editing. No Python code in either package hardcodes a bin or pixel count, since
both come from the msfm config and pixel file.

| where | today | change |
|---|---|---|
| y3-deep-lss `configs/scales/*.yaml` ✓ | `theta_fwhm`, `white_noise_sigma`, `l_max` from the Y3 n(z), signal and n_gal; lists of length 4 | a Y6 scales file from `scale_cuts.ipynb` and `dev/scripts/noise_models/derive_scale_cut_noise.py`. It needs fiducial maps (Q6) or a stand-in |
| y3-deep-lss `deep_lss/utils/configuration.py` ✓ | per-pair ℓ cuts fall back to a default, **silently**, when a bin index exceeds the list; the smoothing channel count comes from the same lists | a length assert against n_z |
| y3-deep-lss `evaluation.py` ✓, `cls_evaluation.py`; msi `observations.py` ✓, the defaults of `run_inference.py`, `coverage.py`, `run_fom_percentile.py`, `run_ppc.py`, `ppc.py`; y3-deep-lss `run_diagnostics.py`; every runs config's `obs_labels` | the observation key `"DESy3"` (`"DESy3_no_sys"`). msi selects only `DESy3*`, so a `DESy6` key gets no chain and **no error** | derive it from `survey.name` |
| y3-deep-lss `submissions/clariden/common.sh` ✓ | the glob `DESy3_grid_dmb_????.tfrecord` | derive it from `survey.name` |
| y3-deep-lss `configs/probes/*.yaml`; msi runs configs, `coverage_round.py`, `plotting.py`, `shifted_posterior.py` | `bg1..bg4`. A subset is accepted, so `bg5` would be marginalized **silently**; `plotting.py` raises a KeyError | take the list from msfm `params.bg.linear` (D3) |
| msi `preprocessing.py` ✓, `coverage_round.py` ✓ | `n_patches = 4`, `n_noise = 3`, `(2500, 80, ...)` | read the msfm `analysis` block |
| both `submissions/clariden/common.sh` | `STORE` on the a0158 store | repoint after the Globus move |
| msi runs configs (`pred_dir`, `scales_config`), `configs/ppc/*.yaml` (`base_dir`) | `runs/v18`, `8wl,32gc.yaml`, `data/v18` | Y6 runs configs |
| y3-deep-lss `INPUT` | `data/$VERSION/$SUBVERSION` on `/iopsstor` scratch | the v1 tfrecords default to `/ritom/scratch/cscs/athomsen/desy6/v1/default/`, or the lp184 copy |

Two more:

- tf_env and torch_env import msfm editable from this checkout. While it is on branch `desy6`, the
  Y3 defaults downstream (`VERSION=v18`, flat config paths) do not resolve.
- A Y6 validation mock, the counterpart of the Buzzard flock, is not chosen.

## 7. Risks

- **A mixed projection.** The `submit.sh` preflight checks only that the files exist. A run before
  the re-scan (section 2, item 2) could mix old and re-projected maps.
- **Few n(z) realizations.** If the grid carries only 20 (Q2), the redshift marginalization rests
  on 20 draws.
- **Scratch purges.** Inputs stay on the store; outputs go to the lp184 store promptly.
- **Unknown cost** until the benchmark: aarch64 healpy throughput is unmeasured at scale.
