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
- All seven product steps have run, `sc_bias` on the 2500 grid cosmologies and, since 2026-11-22,
  the key `fiducial` on `benchmarks/fiducial_bench` (job 5424770; the grid keys byte-identical).
- The config holds every value.
- Executor: `pipelines/daint/common/packed_node.slurm` and `pipelines/daint/desy6/v1/submit.sh`. Smoke
  test, job 5408757: 24 tasks × 12 cores ran indices 0-23 in 6:43.
- Grid smoke test of the v1 config on the test files, with `bias: prior` in place of the missing
  table, with and without the MAIN19 systematics (jobs 5408520, 5408563). Result: 5 records, 17
  labels, 4 + 5 bins, and every example carries its permutation's `shearm`.

**The projection.** Tomasz's re-projection is complete (transfer and permissions fixed 2026-11-20),
under
`/capstor/store/cscs/userlab/lp184/tomaszk/261109_cosmogridv1_desy6/000_test/cosmogrid/processed/desy6/CosmoGrid/bary/grid`.
The re-scan of all 50,000 files (2026-11-20, `claude/tmp/desy6_rescan/`) found:

- nside 512, the labels `metacal1-4` and `maglim1-6`, one layout, and no truncated file;
- perms 0 and 1 of `cosmo_000001` identical to the fixed test files
  (`lp184/athomsen/y6_debug/cosmo_000001/perm_000{0,1}`). So what was measured on those carries over:
  the maps are 4× the old ones, and ⟨κ⟩ matches the analytic value;
- the projection does not apply m, so msfm does: over the 80 bins of `cosmo_000001`, the kernel
  ratio has slope -0.05 against 1 + m;
- each file carries its own realization, rows 0-49,999 of the source ensemble in file order. The
  result is offset from the full ensemble and correlated with cosmology; accepted for v1
  (`differences_to_y6_kp.md`, Realization assignment).

## 2. Remaining work, in order

1. **CPU sizing benchmark.** Done 2026-11-20. Each arm ran one node, one grid cosmology per task:
   24 × 12 took 44:53 (job 5408848) and 12 × 24 took 34:04 (job 5408849). That is 0.031 against 0.047
   node-hours per cosmology, with a peak of 10.5 GB per task. **24 × 12 is the default (user
   decision), ≈ 79 node-hours for the grid** plus the merge (Perlmutter reference: ~150 for the v17
   grid). The numbers are in "Sizing and cost" of the v1 README, and `submit.sh` defaults to them.
2. **Provenance rerun.** Done 2026-11-20/21. All seven steps reran with `--provenance` on msfm `814a500`:
   - `pixels`, `maps` and `noise` as debug jobs 5409505, 5409512 and 5416023;
   - `sc_bias` as debug job 5416036 (9:05);
   - `values` again on `d129dfa`, since an up-to-date config left no record before that commit.

   Every output came out byte-identical to the one already in place, the bias table included, and
   `values --check` passes. `data/desy6/provenance.yaml` holds the seven records.
3. **Production grid tfrecords.** Done 2026-11-21.
   - Array 5416078: all 105 elements completed, 41:49–45:06 each, 77.2 node-hours in total. All 2500
     indices finished, with no failure and no error in any log.
   - The output is 2500 files, 16 TB, under `/ritom/scratch/cscs/athomsen/desy6/v1/default/tfrecords/grid`.
   - Merge 5416079: 3:41, 1.4 GB peak, writing `cls/grid_cls.h5` (282 GB, 45 Cl pairs, 16 labels).
     In that file every cosmology has a single `i_sobol` (2500 distinct), `i_signal` 0-79 × 5,
     `i_noise` 0-4 × 80, and finite labels and binned Cls.
4. **Validation, optional** (section 3). The machinery is built and tested; the runs are left for
   later (user decision, 2026-11-22). The Y3 regression after the shared-code changes of the
   2026-11-22 review (`lensing.read_m_bias`, the mocks' m in `c824650`) ran on `c824650`: job
   5424437, 115 of 115 arrays identical to the baseline (`after_mock_m/`).
5. **Copy the tfrecords to the lp184 store**, since `/ritom` scratch is purged.

Later, and not needed for the grid set:

- the mock observations from the four benchmark projections: all six done 2026-11-22 (one node each,
  under 4 min, checks passed, `/ritom/.../desy6/v1/default/obs/`; the Y3 regression after the
  shared-code change, job 5424920, stayed identical, 115 of 115). Copied to the lp184 store the same
  day, `/capstor/store/cscs/userlab/lp184/athomsen/deep_lss/data/desy6/v1/default/obs` (byte-identical).
  All six
  are defined in `configs/desy6/v1/mock_observations.yaml` (format in `msfm/utils/mocks.py`) and
  submitted with `pipelines/daint/desy6/v1/submit.sh obs <label>|all`;
- the downstream port (section 6).

The fiducial set is not needed (user, 2026-11-22). Where Y3 used the fiducial maps, Y6 uses the
benchmark `fiducial_bench`: the bias-table key `fiducial`, and the noise level of y3-deep-lss's
scales config.

No decision is open (section 4).

## 3. Validating the Y6 tfrecords (optional)

**Machinery** (msfm `3e27d6a`): everything is in `dev/scripts/validation/`, and each docstring lists
its checks. The commands, in run order, are in the "Validation" section of the v1 README.

- `grid_tfrecords.py` runs through the packed executor (`./submit.sh validate`). Per file it checks:
  - schema and labels;
  - finite maps with zero padding;
  - noise and signal from the 5 realizations;
  - the projected inputs (m, n(z));
  - the matching `grid_cls.h5` row.

  Across files: the lens bins against their own `bg` labels, lensing against S8, m and n(z)
  across perms, and a byte comparison against a `--debug` rerun of the grid app.
- `data_vs_mock.py` compares the observation with the forward model. `--grid_cls` gives the observed
  Cls' percentile among all grid Cls (does the training set cover the data?). `--scan` compares the
  grid's lensing noise with the data's in-place noise.

**Done** (2026-11-22): a test on indices 0 and 1 found no failure, and both planted errors failed as
they should. It also turned up two things:
- The rows of `grid_cls.h5` are in shuffled order (`Dataset.list_files`), so the check finds each
  file's row by `i_sobol`.
- The lens noise matches Poisson to 0.989 once the hard band limit is included (ℓ ≤ 1535 keeps 0.75
  of white-noise variance).

**Why optional.** The grid app verifies every record on write, the merge read all 2500 files with
finite labels and Cls, and the code path is the same for every file. So a full scan mostly
re-confirms file integrity. The parts job success cannot answer are scientific: the grid coverage
of the data and the noise level against the data.

**If run later**, in order of value:

1. `data_vs_mock.py --grid_cls` on one production file, with `--scan` from a 72-file scan
   (`ARRAY=0 SKIP_MERGE=1 ./submit.sh validate`, merged with `--partial`): about 0.5 node-hours.
2. The rerun comparison: one debug job on indices 0-23, about 0.2 node-hours. It is the only direct
   check that the stored maps carry each perm's `shearm`, and that HEAD still produces these files
   after the review's change to `lensing.read_m_bias`.
3. The full scan of all 2500 files: about 2-6 node-hours and about 1 h of wall time. This is an
   estimate, since the read rate with 72 files at once per node is not measured (2 at once: 48 s per
   file).

Open from the 2026-11-20 run: the `count` shape noise is 1-2% below the in-place noise in Y6 source
bins 1-2 (Y3: about 1%). Whether that matters is not decided. Not covered by the machinery: a
theory check of the mean κ power (`dev/scripts/wl_audit`).

## 4. Decisions

**Open.**

None.

**Settled.** The difference lists hold the details.

- **D1** (2026-11-18): `configs/desy6/v1/default.yaml`, `pipelines/daint/desy6/v1/`.
- **D2** (2026-11-20): bin labels Y3's, `metacal1-4` and `maglim1-6`, as the projection writes
  them. Revisit only if the production projection changes them.
- **D3** (2026-11-21): bias names by survey bin, `bg1, bg3, bg4, bg5, bg6` (as the DES chains'
  `b1e_bin*`). y3-deep-lss (its D5) and msi (its D3) take them as they are.
- **D4** (2026-11-20): the IA pivot is the KP's, z0 = 0.3. Everything else stays v18's: δ-NLA, the
  boxes and the fiducials.
- **D5** (2026-11-18): shape noise as v18, `count` with `survey_systematics: True`, using the
  MAIN19 map.
- **D6** (2026-11-20): no lens magnification.
- **D7** (2026-11-19): linear lens bias, also the KP baseline.
- **D8** (2026-11-22): the mean shear is not subtracted, as in v18. The KP subtracts c, the
  weighted ⟨e⟩/R per bin (KP Table 2, ×10⁻⁴: (1.6, 0.62), (0.45, −0.58), (2.4, −1.7), (3.1, 1.1));
  step `maps` reproduces it to the printed digit. Our simulations have no additive bias but keep
  the cosmological footprint mean, so subtracting from the data alone would be the inconsistent
  choice. c is about half of Y3's γ1 and 0.6-1% of the per-pixel noise rms (Y3: ~1.5%), and Y3
  measured ≤ 0.05σ on every parameter. Neither KP nor FW models PSF leakage or an additive ξ±
  term, so there is nothing else to forward model. To confirm on Y6: the paired-seed test
  (`claude/tmp/meansub_des/`) once a Y6 lensing network exists.
- **D9** (2026-11-19): coverage threshold 0.73.
- **D10** (2026-11-20): the IA amplitude from the mean n(z) with the 5-95% truncation.
- **D11** (2026-11-22): the lens weights follow our Y3 approach. They correct the observed counts
  only, never the simulations, and no uncertainty is propagated. DES instead adds C + ΔΔᵀ to each
  lens bin's w(θ) covariance, Δ = ŵ(θ)_ENet − ŵ(θ)_ISD on the data (arXiv:2601.14484 Eq. 24); the
  difference is row 5 of `differences_to_y6_kp.md`, to address perhaps later. Measured 2026-11-22 on
  the released weights (`weight`, `WEIGHT_ISD` of `maglim_addl_cols_v1.hdf5`), as nside-64 pixel
  means for bins 1, 3-6: correlation 0.40, 0.68, 0.88, 0.78, 0.93, and the rms of the difference
  0.93, 1.04, 0.47, 0.71, 0.36 × the ENet correction. A later check could be the posterior shift
  between ENet- and ISD-corrected data, which needs an ISD weight map.
- **D12** (2026-11-22): mocks apply m when the config takes it from the projection.
  `forward_model_cosmogrid(shear_biasing=None)` resolves from `source`, so a Y6 mock carries its
  realization's n(z) and m, as the grid does; Y3 mocks keep m = 0 (`c824650`). The tfrecords are
  unaffected.
- **D13** (2026-11-22): step `values` writes the bias fiducials `analysis.fiducial.bg*`: v18's
  power law (bg 1.5, n_bg 0.5, pivot 0.62) on the Y6 lens n(z), the pivot a constant of the step
  rather than a config key. The written values equal the typed ones digit for digit.

## 5. Questions for Tomasz

**Open.** None.

**Answered.**

- Q6: the four CosmoGridV1 benchmarks are projected (`benchmarks/`, written 2026-11-19):
  `fiducial_bench`, `box_size`, `particle_count` and `redshift_resolution`, 20 permutations each,
  nside 512 and `maglim1-6`. All 80 files carry one n(z) and m, the mean over the 80,760
  realizations, which equals the config's fiducial n(z) to 4e-8. No fiducial projection; none is
  needed.

- Q1: the production projection is written in place, at nside 512.
- Q2: every file carries its own realization, source row 20c + p (cosmology c, permutation p),
  found by the re-scan (section 1).
- Q3: all six lens bins are projected, as `maglim1-6`.
- Q4: the lens n(z) are rows of `desy6_lens_redshift.h5`.
- Q5: moot after D6.

## 6. Downstream port (y3-deep-lss, msi)

**y3-deep-lss** is being ported on its branch `desy6` (started 2026-11-20); its plan is
`y3-deep-lss/dev/notes/desy6/plan.md`. It produces the Y6 scales config (smoothing scales, noise
levels, ℓ cuts) through its own product steps, which share their helpers with
`msfm.apps.products.desy6` through `msfm/utils/products.py`.

**msi** is being ported on its branch `desy6` (started 2026-11-21); its plan is
`multiprobe-simulation-inference/dev/notes/desy6/plan.md`, its model differences
`differences_to_y3.md` next to it. Done there: the observation key from `survey.name`, the `bg5`/`bg6`
labels, a guard on the Y3 key-project overlay. The grid-geometry and `STORE` rows below need no
change for Y6 (same geometry; `RUNS_ROOT` is passed). The table is the read-only sweep of
2026-11-15, kept for the record.

| where | today | change |
|---|---|---|
| msi `observations.py` ✓, the defaults of `run_inference.py`, `coverage.py`, `run_fom_percentile.py`, `run_ppc.py`, `ppc.py`; every runs config's `obs_labels` | the observation key `"DESy3"` (`"DESy3_no_sys"`). msi selects only `DESy3*`, so a `DESy6` key gets no chain and **no error** | derive it from `survey.name` |
| msi runs configs, `coverage_round.py`, `plotting.py`, `shifted_posterior.py` | `bg1..bg4`. A subset is accepted, so `bg5` would be marginalized **silently**; `plotting.py` raises a KeyError | take the list from msfm `params.bg.linear` (D3) |
| msi `preprocessing.py` ✓, `coverage_round.py` ✓ | `n_patches = 4`, `n_noise = 3`, `(2500, 80, ...)` | read the msfm `analysis` block |
| msi `submissions/clariden/common.sh` | `STORE` on the a0158 store | repoint after the Globus move |
| msi runs configs (`pred_dir`, `scales_config`), `configs/ppc/*.yaml` (`base_dir`) | `runs/v18`, `8wl,32gc.yaml`, `data/v18` | Y6 runs configs |

Two more:

- tf_env and torch_env import msfm editable from this checkout. While it is on branch `desy6`, the
  Y3 defaults downstream (`VERSION=v18`, flat config paths) do not resolve.
- A Y6 validation mock, the counterpart of the Buzzard flock, is not chosen.

## 7. Risks

- **Ordered n(z) rows.** Accepted for v1 (section 1); the effect on the posteriors is not measured.
- **Scratch purges.** Inputs stay on the store; outputs go to the lp184 store promptly.
- **Cost.** Measured at 24 × 12 (section 2, item 1); the merge is not yet timed on daint.
