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
- All seven product steps have run, `sc_bias` on the 2500 grid cosmologies (job 5409210, no
  `fiducial` key).
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
4. **Validation** (section 3), then the Y3 regression once more. It is due anyway: the review of
   2026-11-22 changed shared code (`lensing.read_m_bias`, now also used by the mock path's
   `shear_biasing`). Y3 never reaches the changed branches, but the rule above asks for the run.
5. **Copy the tfrecords to the lp184 store**, since `/ritom` scratch is purged.

Later, and not needed for the grid set:

- the fiducial set (it needs a fiducial projection, Q6);
- the downstream port (section 6).

The open decisions D8 and D11 (section 4) affect only the observation and the inference, not the
tfrecords.

## 3. Validating the Y6 tfrecords

- **Schema.**
  - `n_z_metacal = 4` and `n_z_maglim = 5`;
  - `n_pix` equal to the Y6 data vector, 457,728;
  - label width and order equal to `parameters.get_parameters`;
  - 2500 files of 80 examples × 5 noise realizations.
- **Nuisances.** m differs across permutations and equals each file's `shearm`. The n(z)
  realization differs per file (section 1).
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

- **D8.** Mean shear. The KP subtracts the residual additive c from the data; v18 does not
  subtract it. Only the observation is affected, not the tfrecords. Not checked yet: whether Y6
  models PSF leakage or other additive terms that we would have to forward model.
- **D11.** Uncertainty of the lens weights. The KP marginalizes the ENet - ISD difference in the
  w(θ) covariance; v1 has none. Raised by `differences_to_y6_kp.md`, not discussed yet.
- **D12.** m in the mocks. `run_single_postprocessing` calls `forward_model_cosmogrid` without
  `shear_biasing`, so its mocks have m = 0 (v18's convention). On Y6 such a mock carries a
  realization's n(z) without that realization's m. Since 2026-11-22, `shear_biasing` takes the
  file's `shearm`; whether Y6 mocks should turn it on is not decided. The tfrecords are unaffected.
- **D13.** The bias fiducials `analysis.fiducial.bg*` are typed into the config, not written by
  step `values`, unlike every other data-derived value. They reproduce exactly (power law bg 1.5,
  n_bg 0.5, pivot 0.62 on the Y6 lens n(z)). Moving them into `values` would need the pivot as a
  constant or key. Only the fiducial set and the mocks read them.

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
- **D9** (2026-11-19): coverage threshold 0.73.
- **D10** (2026-11-20): the IA amplitude from the mean n(z) with the 5-95% truncation.

## 5. Questions for Tomasz

**Open.**

- **Q6.** Are fiducial and benchmark projections planned? They are needed for the fiducial
  tfrecords, the mocks, the downstream scale cuts and the `fiducial` key of the bias table.
  Fiducial derivative pairs need the same realization per permutation for both signs.

**Answered.**

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
