# Packed regular-QOS run of the v18 baseline postprocessing (packed-only, no esub)

Clean-start submission of the v18 baseline dataset on `--qos=regular` (whole CPU nodes), packing
many postprocessing tasks per node, each pinned to a NUMA domain. Uses the version-agnostic
executors in `../../common/packed/` — see that directory's `README.md` for *why* this exists (vs.
esub/shared, vs. unpinned packing) and how the `SLOTS`/`OMP` sizing methodology works in general.
This file only covers what's specific to v18.

**v18 is packed-only.** Unlike v17, there is no esub/shared-QOS sibling for this version — the
retired esub definitions (`pipe.yaml`, `default/obs_commands.sh`) live in
`../../deprecated/v18_esub/` purely as historical reference; every job they defined has a packed
equivalent here (`submit.sh` for grid/fiducial tfrecord production, `submit_obs.sh` for every
single-postprocessing benchmark/mock arm).

## Sizing

`chain_cfg()` in `submit.sh` uses the same `SLOTS=24`/`OMP=10` as v17's grid/fiducial chains (see
`../../v17/packed/README.md` for the measured 24-vs-32-slot bandwidth-knee benchmark that
sizing is based on). **This is an inherited assumption, not an independent v18 measurement** — v18
runs the same apps (`run_grid_postprocessing.py`/`run_fiducial_postprocessing.py`) over the same
CosmoGrid inputs, so the per-task memory/compute profile should be close to v17's, but v18 hasn't
actually been run at packed scale yet. Re-measure node time on a first real `ARRAY=0` submission
before trusting the cost estimate below at scale.

v18 has no `SKIP_INDICES` equivalent to v17's fiducial `index 209` corrupt-input skip — not
because the issue doesn't apply (same CosmoGrid inputs), but because it hasn't been re-checked for
this version; if fiducial index 209 fails here too, add
`SKIP_INDICES_DEFAULT="209"` to `chain_cfg`'s `fiducial` case the same way v17 does.

## Files

- `submit.sh` — control script for grid+fiducial tfrecord production. Builds the index list,
  submits the packed array, then the merge (afterok), routing through
  `../../common/packed/{run_packed.py,packed_node.slurm,merge_node.slurm}`.
- `submit_obs.sh` — packed replacement for the retired `obs_commands.sh`: one `arm_cfg()` table
  covering every benchmark/mock arm (`reference`, `dmo`, `box_size`, `particle_count`,
  `redshift_resolution`, `sc_in_place`, `sc_no_sys`, `sc_gatti`, `eta_shell`, `grid_debug`).

## Estimated cost / wall clock (unverified for v18, see caveat above)

Carried over from v17's measured numbers as a starting estimate only:

- **grid**: 2500 / 24 = **105 nodes** × ~85 min ⇒ **~150 node-hours** (v17 measurement).
- **fiducial**: 1000 / 24 = **42 nodes** × ~38 min ⇒ **~26 node-hours** (v17 measurement).
- **obs benchmarks** (`submit_obs.sh`): 10 arms × 20 indices, `SLOTS=24` fits each arm on one
  node ⇒ ~10 node-hours total at ~1 h/node, generous.

## How to run

```bash
cd pipelines/v18/packed
DRY_RUN=1 ./submit.sh both          # inspect grid+fiducial: prints every sbatch call, touches nothing
./submit.sh both                    # or: grid | fiducial

DRY_RUN=1 ./submit_obs.sh all       # inspect every benchmark/mock arm
./submit_obs.sh sc_gatti            # or any single arm, or `all`
```

**Fresh vs reuse.** A plain `./submit.sh <chain>` / `./submit_obs.sh <arm>` is a *fresh* run: it
submits every index and any pre-existing output file is regenerated (`TFRecordWriter`/`h5py`
truncate on open). Prefer this after a cancelled run.

**Staging overrides** (`submit.sh`, for benchmarking a config or splitting a run):
- `ARRAY=<spec>` submits only those array elements, e.g. `ARRAY=0 ./submit.sh fiducial` runs one
  node (indices 0–SLOTS-1). The index file is still built over *all* indices so element k always
  maps to indices `[k*SLOTS,(k+1)*SLOTS)`.
- `SKIP_MERGE=1` defers the merge; once every stage is done submit it with
  `sbatch --dependency=afterok:<id1>:<id2> --time=<MERGE_TIME> --job-name=<chain>_merge --output=<PACKED_LOGS>/%x_%A.out --export=ALL ../../common/packed/merge_node.slurm`.

## If something fails

- A killed/failed packed element (OOM, walltime, task error) exits non-zero, which — via `afterok`
  — keeps the merge blocked. Delete any 0-byte or short stub the failed task left, then rerun only
  the missing indices:
  ```bash
  ./submit.sh --rerun grid       # missing = all indices with no .tfrecord; resubmits + a fresh merge
  ./submit_obs.sh --rerun sc_gatti
  ```
- Sanity check the output count before merging — normally 2500 (grid) / 1000 (fiducial):
  ```bash
  ls /pscratch/sd/a/athomsen/v11desy3/v18/baseline/tfrecords/grid/*_grid_*.tfrecord | wc -l
  ```
- If fiducial index 209 fails the same way it did in v17 (see "Sizing" above), the fiducial
  `merge()` tolerates a missing file — it globs whatever is present, sorts by stored `i_signal`,
  and merges a correct, self-labelled subset (that permutation is simply absent from the output).
