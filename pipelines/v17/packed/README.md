# Packed regular-QOS run of the v17 baseline postprocessing (no esub)

Clean-start submission of the v17 baseline dataset (grid + fiducial) on `--qos=regular` (whole
CPU nodes), packing many postprocessing tasks per node, each pinned to a NUMA domain. Uses the
version-agnostic executors in `../../common/packed/` — see that directory's `README.md` for *why*
this exists (vs. esub/shared, vs. unpinned packing) and how the `SLOTS`/`OMP` sizing methodology
works in general. This file only covers what's specific to v17.

v17 also still has an esub/shared-QOS path (`../esub/`, driven by `../pipe.yaml`), kept
intentionally in case there's a reason to fall back to it. `submit_single_mock.sh` in this
directory duplicates exactly one arm also defined in `../esub/obs_commands.sh` (`sc_fixed_sys`) —
see the cross-reference comments in both files if you change one.

## This chain's sizing

| chain    | peak RSS/task | SLOTS | per domain       | threads/task | node mem | node time (measured)     |
|----------|---------------|------:|------------------|-------------:|---------:|--------------------------|
| grid     | ~13.5 GB      |    24 | 3 × 13.5 = 40 GB |           10 | ~325 GB  | 85 min (contended)       |
| fiducial | ~11.5 GB      |    24 | 3 × 11.5 = 35 GB |           10 | ~277 GB  | 37.5 min                 |

Memory is near-constant across cosmologies (fixed array shapes), so the peaks are reliable
ceilings; both leave >20 GB/domain of headroom. Walltime caps sit well above the measured node
time: NERSC charges *actual elapsed*, not the cap, so headroom is free and guards against a node
killed at the cap (which fails the element and blocks the `afterok` merge). All per-chain numbers
live in one place — the `chain_cfg()` block in `submit.sh`.

### The 24-vs-32-slot benchmark (why 24)

Both packings were run on one full node under the same concurrent-fleet filesystem contention:

| grid packing        | node time      | nodes (2500) | node-hours |
|---------------------|----------------|-------------:|-----------:|
| **24 slots / 10 thr** | 85.5 min     |          105 |    **150** |
| 32 slots / 8 thr    | 142.8 min (med)|           79 |        188 |

24-slot is ~20% cheaper — the 4th task/domain nearly doubles the node time for a bandwidth-bound
workload. (The same holds for fiducial: 24-slot = 37.5 min → ~26 nh vs 32-slot = 78 min → ~42 nh.)

> **History:** the first v17 production run used grid at **32/8** — before this comparison existed —
> and under full-fleet contention cost ~188 node-hours (nodes ran a median 2 h 23, well above the
> isolated 1 h 39 dense benchmark). `chain_cfg` now defaults both chains to **24/10**.

## Files

- `submit.sh` — control script / single source of truth for grid+fiducial tfrecord production.
  Builds the index list, submits the packed array, then the merge (afterok), routing through
  `../../common/packed/{run_packed.py,packed_node.slurm,merge_node.slurm}`.
- `submit_single_mock.sh` — packed submission for the `sc_fixed_sys` single-postprocessing mock
  (see the cross-reference note above).

## Estimated cost / wall clock (at the current 24/10 default)

NERSC charges *actual elapsed* × nodes, not the requested cap. Node times below are the contended
measurements; an uncontended run (fewer nodes in flight) is faster.

- **grid**: 2500 / 24 = **105 nodes** × ~85 min ⇒ **~150 node-hours**.
- **fiducial**: 1000 / 24 = **42 nodes** × 37.5 min ⇒ **~26 node-hours**.
- **merge**: 2 single-node jobs (grid ~2 h 47, fiducial ~32 min) ⇒ a few node-hours.
- Wall clock per chain: ~1.5–3 h depending on how fast the nodes backfill.

## How to run

```bash
cd pipelines/v17/packed
DRY_RUN=1 ./submit.sh both     # inspect: prints every sbatch call, touches nothing
./submit.sh both               # or: grid | fiducial
```

**Fresh vs reuse.** A plain `./submit.sh <chain>` is a *fresh* run: it submits every index and
TFRecordWriter truncates-on-open, so any pre-existing file is regenerated. Prefer this after a
cancelled run — leftover files can be truncated mid-write, and "full" files can split into size
clusters that can't be certified complete without parsing, so `--rerun`'s file-presence check
could otherwise feed a corrupt file into the merge.

**Staging overrides** (for benchmarking a config or splitting a run):
- `ARRAY=<spec>` submits only those array elements, e.g. `ARRAY=0 ./submit.sh fiducial` runs one
  node (indices 0–SLOTS-1). The index file is still built over *all* indices so element k always
  maps to indices `[k*SLOTS,(k+1)*SLOTS)`; a later `ARRAY=1-41 ./submit.sh fiducial` covers the
  rest, reusing element 0's output.
- `SKIP_MERGE=1` defers the merge; once every stage is done submit it with
  `sbatch --dependency=afterok:<id1>:<id2> --time=<MERGE_TIME> --job-name=<chain>_merge --output=<PACKED_LOGS>/%x_%A.out --export=ALL ../../common/packed/merge_node.slurm`.

## If something fails

- A killed/failed packed element (OOM, walltime, task error) exits non-zero, which — via `afterok`
  — keeps the merge blocked. Delete any 0-byte or short stub the failed task left, then rerun only
  the missing indices:
  ```bash
  ./submit.sh --rerun grid      # missing = all indices with no .tfrecord; resubmits + a fresh merge
  ```
- Sanity check the output count before merging — normally 2500 (grid) / 1000 (fiducial):
  ```bash
  ls /pscratch/sd/a/athomsen/v11desy3/v17/baseline/tfrecords/grid/*_grid_*.tfrecord | wc -l
  ```
- **A missing file is tolerated by the fiducial `merge()`** (verified): it globs whatever is
  present, sorts records by their stored `i_signal`, and the extra `i_noise` tiling entries are
  never indexed — so merging N<1000 files yields a correct, self-labelled subset (that permutation
  is simply absent from the output). If the array already finished, submit the merge directly (no
  `afterok`):
  ```bash
  # export APP / DRIVER / JOB_NAME / APP_ARGS as submit.sh does, then:
  sbatch --time=04:00:00 --job-name=tfr_fidu_v17_merge --output=$PACKED_LOGS/%x_%A.out \
      --export=ALL ../../common/packed/merge_node.slurm
  ```
  This was needed once for the v17 run: input map `cosmo_delta_bary_nu_p/perm_0209` was corrupt
  upstream (the only bad file of 67,260), so fiducial index 209 could not complete and the set was
  merged from 999 files. (The grid `merge()` was not re-verified for gaps; grid inputs were all
  intact, so grid expects the full 2500.)
