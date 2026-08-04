# Packed regular-QOS run of the v17 baseline postprocessing

Clean-start submission of the v17 baseline dataset (grid + fiducial) on `--qos=regular`
(whole CPU nodes), packing many postprocessing tasks per node.

## Why not the default esub/shared chain

`--qos=shared` maps to the 70-node `shared_milan_ss11` pool, which is chronically saturated by
other users' multi-day jobs — only ~5 of our ~1 h tasks run at a time (ETA weeks). The
`regular_milan_ss11` pool is ~2850 nodes with idle capacity, and short single-node jobs
backfill quickly. But regular QOS is **node-exclusive**, so a plain esub jobarray would waste a
128-core node on one ~15 GB task. We therefore hand-pack many tasks onto each node.

## How the packing is sized

The forward model is **memory-bound, not compute-bound** — cores are requested mainly to buy
RAM (see the `resources()` docstring in `run_fiducial_postprocessing.py`). On a node-exclusive
node all ~503 GB is available regardless of thread count, so we pack by **memory** and split the
256 hyperthreads across the slots. Measured peak RSS / max walltime from the cancelled shared
run:

| chain     | peak RSS | max task time | SLOTS/node | threads/task | node mem used | walltime cap |
|-----------|----------|---------------|-----------:|-------------:|--------------:|-------------:|
| grid      | 14.6 GB  | 1 h 45        |         24 |           10 | ~350 GB       | 6 h          |
| fiducial  | 17.2 GB  | 1 h 22        |         20 |           12 | ~344 GB       | 4 h          |

Memory is near-constant across cosmologies (fixed array shapes), so the peaks are reliable
ceilings; both packings leave >90 GB of headroom. The walltime caps sit well above the measured
max task time on purpose: NERSC charges *actual elapsed*, not the requested cap, so headroom is
free and guards against a node killed at the cap (which would fail the element and block the
`afterok` merge). All per-chain numbers live in one place — the `chain_cfg()` block at the top of
`submit.sh`.

## Files

- `submit.sh` — the control script and single source of truth. Builds the index list, submits
  the packed array, then the merge (afterok).
- `packed_node.slurm` — generic executor: runs `$SLOTS` `esub --mode=run --function=main` tasks
  in parallel, one per index in this array element's slice. All config arrives via `--export=ALL`.
- `merge_node.slurm` — generic single-node `esub --mode=run --function=merge`; globs the
  `.tfrecord` files and writes `{grid,fiducial}_cls.h5`.

Each packed task calls the exact same entry point esub uses in jobarray mode
(`esub <app> ... --function=main --tasks=<idx>`), so the forward model, filenames and tfrecord
contents are bit-for-bit identical to the normal pipeline — only the QOS and the packing change.
No Python was modified for this.

## Estimated cost / wall clock

NERSC charges *actual elapsed* × nodes, not the requested cap, so the real cost tracks the
measured task times, not the generous walltimes above.

- grid:     2500 / 24 = **105 node jobs**; actual ~1h10-1h45 each  => ~130-185 node-hours
- fiducial: 1000 / 20 =  **50 node jobs**; actual ~35-82 min each  => ~30-70 node-hours
- merge:    2 single-node jobs (unmeasured)                        => a few node-hours
- **total charged ≈ 165-260 node-hours** (reserved caps would be ~740, but early-finishing jobs
  release the remainder). Wall clock: hours, not days, if a healthy fraction backfill concurrently.

## How to run

```bash
cd pipelines/v17/packed
DRY_RUN=1 ./submit.sh both     # inspect: prints every sbatch call, touches nothing
./submit.sh both               # or: grid | fiducial
```

A fresh run archives any stale `<job>_done.dat` (timestamped `.bak`) so the merge count and any
reruns start from a clean slate. Bookkeeping: each successful task appends its index to
`/pscratch/sd/a/athomsen/run_files/v17/esub_logs/<job>_done.dat`.

## If something fails

- A killed/failed packed element (OOM, walltime) exits non-zero, which — via `afterok` — keeps
  the merge blocked. Its tasks never reached `done.dat`.
- Rerun only the missing indices:
  ```bash
  ./submit.sh --rerun grid      # missing = all - done.dat; resubmits + a fresh merge
  ```
  A rerun overwrites any partial `.tfrecord` (`TFRecordWriter` truncates on open).
- Sanity check before trusting the merge:
  `wc -l /pscratch/sd/a/athomsen/run_files/v17/esub_logs/*_done.dat` must reach 2500 (grid) /
  1000 (fiducial). The merge needs the complete set — with files missing, `merge()` batches
  cosmologies misaligned and produces garbage Cls.

Nothing here has been executed — this is a prepared, decision-pending submission.
