# Packed regular-QOS fallback for the v17 baseline postprocessing

Prepared 2026-07-12. The v17 grid (esub chain 55803045+) and fiducial (55803410+) jobarrays run
on `--qos=shared`, which maps to the 70-node `shared_milan_ss11` pool. That pool is currently
saturated by other users' 1-2 day jobs, so only ~5 of our ~1h tasks run concurrently
(ETA weeks). For contrast, the v16 grid ran through an empty shared pool in 10.5 h on 2026-05-06.
Nothing here has been executed — it is a prepared switch, decision pending.

## What switching changes

- **QOS/partition**: `--qos=shared` (8/16-core slices, currently starved) -> `--qos=regular`
  (whole CPU nodes, large pool, short jobs backfill well).
- **Packing**: regular QOS is node-exclusive, so 8-core tasks must be packed by hand; esub's
  jobarray mode cannot run tasks in parallel within one job. `packed_node.slurm` runs 16
  `esub --mode=run --function=main --tasks=<idx>` processes per node (measured per task:
  grid ~15 GB / ~1h10, fiducial ~12-18 GB / ~30-40 min; 16 x 16 hyperthreads = 256 HT,
  16 x ~15 GB = ~240 of 512 GB).
- **Bookkeeping**: unchanged. The packed runs append finished indices to the same esub
  `<job>_done.dat`, and the original esub *merge* jobs are kept and re-pointed
  (`scontrol update ... Dependency=afterok:<packed array>`), so `grid_cls.h5` /
  `fiducial_cls.h5` still get produced automatically. Only *rerun_missing* is cancelled
  (it would resubmit everything back to shared).

## Cost / duration estimate (at preparation time)

- grid: ~2470 remaining / 16 per node = ~155 node jobs x ~1.5 h  = ~230 CPU-node-hours
- fiducial: ~970 remaining / 16 per node = ~61 node jobs x ~0.75 h = ~45 CPU-node-hours
- wall clock: plausibly < 1 day including queueing (1-2.5 h jobs backfill quickly)

## How to run

```bash
cd pipelines/v17/packed
DRY_RUN=1 ./switch_to_regular.sh both   # inspect: prints scancel/sbatch/scontrol, touches nothing
./switch_to_regular.sh both             # or: grid | fiducial
```

Running shared-QOS elements are left to drain (their indices are excluded from the packed
list), pending ones are cancelled. The merge waits on `afterok` of the packed array plus
`afterany` of the draining main array.

## If something fails

- A failed packed element prints `index <i> FAILED` in
  `/pscratch/sd/a/athomsen/run_files/v17/packed_logs/` and blocks the merge (afterok).
  Rerun the switch script for that chain — it re-derives the missing set from `_done.dat`
  and submits a new (small) packed array; then the merge dependency is updated again by step 5.
- Sanity check before/after: `wc -l /pscratch/sd/a/athomsen/run_files/v17/esub_logs/*_done.dat`
  should reach 2500 (grid) and 1000 (fiducial); the merge requires the complete file set —
  with files missing, `merge()` would batch cosmologies misaligned and produce garbage Cls.
