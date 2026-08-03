# Packed regular-QOS run of the v18 baseline postprocessing (no esub)

Clean-start submission of the v18 baseline dataset (grid + fiducial) on `--qos=regular`
(whole CPU nodes), packing many postprocessing tasks per node, each pinned to a NUMA domain.
This is the simplified sibling of `../packed_esub`: the per-task call goes through a ~30-line
`run_packed.py` instead of `esub --mode=run`.

## Why not the default esub/shared chain

`--qos=shared` maps to the 70-node `shared_milan_ss11` pool, which is chronically saturated by
other users' multi-day jobs — only ~5 of our tasks run at a time (ETA weeks). The
`regular_milan_ss11` pool is ~2850 nodes with idle capacity, and short single-node jobs backfill
quickly. But regular QOS is **node-exclusive**, so a plain jobarray would waste a 128-core node
on one ~14 GB task. We therefore hand-pack many tasks onto each node.

## Why no esub

esub earns its keep on a farm of *independent* shared-QOS jobarray tasks: it maintains a
`done.dat` of finished indices and offers `rerun_missing`. On packed full nodes that buys
nothing — the natural unit of completion is the output `.tfrecord` file, so we track it there:
`./submit.sh --rerun` globs `--dir_out` and resubmits any index whose file is absent. A
failed/killed task leaves either no file or a partial one (`TFRecordWriter` truncates on open, so
a partial can only exist for a task interrupted mid-write); see *If something fails* for how to
handle a partial. `run_packed.py` calls the app's `main`/`merge` with the exact arg list esub
forwards, so tfrecord contents are bit-for-bit identical. No Python in `msfm` was modified.

## Why pinning is mandatory (not just a NUMA nicety)

**Each task must be pinned to one NUMA domain with `numactl` — the run is ~4x slower without
it.** Two independent reasons, both measured on real nodes:

1. **Thread-pool explosion.** `OMP_NUM_THREADS` bounds OpenMP/BLAS but **not** TensorFlow's
   intra/inter-op pools (used by `TFRecordWriter` and the verify-parse), which auto-size to the
   *visible CPU count*. Unpinned, every task sees all 256 CPUs and spawned **~536 threads**; with
   24 tasks that is ~12.9k threads and a load average of **~750** on a 256-thread node — ~3x
   oversubscription, constant context-switching. `numactl --cpunodebind` confines each task to a
   32-CPU domain, so the pools size to 32, and we additionally set `TF_NUM_INTRAOP_THREADS`
   explicitly as a belt-and-suspenders cap.
2. **Memory-bandwidth locality.** The dominant cost is healpy's libsharp SHTs (`map2alm`/
   `alm2map`), which are OpenMP-parallel and **memory-bandwidth heavy** — they scale ~linearly to
   ~8 threads (measured 6.4x from 1→8), then hit a bandwidth knee. `--preferred=<domain>` keeps
   each task's memory on its local controllers, so the 8 domains' controllers are used evenly and
   accesses don't cross the die.

## How the packing is sized — pack to the bandwidth knee, not the RAM ceiling

Each `regular` node is **8 NUMA domains × 16 cores (32 HW threads) × ~64 GB** (2× EPYC 7763,
~503 GB usable). We keep **SLOTS a multiple of 8** so `SLOTS/8` tasks land on each domain and set
`OMP ≈ 32 / (SLOTS/8)` (rounded down) to split each domain's 32 threads across its tasks.

The per-domain 64 GB is only an *upper* bound on SLOTS, and it is **not** the binding constraint:
the SHTs are memory-**bandwidth** heavy, and a domain's controllers saturate at **~3 tasks**, past
which extra tasks/domain raise the node time faster than they cut the node count — so node-hours
*increase*. The right `SLOTS` is the bandwidth-saturation knee (measured **3 tasks/domain → 24
slots** for both chains), not "fill the RAM". Both chains fit 4/domain on memory but are cheaper at
3 (see the benchmark below).

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

- `submit.sh` — control script / single source of truth. Builds the index list, submits the
  packed array, then the merge (afterok). Exports an absolute `DRIVER` path (the executors run
  from Slurm's spool dir and cannot locate the driver themselves).
- `run_packed.py` — ~30-line driver: imports the app by path and calls `main`/`merge` directly.
- `packed_node.slurm` — generic executor: runs `$SLOTS` `numactl`-pinned `run_packed.py` tasks in
  parallel, `SLOTS/8` per NUMA domain. All config arrives via `--export=ALL`.
- `merge_node.slurm` — single-node `run_packed.py --function=merge`; globs the `.tfrecord` files
  and writes `{grid,fiducial}_cls.h5`.

## Estimated cost / wall clock (at the current 24/10 default)

NERSC charges *actual elapsed* × nodes, not the requested cap. Node times below are the contended
measurements; an uncontended run (fewer nodes in flight) is faster.

- **grid**: 2500 / 24 = **105 nodes** × ~85 min ⇒ **~150 node-hours**.
- **fiducial**: 1000 / 24 = **42 nodes** × 37.5 min ⇒ **~26 node-hours**.
- **merge**: 2 single-node jobs (grid ~2 h 47, fiducial ~32 min) ⇒ a few node-hours.
- Wall clock per chain: ~1.5–3 h depending on how fast the nodes backfill.

## How to run

```bash
cd pipelines/v18/packed
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
  `sbatch --dependency=afterok:<id1>:<id2> --time=<MERGE_TIME> --job-name=<chain>_merge --export=ALL merge_node.slurm`.

## If something fails

- A killed/failed packed element (OOM, walltime, task error) exits non-zero, which — via `afterok`
  — keeps the merge blocked. Delete any 0-byte or short stub the failed task left, then rerun only
  the missing indices:
  ```bash
  ./submit.sh --rerun grid      # missing = all indices with no .tfrecord; resubmits + a fresh merge
  ```
- Sanity check the output count before merging — normally 2500 (grid) / 1000 (fiducial):
  ```bash
  ls /pscratch/sd/a/athomsen/v11desy3/v18/baseline/tfrecords/grid/*_grid_*.tfrecord | wc -l
  ```
- **A missing file is tolerated by the fiducial `merge()`** (verified): it globs whatever is
  present, sorts records by their stored `i_signal`, and the extra `i_noise` tiling entries are
  never indexed — so merging N<1000 files yields a correct, self-labelled subset (that permutation
  is simply absent from the output). If the array already finished, submit the merge directly (no
  `afterok`):
  ```bash
  # export APP / DRIVER / JOB_NAME / APP_ARGS as submit.sh does, then:
  sbatch --time=04:00:00 --job-name=tfr_fidu_v18_merge --export=ALL merge_node.slurm
  ```
  This was needed once for the v17 run: input map `cosmo_delta_bary_nu_p/perm_0209` was corrupt
  upstream (the only bad file of 67,260), so fiducial index 209 could not complete and the set was
  merged from 999 files. (The grid `merge()` was not re-verified for gaps; grid inputs were all
  intact, so grid expects the full 2500.)
