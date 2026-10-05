# Shared packed regular-QOS infrastructure

Version-agnostic building blocks for running msfm postprocessing apps on `--qos=regular` (whole
CPU nodes), packing many tasks per node, each pinned to a NUMA domain, instead of esub's
`--qos=shared` jobarrays. Used by `../../v17/packed/{submit.sh,submit_single_mock.sh}` and
`../../v18/packed/{submit.sh,submit_obs.sh}` — those per-version scripts hold the actual chain/arm
configuration (paths, `N_TOTAL`, per-chain `SLOTS`/`OMP`) and just point here for the executors.
See each version's own `packed/README.md` for its measured numbers, cost table and how-to-run
examples.

## Why not the default esub/shared chain

`--qos=shared` maps to the 70-node `shared_milan_ss11` pool, which is chronically saturated by
other users' multi-day jobs — sometimes only a handful of tasks run at a time (ETA can be weeks).
The `regular_milan_ss11` pool is ~2850 nodes with idle capacity, and short single-node jobs
backfill quickly. But regular QOS is **node-exclusive**, so a plain jobarray would waste a
128-core node on one small task. We therefore hand-pack many tasks onto each node.

## Why no esub

esub earns its keep on a farm of *independent* shared-QOS jobarray tasks: it maintains a
`done.dat` of finished indices and offers `rerun_missing`. On packed full nodes that buys
nothing — the natural unit of completion is the app's own output file(s), so each `submit.sh`/
`submit_*.sh` tracks that itself: `--rerun` checks `--dir_out` and resubmits any index whose
output is absent or a short/partial stub. `run_packed.py` calls the app's `main`/`merge` with the
exact arg list esub would forward, so outputs are bit-for-bit identical to the esub pipeline. No
Python in `msfm` was modified.

## Why pinning is mandatory (not just a NUMA nicety)

**Each task must be pinned to one NUMA domain with `numactl` — measured ~4x slower without it.**
Two independent reasons:

1. **Thread-pool explosion.** `OMP_NUM_THREADS` bounds OpenMP/BLAS but **not** TensorFlow's
   intra/inter-op pools, which auto-size to the *visible CPU count*. Unpinned, every task sees all
   256 CPUs and spawns hundreds of threads; with 24+ tasks that's tens of thousands of threads and
   massive oversubscription. `numactl --cpunodebind` confines each task to a 32-CPU domain, so the
   pools size to 32, and `packed_node.slurm` additionally sets `TF_NUM_INTRAOP_THREADS` explicitly
   as a belt-and-suspenders cap.
2. **Memory-bandwidth locality.** The dominant cost in every one of these apps is healpy's
   libsharp SHTs (`map2alm`/`alm2map`), which are OpenMP-parallel and **memory-bandwidth heavy** —
   they scale well up to a handful of threads, then hit a bandwidth knee. `--preferred=<domain>`
   keeps each task's memory on its local controllers, so the 8 domains' controllers are used
   evenly and accesses don't cross the die.

## How the packing is sized — pack to the bandwidth knee, not the RAM ceiling

Each `regular` node is **8 NUMA domains × 16 cores (32 HW threads) × ~64 GB** (2× EPYC 7763,
~503 GB usable). Keep **`SLOTS` a multiple of 8** so `SLOTS/8` tasks land on each domain evenly,
and set `OMP ≈ 32 / (SLOTS/8)` (rounded down) to split each domain's 32 threads across its tasks.
If the real task count for a run isn't itself a multiple of 8, size `SLOTS` (and hence `OMP`) to
the *worst-case* per-domain count (`ceil(n_tasks / 8)`), not the average — `packed_node.slurm`
assigns domains round-robin (`position % 8`), so some domains can get one more task than others.

The per-domain RAM is only an *upper* bound on `SLOTS`, and it is **not** the binding constraint
for the SHT-heavy chains measured so far: the SHTs are memory-**bandwidth** heavy, and a domain's
controllers saturate at **~3 tasks**, past which extra tasks/domain raise the node time faster
than they cut the node count — so node-hours *increase*. The measured knee for the grid/fiducial
tfrecord chains is **3 tasks/domain → `SLOTS=24`, `OMP=10`** (a ~20% node-hour saving over
4/domain); this same 24/10 pairing also happens to be the correct worst-case sizing for smaller,
`N_TOTAL<24` runs (e.g. the single-postprocessing mock/benchmark arms), since `ceil(20/8)=3` lands
on the exact same knee. Don't assume it transfers to a workload that hasn't been profiled — treat
24/10 as the current default, not a law of physics, and remeasure if a new chain's per-task
memory/compute profile looks meaningfully different.

## Files

- `run_packed.py` — ~30-line driver: imports the target app by path and calls its `main`/`merge`
  directly, forwarding all other CLI args verbatim.
- `packed_node.slurm` — generic executor: runs `$SLOTS` `numactl`-pinned `run_packed.py` tasks in
  parallel, `SLOTS/8` per NUMA domain. All config arrives via `--export=ALL`; the caller's
  `sbatch --output=...` overrides this file's fallback `#SBATCH --output`.
- `merge_node.slurm` — single-node `run_packed.py --function=merge`. Reads `TASKS` (default `"0"`,
  a dummy value the grid/fiducial apps' glob-based `merge()` ignores); callers whose `merge()`
  actually needs the index list (`run_single_postprocessing.py`) must `export TASKS=<full list>`
  before submitting.
