# DES Y6 v1 tfrecords on daint

`submit.sh` produces the Y6 v1 training set: `configs/desy6/v1/default.yaml` run through
`msfm/apps/run_grid_postprocessing.py`, one `.tfrecord` per grid cosmology (2500 files, 20 perms × 4
patches × 5 noise realizations each). It is a port of `../../../perlmutter/desy3/v18/packed/submit.sh`
and keeps its interface. The executor is `../../common/packed_node.slurm` + `packed_task.sh`, which
calls the app through `../../../common/run_packed.py`.

## Before the production run

1. The projection is complete and re-scanned: all 50,000 files are the nside-512 re-projection with
   `maglim1-6`. The preflight checks only that the files exist. Done 2026-11-20 (`dev/notes/desy6/plan.md`,
   section 1).
2. Step `sc_bias` has written `data/desy6/desy6_metacal_bias.h5`. The grid app reads it at startup, so
   the preflight refuses to submit without it. Done 2026-11-20.
3. Every product step has rerun with `--provenance` on committed code. Done 2026-11-21.
4. `DRY_RUN=1 ./submit.sh grid` passes.

## How to run

```bash
cd pipelines/daint/desy6/v1
DRY_RUN=1 ./submit.sh grid      # preflight + the exact sbatch lines, submits nothing
./submit.sh grid                # 105 node jobs (25 at a time), then the merge (afterok)
./submit.sh --rerun grid        # only the indices whose .tfrecord is missing or a stub
```

Defaults: inputs from Tomasz's projection on the lp184 store, outputs and logs under
`/ritom/scratch/cscs/athomsen/desy6/v1/default/{tfrecords/grid,packed_logs}`. Every path, the
packing (`SLOTS`, `OMP`), the wall clocks and the partition can be overridden from the environment
(see the header of `submit.sh`). `ARRAY=<spec>` submits only some array elements; element k
processes indices `[k*SLOTS, (k+1)*SLOTS)`.

**A failed element** exits non-zero, which keeps the `afterok` merge blocked. Delete any short stub,
then `./submit.sh --rerun grid` submits the missing indices and a fresh merge. The merge globs every
`.tfrecord` in `DIR_OUT`, so it must see all 2500.

## Pinning

Each of the `SLOTS` tasks pins itself with `taskset` to `OMP` consecutive cores, and checks that they
lie in one Grace NUMA domain, so its memory stays local by first touch. `OMP` must divide 72.
SLURM's own binding is off (`--cpu-bind=none`), since sub-node steps with it fail on Alps.
`TF_NUM_INTRAOP_THREADS`, `OMP_NUM_THREADS` etc. are set to `OMP`, and `CUDA_VISIBLE_DEVICES` is
empty. The node log lists every task's core range.

## Sizing and cost

Measured on 2026-11-20 (jobs 5408848, 5408849). Each arm ran one node on `normal`, one full cosmology
per task on indices 0–23 of the re-projected files, with a stand-in `sc_bias` table (b = 1, which
does not change the cost):

| packing | per Grace | node time | per task | peak RSS per task | node-hours per cosmology | grid (2500) |
|---|---|---|---|---|---|---|
| **24 tasks × 12 cores** (default) | 6 | 44:53 | 40.7–42.8 min | 10.5 GB | **0.031** | 105 nodes → **≈ 79 node-hours** |
| 12 tasks × 24 cores | 3 | 34:04 | 31.2–32.1 min | 10.3 GB | 0.047 | 209 nodes → ≈ 119 node-hours |

- Denser packing still pays on Grace: doubling the tasks per domain made each task only 1.3× slower.
  Unlike Perlmutter, there was no bandwidth knee by 6 per domain.
- Memory would allow 8 per Grace (`SLOTS=32 OMP=9`, ~340 GB), but that packing is untested.
- At 24 × 12, a permutation takes ~64 s, of which ~49 s is the count-based shape noise. The per-signal
  loop then takes ~15 s per signal × 80.
- Each `.tfrecord` is 6.70 GB, so the grid is ≈ 16.8 TB.
- **Full run:** ≈ 79 node-hours for the grid. The merge has not been timed on daint (it took 2 h 47 on
  one Perlmutter node), so add ≈ 3 node-hours.
- **Wall clock:** about 4 h for the array at `ARRAY_THROTTLE=25` (five waves of ~45 min, if the queue
  keeps up), then the merge.
- **Wall-clock caps:** `NODE_TIME` defaults to 01:30:00, twice the measured node time. `MERGE_TIME`
  defaults to 06:00:00.

**The production run (2026-11-21) confirmed the benchmark.** The array (job 5416078) had 105
elements of 41:49–45:06 each, 77.2 node-hours in total. The merge (job 5416079) took 3:41 on one
node. Including the merge, the grid cost ≈ 81 node-hours and took about 8 h of wall time.
`.tfrecord` sizes come in three values, 80 bytes apart. That is one byte per record per step, most
likely a varint-encoded index growing with the cosmology.
