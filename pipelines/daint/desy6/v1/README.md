# DES Y6 v1 tfrecords on daint

`submit.sh` produces the Y6 v1 training set: `configs/desy6/v1/default.yaml` run through
`msfm/apps/run_grid_postprocessing.py`, one `.tfrecord` per grid cosmology (2500 files, 20 perms × 4
patches × 5 noise realizations each). It is a port of `../../../perlmutter/desy3/v18/packed/submit.sh`
and keeps its interface. The executor is `../../common/packed_node.slurm` + `packed_task.sh`, which
calls the app through `../../../common/run_packed.py`.

## Before the production run

1. The projection is complete and re-scanned: all 50,000 files are the nside-512 re-projection with
   `maglim1-6`. The preflight checks only that the files exist, and on 2026-11-20 all 50,000 did,
   while 785 still predated the re-projection.
2. Step `sc_bias` has written `data/desy6/desy6_metacal_bias.h5`. The grid app reads it at startup, so
   the preflight refuses to submit without it.
3. `DRY_RUN=1 ./submit.sh grid` passes.

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

(benchmark pending)
