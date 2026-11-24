# DES Y6 tfrecords on daint

`submit.sh` produces the training set of one DES Y6 version: `VERSION=desy6/v2` runs
`configs/desy6/v2/v2.yaml` through `msfm/apps/run_grid_postprocessing.py`, one `.tfrecord` per grid
cosmology (2500 files, 20 perms × 4 patches × 5 noise realizations each). `VERSION` has no default,
as in the y3-deep-lss and msi scripts, so a run cannot land in another version's folder. The script
is a port of `../../perlmutter/desy3/v18/packed/submit.sh` and keeps its interface. The executor is
`../common/packed_node.slurm` + `packed_task.sh`, which calls the app through
`../../common/run_packed.py`.

## Before a production run

1. The projection is complete: all 50,000 files are the nside-512 re-projection with `maglim1-6`
   (done 2026-11-20). The preflight checks only that the files exist.
2. Step `sc_bias` has written `data/desy6/desy6_metacal_bias.h5` (done 2026-11-20). The grid app reads
   it at startup, so the preflight refuses to submit without it.
3. Every product step has rerun with `--provenance` on committed code (done 2026-11-21), and step
   `values --check` passes on the version's config (`data/desy6/README.md`).
4. `DRY_RUN=1 ./submit.sh grid` passes.

## How to run

```bash
cd pipelines/daint/desy6
export VERSION=desy6/v2
DRY_RUN=1 ./submit.sh grid      # preflight + the exact sbatch lines, submits nothing
./submit.sh grid                # 105 node jobs (25 at a time), then the merge (afterok)
./submit.sh --rerun grid        # only the indices whose .tfrecord is missing or a stub
```

The input projection and the output root come from `configs/$VERSION/locations.yaml`
(`msfm/utils/locations.py`): Tomasz's projection on the lp184 store, and for v2
`/ritom/scratch/cscs/athomsen/desy6/v2` (v1: `.../desy6/v1/default`), under which the script writes
`tfrecords/grid`, `cls`, `obs`, `validation` and `packed_logs`. Every path, the packing (`SLOTS`,
`OMP`), the wall clocks and the partition can be overridden from the environment (`DIR_IN` and
`DATA_ROOT` for the two roots; see the header of `submit.sh`). `ARRAY=<spec>` submits only some array elements; element k
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

Measured for v1 on 2026-11-20 (jobs 5408848, 5408849). Each arm ran one node on `normal`, one full
cosmology per task on indices 0–23 of the re-projected files, with a stand-in `sc_bias` table (b = 1,
which does not change the cost):

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

**The v1 production run (2026-11-21) confirmed the benchmark.** The array (job 5416078) had 105
elements of 41:49–45:06 each, 77.2 node-hours in total. The merge (job 5416079) took 3:41 on one
node. Including the merge, the grid cost ≈ 81 node-hours and took about 8 h of wall time.
`.tfrecord` sizes come in three values, 80 bytes apart. That is one byte per record per step, most
likely a varint-encoded index growing with the cosmology.

## Mock observations

`./submit.sh obs <label>|all` makes the mock observations the networks are tested on, one
`<label>_obs_maps.h5` each in the `obs/` of the output root. Each mock runs
`msfm/apps/run_single_postprocessing.py` over the 20 permutations of one CosmoGrid cosmology (4 patches
each) on one node, followed by a merge.

Every mock is defined in `configs/$VERSION/mock_observations.yaml` and nowhere else: its input below
the projection root, the keys of the version's config it changes, and its forward-model flags (format
in `msfm/utils/mocks.py`). A new mock is a new entry there; this script reads only the labels.

```bash
cd pipelines/daint/desy6
export VERSION=desy6/v2
DRY_RUN=1 ./submit.sh obs all                   # checks every mock, prints the sbatch lines
PARTITION=debug ./submit.sh obs fiducial_bench  # one mock
./submit.sh obs all                             # one array and one merge per mock
```

The preflight runs `python -m msfm.utils.mocks check` on every selected mock before submitting any. It
checks the config the way the pipelines do, the files the config names, every permutation's maps file
and the bias-table key; the benchmarks read `fiducial`, fit on `fiducial_bench`. The merged file stores
the mock's label, input, flags and resolved config as attributes. A merged mock counts as complete for
`--rerun`, since the merge deletes the per-permutation files.

The six v1 mocks are `fiducial_bench`, `box_size`, `particle_count`, `redshift_resolution`, and the
source-clustering arms `fiducial_bench_source_clustering_in_place` and `..._no_sys`. Y3's
dark-matter-only, Gatti, IA-shell and Buzzard mocks have no Y6 input.

All six ran on 2026-11-22 (`fiducial_bench`: jobs 5424945/5424963 on `debug`; the other five: arrays
5425225-5425233 on `normal`). Each array took 3:33-3:43 on one node and each merge about 25 s; every
file is 1.3 GB, 80 examples (20 permutations × 4 patches) of 9 channels on the 309,226 footprint
pixels. Every value is finite, no example is empty or a duplicate, and each stored config equals the
registry's. Against `fiducial_bench`, the three simulation benchmarks differ in every footprint pixel
of both probes, while the two source-clustering arms differ in lensing only: their clustering
channels are identical, since the noise seeds are shared and the shape-noise model touches only the
sources.

The six files are copied to the lp184 store, since `/ritom` is purged:
`/capstor/store/cscs/userlab/lp184/athomsen/deep_lss/data/desy6/v1/default/obs` (2026-11-22,
byte-identical, group `lp184`), the `archive` of `configs/desy6/v1/locations.yaml`. y3-deep-lss reads
mocks from the `obs_dir` of that file, the `/ritom` copy; `OBS_DIR=<directory of the h5 files>`
points one evaluation at the store copy instead.

The debug QOS takes two submitted jobs per user and runs one, so a mock there needs both slots free;
several mocks go to `normal`.

## Validation

The checks live in `dev/scripts/validation/`, and each script's docstring lists what it tests:

- `grid_tfrecords.py` checks the `.tfrecords` against the config, the metainfo, the projection and
  `grid_cls.h5`, file by file and across files. It runs through the same executor as the grid.
- `data_vs_mock.py` compares the observation with the forward model: one mock, the grid's noise and
  the grid's Cl range.

Run them in this order, here for v1, outputs under `/ritom/.../desy6/v1/default/validation`:

```bash
cd pipelines/daint/desy6
export VERSION=desy6/v1
# 1. reproduction: indices 0-23 rerun with --debug at HEAD, records 0-4 each (perms 0 and 1).
#    JOB_NAME keeps its logs and index file apart from the production ones
JOB_NAME=tfr_reproduce_desy6_v1 DIR_OUT=/ritom/scratch/cscs/athomsen/desy6/v1/default/validation/reproduce \
    EXTRA_ARGS=--debug PARTITION=debug NODE_TIME=00:30:00 ARRAY=0 SKIP_MERGE=1 ./submit.sh grid
# 2. the scan: one node first (72 files), to measure the I/O rate, then the rest and the merge.
#    The merge compares the rerun of step 1 if validation/reproduce/debug exists at submission
ARRAY=0 SKIP_MERGE=1 ./submit.sh validate
./submit.sh --rerun validate
# 3. the observation against one mock and the whole grid, on a compute node
python dev/scripts/validation/data_vs_mock.py --config configs/desy6/v1/v1.yaml \
    --tfrecord <a grid .tfrecord> --out <validation>/data_vs_mock \
    --scan <validation>/scan.h5 --grid_cls <...>/cls/grid_cls.h5
```

The merge writes `report.txt` and exits non-zero on any FAIL, which marks the merge job as failed.
