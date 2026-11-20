> **DEPRECATED — never used in production.** Grid postprocessing runs on Perlmutter
> (`../../v18/packed/`), not Clariden; this port was written and sized but never adopted. The
> esub-free sibling `../v18_packed_clariden/` superseded it even within the Clariden experiment.
> Paths below still name its old location `pipelines/v18/packed_clariden_esub/`. Kept for
> historical reference only -- do not use for new runs.

# Packed whole-node v18 grid postprocessing for Clariden (CSCS)

Clariden analogue of `pipelines/v18/packed/` (which was the Perlmutter regular-QOS fallback).

## Why packing is required here

Clariden has **no shared QOS** — every partition is `OverSubscribe=EXCLUSIVE`, so any allocation
is a whole **Grace-Hopper node (288 ARM cores, 1 thread/core, ~870 GB)**. The Perlmutter pipeline
([../pipe.yaml](../pipe.yaml)) relied on `--qos=shared` to pack ~32 single-cosmology `esub`
jobarray elements onto one node at 8 cores each. That is impossible on Clariden: an esub jobarray
element cannot share a node, so a naive `--n_files=2500` jobarray would give **each cosmology its
own 288-core node using only 8 cores (~36× waste, ≈1500 node-h)**.

Instead, each array element owns one exclusive node and runs `SLOTS` cosmologies **concurrently**
via `esub --mode=run` (local execution, no SLURM submission, `resources()` never consulted).

## Node budget

Measured (v17, `cosmo_000001`, 8 threads): **~0.59 h/cosmo** (~1072 s fixed projection + 80 ×
~13.2 s signal SHTs) and **~15 GB peak/cosmo**.

    SLOTS * OMP <= 288 cores        SLOTS * ~15 GB <= ~870 GB
    default: SLOTS=36, OMP=8  ->  288 threads, ~540 GB  ->  ~36 cosmos / ~0.6 h / node

More concurrency with fewer threads (e.g. `SLOTS=48 OMP=6`) squeezes ~10–15 % more throughput but
uses ~720 GB — tighter, and only worth it if the fixed (weakly-threaded) projection dominates.

## Total cost (2500 cosmologies)

~2500 × 0.59 h / 36 per node ≈ **~41 node-hours** (≈45–50 with overhead). On the 12 h `normal`
partition: 4 nodes ≈ 10 h wall, 8 nodes ≈ 5–6 h.

## How to run

```bash
cd pipelines/v18/packed_clariden

# main postprocessing: ceil(2500 / SLOTS) array elements (2500/36 -> 70 -> 0..69)
sbatch --array=0-69 packed_node.slurm
# retune (fewer, fuller nodes): SLOTS=48 OMP=6 -> ceil(2500/48)=53 -> 0..52
sbatch --array=0-52 --export=ALL,SLOTS=48,OMP=6 packed_node.slurm

# after ALL elements succeed, produce grid_cls.h5 (single-process merge in the container):
srun -A a0158 -p normal -N1 --ntasks=1 --time=00:30:00 --environment=tensorflow bash -c '
  source ~/dlss/tf_env/bin/activate
  esub /users/athomsen/dlss/repos/multiprobe-simulation-forward-model/msfm/apps/run_grid_postprocessing.py \
    --n_files=2500 --cluster=clariden \
    --config=/users/athomsen/dlss/repos/multiprobe-simulation-forward-model/configs/v18/default.yaml \
    --dir_out=/iopsstor/scratch/cscs/athomsen/v11desy3/v18/baseline/tfrecords/grid \
    --mode=run --function=merge --tasks=0 \
    --job_name=tfr_grid_v18 --log_dir=/iopsstor/scratch/cscs/athomsen/run_files/v18/esub_logs'
```

## Notes / prerequisites

- **Input data**: expects the CosmoGrid projections at
  `$DIR_IN/grid/cosmo_XXXXXX/perm_YYYY/projected_probes_maps_v11dmb.h5` with
  `DIR_IN=/iopsstor/scratch/cscs/athomsen/v11desy3`. Only `cosmo_000001` is copied so far —
  copy the rest before a full run. (Perlmutter read these straight from CFS; Clariden has no
  equivalent mount, so they must be staged onto `/iopsstor` scratch.)
- **Bookkeeping**: finished indices are appended to `<log_dir>/tfr_grid_v18_done.dat`. To rerun
  only missing indices, submit a small array whose elements cover `{0..2499} \ done`.
- **Merge completeness**: `merge()` batches cosmologies by file order — it requires the complete
  set of 2500 `.tfrecord` files, or it silently misaligns and produces garbage Cls. Check
  `wc -l tfr_grid_v18_done.dat` == 2500 before merging.
- `esub` is on `PATH` after `source ~/dlss/tf_env/bin/activate` inside the container.
- Paths (`REPO`, `DIR_IN`, `DIR_OUT`, `LOG_DIR`) are set at the top of `packed_node.slurm`.
