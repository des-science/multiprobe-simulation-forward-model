> **DEPRECATED — never used in production.** Grid postprocessing runs on Perlmutter
> (`../../v17/packed/`), not Clariden; this port was written and sized but never adopted. Paths
> below still name its old location `pipelines/v17/packed_clariden/`. Kept for historical
> reference only -- do not use for new runs.

# Packed whole-node v17 grid postprocessing for Clariden (CSCS) — esub-free

Simpler alternative to `../v17_packed_clariden_esub/`. Same idea (pack ~36 cosmologies onto each
exclusive Grace-Hopper node), but **without esub**: SLURM does the intra-node fan-out and CPU
pinning, and a tiny driver calls the app's `main()` directly.

## Why drop esub here

esub's value on Perlmutter was **per-core accounting + rerun-missing bookkeeping**: with
`--qos=shared` you paid only for the few cores a rerun of the missing cosmologies used. On Clariden
there is **no shared QOS** — every allocation is a whole node, billed per node — so re-running a
handful of cosmologies still costs a whole node. esub's fine-grained bookkeeping therefore buys
nothing, and we can replace `esub --mode=run` with a direct function call.

The app's Python logic is untouched. `run_grid_postprocessing.py` has no `__main__` guard and its
`main()`/`merge()` are generators that esub simply imports and drives — so `run_cosmo.py` does the
same in ~10 lines.

## How the parallelism works

```
sbatch --array=0-69                       70 exclusive nodes (one per array element)
   │
   └─ node k: srun --ntasks=36 --cpus-per-task=8 --environment=tensorflow run_slot.sh
                 ├─ PROCID 0  -> idx k*36+0   python run_cosmo.py --tasks=<idx>  (8 cores)
                 ├─ PROCID 1  -> idx k*36+1   ...
                 └─ PROCID 35 -> idx k*36+35
```

SLURM starts `SLOTS` tasks per node and pins `OMP` cores to each (`--cpus-per-task`), replacing the
manual `& ... wait` loop and the `_done.dat` file of the esub variant. Each task computes its own
cosmology index `idx = SLURM_ARRAY_TASK_ID * SLOTS + SLURM_PROCID`; tasks with `idx >= N_TOTAL`
no-op (only the last array element has any).

## Files

- `run_cosmo.py` — esub-free driver: imports the app, expands `--tasks`, passes all other flags
  through to the app's argparse, and exhausts `main()`/`merge()`.
- `run_slot.sh` — per-task body: sets threading, computes `idx`, runs one cosmology.
- `packed_node.slurm` — the SLURM array job (one exclusive node per element).

## Node budget

Measured (v17, `cosmo_000001`, 8 threads): **~0.59 h/cosmo**, **~15 GB peak/cosmo**.

    SLOTS * OMP <= 288 cores        SLOTS * ~15 GB <= ~870 GB
    default: SLOTS=36, OMP=8  ->  288 threads, ~540 GB  ->  ~36 cosmos / ~0.6 h / node

Total for 2500 cosmologies: ~2500 × 0.59 h / 36 ≈ **~41 node-hours** (≈45–50 with overhead).
On the 12 h `normal` partition: 4 nodes ≈ 10 h wall, 8 nodes ≈ 5–6 h.

## Run

```bash
cd pipelines/v17/packed_clariden

# main postprocessing: ceil(2500 / SLOTS) array elements (2500/36 -> 70 -> 0..69)
sbatch --array=0-69 packed_node.slurm

# retune (fewer, fuller nodes): ceil(2500/48)=53
sbatch --array=0-52 --export=ALL,SLOTS=48,OMP=6 packed_node.slurm

# after ALL indices succeed, produce grid_cls.h5 (single process, one node):
srun -A a0158 -p normal -N1 --ntasks=1 --time=00:30:00 --environment=tensorflow bash -c '
  source ~/dlss/tf_env/bin/activate
  python -u /users/athomsen/dlss/repos/multiprobe-simulation-forward-model/pipelines/v17/packed_clariden/run_cosmo.py \
    --tasks=0 --function=merge \
    --n_files=2500 --cluster=clariden \
    --config=/users/athomsen/dlss/repos/multiprobe-simulation-forward-model/configs/v17/baseline.yaml \
    --dir_in=/iopsstor/scratch/cscs/athomsen/v11desy3 \
    --dir_out=/iopsstor/scratch/cscs/athomsen/v11desy3/v17/baseline/tfrecords/grid'
```

## Reruns

Because the index→node mapping is deterministic and contiguous, a failed node's block is known:
resubmit just that array element, e.g. `sbatch --array=12 packed_node.slurm`. `TFRecordWriter`
overwrites, so re-running a completed index is safe (idempotent). To find gaps, list the written
`.tfrecord`s in `$DIR_OUT` and compare against `0..2499`.

## Prerequisites / notes

- **Input data**: expects `$DIR_IN/grid/cosmo_XXXXXX/perm_YYYY/projected_probes_maps_v11dmb.h5`
  with `DIR_IN=/iopsstor/scratch/cscs/athomsen/v11desy3`. Only `cosmo_000001` is staged so far —
  copy the rest onto `/iopsstor` scratch before a full run (Clariden has no CFS/SAN mount).
- **Merge completeness**: `merge()` batches cosmologies by sorted file order and needs the full set
  of 2500 `.tfrecord` files, or it silently misaligns and produces garbage Cls. Confirm all 2500
  outputs exist before merging.
- Per-index logs land in `$LOG_DIR/index<idx>.log`; the node-level srun log is
  `$LOG_DIR/tfr_grid_v17_<jobid>_<element>.out`.
