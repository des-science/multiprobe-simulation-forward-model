#!/bin/bash
# Per-slot task body. srun launches this once per task (SLURM_PROCID = 0..SLOTS-1) inside the
# tensorflow container, with SLURM pinning OMP cores to each task. Each task processes exactly
# ONE cosmology:  idx = array_element * SLOTS + SLURM_PROCID.  Tasks whose idx >= N_TOTAL no-op.
set -uo pipefail

source ~/dlss/tf_env/bin/activate

idx=$(( SLURM_ARRAY_TASK_ID * SLOTS + SLURM_PROCID ))
if [ "$idx" -ge "$N_TOTAL" ]; then
    echo "slot $SLURM_PROCID -> idx $idx >= N_TOTAL=$N_TOTAL, nothing to do"
    exit 0
fi

# threading for this cosmology; srun already gave us OMP cpus (--cpus-per-task=OMP)
export OMP_NUM_THREADS=$OMP OPENBLAS_NUM_THREADS=$OMP MKL_NUM_THREADS=$OMP \
       VECLIB_MAXIMUM_THREADS=$OMP NUMEXPR_NUM_THREADS=$OMP
export OMP_PLACES=threads OMP_PROC_BIND=false

python -u "$REPO/pipelines/v17/packed_clariden/run_cosmo.py" \
    --tasks="$idx" --function=main \
    --n_files="$N_TOTAL" \
    --config="$REPO/configs/v17/baseline.yaml" \
    --dir_in="$DIR_IN" \
    --dir_out="$DIR_OUT" \
    --cosmogrid_version=1.1 \
    --cluster=clariden \
    > "$LOG_DIR/index${idx}.log" 2>&1
