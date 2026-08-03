# esub
export CORES_PER_NODE=256 # because of hyperthreading
export MEM_PER_NODE=499712

# OpenMP
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

if [[ -n "${SLURM_CPUS_PER_TASK}" ]]; then
    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
    export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
    export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
    export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK}
    export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
    echo "the number of threads is set to ${SLURM_CPUS_PER_TASK}"
else
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export TF_NUM_INTRAOP_THREADS=1
    export TF_NUM_INTEROP_THREADS=1
    echo "!!! the number of threads is set to 1 !!!"
fi

echo "Perlmutter setup complete"
