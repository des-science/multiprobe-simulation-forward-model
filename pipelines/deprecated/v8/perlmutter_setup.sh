# esub
export CORES_PER_NODE=128
export MEM_PER_NODE=512000
export HYPER_THREADS=2

# OpenMP
export OMP_NUM_THREADS=$HYPER_THREADS
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

echo "Perlmutter setup complete"