# daint (CSCS Alps)

Scripts for daint's GH200 nodes. msfm uses only their CPUs. A node is 4 Grace CPUs × 72 cores with
476 GB of host DRAM in total (~119 GB per Grace NUMA domain). SLURM and `free` report more because
they count the GPUs' HBM, so budget memory against 476 GB.

- **Account** `lp184`. Partitions: `normal` (24 h, the default) and `debug` (30 min, one job per user
  at a time, for smoke tests).
- **Allocations are whole exclusive nodes**, billed per node-hour. There is no shared queue, so
  anything smaller than a node is packed onto one by hand.
- **Environments.** The Y6 product steps run in `torch_env` inside the `pytorch/v2.9.1:v2` uenv
  (healsparse lives there). The tfrecord postprocessing runs in the `tensorflow` EDF container with
  `tf_env`. That container mounts `/iopsstor` and `/ritom` scratch, `$HOME/dlss`, and the a0158 and
  lp184 stores (`~/.edf/tensorflow.toml`).
- **Sub-node `srun` steps** need `--cpu-bind=none` on Alps. With SLURM's binding, the step is
  CANCELLED within seconds, its log stays empty, and the error appears only in `slurm-<jid>.out`.

| path | what |
|---|---|
| `common/packed_node.slurm` | generic packed executor: one srun step of `SLOTS` tasks in the tensorflow container, `--cpu-bind=none`; `FUNCTION=merge` runs the merge on the whole node |
| `common/packed_task.sh` | one task: takes its index from `INDEX_FILE`, pins itself with `taskset` to `OMP` cores inside one Grace, calls `../common/run_packed.py` |
| `desy6/products/run.sh <pixels\|maps\|noise>` | one compute-node step of the Y6 data products (`msfm.apps.products.desy6`), after its `--dry-run` on the login node |
| `desy6/v1/submit.sh grid` | the Y6 v1 grid tfrecords: login-node preflight, the packed array, the merge; see `desy6/v1/README.md` |
