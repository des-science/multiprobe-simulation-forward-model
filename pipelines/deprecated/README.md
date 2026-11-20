# Deprecated pipelines

Retired pipeline generations and superseded submission methods, sorted by the cluster they were
written for: `deprecated/<cluster>/<survey>/<version>/`. Kept for historical reference only; nothing
here should be used for new runs. See `../README.md` for the active scripts.

**Paths inside these files are stale.** They name the locations the scripts had when they were
retired (`pipelines/vNN/...`, before the per-survey move of 2026-11-18 and the per-cluster move of
2026-11-20), and none were updated for either move.

## Euler (ETH, LSF)

- **`euler/desy3/v7`** — earliest pipeline in this tree (`bsub`-style `esub`).

## Perlmutter (NERSC)

- **`perlmutter/desy3/marcel`** — one-off pipeline for a collaborator's run, not part of the numbered
  version line.
- **`perlmutter/desy3/v8`** — transitional Euler → Perlmutter port; includes `globus.sh` (data
  transfer) and an `esub_debug/` subdir, predating the later per-version directory conventions.
- **`perlmutter/desy3/v9`, `v10`** — first Perlmutter-native versions. Their `pipe.yaml` already has
  an "UNFAVORED postprocessing, `--qos=regular`" block using esub's own
  `--max_nnodes`/`--per_node_accounting` flags — the direct precursor of the later hand-rolled
  `packed/` pattern. Superseded by v11+.
- **`perlmutter/desy3/v11`–`v15`** — flat single-`pipe.yaml` versions, no named subdirectories yet.
  Superseded by v16's `rot_in_place` shape-noise model and later versions.
- **`perlmutter/desy3/v16`** — introduced the `rot_in_place` shape-noise variant, the `m5030`
  account, and the first named-subdirectory convention (`rot_in_place/obs_commands.sh`). v17 reuses
  v16's fiducial tfrecords/obs products directly (bit-identical at `bta=0`) rather than regenerating
  them.
- **`perlmutter/desy3/v17_packed_esub/`** — the first regular-QOS "packed" attempt for v17, routing
  each packed task through `esub --mode=run` rather than calling the app directly. Superseded by
  `../perlmutter/desy3/v17/packed/` (adds mandatory NUMA pinning, discovered afterward to matter
  ~4x, plus bandwidth-knee sizing) and by the Perlmutter executors in `../perlmutter/common/`.
- **`perlmutter/desy3/v18_esub/`** — v18's esub/shared-QOS definitions (`pipe.yaml`,
  `obs_commands.sh`), retired when v18 moved to packed-only submission. Every job they defined has a
  packed equivalent in `../perlmutter/desy3/v18/packed/`.

## Clariden (CSCS) — an abandoned port of the Y3 grid postprocessing

The DES Y3 grid postprocessing ran on Perlmutter. The four `clariden/desy3/v1N_packed_clariden*`
directories are a port to Clariden (CSCS Grace-Hopper) that was written and sized, then never
adopted: no production grid was produced with them. They are kept only for their Clariden-specific
reasoning (no shared QOS, so every allocation is a whole 288-core node, which forces many
cosmologies to be hand-packed per node).

**Their sizing is wrong**: 36 tasks × ~15 GB = 540 GB assumed the ~870 GB that SLURM and `free`
report, which counts the GPUs' HBM. A GH200 node has 476 GB of host DRAM. The daint pipeline
(`../daint/`) replaces this port for DES Y6.

- **`clariden/desy3/v17_packed_clariden_esub/`, `v18_packed_clariden_esub/`** — first attempt,
  routing each packed task through `esub --mode=run` plus a `_done.dat` barrier.
- **`clariden/desy3/v17_packed_clariden/`, `v18_packed_clariden/`** — esub-free rewrite of the
  above: SLURM does the intra-node fan-out and core pinning (`--ntasks=36 --cpus-per-task=8`) and
  `run_cosmo.py` calls the app's `main()` directly. `run_cosmo.py` is identical between v17 and v18;
  the READMEs, `packed_node.slurm` and `run_slot.sh` differ only in version paths and config.
