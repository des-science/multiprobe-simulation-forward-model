# Deprecated pipelines

Retired pipeline generations and superseded submission methods, kept for historical reference
only. Nothing here should be used for new runs — see `../README.md` for the active versions.

## Retired version generations

- **`marcel`** — one-off pipeline for a collaborator's run, not part of the numbered version line.
- **`v7`** — earliest pipeline in this tree; targeted Euler (`bsub`-style `esub`), not Perlmutter.
- **`v8`** — transitional Perlmutter port; includes `globus.sh` (data transfer) and an
  `esub_debug/` subdir, predating the later per-version directory conventions.
- **`v9`, `v10`** — first Perlmutter-native versions. Their `pipe.yaml` already has an "UNFAVORED
  postprocessing, `--qos=regular`" block using esub's own `--max_nnodes`/`--per_node_accounting`
  flags — the direct precursor of the later hand-rolled `packed/` pattern. Superseded by v11+.
- **`v11`–`v15`** — flat single-`pipe.yaml` versions, no named subdirectories yet. Superseded by
  v16's `rot_in_place` shape-noise model and later versions.
- **`v16`** — introduced the `rot_in_place` shape-noise variant, the `m5030` account, and the
  first named-subdirectory convention (`rot_in_place/obs_commands.sh`). v17 reuses v16's fiducial
  tfrecords/obs products directly (bit-identical at `bta=0`) rather than regenerating them.

## Superseded submission methods

- **`v17_packed_esub/`** — the first regular-QOS "packed" attempt for v17, routing each packed
  task through `esub --mode=run` rather than calling the app directly. Superseded by
  `../v17/packed/` (adds mandatory NUMA pinning, discovered afterward to matter ~4x, plus
  bandwidth-knee sizing) and by the shared `../common/packed/`.
- **`v18_esub/`** — v18's esub/shared-QOS definitions (`pipe.yaml`, `obs_commands.sh`), retired
  when v18 moved to packed-only submission. Every job they defined has a packed equivalent in
  `../v18/packed/`.

## Abandoned port: grid postprocessing on Clariden

Grid postprocessing runs on **Perlmutter**. The four `v1N_packed_clariden*` directories are a
port to Clariden (CSCS Grace-Hopper) that was written, sized against measured timings, and then
never adopted — no production grid was produced with them. They are kept only for the sizing
reasoning, which is genuinely Clariden-specific (no shared QOS, so every allocation is a whole
288-core node, forcing ~36 cosmologies to be hand-packed per node).

- **`v17_packed_clariden_esub/`, `v18_packed_clariden_esub/`** — first attempt, routing each
  packed task through `esub --mode=run` plus a `_done.dat` barrier.
- **`v17_packed_clariden/`, `v18_packed_clariden/`** — esub-free rewrite of the above: SLURM does
  the intra-node fan-out and core pinning (`--ntasks=36 --cpus-per-task=8`) and `run_cosmo.py`
  calls the app's `main()` directly. `run_cosmo.py` is identical between v17 and v18; the READMEs,
  `packed_node.slurm` and `run_slot.sh` differ only in version paths and config.

Paths inside these directories still name their pre-move locations (`pipelines/vNN/packed_clariden*/`).
