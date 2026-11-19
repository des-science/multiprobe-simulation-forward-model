# Pipelines

Submission scripts for the msfm forward-model postprocessing, one directory per survey and pipeline
version: `pipelines/<survey>/<version>/`. Configs live in `../configs/<survey>/<version>/`, which
mirrors this layout 1:1. `common/` holds the survey-agnostic executors.

Two ways to submit a job:
- **esub / shared QOS** — `pipe.yaml` (grid+fiducial tfrecord production, run via `epipe`) and
  `esub/obs_commands.sh` (ad hoc single-postprocessing benchmark/mock commands) are copy-paste
  command listings, always run with `cwd = pipelines/<survey>/vNN`. Their `../../../msfm` and
  `../../../configs/<survey>/vNN` relative paths only resolve from that depth, whichever file the
  command text lives in. Queues on NERSC's `shared` QOS, which can have multi-day wait times.
- **packed / regular QOS** — `packed/submit.sh` (and version-specific sibling scripts) are real
  executables that hand-pack many tasks onto whole `regular`-QOS nodes, each pinned to a NUMA
  domain, instead of waiting in the shared-QOS queue. See `common/packed/README.md` for why this
  exists and how it's sized; each version's `packed/README.md` for its own numbers.

The DES Y3 pipelines moved from `pipelines/vNN/` to `pipelines/desy3/vNN/` on 2026-11-18, with
their configs (`configs/vNN/` to `configs/desy3/vNN/`). They are Perlmutter-era scripts
(`REPO=/global/homes/...`); `main` still has the flat layout.

## Versions

### DES Y6 (`desy6/`)

| version | status | configs | notes |
|---|---|---|---|
| v1 | in preparation | `configs/desy6/v1/` | First DES Y6 forward model. |

`desy6/products/run.sh <pixels|maps|noise>` submits a compute-node step of the Y6 data products
(`msfm.apps.products.desy6`, see `data/desy6/README.md`) after running its `--dry-run` on the login node.
It is a daint script (account `lp184`, `torch_env` in the `pytorch/v2.9.1:v2` uenv), not a
Perlmutter one.

### DES Y3 (`desy3/`, frozen)

| version | status | configs | esub | packed | notes |
|---|---|---|---|---|---|
| v18 | **packed-only** | `configs/desy3/v18/` | — (retired, see `desy3/deprecated/v18_esub/`) | `desy3/v18/packed/` | Count-based shape noise with the DES Y3 imaging systematics imprinted on the source density. |
| v17 | **esub + packed** | `configs/desy3/v17/` | `desy3/v17/esub/`, `desy3/v17/pipe.yaml` | `desy3/v17/packed/` | Standard NLA (no `bta`, no `ds` map); bit-identical to v16/rot_in_place at the fiducial. Esub kept deliberately in case of a future esub need. |
| v16 | deprecated | `configs/desy3/v16/` (still referenced by v17's "bit-identical at fiducial" reuse) | `desy3/deprecated/v16/` | — | Introduced the `rot_in_place` shape-noise variant and the `m5030` account. |
| v15 and earlier, `marcel` | deprecated | `configs/desy3/deprecated/vNN/` | `desy3/deprecated/vNN/`, `desy3/deprecated/marcel/` | — | Superseded by later versions; kept for historical reference only. |

`cl_white_noise` (white-noise Cls generation, `run_power_spectra_noise.py`) is obsolete for both
v17 and v18 now that the Cls scale cut is `hard_rebinned` — it has no packed port and isn't
expected to need one. Removed from v17's active `pipe.yaml`; it survives only inside
`desy3/deprecated/v18_esub/pipe.yaml` as historical record. The scripts under `desy3/deprecated/`
were not updated for the move, so their relative paths are one level short.

## Layout

```
pipelines/
  README.md                    this file
  common/
    perlmutter_setup.sh        Perlmutter env setup (OMP/MKL/etc thread counts), sourced by esub
                               commands via --source_file=; identical across v16/v17/v18
    packed/                    survey- and version-agnostic packed executors, see its own README.md
  desy3/
    deprecated/                retired pipeline generations and superseded submission methods
    v17/
      pipe.yaml                esub: grid+fiducial tfrecord production
      esub/obs_commands.sh     esub: ad hoc single-postprocessing benchmark/mock commands
      packed/                  regular-QOS equivalents of both of the above
    v18/
      packed/                  the only submission path -- submit.sh (grid+fiducial) and
                               submit_obs.sh (every benchmark/mock arm)
  desy6/
    products/run.sh            submits a compute-node step of the Y6 data products
    v1/                        not yet created
```
