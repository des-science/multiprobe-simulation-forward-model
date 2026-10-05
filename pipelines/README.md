# Pipelines

Submission scripts for the msfm forward-model postprocessing, one directory per pipeline version.
Configs live in `../configs/vNN/`, mirroring this directory's version numbers 1:1.

Two ways to submit a job:
- **esub / shared QOS** — `pipe.yaml` (grid+fiducial tfrecord production, run via `epipe`) and
  `esub/obs_commands.sh` (ad hoc single-postprocessing benchmark/mock commands) are copy-paste
  command listings, always run with `cwd = pipelines/vNN` (their `../../msfm`, `../../configs/vNN`
  relative paths only resolve from that depth, regardless of which file the command text lives
  in). Queues on NERSC's `shared` QOS, which can have multi-day wait times.
- **packed / regular QOS** — `packed/submit.sh` (and version-specific sibling scripts) are real
  executables that hand-pack many tasks onto whole `regular`-QOS nodes, each pinned to a NUMA
  domain, instead of waiting in the shared-QOS queue. See `common/packed/README.md` for why this
  exists and how it's sized; each version's `packed/README.md` for its own numbers.

## Versions

| version | status | configs | esub | packed | notes |
|---|---|---|---|---|---|
| v18 | **active, packed-only** | `configs/v18/` | — (retired, see `deprecated/v18_esub/`) | `v18/packed/` | Count-based shape noise with the DES Y3 imaging systematics imprinted on the source density. |
| v17 | **active, esub + packed** | `configs/v17/` | `v17/esub/`, `v17/pipe.yaml` | `v17/packed/` | Standard NLA (no `bta`, no `ds` map); bit-identical to v16/rot_in_place at the fiducial. Esub kept deliberately in case of a future esub need. |
| v16 | deprecated | `configs/v16/` (still referenced by v17's "bit-identical at fiducial" reuse) | `deprecated/v16/` | — | Introduced the `rot_in_place` shape-noise variant and the `m5030` account. |
| v15 and earlier, `marcel` | deprecated | `configs/vNN/` | `deprecated/vNN/`, `deprecated/marcel/` | — | Superseded by later versions; kept for historical reference only. |

`cl_white_noise` (white-noise Cls generation, `run_power_spectra_noise.py`) is obsolete for both
v17 and v18 now that the Cls scale cut is `hard_rebinned` — it has no packed port and isn't
expected to need one. Removed from v17's active `pipe.yaml`; it survives only inside
`deprecated/v18_esub/pipe.yaml` as historical record.

## Layout

```
pipelines/
  README.md                  this file
  deprecated/                 retired pipeline generations and superseded submission methods
  common/
    perlmutter_setup.sh        Perlmutter env setup (OMP/MKL/etc thread counts), sourced by esub
                                commands via --source_file=; identical across v16/v17/v18
    packed/                    version-agnostic packed executors, see its own README.md
  v17/
    pipe.yaml                  esub: grid+fiducial tfrecord production
    esub/obs_commands.sh        esub: ad hoc single-postprocessing benchmark/mock commands
    packed/                     regular-QOS equivalents of both of the above
  v18/
    packed/                     the only submission path -- submit.sh (grid+fiducial) and
                                submit_obs.sh (every benchmark/mock arm)
```
