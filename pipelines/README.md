# Pipelines

Submission scripts for the msfm forward-model postprocessing, sorted **by cluster first**:
`pipelines/<cluster>/<survey>/`. The cluster decides whether a script can run at all (accounts,
filesystems, queues, containers), so it is the top level. Below it, the survey mirrors the configs.
DES Y6 has one script for all its versions, `daint/desy6/submit.sh`, which takes the version as
`VERSION` (`desy6/v2` for `configs/desy6/v2/`). The DES Y3 versions each have their own folder,
`pipelines/<cluster>/desy3/<version>/`.

| cluster | status | surveys and versions | how jobs are submitted |
|---|---|---|---|
| [`daint/`](daint/README.md) (CSCS Alps, GH200) | **active** | DES Y6: `desy6/products/` (data products), `desy6/submit.sh` (tfrecords and mocks of every version) | packed whole exclusive nodes, CPUs only; `lp184` |
| [`perlmutter/`](perlmutter/README.md) (NERSC, CPU nodes) | frozen | DES Y3: `desy3/v17/` (esub + packed), `desy3/v18/packed/` | esub on the `shared` QOS, or packed whole nodes on `regular` |
| [`deprecated/`](deprecated/README.md) | never run again | Euler, Perlmutter and Clariden scripts of older versions | — |

`common/` holds the only cluster-agnostic file, `run_packed.py`: an esub-free driver that imports an
msfm app by path and calls its `main`/`merge` with the arguments esub would forward, so the outputs
are identical to an esub run. Every packed executor on every cluster calls it.

## Layout

```
pipelines/
  README.md
  common/
    run_packed.py              cluster-agnostic driver, see above
  daint/                       CSCS Alps daint (GH200 nodes, CPUs only), account lp184
    README.md
    common/
      packed_node.slurm        packed executor (srun step of taskset-pinned tasks, tensorflow container)
      packed_task.sh           one task of it
    desy6/
      products/run.sh          submits a compute-node step of the Y6 data products
      submit.sh                tfrecords and mocks of VERSION (preflight, packed array, merge)
      README.md                how to run it, sizing and cost, the v1 records
  perlmutter/                  NERSC Perlmutter CPU nodes; the DES Y3 production, frozen
    README.md                  the packed regular-QOS rationale: pinning, bandwidth-knee sizing
    common/
      perlmutter_setup.sh      thread-count environment, sourced by the esub commands via --source_file=
      packed_node.slurm        packed executor (numactl-pinned tasks on one regular-QOS node)
      merge_node.slurm         single-node merge
    desy3/
      v17/
        pipe.yaml              esub: grid+fiducial tfrecord production
        esub/obs_commands.sh   esub: ad hoc single-postprocessing benchmark/mock commands
        packed/                regular-QOS equivalents of both of the above
      v18/
        packed/                the only v18 path: submit.sh (grid+fiducial), submit_obs.sh (benchmark/mock arms)
  deprecated/<cluster>/<survey>/<version>/
```

## DES Y6 (`daint/desy6/`)

| version | status | configs | notes |
|---|---|---|---|
| v1 | done (2026-11-21) | `configs/desy6/v1/` | First DES Y6 forward model. `VERSION=desy6/v1 daint/desy6/submit.sh grid`. |
| v2 | in preparation | `configs/desy6/v2/` | The KP's IA prior and fiducials, the source clustering bias marginalized. `VERSION=desy6/v2 daint/desy6/submit.sh grid`. |

`daint/desy6/products/run.sh <pixels|maps|noise>` submits a compute-node step of the Y6 data
products (`msfm.apps.products.desy6`, see `data/desy6/README.md`) after running its `--dry-run` on
the login node.

## DES Y3 (`perlmutter/desy3/`, frozen)

| version | status | configs | esub | packed | notes |
|---|---|---|---|---|---|
| v18 | **packed-only** | `configs/desy3/v18/` | — (retired, see `deprecated/perlmutter/desy3/v18_esub/`) | `perlmutter/desy3/v18/packed/` | Count-based shape noise with the DES Y3 imaging systematics imprinted on the source density. |
| v17 | **esub + packed** | `configs/desy3/v17/` | `perlmutter/desy3/v17/esub/`, `perlmutter/desy3/v17/pipe.yaml` | `perlmutter/desy3/v17/packed/` | Standard NLA (no `bta`, no `ds` map); bit-identical to v16/rot_in_place at the fiducial. Esub kept deliberately in case of a future esub need. |
| v16 | deprecated | `configs/desy3/v16/` (still referenced by v17's "bit-identical at fiducial" reuse) | `deprecated/perlmutter/desy3/v16/` | — | Introduced the `rot_in_place` shape-noise variant and the `m5030` account. |
| v15 and earlier, `marcel` | deprecated | `configs/desy3/deprecated/vNN/` | `deprecated/{euler,perlmutter}/desy3/vNN/`, `deprecated/perlmutter/desy3/marcel/` | — | Superseded by later versions; kept for historical reference only. |

The esub command listings (`pipe.yaml`, `esub/obs_commands.sh`) are copy-paste text, always run with
`cwd = pipelines/perlmutter/desy3/vNN`: their `../../../../msfm` and `../../../../configs/desy3/vNN`
paths only resolve from that depth, whichever file the command text lives in.

`cl_white_noise` (white-noise Cls generation, `run_power_spectra_noise.py`) is obsolete for both
v17 and v18 now that the Cls scale cut is `hard_rebinned` — it has no packed port and isn't
expected to need one. Removed from v17's active `pipe.yaml`; it survives only inside
`deprecated/perlmutter/desy3/v18_esub/pipe.yaml` as historical record.

## History

- 2026-11-18: the DES Y3 pipelines moved from `pipelines/vNN/` to `pipelines/desy3/vNN/`, with their
  configs (`configs/vNN/` to `configs/desy3/vNN/`). `main` still has the flat layout.
- 2026-11-20: sorted by cluster. `desy3/{v17,v18}` → `perlmutter/desy3/`,
  `common/{perlmutter_setup.sh,packed/*.slurm}` → `perlmutter/common/`,
  `common/packed/README.md` → `perlmutter/README.md`, `common/packed/run_packed.py` →
  `common/run_packed.py`, `desy6/products` → `daint/desy6/products`, `desy3/deprecated/*` →
  `deprecated/<cluster>/desy3/*`.
