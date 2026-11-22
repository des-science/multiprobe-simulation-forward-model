# Data locations: one file per dataset

Plan, 2026-11-22. Covers the `desy6` branches of msfm, y3-deep-lss and msi.

## Goal

Every out-of-repo data location of a dataset (survey + version) is written once, in msfm next to
the dataset's config, and msfm, y3-deep-lss and msi all read it from there. No script defaults to a
dataset: a run without one stops.

Decided with the user on 2026-11-22: the file lives in msfm per dataset (not one file per machine);
Y3 gets one too, reproducing today's paths; no dataset, no run.

## Today

- **msfm.** `pipelines/daint/desy6/v1/submit.sh` hardcodes the projection root (`DIR_IN`,
  overridable) and the output root (`DATA_ROOT=/ritom/...`, not overridable). The products take the
  projection as a hand-typed `--source`, which is repeated in `provenance.yaml`. The DES release is
  an absolute path inside the science config (`dirs.catalog`), and the Y3 catalog path is copied into
  5 configs.
- **y3-deep-lss.** `submissions/clariden/common.sh` defines `MYSCRATCH` (`/iopsstor`), `STORE`
  (a0158), then `INPUT`, `OBS_INPUT` and `RUNS_ROOT` from `VERSION/SUBVERSION`, all overridable.
  `SURVEY` picks the config directories but not the data roots. The apps take every path as a CLI
  argument, and no YAML holds a path.
- **msi.** `common.sh` repeats both roots. Every runs config has a literal absolute `pred_dir`
  (`/iopsstor/...`) and the Y3 `scales_config`.

What goes wrong:

1. Producer and consumer don't share a definition. msfm writes Y6 data to `/ritom`, while
   y3-deep-lss defaults to `/iopsstor`. The maps path reads mocks from `$OBS_INPUT/obs` (the store),
   the Cls path from `$INPUT/obs` (scratch). y3-deep-lss writes runs to scratch, while msi's
   `inference.sh` reads the store.
2. One root is written in many places: `/iopsstor` in two `common.sh` and every `pred_dir`, the
   projection in `submit.sh` and two `--source`s.
3. With `VERSION` unset, msi's `inference.sh` resolves to the live Y3 run tree on the store, where a
   Y6 run of the same name would overwrite Y3 chains.
4. The Y3 layouts don't match across repos: msfm `configs/desy3/v18`, but `runs/v18` and msi
   `configs/runs/v18`. No single `VERSION` resolves Y3 on the `desy6` branches.

## Design

### The file

`configs/<survey>/<version>/locations.yaml` in msfm, next to the configs of that version. It holds
machine paths only, never a science choice, so moving a directory never changes `default.yaml`
(whose sha256 the provenance records hold, and onto which the mock registry merges).

| key | role | below it |
|---|---|---|
| `projection` | CosmoGrid projection, input, read-only | `grid/`, `benchmarks/` |
| `catalog` | the survey release, input, read-only (phase 2) | release files |
| `data` | msfm outputs, read by y3-deep-lss | `<config>/{tfrecords/grid, cls/grid_cls.h5, obs, validation, packed_logs}` |
| `obs` | mock observations, if not under `data` | `<config>/obs/*_obs_maps.h5` |
| `runs` | y3-deep-lss runs, read by msi | `<config>/<representation>/<probe>/<run>` |
| `archive` | durable copy of `data`; a copy target no script reads | as `data` |
| `runs_archive` | durable copy of `runs`; a copy target no script reads | as `runs` |

`<config>` is the msfm config's stem, today's `SUBVERSION` (`default`). Every key is optional, since a
dataset may lack one (Y3 on daint has no projection). Values must be absolute. An unknown key is an
error, and a consumer that needs a missing key stops and names the file.

Y6 (`configs/desy6/v1/locations.yaml`):

```yaml
projection: /capstor/store/cscs/userlab/lp184/tomaszk/261109_cosmogridv1_desy6/000_test/cosmogrid/processed/desy6/CosmoGrid/bary
data: /ritom/scratch/cscs/athomsen/desy6/v1
archive: /capstor/store/cscs/userlab/lp184/athomsen/deep_lss/data/desy6/v1
runs: /ritom/scratch/cscs/athomsen/deep_lss/runs/desy6/v1
runs_archive: /capstor/store/cscs/userlab/lp184/athomsen/deep_lss/runs/desy6/v1
```

Y3 (`configs/desy3/v18/locations.yaml`), today's resolved paths, with the runs on the store (D1) and the
data on `/ritom` (D4):

```yaml
data: /ritom/scratch/cscs/athomsen/v18
obs: /capstor/store/cscs/swissai/a0158/athomsen/deep_lss/data/v18
runs: /capstor/store/cscs/swissai/a0158/athomsen/deep_lss/runs/v18
```

### Resolver

`msfm/utils/locations.py`:

- `load(msfm_config_path)` finds the sibling `locations.yaml`, validates it, and returns the resolved
  paths of that config: `data_dir = <data>/<config>`, `obs_dir = <obs or data>/<config>/obs`,
  `runs_dir`, `archive_dir`, `projection`, `catalog`.
- CLI `python -m msfm.utils.locations <msfm config> [--key K]` prints `key=value` lines, or one value.
  Scripts read it through the CLI, as `common.sh` already does with `config_check resolve --flat`,
  never by grepping YAML.
- An environment variable still overrides any one path for a one-off run.

### Consumers

**msfm**
- `pipelines/daint/desy6/v1/submit.sh`: `DIR_IN` from `projection`, `DATA_ROOT` from `data_dir`.
- `msfm.apps.products.desy6 sc_bias`: `--source` defaults to `projection`; the provenance record keeps
  the path actually used. `systematics --source` (an `lss_sys` export) stays a CLI argument.
- `msfm.utils.mocks check`: `--projection` defaults to `projection`.

**y3-deep-lss**
- `common.sh`: `VERSION` has no default. `INPUT`, `OBS_INPUT` and `RUNS_ROOT` come from the resolver.
  `STORE` goes; `MYSCRATCH` stays for `claude/bench` only.
- `run_evaluation.py` and `run_cls_training+evaluation.py` get `--obs_dir` (the directory holding
  `*_obs_maps.h5`). `prod/maps.sh` and `prod/cls.sh` pass the same resolved `obs_dir`. The Cls app's
  `--data_dir` keeps meaning the Cls cache root; `run_evaluation.py --data_dir` stays as a fallback
  (`obs = data_dir/obs`).
- `deep_lss.apps.products.desy6 signal`: `--source` defaults to `projection`.

**msi**
- `common.sh` and `inference.sh`: `VERSION` has no default; `RUNS_ROOT` from the resolver.
- Runs configs: a new top-level key `dataset: desy6/v1/default`. A relative `pred_dir` resolves against
  that dataset's `runs_dir`; an absolute one is used as is, so every existing Y3 config is unchanged.
  One loader, `msi/utils/runs.py: load_runs_config(path)`, replaces the `read_yaml` in `run_ppc.py`,
  `run_tension_chains.py`, `run_tension_values.py` and `run_fom_percentile.py`.
- `configs/runs/v18/` moves to `configs/runs/desy3/v18/` (and `v17` to `desy3/v17`), contents
  unchanged, so `RUNS_CONFIG` resolves from `VERSION=desy3/v18` like `MSFM_CONFIG`.

### No dataset, no run

The y3-deep-lss and msi scripts stop when `VERSION` is unset, listing the
`configs/*/*/locations.yaml` they could use. `DRYRUN=1` too.

## Decisions

All by the user, 2026-11-22.

- **D1. The Y3 runs root is the a0158 store.** One key serves y3-deep-lss (which wrote to scratch, since
  purged) and msi (which reads the store, the only complete copy of the frozen runs). So a y3-deep-lss
  `eval` or `infer` on a frozen Y3 run would write into the store copy: `prod/maps.sh` and `prod/cls.sh`
  refuse either stage on a run directory that already holds predictions unless `OVERWRITE=1`, the
  variable that already guards training. (`infer` was added at implementation: msi's `inference.sh`
  rewrites a run's chains by design.)
- **D2. Y6 runs go to `/ritom` scratch** (`deep_lss/runs/desy6/v1`), next to the Y6 data. It is purged,
  so finished runs are copied to `runs_archive` on the lp184 store.
- **D4. The Y3 data are on `/ritom`** (`/ritom/scratch/cscs/athomsen/v18/default`: tfrecords, Cls,
  mocks, 14 TB). The old default, `/iopsstor/.../deep_lss/data/v18/default`, was found purged to empty
  directories during step 3, so copying it would have pointed Y3 at nothing.
- **D3. Y3 Cls mocks come from the shared `obs_dir`**, the store. Today a Y3 Cls evaluation reads
  `$INPUT/obs` on scratch, which holds none; only future Y3 Cls evaluations change.

## Not in this round

- `dirs.catalog` into `catalog` (phase 2). Four readers in `catalog.py` and one in `catalog_desy6.py`
  get a config dict without its path, so the locations would have to travel with the dict (e.g.
  attached by `files.load_config`). The Y6 `default.yaml` would lose a line, so step `values` re-records
  its provenance. Y3 configs keep `dirs.catalog`.
- Environment paths (`UENV`, `PY`, `tf_env`) and the absolute `#SBATCH --output` lines: per machine,
  not per dataset.
- msi `configs/ppc/*.yaml` `base_dir` and the runs configs' `scales_config` (a repo path, not data).

## Order and verification

1. msfm: the resolver, the two files, tests (missing key, unknown key, relative value, the CLI). Done
   2026-11-22: `tests/test_locations.py`, 13 passed in torch_env; every resolved path equals today's
   script default. `dev/scripts/data_audit/gate.sh` now skips `data/` inside absolute paths, which it
   had listed as missing repo files (OK count unchanged, 23).
2. msfm consumers. `DRY_RUN=1 ./submit.sh grid|validate|obs all` prints the same commands as before.
   Done 2026-11-22: all three dry runs byte-identical; `DATA_ROOT` is now overridable too, and a config
   whose dataset lacks the projection or `data` stops (tested on the Y3 file). `sc_bias --source` and
   `mocks check --projection` default to the projection; `systematics --source` stays required.
3. y3-deep-lss: `common.sh`, `--obs_dir`, the products default. Acceptance: the `DRYRUN=1` output of
   every `prod/` script for `VERSION=desy3/v18` equals today's with the old defaults set explicitly
   (`INPUT=`, `OBS_INPUT=`, `RUNS_ROOT=`), except the changes D1 and D3 make. The Y6 dry runs resolve
   to the `/ritom` data and runs. Done 2026-11-22 (y3-deep-lss `d61fc55`): 18 of 18 dry runs (nine
   modes per survey) differ by exactly the runs root (Y3), `--data_dir` -> `--obs_dir` in the maps
   evaluation and the Cls app's new `--obs_dir`; debug job 5425620 resolved the same paths through
   the srun step. `OBS_INPUT` is replaced by `OBS_DIR` (the obs directory itself).
4. msi: `common.sh`/`inference.sh`, the runs loader, the directory move. Acceptance: the same dry-run
   comparison, and the 18 v18 runs still select identical keys. Done 2026-11-22 (msi `20614fa`): the
   dataset is a function, `use_dataset`, since `coverage_round.sh` needs none; every existing runs
   config loads identically through `load_runs_config` (tested); dry runs differ only by the moved runs
   config and, for `fisher_cls.sh`, the store runs root and the `/ritom` data; the suite passes, 20 of 20,
   after the integration test's stale msfm path was fixed. Found: every Y3 runs config's absolute
   `pred_dir` points at the purged `/iopsstor` copies of the runs (no files left, only dangling links).
5. Docs: each repo's README and notes, the workspace `CLAUDE.md` section on filesystem paths (with the
   user's OK), the memory.
