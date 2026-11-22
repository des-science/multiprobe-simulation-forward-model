# Configs

One forward-model config per survey, version and variant:

```
configs/<survey>/<version>/<subversion>.yaml
```

| | |
|---|---|
| `desy6/v1/` | DES Y6, first version. `default.yaml` is based on `desy3/v18/default.yaml`; its data-derived values, on the lines marked `# <- step values`, are written by `python -m msfm.apps.products.desy6 values`. Which products exist yet: `data/desy6/README.md` |
| `desy3/v18/` | DES Y3 production: `default.yaml`, plus the mock-observation variants in `obs/`. **Frozen** |
| `desy3/v17/`, `desy3/v16/` | earlier DES Y3 generations, still read by their pipelines and studies |
| `desy3/deprecated/` | v5 to v15 and `marcel` |

Versions are numbered per survey, so Y6 starts again at `v1`. `pipelines/<cluster>/<survey>/<version>/`
holds the submission scripts for the config of the same name, on the cluster it runs on (DES Y3 on
`perlmutter`, DES Y6 on `daint`).

A config's repo paths (`files:`, `dirs:`) lie either directly in `data/`, for the inputs shared by
all surveys, or under `data/<survey.name in lower case>/`. msfm raises on any other `data/` path
(`msfm.utils.configuration.check_survey_paths`).

## Data locations

Where a dataset's data live outside the repo is written once, in `locations.yaml` next to its
configs (`desy6/v1/locations.yaml`, `desy3/v18/locations.yaml`). msfm, y3-deep-lss and msi all
read it from there. It holds machine paths only, never a science choice, so moving a directory
leaves the configs, and the sha256 of `default.yaml` in the provenance records, unchanged.

| key | what | read by |
|---|---|---|
| `projection` | the CosmoGrid projection (`grid/`, `benchmarks/`), read-only | msfm pipelines and products |
| `data` | msfm outputs, per config `<config>/{tfrecords, cls, obs, validation}` | y3-deep-lss |
| `obs` | mock observations, if they are not under `data` | y3-deep-lss |
| `runs` | y3-deep-lss runs, per config `<config>/<representation>/<probe>/<run>` | y3-deep-lss, msi |
| `archive`, `runs_archive` | durable copies of `data` and `runs`; copy targets no script reads | |

`<config>` is the config's stem (`default`). Every key is optional, values must be absolute, and an
unknown key is an error. `msfm.utils.locations` resolves the paths of one config:

```bash
python -m msfm.utils.locations configs/desy6/v1/default.yaml             # key=value lines
python -m msfm.utils.locations configs/desy6/v1/default.yaml --key obs_dir
```

The resolved keys are `projection`, `data_dir`, `obs_dir` (`<obs or data>/<config>/obs`), `runs_dir`,
`archive_dir` and `runs_archive_dir`; a script that needs a key the file lacks stops and names the
file. Scripts read paths through this CLI, never by parsing the YAML, and an environment variable
still overrides one path for a one-off run.

The y3-deep-lss and msi submission scripts take the dataset as `VERSION` (and `SUBVERSION`,
default `default`), i.e. `configs/$VERSION/$SUBVERSION.yaml`: `VERSION=desy6/v1` or
`VERSION=desy3/v18`. `VERSION` has no default, so a script run without it stops rather than fall
back to another survey's data.

## History

The DES Y3 configs moved here from `configs/vNN/` on 2026-11-18 (branch `desy6`); `main` still has
the flat layout. Run directories made before the move embed a copy of their config, whose data
paths are the old flat ones (see `data/desy3/README.md`).
