# Configs

A dataset is one version of a survey, with one forward-model config named after the version:

```
configs/<survey>/<version>/
    <version>.yaml            the forward-model config, e.g. desy6/v2/v2.yaml
    locations.yaml            where the dataset's data lie on disk (below)
    mock_observations.yaml    its mock observations (msfm/utils/mocks.py)
```

A version is a training set. Any change to the forward model, a prior included, means new tfrecords, and everything
downstream is trained on those, so it is a new version. A variant to train on is a version of its own (e.g.
`v2_dmo`), never a second config inside one. Naming the config after its version keeps its file name unique in the
repository.

| | |
|---|---|
| `desy6/v2/` | DES Y6, second version: v1 with choices of the DES Y6 3x2pt key project, starting with its intrinsic-alignment prior. In preparation |
| `desy6/v1/` | DES Y6, first version. `v1.yaml` is based on `desy3/v18/v18.yaml`; its data-derived values, on the lines marked `# <- step values`, are written by `python -m msfm.apps.products.desy6 values`. Which products exist yet: `data/desy6/README.md` |
| `desy3/v18/` | DES Y3 production: `v18.yaml`, plus the mock-observation variants in `obs/`. **Frozen** |
| `desy3/v17/`, `desy3/v16/` | earlier DES Y3 generations, from before one config per version, still read by their pipelines and studies. They have no `locations.yaml` |
| `desy3/deprecated/` | v5 to v15 and `marcel` |

Versions are numbered per survey, so Y6 starts again at `v1`. `pipelines/<cluster>/<survey>/<version>/`
holds the submission scripts for the version of the same name, on the cluster it runs on (DES Y3 on
`perlmutter`, DES Y6 on `daint`).

A config's repo paths (`files:`, `dirs:`) lie either directly in `data/`, for the inputs shared by
all surveys, or under `data/<survey.name in lower case>/`. msfm raises on any other `data/` path
(`msfm.utils.configuration.check_survey_paths`).

## Data locations

Where a dataset's data live outside the repo is written once, in its `locations.yaml`. msfm,
y3-deep-lss and msi all read it from there. It holds machine paths only, never a science choice, so
moving a directory leaves the config, and its sha256 in the provenance records, unchanged.

| key | what | read by |
|---|---|---|
| `projection` | the CosmoGrid projection (`grid/`, `benchmarks/`), read-only | msfm pipelines and products |
| `data` | msfm outputs, `{tfrecords, cls, obs, validation}` | y3-deep-lss |
| `obs` | the directory holding the mock observations' `obs/`, if it is not `data` | y3-deep-lss |
| `runs` | y3-deep-lss runs, `<representation>/<probe>/<run>` | y3-deep-lss, msi |
| `archive`, `runs_archive` | durable copies of `data` and `runs`; copy targets no script reads | |

Every key is optional, values must be absolute, and an unknown key is an error.
`msfm.utils.locations` resolves the paths of a dataset's config:

```bash
python -m msfm.utils.locations configs/desy6/v2/v2.yaml             # key=value lines
python -m msfm.utils.locations configs/desy6/v2/v2.yaml --key obs_dir
```

The resolved keys are `projection`, `data_dir`, `obs_dir` (`<obs or data>/obs`), `runs_dir`,
`archive_dir` and `runs_archive_dir`; a script that needs a key the file lacks stops and names the
file. Scripts read paths through this CLI, never by parsing the YAML, and an environment variable
still overrides one path for a one-off run.

The y3-deep-lss and msi submission scripts take the dataset as `VERSION`, i.e.
`configs/$VERSION/<version>.yaml`: `VERSION=desy6/v2` or `VERSION=desy3/v18`. `VERSION` has no
default, so a script run without it stops rather than fall back to another survey's data. In
Python, `msfm.utils.locations.dataset_config("desy6/v2")` gives the config's path.

## History

The DES Y3 configs moved here from `configs/vNN/` on 2026-11-18 (branch `desy6`); `main` still has
the flat layout. Run directories made before the move embed a copy of their config, whose data
paths are the old flat ones (see `data/desy3/README.md`).

Until 2026-11-24 a version's config was `default.yaml`, and every data and runs path appended that
name. DES Y3 v18 and DES Y6 v1 were produced then, so their data and runs keep a `default/`
directory, which their `locations.yaml` names.
