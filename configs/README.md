# Configs

One forward-model config per survey, version and variant:

```
configs/<survey>/<version>/<subversion>.yaml
```

| | |
|---|---|
| `desy6/v1/` | DES Y6, first version. `default.yaml` is a skeleton based on `desy3/v18/default.yaml`; its data-derived values, on the lines marked `# <- step values`, are written by `python -m msfm.apps.products.desy6 values` (`data/desy6/README.md`) |
| `desy3/v18/` | DES Y3 production: `default.yaml`, plus the mock-observation variants in `obs/`. **Frozen** |
| `desy3/v17/`, `desy3/v16/` | earlier DES Y3 generations, still read by their pipelines and studies |
| `desy3/deprecated/` | v5 to v15 and `marcel` |

Versions are numbered per survey, so Y6 starts again at `v1`. `pipelines/<cluster>/<survey>/<version>/`
holds the submission scripts for the config of the same name, on the cluster it runs on (DES Y3 on
`perlmutter`, DES Y6 on `daint`).

A config's repo paths (`files:`, `dirs:`) lie either directly in `data/`, for the inputs shared by
all surveys, or under `data/<survey.name in lower case>/`. msfm raises on any other `data/` path
(`msfm.utils.configuration.check_survey_paths`).

The y3-deep-lss and msi submission scripts build `configs/$VERSION/$SUBVERSION.yaml`,
`data/$VERSION/$SUBVERSION` and `runs/$VERSION/$SUBVERSION`, so the survey goes into `VERSION`:
`VERSION=desy6/v1 SUBVERSION=default`.

The DES Y3 configs moved here from `configs/vNN/` on 2026-11-18 (branch `desy6`); `main` still has
the flat layout. Run directories made before the move embed a copy of their config, whose data
paths are the old flat ones (see `data/desy3/README.md`).
