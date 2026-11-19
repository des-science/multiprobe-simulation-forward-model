#!/usr/bin/env bash
# Submits one compute-node step of the DES Y6 data products (msfm.apps.products.desy6) as a single-node CPU job.
#
#   pipelines/desy6/products/run.sh <pixels|maps|noise> [extra step arguments]
#   DRYRUN=1 pipelines/desy6/products/run.sh maps      # print the sbatch command, submit nothing
#
# nz, values and config run on the login node without this wrapper (data/desy6/README.md).
# Before submitting, the step's --dry-run runs on the login node, so a missing input or a broken environment fails
# here and not in the queue.
#
# Variables:
#   CONFIG        config relative to the repo, default configs/desy6/v1/default.yaml
#   TIME          wall clock, default 00:30:00
#   PARTITION     default debug: each step fits its 30 min limit (one job at a time there)
#   ACCOUNT       default lp184
#   UENV          default pytorch/v2.9.1:v2, with view default
#   PY            default ~/dlss/torch_env/bin/python
#   LOGDIR        default $SCRATCH/msfm_desy6_products
#   DRYRUN=1      print the sbatch command instead of submitting
#
# Extra arguments go to the step, e.g. --provenance for a production run: the preflight then also refuses uncommitted
# msfm/ or configs/.
set -euo pipefail

die() { echo "run.sh: $*" >&2; exit 1; }

STEP=${1:-}
case $STEP in
    pixels|maps|noise) shift ;;
    *) die "usage: run.sh <pixels|maps|noise> [extra step arguments]; nz, values and config run on the login node" ;;
esac

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CONFIG=${CONFIG:-configs/desy6/v1/default.yaml}
TIME=${TIME:-00:30:00}
PARTITION=${PARTITION:-debug}
ACCOUNT=${ACCOUNT:-lp184}
UENV=${UENV:-pytorch/v2.9.1:v2}
PY=${PY:-$HOME/dlss/torch_env/bin/python}
LOGDIR=${LOGDIR:-${SCRATCH:?SCRATCH is not set, set LOGDIR}/msfm_desy6_products}

cd "$REPO_DIR"
[[ -f $CONFIG ]] || die "no config $CONFIG in $REPO_DIR"

STEP_CMD=("$PY" -m msfm.apps.products.desy6 "$STEP" --config "$CONFIG" "$@")

# inside a uenv session the environment is already there; uenv run does not nest
IN_UENV=${UENV_VIEW:-}${UENV_MOUNT_LIST:-}
if [[ -n $IN_UENV ]]; then
    RUN=()
else
    RUN=(uenv run --view=default "$UENV" --)
fi

echo "run.sh: preflight on the login node"
${RUN[@]+"${RUN[@]}"} "${STEP_CMD[@]}" --dry-run || die "the dry run of $STEP failed"
if [[ $STEP == pixels ]]; then
    ${RUN[@]+"${RUN[@]}"} "$PY" -c "import healsparse" || die "step pixels needs healsparse in $PY"
fi

mkdir -p "$LOGDIR"
SB=(sbatch
    --account="$ACCOUNT"
    --job-name="desy6_$STEP"
    --partition="$PARTITION"
    --time="$TIME"
    --nodes=1
    --ntasks=1
    --cpus-per-task=288
    --chdir="$REPO_DIR"
    --output="$LOGDIR/desy6_${STEP}_%j.log"
    --uenv="$UENV"
    --view=default)
# sbatch is disabled inside a uenv session unless the session's uenv is ignored; the job brings its own
[[ -n $IN_UENV ]] && SB+=(--uenv-passthrough=ignore)
SB+=(--wrap="$(printf '%q ' "${STEP_CMD[@]}")")

if [[ ${DRYRUN:-0} == 1 ]]; then
    printf '%q ' "${SB[@]}"
    echo
    exit 0
fi
echo "run.sh: desy6 $STEP -> $PARTITION ($TIME), log $LOGDIR/desy6_${STEP}_<jobid>.log"
exec "${SB[@]}"
