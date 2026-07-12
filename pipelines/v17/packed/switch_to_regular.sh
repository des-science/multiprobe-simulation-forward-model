#!/bin/bash
# Switch the v17 baseline postprocessing from the saturated shared QOS to packed regular-QOS
# node jobs (prepared 2026-07-12, see README.md). Safe to inspect any time; MUTATES THE QUEUE
# when run without DRY_RUN.
#
# Usage:
#   DRY_RUN=1 ./switch_to_regular.sh both    # print what would happen, touch nothing
#   ./switch_to_regular.sh grid|fiducial|both
#
# Per chain this
#   1. cancels the esub rerun_missing job (it would resubmit everything to shared),
#   2. cancels only the PENDING elements of the main array (running ones drain and finish),
#   3. derives the missing indices = all - done (<job>_done.dat) - currently running,
#   4. submits packed_node.slurm as a regular-QOS array (16 tasks per node),
#   5. re-points the original esub merge job at the packed array (afterok) so it still runs
#      automatically; the merge_log_files job follows with afterany.
#
# If a packed element fails, the merge stays pending (afterok). Rerun the missing indices
# (step 3 lists them again) and then: scontrol update job <merge_id> Dependency=afterok:<new_id>

set -euo pipefail
cd "$(dirname "$0")"

LOG_DIR=/pscratch/sd/a/athomsen/run_files/v17/esub_logs
PACKED_LOGS=/pscratch/sd/a/athomsen/run_files/v17/packed_logs
SLOTS=16

DRY_RUN=${DRY_RUN:-}
run() {
    if [ -n "$DRY_RUN" ]; then
        echo "DRY RUN + $*"
    else
        "$@"
    fi
}

mkdir -p "$PACKED_LOGS"

job_id_by_name() {
    squeue -u "$USER" -h -o "%F %j" | awk -v n="$1" '$2 == n {print $1}' | sort -u | head -1
}

switch_chain() {
    local chain=$1 job_name n_total
    if [ "$chain" = grid ]; then
        job_name=tfr_grid_v17
        n_total=2500
    else
        job_name=tfr_fidu_v17
        n_total=1000
    fi

    local main_id rerun_id merge_id mlog_id
    main_id=$(job_id_by_name "${job_name}_main")
    rerun_id=$(job_id_by_name "${job_name}_rerun_missing")
    merge_id=$(job_id_by_name "${job_name}_merge")
    mlog_id=$(job_id_by_name "${job_name}_merge_log_files")
    echo "[$chain] main=$main_id rerun_missing=$rerun_id merge=$merge_id merge_log=$mlog_id"

    # 1. rerun_missing must never fire
    [ -n "$rerun_id" ] && run scancel "$rerun_id"
    # 2. keep the running elements, cancel the pending ones
    [ -n "$main_id" ] && run scancel --state=PENDING "$main_id"

    # 3. missing = all - done - running (running elements finish and append to done.dat).
    # esub submits slurm arrays 1-based (--array=1-N) but tasks are 0-based: element K runs
    # esub index K-1, hence the "- 1" on the squeue %K output.
    local excl=$PACKED_LOGS/${job_name}_excluded.txt
    local missing=$PACKED_LOGS/${job_name}_missing.txt
    { [ -n "$main_id" ] && squeue -j "$main_id" -h -t RUNNING -o "%K" 2>/dev/null | awk '{print $1 - 1}';
      cat "$LOG_DIR/${job_name}_done.dat" 2>/dev/null; } | grep -E '^[0-9]+$' | sort -n -u > "$excl" || true
    seq 0 $((n_total - 1)) | grep -vxF -f "$excl" > "$missing" || true

    local n_missing n_elements
    n_missing=$(wc -l < "$missing")
    n_elements=$(( (n_missing + SLOTS - 1) / SLOTS ))
    echo "[$chain] $(wc -l < "$excl") done/running excluded, $n_missing missing -> $n_elements packed node jobs"
    if [ "$n_missing" -eq 0 ]; then
        echo "[$chain] nothing to do"
        return 0
    fi

    # 4. packed regular-QOS array
    local packed_id
    if [ -n "$DRY_RUN" ]; then
        echo "DRY RUN + sbatch --parsable --array=0-$((n_elements - 1)) --job-name=${job_name}_packed" \
             "--export=ALL,CHAIN=$chain,INDEX_FILE=$missing packed_node.slurm"
        packed_id=DRYRUN
    else
        packed_id=$(sbatch --parsable \
            --array=0-$((n_elements - 1)) \
            --job-name="${job_name}_packed" \
            --export=ALL,CHAIN=$chain,INDEX_FILE="$missing" \
            packed_node.slurm)
    fi
    echo "[$chain] packed array: $packed_id"

    # 5. the original esub merge now waits for the packed array (and the draining main
    #    elements); afterok blocks the merge if any packed element fails
    if [ -n "$merge_id" ]; then
        local dep="afterok:$packed_id"
        [ -n "$main_id" ] && dep="$dep,afterany:$main_id"
        run scontrol update job "$merge_id" Dependency="$dep"
    fi
    [ -n "$mlog_id" ] && run scontrol update job "$mlog_id" Dependency="afterany:$packed_id"
    return 0
}

case "${1:-}" in
    grid)     switch_chain grid ;;
    fiducial) switch_chain fiducial ;;
    both)     switch_chain grid; switch_chain fiducial ;;
    *)        echo "usage: [DRY_RUN=1] $0 grid|fiducial|both" && exit 1 ;;
esac
