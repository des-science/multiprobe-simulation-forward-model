#!/usr/bin/env python
"""Minimal, esub-free launcher for run_grid_postprocessing on Clariden.

Why this exists: on Clariden every allocation is a whole exclusive node (billed per node, not
per core), so esub's per-core accounting and rerun-missing bookkeeping buy nothing here. This
launcher skips esub entirely -- no jobarray/resources() machinery, no _done.dat. It imports the
app UNCHANGED and calls its main()/merge() generators directly for the requested task index.

Everything after this launcher's own two flags (--tasks, --function) is passed straight through
to the app's own argparse (--n_files, --config, --dir_in, --dir_out, --cluster, ...), so the app
sees exactly the arguments it would from esub.
"""
import argparse

from msfm.apps import run_grid_postprocessing as app


def parse_tasks(spec):
    """'5' -> [5];  '0,3,7' -> [0, 3, 7];  '0-9' -> 0..9 inclusive;  '0>10' -> 0..9 (esub style)."""
    spec = spec.strip()
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    if ">" in spec:
        lo, hi = spec.split(">")
        return list(range(int(lo), int(hi)))
    return [int(x) for x in spec.replace(",", " ").split()]


ap = argparse.ArgumentParser(add_help=False, description=__doc__)
ap.add_argument("--tasks", required=True, help="index (5), list (0,3,7), or range (0-9 / 0>10)")
ap.add_argument("--function", default="main", choices=("main", "merge"), help="app entry point to run")
launcher_args, passthrough = ap.parse_known_args()

indices = parse_tasks(launcher_args.tasks)
fn = getattr(app, launcher_args.function)

# main() and merge() are generators (esub drives them the same way) -- exhaust to run.
for _ in fn(indices, passthrough):
    pass
