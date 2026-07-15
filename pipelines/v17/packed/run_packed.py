#!/usr/bin/env python
"""Minimal esub-free driver for the v17 postprocessing apps.

esub `--mode=run` ultimately does just two things with an app (see esub.py):
  - function main : consume the `main(indices, args)` generator
  - function merge : call `merge(indices, args)`
plus a layer of done.dat bookkeeping and rerun_missing logic. That layer is what
makes esub worth it for a farm of independent shared-QOS jobarray tasks, but it
is pointless here: we pack whole nodes and track completion by the .tfrecord
files themselves (a missing output file == a task to (re)run, see submit.sh).

So this script replicates only the two calls above and nothing else. The app is
imported by path and handed the leftover CLI args verbatim -- exactly the arg
list esub would forward -- so the forward model, filenames and tfrecord contents
are bit-for-bit identical to the normal pipeline. No Python in msfm was changed.
"""
import argparse
import importlib.util
import sys


def load_app(path):
    spec = importlib.util.spec_from_file_location("packed_app", path)
    module = importlib.util.module_from_spec(spec)
    # register so dataclasses / pickle inside the app can find it by name
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(
        description="Call an msfm postprocessing app's main/merge directly, without esub."
    )
    parser.add_argument("--app", required=True, help="path to the app, e.g. msfm/apps/run_grid_postprocessing.py")
    parser.add_argument("--function", default="main", choices=("main", "merge"))
    parser.add_argument(
        "--tasks",
        required=True,
        help="comma-separated 0-based indices, e.g. '7' or '7,8,9' (one .tfrecord file each)",
    )
    # everything else is forwarded untouched to the app's argparse (parse_known_args)
    args, app_args = parser.parse_known_args()

    indices = [int(t) for t in args.tasks.split(",") if t != ""]
    app = load_app(args.app)

    if args.function == "main":
        for index in app.main(indices, app_args):
            print(f"##### finished index {index} #####", flush=True)
    else:
        app.merge(indices, app_args)


if __name__ == "__main__":
    main()
