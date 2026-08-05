# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created July 2026
Author: Arne Thomsen

Driver for the unified Gaussian Cls Fisher forecasts (`msfm.utils.fisher`). Reads a small YAML spec
that names data sources and the forecasts to run over them, evaluates each forecast, and prints the
marginalized constraints plus any requested pairwise comparisons.

A forecast is `{source, blocks}`: a covariance/jacobian source pair and an ordered list of Cl blocks
(E, B, ...). The "standard" E-mode-only forecast is `blocks: [E]`; the shear-equivalent E+B is
`blocks: [E, B]`. Comparing the same block set across two sources (e.g. an ell_min=30 baseline tree vs
an ell_min=None tree) isolates an analysis choice; comparing block sets within one source isolates the
information a block carries.

Spec format (see configs/v17/fisher_bmode.yaml):

    config: configs/v17/baseline_bmode.yaml   # for the Gaussian prior widths (priors: true only)
    options: {priors: true, hartlap: true, normalize: true}
    # optional custom block registry; defaults to fisher.BLOCKS_DEFAULT
    blocks:
      E: {cov: cls/binned,       jac: pert_binned_E}
      B: {cov: cls/bmode_binned, jac: pert_binned_B}
    sources:
      baseline: {cov: /.../baseline/cls/fiducial_cls.h5,       jac: /.../baseline/cls/forecast_inputs.h5}
      bmode:    {cov: /.../baseline_bmode/cls/fiducial_cls.h5, jac: /.../baseline_bmode/cls/forecast_inputs.h5}
    forecasts:
      baseline_E: {source: baseline, blocks: [E]}
      bmode_E:    {source: bmode,    blocks: [E]}
      bmode_EB:   {source: bmode,    blocks: [E, B]}
    compare:                       # optional; each pair prints a (from -> to) delta
      - [baseline_E, bmode_E]      # l_min ablation
      - [bmode_E, bmode_EB]        # B-mode information gain

Usage:
    python -m msfm.apps.run_fisher_forecast --spec configs/v17/fisher_bmode.yaml
    python -m msfm.apps.run_fisher_forecast --spec ... --only bmode_E bmode_EB   # subset
    python -m msfm.apps.run_fisher_forecast --spec ... --no-priors               # override options
"""

import argparse

from msfm.utils import files, fisher, input_output, logger

LOGGER = logger.get_logger(__file__)


def resolve(spec):
    """Fill in defaults and validate the spec, returning (block_defs, sources, forecasts, compare,
    options, conf)."""
    block_defs = spec.get("blocks", fisher.BLOCKS_DEFAULT)
    sources = spec["sources"]
    forecasts = spec["forecasts"]
    compare = spec.get("compare", [])
    options = {"priors": False, "hartlap": True, "normalize": True}
    options.update(spec.get("options", {}))
    conf = files.load_config(spec["config"]) if spec.get("config") else None

    for name, fc in forecasts.items():
        if fc["source"] not in sources:
            raise KeyError(f"forecast '{name}': unknown source '{fc['source']}'")
        for b in fc["blocks"]:
            if b not in block_defs:
                raise KeyError(f"forecast '{name}': unknown block '{b}' (have {list(block_defs)})")
    for pair in compare:
        for name in pair:
            if name not in forecasts:
                raise KeyError(f"compare pair {pair}: unknown forecast '{name}'")
    return block_defs, sources, forecasts, compare, options, conf


def print_forecast(name, r):
    tag = " + priors" if r["priors"] else ""
    print(
        f"\n=== {name}  [blocks={'+'.join(r['blocks'])}]{tag} ===  "
        f"(p={r['p']}, n_real={r['n_real']}, cov cond={r['cond']:.2e})"
    )
    print(
        f"  sigma(Om) = {r['sigma_Om']:.5f}   sigma(s8) = {r['sigma_s8']:.5f}   "
        f"sigma(S8) = {r['sigma_S8']:.5f}   FoM(Om,S8) = {r['fom_Om_S8']:.2f}"
    )


def print_comparison(name_a, ra, name_b, rb):
    print(
        f"\n  --> {name_a} -> {name_b}:  "
        f"sigma(S8) {ra['sigma_S8']:.5f} -> {rb['sigma_S8']:.5f}  "
        f"({100*(rb['sigma_S8']/ra['sigma_S8']-1):+.2f} %),   "
        f"FoM {ra['fom_Om_S8']:.2f} -> {rb['fom_Om_S8']:.2f}  "
        f"({100*(rb['fom_Om_S8']/ra['fom_Om_S8']-1):+.2f} %)"
    )
    if ra["params"] == rb["params"]:
        print("  per-param sigma ratio (to/from):")
        for i, p in enumerate(ra["params"]):
            print(
                f"    {p:10s} {ra['sigma'][i]:.4e} -> {rb['sigma'][i]:.4e}  " f"({rb['sigma'][i]/ra['sigma'][i]:.3f})"
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, help="YAML forecast spec")
    ap.add_argument(
        "--only", nargs="+", default=None, help="run only these named forecasts (still honours --compare among them)"
    )
    ap.add_argument("--no-priors", dest="no_priors", action="store_true", help="override options.priors -> False")
    ap.add_argument("--priors", dest="force_priors", action="store_true", help="override options.priors -> True")
    ap.add_argument("-v", "--verbosity", default="warning")
    args = ap.parse_args()
    logger.set_all_loggers_level(args.verbosity)

    spec = input_output.read_yaml(args.spec)
    block_defs, sources, forecasts, compare, options, conf = resolve(spec)
    if args.no_priors:
        options["priors"] = False
    if args.force_priors:
        options["priors"] = True

    names = list(forecasts) if args.only is None else args.only
    for n in names:
        if n not in forecasts:
            raise KeyError(f"--only: unknown forecast '{n}'")

    results = {}
    for name in names:
        fc = forecasts[name]
        src = sources[fc["source"]]
        LOGGER.info(f"running forecast '{name}' (source={fc['source']}, blocks={fc['blocks']})")
        results[name] = fisher.run_forecast(
            cov_h5=src["cov"],
            jac_h5=src["jac"],
            blocks=fc["blocks"],
            conf=conf,
            priors=options["priors"],
            hartlap=options["hartlap"],
            normalize=options["normalize"],
            block_defs=block_defs,
        )
        print_forecast(name, results[name])

    if compare:
        print("\n" + "=" * 78)
        print("COMPARISONS")
        print("=" * 78)
        for pair in compare:
            a, b = pair
            if a in results and b in results:
                print_comparison(a, results[a], b, results[b])


if __name__ == "__main__":
    main()
