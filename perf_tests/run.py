"""
Orchestrator for the perf-test suite.

Discovers every ``bench_*.py`` module in this folder, collects their ``BENCHES``, and runs the selected
group against a problem size -- dumping/comparing per-machine golden statistics.

    python run.py                              # cheap group, default size, live aligned report
    python run.py --group all --save rtx3080   # include the expensive solve/grad benches
    python run.py --benches mass.mv,moment.mv
    python run.py --compare rtx3080

Each ``bench_*.py`` is also runnable on its own (``python bench_mass.py``); it just calls
``run_module`` to run that file's benches with the same CLI.
"""
import argparse
import glob
import importlib
import os
import sys

import jax

import problems
from _common import save_golden, load_golden, compare_golden


def discover_benches(only_module: str = None):
    """Import every ``bench_*.py`` (or just ``only_module``) and return a flat list of Bench objects."""
    here = os.path.dirname(__file__)
    if here not in sys.path:
        sys.path.insert(0, here)
    names = ([only_module] if only_module
             else [os.path.splitext(os.path.basename(p))[0]
                   for p in sorted(glob.glob(os.path.join(here, "bench_*.py")))])
    benches = []
    for name in names:
        mod = importlib.import_module(name)
        benches.extend(getattr(mod, "BENCHES", []))
    return benches


def select(benches, group: str, wanted: str):
    if group != "all":
        benches = [b for b in benches if b.group == group]
    if wanted:
        names = set(wanted.split(","))
        benches = [b for b in benches if b.name in names]
    return benches


def _problem_for(bench, override: str) -> str:
    if override:
        return override
    return problems.DEFAULT_SOLVE if bench.group == "expensive" else problems.DEFAULT_MV


def run_benches(benches, args) -> dict:
    """Group benches by resolved problem size, build each problem once, run, and report."""
    results = {}
    by_problem = {}
    for b in benches:
        by_problem.setdefault(_problem_for(b, args.problem), []).append(b)

    for pname, group in by_problem.items():
        sd, qs, info = problems.build(pname)
        print(f"\n>>> {info}", flush=True)
        for b in group:
            results[b.name] = b.run(sd, qs)   # each bench prints its own live aligned result block
    return results


def _add_common_args(p):
    p.add_argument("--problem", default="", help=f"problem name (default per group); one of {list(problems.PROBLEMS)}")
    p.add_argument("--save", default="", help="golden name to write/merge into")
    p.add_argument("--compare", default="", help="golden name to compare against (exit 1 on peak-mem drift)")


def _finish(results, args) -> int:
    if args.save:
        print(f"\nwrote golden -> {save_golden(args.save, results)}")
    if args.compare:
        fails, warns = compare_golden(results, load_golden(args.compare))
        print("\n================ golden compare ================")
        print("\n".join("  FAIL " + f for f in fails) or "peak memory within tolerance of golden.")
        if warns:
            print("time drift (advisory):")
            print("\n".join("  warn " + w for w in warns))
        return 1 if fails else 0
    return 0


def run_module(module_name: str):
    """Entry point for a standalone ``python bench_x.py`` run -- runs only that module's benches."""
    p = argparse.ArgumentParser()
    _add_common_args(p)
    args = p.parse_args()
    print("devices:", jax.devices())
    benches = discover_benches(only_module=module_name)
    results = run_benches(benches, args)
    raise SystemExit(_finish(results, args))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", choices=["cheap", "expensive", "all"], default="cheap")
    p.add_argument("--benches", default="", help="comma list of bench names (overrides group)")
    _add_common_args(p)
    args = p.parse_args()

    print("devices:", jax.devices())
    benches = select(discover_benches(), args.group, args.benches)
    if not benches:
        print("no benches selected.")
        raise SystemExit(0)
    results = run_benches(benches, args)
    raise SystemExit(_finish(results, args))


if __name__ == "__main__":
    main()
