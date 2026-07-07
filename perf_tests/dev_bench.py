"""
Fast single-operator, single-resolution benchmark for iterating on one strategy implementation.

``run.py`` always sweeps *every* discovered strategy of a bench axis over one of the five fixed
``problems.PROBLEMS`` presets -- great for a full sweep, slow and noisy for "does my one new
InversionStrategy get faster after this change, on a mesh small enough to iterate on". This script
pins both: one bench axis (``--bench``), optionally narrowed to one strategy class (``--strategy``),
against a resolution you set directly (``--dim/--degree/--n-other/--res-mult/--tn-order``) instead of
picking from the presets. It reuses each ``bench_*.py`` module's own ``Bench`` definition (extract /
make_call / overrides), so it stays wired to the same construction path as the full suite.

Examples
--------
    # list available axes
    python dev_bench.py --list

    # every discovered forward strategy, default mv-d1-60k resolution
    python dev_bench.py --bench transport.forward

    # only your new strategy, tiny mesh for a fast dev loop
    python dev_bench.py --bench transport.inv --strategy MySweep --n-other 2 --tn-order 2

    # custom resolution across the board, one named preset instead
    python dev_bench.py --bench transport.forward --problem tiny-d1

    # one (inversion, angle-lifting) pair from the multiangle product
    python dev_bench.py --bench transport.inv.multiangle --strategy MySweep --lifting VmapAngles

    # Warp example
    python dev_bench.py --bench transport.inv.multiangle \
    --strategy WarpLevelSweep --lifting WarpLevelAngles \
    --degree 1 --n-other 5 --tn-order 3 --res-mult 4
"""
import argparse

from _common import measure, LiveTable
from _create_problem import build_problem
from _discovery import discover_variants, serializes
from run import discover_benches


def _filter_by(variants, spec):
    """``spec`` is a name/substring, or a comma list of several -- any match keeps the variant."""
    patterns = [s.strip().lower() for s in spec.split(",") if s.strip()]
    return [v for v in variants if any(p in v.label.lower() for p in patterns)]


def _resolve_problem(args):
    if args.problem:
        import problems
        return problems.build(args.problem)
    params = dict(dim=args.dim, degree=args.degree, n_other=args.n_other,
                  res_mult=args.res_mult, tn_order=args.tn_order)
    sd, qs = build_problem(**params)
    info = (f"custom: dim {params['dim']} deg {params['degree']} "
            f"elems {sd.domain.n_elements} angles {qs.angles.shape[0]} "
            f"groups {sd.domain.cross_sections_indexed.n_energy_groups}")
    return sd, qs, info


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", default="", help="axis name, e.g. transport.forward, transport.inv, mass.mv")
    p.add_argument("--list", action="store_true", help="list available --bench axes and exit")
    p.add_argument("--strategy", default="",
                    help="restrict to discovered strategy class(es): name/substring, or a comma list for several "
                         "(e.g. 'BatchedSweep,VmapSweep'); default: sweep all")
    p.add_argument("--lifting", default="",
                    help="transport.inv.multiangle only: restrict to discovered AngleLifting(s), same comma-list "
                         "syntax as --strategy")
    p.add_argument("--problem", default="", help="use a named preset from problems.PROBLEMS instead of the --dim/etc flags")
    p.add_argument("--dim", type=int, default=3)
    p.add_argument("--degree", type=int, default=1)
    p.add_argument("--n-other", type=int, default=3)
    p.add_argument("--res-mult", type=int, default=1)
    p.add_argument("--tn-order", type=int, default=2)
    args = p.parse_args()

    benches = {b.name: b for b in discover_benches() if hasattr(b, "base")}
    axis_names = sorted(benches) + ["transport.inv.multiangle"]  # bespoke two-axis bench (see below)
    if args.list or not args.bench:
        print("available --bench axes (transport.inv.multiangle also takes --lifting; solve.* is not covered here):")
        for name in axis_names:
            print(f"  {name}")
        raise SystemExit(0)
    if args.bench == "transport.inv.multiangle":
        _run_multiangle(args)
        return
    if args.bench not in benches:
        raise SystemExit(f"unknown --bench {args.bench!r}; choices: {axis_names}")
    bench = benches[args.bench]

    sd, qs, info = _resolve_problem(args)
    print(f">>> {info}", flush=True)

    variants = discover_variants(bench.base, bench.overrides, bench.instantiate)
    if args.strategy:
        variants = _filter_by(variants, args.strategy)
        if not variants:
            raise SystemExit(f"no discovered variant of {bench.base.__name__} matches {args.strategy!r}")

    from jax_sn.operators.multi_group_operator.multi_group_operator_creation import create_multi_group_operator

    table = LiveTable(bench.name, [v.label for v in variants])
    for v in variants:
        table.start(v.label)
        if v.instance is None:
            table.skip(v.skip_reason)
            continue

        def build_fn(strat=v.instance):
            mgop = create_multi_group_operator(sd, qs, **{bench.kwarg: strat})
            return bench.make_call(bench._extract(mgop))

        m = measure(build_fn)
        table.ok(m.peak_mb, m.t_ms) if m.status == "ok" else table.skip(m.reason)


def _run_multiangle(args):
    """``transport.inv`` swept over the (InversionStrategy x AngleLifting) product -- mirrors
    ``MultiAngleInvBench.run`` in ``bench_transport_multiangle.py``, but filterable to one pair via
    ``--strategy``/``--lifting`` and buildable at a custom resolution."""
    from bench_transport import _INV_OVERRIDES
    from jax_sn.operators.transport_operator import InversionStrategy, AngleLifting, BatchedForward
    from jax_sn.operators.multi_group_operator.multi_group_operator_creation import create_multi_group_operator
    from _bench import inv_call

    sd, qs, info = _resolve_problem(args)
    print(f">>> {info}", flush=True)

    invs = [v for v in discover_variants(InversionStrategy, _INV_OVERRIDES) if v.instance is not None]
    lifts = [v for v in discover_variants(AngleLifting) if v.instance is not None]
    if args.strategy:
        invs = _filter_by(invs, args.strategy)
        if not invs:
            raise SystemExit(f"no discovered InversionStrategy matches --strategy {args.strategy!r}")
    if args.lifting:
        lifts = _filter_by(lifts, args.lifting)
        if not lifts:
            raise SystemExit(f"no discovered AngleLifting matches --lifting {args.lifting!r}")

    pairs = [(i, l) for i in invs for l in lifts if not serializes(l.instance, i.instance)]
    if not pairs:
        raise SystemExit("no (inversion, lifting) pair survived the serializes() filter for this selection")

    table = LiveTable("transport.inv.multiangle", [f"{i.label} + {l.label}" for i, l in pairs])
    for inv, lift in pairs:
        label = f"{inv.label} + {lift.label}"
        table.start(label)

        def build_fn(i=inv.instance, l=lift.instance):
            mgop = create_multi_group_operator(
                sd, qs, forward_strategy=BatchedForward(1000),
                inversion_strategy=i, angle_lifting=l)
            return inv_call(mgop.transport_operator)

        m = measure(build_fn)
        table.ok(m.peak_mb, m.t_ms) if m.status == "ok" else table.skip(m.reason)


if __name__ == "__main__":
    main()
