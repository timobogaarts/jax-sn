"""
Fast single-operator, single-resolution benchmark for iterating on one strategy implementation.

``run.py`` always sweeps *every* discovered strategy of a bench axis over one of the five fixed
``problems.PROBLEMS`` presets -- great for a full sweep, slow and noisy for "does my one new
SingleAngleInversionStrategy get faster after this change, on a mesh small enough to iterate on". This script
pins both: one bench axis (``--bench``), optionally narrowed to one strategy class (``--strategy``),
against a resolution you set directly (``--dim/--degree/--n-other/--res-mult/--tn-order``) instead of
picking from the presets. It reuses each ``bench_*.py`` module's own ``Bench`` definition (chain_field /
lift / extract / make_call / overrides), so it stays wired to the same construction path as the full suite.

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

    # one inversion with an explicit angle inversion (combinator wraps the inversion; native stands alone)
    python dev_bench.py --bench transport.inv.multiangle --strategy MySweep --lifting ScanAnglesInversion

    # Warp example
    python dev_bench.py --bench transport.inv.multiangle \
    --strategy WarpLevelSweep --lifting WarpLevelAngles \
    --degree 1 --n-other 5 --tn-order 3 --res-mult 4
"""
import argparse

from _common import measure, LiveTable
from _create_problem import build_problem
from _discovery import discover_variants
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
                    help="transport.inv.multiangle only: override the angle inversion with discovered MultiAngleInversionStrategy "
                         "class(es) (native, or combinator wrapping the inversion), same comma-list syntax as --strategy; "
                         "default: each strategy's auto-resolved natural lift")
    p.add_argument("--problem", default="", help="use a named preset from problems.PROBLEMS instead of the --dim/etc flags")
    p.add_argument("--dim", type=int, default=3)
    p.add_argument("--degree", type=int, default=1)
    p.add_argument("--n-other", type=int, default=20)
    p.add_argument("--res-mult", type=int, default=1)
    p.add_argument("--tn-order", type=int, default=3)
    args = p.parse_args()

    benches = {b.name: b for b in discover_benches() if hasattr(b, "base")}
    axis_names = sorted(set(benches) | {"transport.inv.multiangle"})  # latter also takes --lifting (see below)
    if args.list or not args.bench:
        print("available --bench axes (per-strategy discovery -- transport.inv.multiangle also takes --lifting):")
        for name in axis_names:
            print(f"  {name}")
        print("MGOp.*/SGOp.*/solve.*/grad.solve.* sweep the hand-picked composed strategies "
              "(_bench.default_multi_group_strategies) instead and aren't covered here.")
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

    from _bench import compose_chain_strategy, create_multi_group_operator

    table = LiveTable(bench.name, [v.label for v in variants])
    for v in variants:
        table.start(v.label)
        if v.instance is None:
            table.skip(v.skip_reason)
            continue

        def build_fn(strat=v.instance):
            strategy = compose_chain_strategy(bench.chain_field, strat, bench.lift, bench.top_level)
            mgop = create_multi_group_operator(sd, qs, strategy)
            return bench.make_call(bench.extract(mgop))

        m = measure(build_fn)
        table.ok(m.peak_mb, m.t_ms) if m.status == "ok" else table.skip(m.reason)


def _multi_angle_inversion_strategy_for(lift_cls, inv_instance):
    """Build an :class:`MultiAngleInversionStrategy` of class ``lift_cls`` for a single-angle inversion instance: a
    native (e.g. warp) default-constructs; a combinator (``VmapAnglesInversion``/``ScanAnglesInversion``) wraps the
    inversion as its ``inner``. Returns ``None`` if the pairing is illegal (e.g. ``VmapAnglesInversion`` over an
    opaque kernel, which ``__check_init__`` rejects)."""
    try:
        return lift_cls()                 # native: owns no inner
    except TypeError:
        pass
    try:
        return lift_cls(inv_instance)     # combinator: wrap this inversion
    except Exception:
        return None                       # illegal pairing (e.g. VmapAnglesInversion over opaque)


def _run_multiangle(args):
    """``transport.inv`` swept over every ``SingleAngleInversionStrategy``, each using its auto-resolved natural
    angle inversion -- mirrors ``bench_transport_multiangle.py``, but filterable via ``--strategy`` and buildable
    at a custom resolution. ``--lifting`` overrides the angle inversion with a specific
    ``MultiAngleInversionStrategy`` class (a native, or a combinator wrapping the inversion), so an off-default
    pairing like ``ScanAnglesInversion`` serialisation is benchmarkable."""
    from bench_transport import _INV_OVERRIDES
    from bench_transport_multiangle import multi_angle
    from jax_sn.operators.transport_operator import (
        SingleAngleInversionStrategy, MultiAngleInversionStrategy, VmapAnglesForward, BatchedForward,
    )
    from _bench import create_multi_group_operator, default_multi_group_strategy, inv_call, lift_inversion
    import equinox as eqx

    sd, qs, info = _resolve_problem(args)
    print(f">>> {info}", flush=True)

    invs = [v for v in discover_variants(SingleAngleInversionStrategy, _INV_OVERRIDES) if v.instance is not None]
    if args.strategy:
        invs = _filter_by(invs, args.strategy)
        if not invs:
            raise SystemExit(f"no discovered SingleAngleInversionStrategy matches --strategy {args.strategy!r}")

    lift_classes = None
    if args.lifting:
        lift_classes = [v.instance for v in _filter_by(
            discover_variants(MultiAngleInversionStrategy, instantiate=False), args.lifting)]
        if not lift_classes:
            raise SystemExit(f"no discovered MultiAngleInversionStrategy matches --lifting {args.lifting!r}")

    # (label, angle inversion) rows; default = auto-resolved natural lift.
    rows = []
    for inv in invs:
        if lift_classes is None:
            rows.append((inv.label, lift_inversion(inv.instance)))
        else:
            for lc in lift_classes:
                ai = _multi_angle_inversion_strategy_for(lc, inv.instance)
                if ai is not None:
                    rows.append((f"{inv.label} + {lc.__name__}", ai))
    if not rows:
        raise SystemExit("no (inversion, angle-inversion) row for this selection")

    table = LiveTable("transport.inv.multiangle", [label for label, _ in rows])
    for label, ai in rows:
        table.start(label)

        def build_fn(a=ai):
            base = default_multi_group_strategy()
            strategy = eqx.tree_at(
                lambda s: (s.single_group_chain_strategy.transport_strategy,
                           s.single_group_chain_strategy.transport_inversion_strategy),
                base, (VmapAnglesForward(BatchedForward(1000)), a))
            mgop = create_multi_group_operator(sd, qs, strategy)
            return inv_call(multi_angle(mgop))

        m = measure(build_fn)
        table.ok(m.peak_mb, m.t_ms) if m.status == "ok" else table.skip(m.reason)


if __name__ == "__main__":
    main()
