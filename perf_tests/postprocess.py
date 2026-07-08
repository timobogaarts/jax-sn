"""
Breaks the aggregate ``SGOp.mv``/``MGOp.mv`` golden numbers down into their constituent operator costs,
so a timing/memory regression can be pinned on the operator that actually changed instead of just the
end-to-end number moving.

``SGOp.mv`` (see ``single_group_operator_imp.py:_mv_single_group_operator``) applies, per matvec::

    x - moment(transport.inv(discrete(mass(inscatter(x)))))

i.e. one inscatter + mass + discrete + transport-inverse (single energy group, full angle set) + moment.
``MGOp.mv`` (``multi_group_operator_imp.py:_multi_group_operator_scan``) is the same pipeline scanned
once per energy group, with that group's ``gather_forward`` (``GatherScatterOperator``) standing in for
the single-group inscatter. In both, mass is applied immediately adjacent to inscatter/gather_forward,
i.e. in moment space rather than discrete/angle space (mass and discrete commute -- independent tensor
factors -- so this ordering is exact, not approximate; moment space is just the smaller of the two). See
``bench_mass.py``'s ``mass.mv.moment`` axis, sized accordingly.

The perf-test suite already benchmarks every one of those constituents standalone (``mass.mv.moment``,
``discrete.mv``, ``moment.mv``, ``scatter.gather``, ``transport.inv.multiangle``) as well as the two
aggregates, all keyed by the same ``InversionStrategy`` label -- so this script only has to look up and
sum the matching rows of a golden JSON. It needs no jax/GPU: it is pure post-processing of
``perf_tests/golden/<name>.json``.

Two facts are not recorded in the golden JSON itself and have to be hard-coded here:

- the ``InversionStrategy -> default AngleLifting`` pairing that ``transport.inv.multiangle`` is keyed
  by (mirrors each strategy's ``default_angle_lifting()``; see ``operators/transport_operator/
  implementations/*.py``);
- the number of energy groups the default MV problem is built with -- the breeding-blanket
  cross-section set is always 4-group (see ``_create_problem.bb_data_dict``), independent of
  ``--n-other``/``--res-mult``/etc.

There is no standalone bench for the plain (non-gather) ``ScatterOperator`` used inside ``SGOp`` --
it's a cheap element-wise multiply, not worth its own axis -- so it, plus any XLA fusion gain/loss from
compiling the whole pipeline as one graph instead of five separate ones, falls into the reported
"unaccounted" residual.

This module also has a ``compare`` subcommand for diffing two already-saved golden files directly
(e.g. before/after a kernel change on the same machine) -- no re-running/GPU needed, just a row-by-row
``peak_mb``/``t_ms`` diff, reusing ``_common.compare_golden`` for the same fail/warn thresholds
``run.py --compare`` uses.

Usage
-----
    python postprocess.py breakdown                # perf_tests/golden/rtx3080.json
    python postprocess.py breakdown mymachine       # perf_tests/golden/mymachine.json
    python postprocess.py compare rtx3080_mass_switch rtx3080   # new vs. baseline, row-by-row diff
"""
import argparse

from _common import load_golden, compare_golden

N_ENERGY_GROUPS = 4  # fixed by _create_problem.bb_data_dict; not swept by any --problem parameter

# create_multi_group_operator's defaults for every strategy axis SGOp.mv/MGOp.mv don't override
# (bench_machinery.py only ever passes inversion_strategy) -- the label each shows up under in its
# own standalone bench.
DEFAULT_STRATEGY_LABEL = dict(
    mass="VmapMass", discrete="BatchedDiscrete", moment="BatchedMoment",
    scatter="GatherScatterOperatorBroadcasted",
)

MASS_BENCH = "mass.mv.moment"  # moment-space sizing -- see bench_mass.py; matches actual SGOp/MGOp usage

# InversionStrategy label (as it appears in SGOp.mv/MGOp.mv) -> its default_angle_lifting(), i.e. the
# other half of the "<inv> + <lift>" label transport.inv.multiangle uses for the same combination.
DEFAULT_LIFT = {
    "VmapSweep": "VmapAngles",
    "BatchedSweep(1000)": "VmapAngles",
    "VmapPallasSweep": "VmapAngles",
    "BatchedPallasSweep(1000)": "VmapAngles",
    "WarpSequentialSweep": "WarpSequentialAngles",
    "WarpLevelSweep": "WarpLevelAngles",
}

PIPELINE = ("scatter", "mass", "discrete", "moment", "transport")


def _point(golden: dict, bench: str, label: str):
    """``(peak_mb, t_ms)`` for one golden row, or ``None`` if missing / skipped / OOM."""
    row = golden.get(bench, {}).get(label)
    if not row or row.get("status") != "ok":
        return None
    mb, t = row.get("peak_mb"), row.get("t_ms")
    if not isinstance(mb, (int, float)) or not isinstance(t, (int, float)):
        return None
    return mb, t


def _constituents(golden: dict, inv_label: str) -> dict:
    """``{name: (peak_mb, t_ms) | None}`` for the per-group pipeline pieces at this inversion
    strategy. ``None`` marks a piece that's missing or skipped (e.g. an incompatible warp combo)."""
    lift = DEFAULT_LIFT.get(inv_label)
    transport_label = f"{inv_label} + {lift}" if lift else None
    return {
        "mass":      _point(golden, MASS_BENCH, DEFAULT_STRATEGY_LABEL["mass"]),
        "discrete":  _point(golden, "discrete.mv", DEFAULT_STRATEGY_LABEL["discrete"]),
        "moment":    _point(golden, "moment.mv", DEFAULT_STRATEGY_LABEL["moment"]),
        "scatter":   _point(golden, "scatter.gather", DEFAULT_STRATEGY_LABEL["scatter"]),
        "transport": _point(golden, "transport.inv.multiangle", transport_label) if transport_label else None,
    }


def _time_row(name: str, t: float, total_t: float):
    pct = f"{100 * t / total_t:5.1f}%" if total_t else "    - "
    print(f"    {name:<12} {t:9.3f} ms {pct}")


def _mem_row(name: str, mb: float):
    print(f"    {name:<12} {mb:9.1f} MB")


def _missing_row(name: str):
    print(f"    {name:<12} (missing)")


def _sgop_breakdown(golden: dict, inv_label: str, parts: dict):
    sgop = _point(golden, "SGOp.mv", inv_label)
    if sgop is None:
        print("  SGOp.mv: skip/OOM")
        return
    total_mb, total_t = sgop
    have = {n: parts[n] for n in ("mass", "discrete", "moment", "transport") if parts.get(n) is not None}

    print(f"  SGOp.mv actual peak memory: {total_mb:9.1f} MB")
    for name, (mb, _) in have.items():
        _mem_row(name, mb)
    print(f"    {'(inscatter)':<12} not benchmarked separately -- plain element-wise ScatterOperator")
    print("    peak memory is a max-concurrent-usage figure, not additive across ops -- the standalone "
          "component peaks above are each that op's *own* full peak (inputs+outputs+temporaries), not an "
          "incremental contribution to the fused SGOp graph, so they are shown for comparison (which op "
          "is the memory bottleneck) rather than summed.")

    print(f"\n  SGOp.mv actual time: {total_t:9.3f} ms  (time is roughly additive across sequential ops)")
    acc_t = 0.0
    for name in ("mass", "discrete", "moment", "transport"):
        pt = parts.get(name)
        if pt is None:
            _missing_row(name)
            continue
        _, t = pt
        acc_t += t
        _time_row(name, t, total_t)
    _time_row("unaccounted (inscatter + fusion delta)", total_t - acc_t, total_t)


def _mgop_breakdown(golden: dict, inv_label: str, parts: dict):
    mgop = _point(golden, "MGOp.mv", inv_label)
    if mgop is None:
        print("  MGOp.mv: skip/OOM")
        return
    total_mb, total_t = mgop
    print(f"  MGOp.mv actual peak memory: {total_mb:9.1f} MB  (scan over {N_ENERGY_GROUPS} groups)")
    missing = [name for name in PIPELINE if parts.get(name) is None]
    for name in PIPELINE:
        if parts.get(name) is not None:
            _mem_row(name, parts[name][0])
    if missing:
        print(f"    missing constituent(s), can't reconstruct time breakdown: {', '.join(missing)}")
        return
    # Memory does NOT scale with groups: the scan reuses one group's temporaries (see
    # _multi_group_operator_scan's docstring), so the single-group components above are the right
    # memory comparison, not N_ENERGY_GROUPS x them -- and, as for SGOp, not additive either.
    print("    (not additive -- see SGOp note above; also not scaled by groups, since the scan reuses "
          "one group's temporaries)")

    pg_t = sum(parts[name][1] for name in PIPELINE)
    expected_t = pg_t * N_ENERGY_GROUPS  # scan runs the pipeline once per group -> time scales with groups
    print(f"\n  MGOp.mv actual time: {total_t:9.3f} ms  "
          f"(expect ~= {N_ENERGY_GROUPS} x one pass through scatter+mass+discrete+moment+transport)")
    for name in PIPELINE:
        _, t = parts[name]
        _time_row(name, t * N_ENERGY_GROUPS, expected_t)
    _time_row("unaccounted (scan overhead + fusion delta)", total_t - expected_t, expected_t)


def report(golden_name: str):
    golden = load_golden(golden_name)
    inv_labels = sorted(set(golden.get("SGOp.mv", {})) | set(golden.get("MGOp.mv", {})))
    if not inv_labels:
        print(f"no SGOp.mv/MGOp.mv rows found in {golden_name}.json")
        return
    for inv_label in inv_labels:
        print(f"\n=== {inv_label} ===")
        parts = _constituents(golden, inv_label)
        _sgop_breakdown(golden, inv_label, parts)
        print()
        _mgop_breakdown(golden, inv_label, parts)


def _diff_rows(new: dict, old: dict):
    """Yield ``(bench, label, old_row, new_row)`` for every row present in either file, sorted."""
    keys = {(bench, label) for d in (old, new) for bench, labels in d.items() for label in labels}
    for bench, label in sorted(keys):
        yield bench, label, old.get(bench, {}).get(label), new.get(bench, {}).get(label)


def compare(new_name: str, old_name: str, mem_tol: float, time_tol: float, all_rows: bool) -> int:
    """Row-by-row ``peak_mb``/``t_ms`` diff of two already-saved golden files -- no re-running.
    Returns a process exit code (1 if any row exceeds ``mem_tol``, mirroring ``run.py --compare``)."""
    new, old = load_golden(new_name), load_golden(old_name)

    print(f"comparing {new_name}.json (new) vs {old_name}.json (baseline)")
    print(f"{'bench':<28}{'label':<32}{'peak_mb old':>12}{'peak_mb new':>12}{'d_mb':>8}"
          f"{'t_ms old':>12}{'t_ms new':>12}{'d_t':>8}")
    for bench, label, o, n in _diff_rows(new, old):
        if o is None:
            print(f"{bench:<28}{label:<32}  new-only row")
        elif n is None:
            print(f"{bench:<28}{label:<32}  missing in {new_name}")
        elif o.get("status") != "ok" or n.get("status") != "ok":
            if all_rows:
                print(f"{bench:<28}{label:<32}  status: {o.get('status')} -> {n.get('status')}")
        else:
            omb, nmb, ot, nt = o["peak_mb"], n["peak_mb"], o["t_ms"], n["t_ms"]
            dmb = 100 * (nmb - omb) / omb if omb else 0.0
            dt = 100 * (nt - ot) / ot if ot else 0.0
            if all_rows or abs(dmb) > 1 or abs(dt) > 1:
                print(f"{bench:<28}{label:<32}{omb:12.1f}{nmb:12.1f}{dmb:7.1f}%{ot:12.3f}{nt:12.3f}{dt:7.1f}%")

    fails, warns = compare_golden(new, old, mem_tol, time_tol)
    print(f"\n================ golden compare ({new_name} vs {old_name}) ================")
    print("\n".join("  FAIL " + f for f in fails) or "  peak memory within tolerance.")
    print("\n".join("  warn " + f for f in warns) or "  time within tolerance.")
    return 1 if fails else 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_break = sub.add_parser("breakdown", help="per-operator SGOp.mv/MGOp.mv cost breakdown")
    p_break.add_argument("golden", nargs="?", default="rtx3080",
                          help="golden file name, i.e. perf_tests/golden/<name>.json (default: rtx3080)")

    p_cmp = sub.add_parser("compare", help="row-by-row diff of two already-saved golden files")
    p_cmp.add_argument("new", help="golden file name to treat as the new/current result")
    p_cmp.add_argument("old", nargs="?", default="rtx3080",
                        help="golden file name to treat as the baseline (default: rtx3080)")
    p_cmp.add_argument("--mem-tol", type=float, default=0.05,
                        help="peak_mb relative drift that counts as FAIL (default 0.05)")
    p_cmp.add_argument("--time-tol", type=float, default=0.25,
                        help="t_ms relative drift that counts as warn (default 0.25)")
    p_cmp.add_argument("--all", action="store_true", help="print every row, not just ones that moved >1%%")

    args = p.parse_args()
    if args.cmd == "compare":
        raise SystemExit(compare(args.new, args.old, args.mem_tol, args.time_tol, args.all))
    report(args.golden)


if __name__ == "__main__":
    main()
