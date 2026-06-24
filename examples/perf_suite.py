"""
Performance suite for the jax-sn operator machinery: a matrix of **operator configurations** (transport
forward/inversion/lift strategies, mass / moment / discrete strategies, MGOp class) x **problem sizes**,
reporting predicted peak memory + steady-state wall time per operation.

Lives in `examples/` (not `tests/`): the timing/memory numbers are machine-specific, so they are checked
against a per-machine golden you record yourself, not asserted in CI.

Two regimes (different sizes, because the right size differs by op cost):
  - MV regime    : the single-application ops (MGOp/SGOp/transport/mass/moment/discrete mv + the sweep
                   `transport.inv`) at a high element count, swept over degree 1/2/3.
  - SOLVE regime : the iterative ops (full BiCGStab solve, lower-triangular solve, grad-through-solve)
                   at a smaller element count + fewer angles. Also reports the BiCGStab iteration counts
                   and an implied `ms/iter`, so a solve's cost can be sanity-checked against the cost of
                   one MGOp.mv (full solve) or one transport sweep (lower-tri, per group).

Two passes:
  1. CONSISTENCY (machine-independent): every buildable config's `mv`/`inv` == the `vmap` reference to fp
     tolerance -- catches a broken warp/pallas/lax/streaming kernel on any machine. `--strict` -> exit 1.
  2. PERFORMANCE (machine-specific): peak memory (compiler-predicted -> never OOMs) + min-of-N wall time.

Usage:
    python examples/perf_suite.py --regime solve --progress      # quick-ish, watch progress
    python examples/perf_suite.py --regime mv                    # the 60k deg1/2/3 sweep (heavy)
    python examples/perf_suite.py --configs vmap,warp-levels
    python examples/perf_suite.py --save rtx3080 / --compare rtx3080
"""
import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as onp
import lineax as lx
import equinox as eqx

jax.config.update("jax_enable_x64", True)

from _perf_common import peak_mb, time_ms, build_problem
from jax_sn.operators.multi_group_operator.multi_group_operator_creation import create_multi_group_operator
from jax_sn.operators.multi_group_operator import (
    MultiGroupLowerTriangularBlockSolver, MultiGroupOperatorScanCheckpoint,
)
from jax_sn.parametrisation import MultigroupOperatorParametrisedVerticesXS

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "perf_golden")

# Problem sizes per regime. n_other controls element count (6 tets / hypercube * sum(base_res)*res_mult
# * n_other**(dim-1)); tn_order 2 -> 32 angles, 3 -> 72 angles.
MV_PROBLEMS: Dict[str, dict] = {
    "mv-d1-60k": dict(dim=3, degree=1, n_other=20, res_mult=1, tn_order=3),
    "mv-d2-60k": dict(dim=3, degree=2, n_other=20, res_mult=1, tn_order=3),
    "mv-d3-60k": dict(dim=3, degree=3, n_other=20, res_mult=1, tn_order=3),
}
SOLVE_PROBLEMS: Dict[str, dict] = {
    "solve-d1-10k": dict(dim=3, degree=1, n_other=8, res_mult=1, tn_order=2),   # ~9.6k elems, 32 angles
    "solve-d2-10k": dict(dim=3, degree=2, n_other=8, res_mult=1, tn_order=2),
}


# --------------------------------------------------------------------------------------------------
# Configurations (named bundles of create_multi_group_operator kwargs)
# --------------------------------------------------------------------------------------------------
@dataclass
class OpConfig:
    name: str
    kwargs: dict = field(default_factory=dict)
    requires: Tuple[str, ...] = ()   # "warp", "gpu", "tet1" -> skipped if not satisfied


def make_configs():
    from jax_sn.operators.transport_operator import (
        BatchedForward, BatchedSweep, VmapPallasSweep, BatchedPallasSweep,
    )
    from jax_sn.operators.mass_operator import LaxMass
    from jax_sn.operators.moment_operator import StreamingMoment

    cfgs = [
        OpConfig("vmap"),                                                              # all defaults (reference)
        OpConfig("batched-100", dict(forward_strategy=BatchedForward(100),
                                     inversion_strategy=BatchedSweep(100))),
        OpConfig("pallas-vmap", dict(inversion_strategy=VmapPallasSweep()), requires=("gpu",)),
        OpConfig("pallas-batched-100", dict(forward_strategy=BatchedForward(100),
                                            inversion_strategy=BatchedPallasSweep(100)), requires=("gpu",)),
        OpConfig("lax-mass", dict(mass_strategy=LaxMass())),
        OpConfig("streaming-moment", dict(moment_strategy=StreamingMoment())),
        OpConfig("checkpoint", dict(multi_group_operator_cls=MultiGroupOperatorScanCheckpoint)),
    ]
    try:
        from jax_sn.operators.transport_operator.transport_operator_warp import (
            WarpSequentialSweep, WarpSequentialAngles, WarpLevelSweep, WarpLevelAngles,
        )
        cfgs += [
            OpConfig("warp-sequential", dict(forward_strategy=BatchedForward(100),
                                             inversion_strategy=WarpSequentialSweep(),
                                             angle_lifting=WarpSequentialAngles()), requires=("warp", "gpu", "tet1")),
            OpConfig("warp-levels", dict(forward_strategy=BatchedForward(100),
                                         inversion_strategy=WarpLevelSweep(),
                                         angle_lifting=WarpLevelAngles(),
                                         sweep_data_kwargs={"maximum_number_of_levels": None}),
                     requires=("warp", "gpu", "tet1")),
        ]
    except ImportError:
        pass
    return cfgs


def _capabilities(problem: dict):
    caps = set()
    if any("cuda" in d.platform.lower() or d.platform == "gpu" for d in jax.devices()):
        caps.add("gpu")
    try:
        import jax_sn.operators.transport_operator.transport_operator_warp  # noqa: F401
        caps.add("warp")
    except ImportError:
        pass
    if problem["dim"] == 3 and problem["degree"] == 1:
        caps.add("tet1")  # warp kernels are linear-tetrahedron only
    return caps


# --------------------------------------------------------------------------------------------------
# Operations
# --------------------------------------------------------------------------------------------------
def mv_ops(mgop):
    """Single-application ops: {name: (callable, args_tuple)}."""
    x_full = jnp.ones(mgop.in_structure().shape)
    sgop = mgop.to_single_group_operator(0)
    top = mgop.transport_operator
    x_ang = jnp.ones(top.in_structure().shape)
    x_mom = jnp.ones(mgop.discrete_operator.in_structure().shape)
    return {
        "MGOp.mv":       (lambda op, v: op.mv(v),             (mgop, x_full)),
        "MGOp.Tmv":      (lambda op, v: op.transpose().mv(v), (mgop, x_full)),
        "SGOp.mv":       (lambda op, v: op.mv(v),             (sgop, jnp.ones(sgop.in_structure().shape))),
        "transport.mv":  (lambda op, v: op.mv(v),             (top, x_ang)),
        "transport.inv": (lambda op, v: op.inv(v),            (top, x_ang)),
        "mass.mv":       (lambda op, v: op.mv(v),             (mgop.mass_operator, x_ang)),
        "moment.mv":     (lambda op, v: op.mv(v),             (mgop.moment_operator, x_ang)),
        "discrete.mv":   (lambda op, v: op.mv(v),             (mgop.discrete_operator, x_mom)),
    }


def _bicgstab():
    return lx.BiCGStab(1e-8, 1e-8, max_steps=400)


def _n_iter_full(mgop, rhs):
    out = eqx.filter_jit(lambda op, b: lx.linear_solve(op, b, _bicgstab(), throw=False))(mgop, rhs)
    return int(onp.asarray(out.stats.get("num_steps", -1)).sum())


def _n_iter_ltri(mgop, rhs):
    solver = MultiGroupLowerTriangularBlockSolver(_bicgstab())
    out = eqx.filter_jit(lambda op, b: lx.linear_solve(op, b, solver, throw=False))(mgop, rhs)
    stats = out.stats
    key = "num_steps_per_group" if "num_steps_per_group" in stats else "num_steps"
    return int(onp.asarray(stats.get(key, -1)).sum())


def solve_ops(mgop, include_grad=True):
    """Iterative ops. Returns {name: (callable, args, kind)} where kind labels how to get n_iter."""
    rhs = jnp.ones(mgop.in_structure().shape)
    theta = MultigroupOperatorParametrisedVerticesXS.theta_base(mgop)
    ltri = MultiGroupLowerTriangularBlockSolver(_bicgstab())

    def solve_iter(op, b):
        return jnp.sum(lx.linear_solve(op, b, _bicgstab(), throw=False).value)

    def solve_ltri(op, b):
        return jnp.sum(lx.linear_solve(op, b, ltri, throw=False).value)

    def reconstruct_solve(th, op, b):
        m = MultigroupOperatorParametrisedVerticesXS.reconstruct_operator(op, th)
        return jnp.sum(lx.linear_solve(m, b, ltri, throw=False).value)

    ops = {
        "solve.iter":      (solve_iter, (mgop, rhs), "full"),
        "solve.ltri":      (solve_ltri, (mgop, rhs), "ltri"),
    }
    if include_grad:
        ops["grad.solve.ltri"] = (lambda th, op, b: jax.value_and_grad(reconstruct_solve)(th, op, b),
                                  (theta, mgop, rhs), "ltri")
    return ops


# --------------------------------------------------------------------------------------------------
# Build / consistency / perf
# --------------------------------------------------------------------------------------------------
def _short(exc: Exception) -> str:
    s = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return s[:80]


def build_configs(sd, qs, configs, caps):
    built = {}
    for cfg in configs:
        missing = [r for r in cfg.requires if r not in caps]
        if missing:
            built[cfg.name] = ("skip", f"requires {','.join(missing)}")
            continue
        try:
            built[cfg.name] = ("ok", create_multi_group_operator(sd, qs, **cfg.kwargs))
        except Exception as e:
            built[cfg.name] = ("skip", _short(e))
    return built


def consistency_pass(built, progress, atol=1e-9, rtol=1e-6):
    out = {}
    if built.get("vmap", (None,))[0] != "ok":
        return {"_error": "vmap reference did not build"}
    ref = built["vmap"][1]
    x = jax.random.normal(jax.random.PRNGKey(0), ref.in_structure().shape)
    b = jax.random.normal(jax.random.PRNGKey(1), ref.transport_operator.in_structure().shape)
    ref_mv = jax.block_until_ready(ref.mv(x))
    ref_inv = jax.block_until_ready(ref.transport_operator.inv(b))
    for name, (status, obj) in built.items():
        if status != "ok":
            out[name] = status
            continue
        if progress:
            print(f"    [consistency] {name} ...", flush=True)
        try:
            e_mv = float(jnp.max(jnp.abs(obj.mv(x) - ref_mv)))
            e_inv = float(jnp.max(jnp.abs(obj.transport_operator.inv(b) - ref_inv)))
            ok = bool(onp.allclose(obj.mv(x), ref_mv, atol=atol, rtol=rtol)
                      and onp.allclose(obj.transport_operator.inv(b), ref_inv, atol=atol, rtol=rtol))
            out[name] = {"max_err_mv": e_mv, "max_err_inv": e_inv, "ok": ok}
        except Exception as e:
            out[name] = f"err: {_short(e)}"
    return out


def perf_pass_mv(built, number, repeats, progress, problem_name):
    out = {}
    for name, (status, obj) in built.items():
        if status != "ok":
            out[name] = {"_status": status}
            continue
        res = {"_status": "ok"}
        for op_name, (f, args) in mv_ops(obj).items():
            if progress:
                print(f"    [{problem_name}/{name}/{op_name}] ...", flush=True)
            res[op_name] = {"peak_mb": peak_mb(f, *args),
                            "t_ms": time_ms(f, *args, number=number, repeats=repeats)}
            if progress:
                print(f"      -> peak {res[op_name]['peak_mb']}  t {res[op_name]['t_ms']}", flush=True)
        out[name] = res
    return out


def perf_pass_solve(built, number, repeats, progress, problem_name, include_grad=True):
    out = {}
    for name, (status, obj) in built.items():
        if status != "ok":
            out[name] = {"_status": status}
            continue
        res = {"_status": "ok"}
        # reference per-iteration costs (one mv = one full-solve iter; one sweep ~ one lower-tri iter)
        refs = mv_ops(obj)
        for ref_name in ("MGOp.mv", "SGOp.mv", "transport.inv"):
            f, args = refs[ref_name]
            if progress:
                print(f"    [{problem_name}/{name}/{ref_name}] ...", flush=True)
            res[ref_name] = {"peak_mb": peak_mb(f, *args),
                             "t_ms": time_ms(f, *args, number=number, repeats=repeats)}
        rhs = jnp.ones(obj.in_structure().shape)
        n_iter_full = _n_iter_full(obj, rhs)
        n_iter_ltri = _n_iter_ltri(obj, rhs)
        for op_name, (f, args, kind) in solve_ops(obj, include_grad).items():
            if progress:
                print(f"    [{problem_name}/{name}/{op_name}] ...", flush=True)
            t = time_ms(f, *args, number=1, repeats=1)   # solves are long + low-variance: 1 warm-up + 1 timed
            n_iter = n_iter_full if kind == "full" else n_iter_ltri
            res[op_name] = {"peak_mb": peak_mb(f, *args), "t_ms": t, "n_iter": n_iter,
                            "ms_per_iter": (t / n_iter if (t and n_iter > 0) else None)}
            if progress:
                print(f"      -> peak {res[op_name]['peak_mb']}  t {t}  n_iter {n_iter}", flush=True)
        out[name] = res
    return out


# --------------------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------------------
def _fmt(v, suffix):
    if v is None:
        return f"{'OOM':>8}"
    if isinstance(v, str):
        return f"{v:>8}"
    return f"{v:7.1f}{suffix}"


def _print_consistency(cons):
    print("  consistency vs vmap:")
    for name, r in cons.items():
        if name.startswith("_"):
            print(f"    {name}: {r}")
            continue
        if isinstance(r, dict):
            print(f"    {name:<22} {'OK ' if r['ok'] else 'MISMATCH'}  "
                  f"max|Δmv|={r['max_err_mv']:.1e} max|Δinv|={r['max_err_inv']:.1e}")
        else:
            print(f"    {name:<22} {r}")


def _print_matrix(perf, op_names):
    hdr = "    {:<16}".format("op") + "".join(f"{c:>20}" for c in perf)
    print(hdr)
    for op in op_names:
        row = "    {:<16}".format(op)
        for r in perf.values():
            if r.get("_status") != "ok" or op not in r:
                cell = "skip" if r.get("_status") != "ok" else "-"
            else:
                cell = f"{_fmt(r[op]['peak_mb'], 'M')}{_fmt(r[op]['t_ms'], 'm')}"
            row += f"{cell:>20}"
        print(row)


def print_mv_report(results):
    for problem, pdata in results.items():
        print(f"\n############ {problem}  ({pdata['_info']}) ############")
        _print_consistency(pdata["consistency"])
        print("\n  peak memory (MB) / time (ms):")
        _print_matrix(pdata["perf"], ["MGOp.mv", "MGOp.Tmv", "SGOp.mv", "transport.mv",
                                      "transport.inv", "mass.mv", "moment.mv", "discrete.mv"])


def print_solve_report(results):
    for problem, pdata in results.items():
        print(f"\n############ {problem}  ({pdata['_info']}) ############")
        _print_consistency(pdata["consistency"])
        perf = pdata["perf"]
        print("\n  reference per-application cost (MB / ms):")
        _print_matrix(perf, ["MGOp.mv", "SGOp.mv", "transport.inv"])
        print("\n  solves -- iteration count + implied applications/iter")
        print("  (sanity: BiCGStab does ~2 matvecs/iter; full-solve ref=MGOp.mv, lower-tri ref=transport.inv sweep):")
        for cfg, r in perf.items():
            if r.get("_status") != "ok":
                print(f"    {cfg:<22} {r['_status']}")
                continue
            mgmv = r["MGOp.mv"]["t_ms"]
            sweep = r["transport.inv"]["t_ms"]
            for op in [o for o in ("solve.iter", "solve.ltri", "grad.solve.ltri") if o in r]:
                d = r[op]
                ref = mgmv if op == "solve.iter" else sweep
                refname = "MGOp.mv" if op == "solve.iter" else "sweep"
                ref_s = f"{ref:.1f}" if isinstance(ref, (int, float)) else str(ref)
                mpi = d.get("ms_per_iter")
                mpi_s = f"{mpi:.1f}" if isinstance(mpi, (int, float)) else "n/a"
                t_s = f"{d['t_ms']:.0f}" if isinstance(d["t_ms"], (int, float)) else "OOM"
                appiter = (f"{mpi/ref:.1f}" if isinstance(mpi, (int, float)) and isinstance(ref, (int, float)) and ref > 0 else "n/a")
                print(f"    {cfg:<18} {op:<16} t={t_s:>7}ms  n_iter={d['n_iter']:>4}  "
                      f"ms/iter={mpi_s:>6}  app/iter={appiter:>4} (ref {refname}={ref_s}ms)")


def compare_golden(results, golden, mem_tol=0.05, time_tol=0.25):
    fails, warns = [], []
    for problem, pdata in results.items():
        gp = golden.get(problem, {}).get("perf", {})
        for cfg, r in pdata["perf"].items():
            if r.get("_status") != "ok":
                continue
            gr = gp.get(cfg, {})
            for op, vals in r.items():
                if op.startswith("_") or not isinstance(vals, dict):
                    continue
                gv = gr.get(op)
                if not gv:
                    continue
                cur_m, ref_m = vals.get("peak_mb"), gv.get("peak_mb")
                if isinstance(cur_m, (int, float)) and isinstance(ref_m, (int, float)) and ref_m > 0:
                    d = (cur_m - ref_m) / ref_m
                    if abs(d) > mem_tol:
                        fails.append(f"{problem}/{cfg}/{op} peak_mb {ref_m:.1f}->{cur_m:.1f} ({d:+.0%})")
                cur_t, ref_t = vals.get("t_ms"), gv.get("t_ms")
                if isinstance(cur_t, (int, float)) and isinstance(ref_t, (int, float)) and ref_t > 0:
                    d = (cur_t - ref_t) / ref_t
                    if abs(d) > time_tol:
                        warns.append(f"{problem}/{cfg}/{op} t_ms {ref_t:.1f}->{cur_t:.1f} ({d:+.0%})")
    print("\n================ golden compare ================")
    print("\n".join("  FAIL " + f for f in fails) or f"peak memory within {mem_tol:.0%} of golden.")
    if warns:
        print(f"time drift (>{time_tol:.0%}, advisory):")
        print("\n".join("  warn " + w for w in warns))
    return len(fails)


# --------------------------------------------------------------------------------------------------
def run_regime(problems_dict, configs, perf_fn, report_fn, args, is_solve):
    results = {}
    consistency_failed = False
    for problem, params in problems_dict.items():
        sd, qs = build_problem(**params)
        info = (f"dim {params['dim']} deg {params['degree']} elems {sd.domain.n_elements} "
                f"angles {qs.angles.shape[0]} groups {sd.domain.cross_sections_indexed.n_energy_groups}")
        if args.progress:
            print(f"\n>>> building {problem} ({info})", flush=True)
        built = build_configs(sd, qs, configs, _capabilities(params))
        cons = consistency_pass(built, args.progress)
        for _, r in cons.items():
            if isinstance(r, dict) and not r.get("ok", True):
                consistency_failed = True
        if is_solve:
            perf = perf_fn(built, args.number, args.repeats, args.progress, problem,
                           include_grad=not args.skip_grad)
        else:
            perf = perf_fn(built, args.number, args.repeats, args.progress, problem)
        results[problem] = {"_info": info, "consistency": cons, "perf": perf}
    report_fn(results)
    return results, consistency_failed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regime", choices=["mv", "solve", "both"], default="both")
    p.add_argument("--problems", default="", help="comma list of problem names (overrides regime sizes)")
    p.add_argument("--configs", default="", help="comma list of config names (default: all)")
    p.add_argument("--number", type=int, default=3, help="timed calls per measurement block")
    p.add_argument("--repeats", type=int, default=2, help="measurement blocks (min-of-N)")
    p.add_argument("--progress", action="store_true", help="print each measurement as it runs")
    p.add_argument("--skip-grad", action="store_true", help="skip grad.solve.ltri (the most expensive solve op)")
    p.add_argument("--save", default="")
    p.add_argument("--compare", default="")
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()

    print("devices:", jax.devices())
    configs = make_configs()
    if args.configs:
        wanted = set(args.configs.split(","))
        configs = [c for c in configs if c.name in wanted]

    all_results = {}
    cons_failed = False
    regimes = []
    if args.regime in ("mv", "both"):
        regimes.append((MV_PROBLEMS, perf_pass_mv, print_mv_report, False))
    if args.regime in ("solve", "both"):
        regimes.append((SOLVE_PROBLEMS, perf_pass_solve, print_solve_report, True))

    for problems_dict, perf_fn, report_fn, is_solve in regimes:
        if args.problems:
            wanted = set(args.problems.split(","))
            problems_dict = {k: v for k, v in problems_dict.items() if k in wanted}
            if not problems_dict:
                continue
        res, cf = run_regime(problems_dict, configs, perf_fn, report_fn, args, is_solve)
        all_results.update(res)
        cons_failed = cons_failed or cf

    if args.save:
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        path = os.path.join(GOLDEN_DIR, args.save + ".json")
        with open(path, "w") as fh:
            json.dump(all_results, fh, indent=2, default=str)
        print(f"\nwrote golden -> {path}")

    exit_code = 0
    if args.compare:
        with open(os.path.join(GOLDEN_DIR, args.compare + ".json")) as fh:
            exit_code = 1 if compare_golden(all_results, json.load(fh)) else 0
    if args.strict and cons_failed:
        print("\nSTRICT: a consistency check mismatched.")
        exit_code = 1
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
