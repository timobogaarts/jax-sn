"""
EXPENSIVE (opt-in) -- iterative solves swept over the hand-picked default composed strategies (see
``_bench.default_multi_group_strategies``: ``DefaultMultiGroupStrategy``'s jax/pallas/warp recipes, batched
and non) -- not an auto-discovered per-field sweep, for the same reason as ``bench_machinery.py``: a solve
exercises the whole pipeline, so only the actual, internally-consistent backend recipes are worth timing.

  - ``solve.iter`` : full-operator BiCGStab solve (each iteration ~ one ``MGOp.mv``).
  - ``solve.ltri`` : lower-triangular block solve (each iteration ~ one transport sweep, per group).

Records peak memory, wall time, and the BiCGStab iteration count (so a solve's cost can be sanity-checked
against the per-application reference costs from ``bench_machinery``). Run via ``run.py --group expensive``
or ``python bench_solve.py``.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as onp

from _common import peak_mb, time_ms, safe_run, short_exc, Measurement, LiveTable
from _bench import default_multi_group_strategies
from jax_sn.solvers.base_solvers import MultiGroupLowerTriangularBlockSolver
from jax_sn.operator_creation.multi_group_operator_creation import create_multi_group_operator


def _bicgstab():
    return lx.BiCGStab(1e-8, 1e-8, max_steps=400)


def _n_iter(mgop, rhs, solver, key_pref):
    out, oom = safe_run(eqx.filter_jit(lambda op, b: lx.linear_solve(op, b, solver, throw=False)), mgop, rhs)
    if oom:
        return -1
    stats = out.stats
    key = key_pref if key_pref in stats else "num_steps"
    return int(onp.asarray(stats.get(key, -1)).sum())


def _measure_solve(f, mgop, rhs, niter_solver, niter_key) -> dict:
    n_iter = _n_iter(mgop, rhs, niter_solver, niter_key)        # the slow iteration count, timed under its row
    t = time_ms(f, mgop, rhs)  # adaptive: a multi-second solve self-sizes to a single timed run
    return dict(status="ok", peak_mb=peak_mb(f, mgop, rhs), t_ms=t, n_iter=n_iter,
                ms_per_iter=(t / n_iter if (t and n_iter and n_iter > 0) else None), reason="")


class SolveBench:
    name = "solve"
    group = "expensive"

    def run(self, sd, qs):
        out = {}
        strategies = default_multi_group_strategies()
        labels = [f"{label} / {op}" for label in strategies for op in ("solve.iter", "solve.ltri")]
        table = LiveTable(self.name, labels, extra_header=" / n_iter")
        for label, make_strategy in strategies.items():
            try:
                mgop = create_multi_group_operator(sd, qs, make_strategy())
            except Exception as e:
                for op in ("solve.iter", "solve.ltri"):
                    row_label = f"{label} / {op}"
                    table.start(row_label); table.skip(short_exc(e))
                    out[row_label] = Measurement.skip(short_exc(e)).__dict__
                continue
            rhs = jnp.ones(mgop.in_structure().shape)
            ltri = MultiGroupLowerTriangularBlockSolver.from_composed(mgop.strategy, _bicgstab())
            specs = {
                "solve.iter": (lambda op, b: jnp.sum(lx.linear_solve(op, b, _bicgstab(), throw=False).value),
                               _bicgstab(), "num_steps"),
                "solve.ltri": (lambda op, b: jnp.sum(lx.linear_solve(op, b, ltri, throw=False).value),
                               ltri, "num_steps_per_group"),
            }
            for op_name, (f, niter_solver, niter_key) in specs.items():
                row_label = f"{label} / {op_name}"
                table.start(row_label)
                try:
                    rec = _measure_solve(f, mgop, rhs, niter_solver, niter_key)
                    table.ok(rec["peak_mb"], rec["t_ms"], suffix=f"  n_iter={rec['n_iter']}")
                except Exception as e:
                    rec = Measurement.skip(short_exc(e)).__dict__
                    table.skip(rec["reason"])
                out[row_label] = rec
        return out


BENCHES = [SolveBench()]

if __name__ == "__main__":
    from run import run_module
    run_module(__name__)
