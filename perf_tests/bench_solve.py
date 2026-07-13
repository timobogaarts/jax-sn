"""
EXPENSIVE (opt-in) -- iterative solves swept over the transport ``SingleAngleInversionStrategy``:

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
from _discovery import discover_variants
from bench_transport import _INV_OVERRIDES
from jax_sn.operators.transport_operator import SingleAngleInversionStrategy
from jax_sn.operators.multi_group_operator import MultiGroupLowerTriangularBlockSolver
from jax_sn.operators.multi_group_operator.multi_group_operator_creation import create_multi_group_operator


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
        variants = discover_variants(SingleAngleInversionStrategy, _INV_OVERRIDES)
        labels = [f"{v.label} / {op}" for v in variants for op in ("solve.iter", "solve.ltri")]
        table = LiveTable(self.name, labels, extra_header=" / n_iter")
        for v in variants:
            if v.instance is None:
                for op in ("solve.iter", "solve.ltri"):
                    label = f"{v.label} / {op}"
                    table.start(label); table.skip(v.skip_reason)
                    out[label] = Measurement.skip(v.skip_reason).__dict__
                continue
            mgop = create_multi_group_operator(sd, qs, inversion_strategy=v.instance)
            rhs = jnp.ones(mgop.in_structure().shape)
            ltri = MultiGroupLowerTriangularBlockSolver(_bicgstab())
            specs = {
                "solve.iter": (lambda op, b: jnp.sum(lx.linear_solve(op, b, _bicgstab(), throw=False).value),
                               _bicgstab(), "num_steps"),
                "solve.ltri": (lambda op, b: jnp.sum(lx.linear_solve(op, b, ltri, throw=False).value),
                               ltri, "num_steps_per_group"),
            }
            for op_name, (f, niter_solver, niter_key) in specs.items():
                label = f"{v.label} / {op_name}"
                table.start(label)
                try:
                    rec = _measure_solve(f, mgop, rhs, niter_solver, niter_key)
                    table.ok(rec["peak_mb"], rec["t_ms"], suffix=f"  n_iter={rec['n_iter']}")
                except Exception as e:
                    rec = Measurement.skip(short_exc(e)).__dict__
                    table.skip(rec["reason"])
                out[label] = rec
        return out


BENCHES = [SolveBench()]

if __name__ == "__main__":
    from run import run_module
    run_module(__name__)
