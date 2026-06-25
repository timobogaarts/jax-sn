"""
EXPENSIVE (opt-in) -- gradient through the lower-triangular solve, swept over the transport
``InversionStrategy``. This is the heaviest op and the one that actually exercises the parametrisation
module (reconstruct-operator + reverse-mode through the block solve). Records peak memory, wall time and
the underlying solve's iteration count.
"""
import jax
import jax.numpy as jnp
import lineax as lx

from _common import peak_mb, time_ms, short_exc, Measurement, LiveTable
from _discovery import discover_variants
from bench_transport import _INV_OVERRIDES
from bench_solve import _bicgstab, _n_iter
from jax_sn.operators.transport_operator import InversionStrategy
from jax_sn.operators.multi_group_operator import MultiGroupLowerTriangularBlockSolver
from jax_sn.operators.multi_group_operator.multi_group_operator_creation import create_multi_group_operator
from jax_sn.parametrisation import MultigroupOperatorParametrisedVerticesXS


class GradSolveBench:
    name = "grad.solve.ltri"
    group = "expensive"

    def run(self, sd, qs):
        out = {}
        variants = discover_variants(InversionStrategy, _INV_OVERRIDES)
        table = LiveTable(self.name, [v.label for v in variants], extra_header=" / n_iter")
        for v in variants:
            table.start(v.label)
            if v.instance is None:
                table.skip(v.skip_reason)
                out[v.label] = Measurement.skip(v.skip_reason).__dict__
                continue
            try:
                mgop = create_multi_group_operator(sd, qs, inversion_strategy=v.instance)
                rhs = jnp.ones(mgop.in_structure().shape)
                theta = MultigroupOperatorParametrisedVerticesXS.theta_base(mgop)
                ltri = MultiGroupLowerTriangularBlockSolver(_bicgstab())

                def reconstruct_solve(th, op, b):
                    m = MultigroupOperatorParametrisedVerticesXS.reconstruct_operator(op, th)
                    return jnp.sum(lx.linear_solve(m, b, ltri, throw=False).value)

                f = lambda th, op, b: jax.value_and_grad(reconstruct_solve)(th, op, b)
                args = (theta, mgop, rhs)
                n_iter = _n_iter(mgop, rhs, ltri, "num_steps_per_group")
                t = time_ms(f, *args)
                rec = dict(status="ok", peak_mb=peak_mb(f, *args), t_ms=t, n_iter=n_iter,
                           ms_per_iter=(t / n_iter if (t and n_iter and n_iter > 0) else None), reason="")
                table.ok(rec["peak_mb"], rec["t_ms"], suffix=f"  n_iter={n_iter}")
            except Exception as e:
                rec = Measurement.skip(short_exc(e)).__dict__
                table.skip(rec["reason"])
            out[v.label] = rec
        return out


BENCHES = [GradSolveBench()]

if __name__ == "__main__":
    from run import run_module
    run_module(__name__)
