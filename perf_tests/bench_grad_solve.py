"""
EXPENSIVE (opt-in) -- gradient through the lower-triangular solve, swept over the hand-picked default
composed strategies (see ``_bench.default_multi_group_strategies``: ``DefaultMultiGroupStrategy``'s
jax/pallas/warp recipes, batched and non) -- same reasoning as ``bench_machinery.py``/``bench_solve.py``.
This is the heaviest op and the one that actually exercises the builder's storage policy
(rebuild-from-primitives + reverse-mode through the block solve). Records peak memory, wall time and
the underlying solve's iteration count.
"""
import jax
import jax.numpy as jnp
import lineax as lx

from _common import peak_mb, time_ms, short_exc, Measurement, LiveTable
from _bench import create_multi_group_builder, default_multi_group_strategies
from bench_solve import _bicgstab, _n_iter
from jax_sn.domain.cross_sections import CrossSectionsIndexed
from jax_sn.solvers.base_solvers import MultiGroupLowerTriangularBlockSolver


class GradSolveBench:
    name = "grad.solve.ltri"
    group = "expensive"

    def run(self, sd, qs):
        out = {}
        strategies = default_multi_group_strategies()
        table = LiveTable(self.name, list(strategies), extra_header=" / n_iter")
        for label, make_strategy in strategies.items():
            table.start(label)
            try:
                # vertices + cross sections free, the structural tier stored: the whole theta-dependent
                # layer stays pure JAX, so the rebuild differentiates.
                # (the leveled sweeps the warp strategies use need a pinned level count to rebuild
                # under a trace -- `create_multi_group_builder` supplies it.)
                builder = create_multi_group_builder(sd, qs, make_strategy())
                mgop = builder.multi_group_operator
                rhs = jnp.ones(mgop.in_structure().shape)
                # the material indices are integers, so theta carries only the cross-section *data*.
                theta = (builder.vertices, builder.cross_sections_indexed.cross_section_data)
                ltri = MultiGroupLowerTriangularBlockSolver.from_composed(mgop.strategy, _bicgstab())

                def reconstruct_solve(th, bld, b):
                    vertices, xs_data = th
                    xs = CrossSectionsIndexed(indices=bld.cross_sections_indexed.indices,
                                              cross_section_data=xs_data)
                    m = bld.updated(vertices=vertices, cross_sections_indexed=xs).multi_group_operator
                    return jnp.sum(lx.linear_solve(m, b, ltri, throw=False).value)

                f = lambda th, bld, b: jax.value_and_grad(reconstruct_solve)(th, bld, b)
                args = (theta, builder, rhs)
                n_iter = _n_iter(mgop, rhs, ltri, "num_steps_per_group")
                t = time_ms(f, *args)
                rec = dict(status="ok", peak_mb=peak_mb(f, *args), t_ms=t, n_iter=n_iter,
                           ms_per_iter=(t / n_iter if (t and n_iter and n_iter > 0) else None), reason="")
                table.ok(rec["peak_mb"], rec["t_ms"], suffix=f"  n_iter={n_iter}")
            except Exception as e:
                rec = Measurement.skip(short_exc(e)).__dict__
                table.skip(rec["reason"])
            out[label] = rec
        return out


BENCHES = [GradSolveBench()]

if __name__ == "__main__":
    from run import run_module
    run_module(__name__)
