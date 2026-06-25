"""Gather-scatter operator: ``gather_forward`` swept over every ``GatherScatterOperator`` class.

This axis is *class*-valued (``gather_scatter_operator_cls``), not a strategy instance, so discovery runs
with ``instantiate=False``. The input is the 4D group-moment array the operator gathers over, so the
bench extracts the whole MGOP to size the input from its moment structure.
"""
import jax.numpy as jnp

from _bench import Bench
from jax_sn.operators.scatter_operator import GatherScatterOperator


def scatter_call(mgop):
    gso = mgop.gather_scatter_operator
    x = jnp.ones((mgop.n_energy_groups,) + mgop.moment_operator.out_structure().shape)
    return (lambda op, v: op.gather_forward(v, 0), (gso, x))


BENCHES = [
    Bench(name="scatter.gather", base=GatherScatterOperator, kwarg="gather_scatter_operator_cls",
          extract=lambda m: m, make_call=scatter_call, instantiate=False),
]

if __name__ == "__main__":
    from run import run_module
    run_module(__name__)
