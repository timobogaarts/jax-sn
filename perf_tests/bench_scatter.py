"""Gather-scatter operator: ``mv`` swept over every discovered ``SingleGroupGatherScatterStrategy`` kernel,
lifted over the group axis by ``VmapGatherScatterStrategy`` (the group-axis analogue of ``VmapAnglesForward``).

The input is the 4D group-moment array the operator gathers over, sized from the whole MGOP's
``in_structure`` (matches ``[n_groups, n_moments, n_elements, n_basis]``, see ``GatherScatterOperator``).
"""
from _bench import Bench, ones_in
from jax_sn.operators.scatter_operator import (
    SingleGroupGatherScatterStrategy, GatherScatterData, GatherScatterOperator, VmapGatherScatterStrategy,
)


def _gather_scatter_operator(mgop) -> GatherScatterOperator:
    gs_data = GatherScatterData(mgop.data.domain.cross_sections_indexed)
    single_group_strategy = mgop.strategy.single_group_gather_scatter_strategy
    return GatherScatterOperator(gs_data, VmapGatherScatterStrategy(single_group_strategy), mgop.in_structure())


def scatter_call(mgop):
    op = _gather_scatter_operator(mgop)
    return (lambda o, v: o.mv(v), (op, ones_in(op)))


BENCHES = [
    Bench(name="scatter.gather", base=SingleGroupGatherScatterStrategy,
          chain_field="single_group_gather_scatter_strategy", top_level=True,
          extract=lambda mgop: mgop, make_call=scatter_call),
]

if __name__ == "__main__":
    from run import run_module
    run_module(__name__)
