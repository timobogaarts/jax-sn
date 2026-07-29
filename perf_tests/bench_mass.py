"""
Mass operator, two axes:

  - ``mass.mv``        : the group's mass sub-operator as pulled from the chain (sized like the transport
    operator's discrete/angle space). Still the real cost for call sites that feed it discrete-space
    data directly with no ``discrete_operator`` step, e.g.
    ``multi_group_operator_imp._solve_multigroup_operator_rhs_scan``.
  - ``mass.mv.moment`` : the same strategies, but sized over the (smaller) moment axis instead --
    matching how mass is actually applied everywhere mass and discrete are adjacent (SingleGroupOperator
    and MultiGroupOperator's forward/transpose mv, the adjoint operator, the lower-triangular solver,
    and ``right_hand_side.solve_multigroup_operator_rhs``), since mass and discrete commute (independent
    tensor factors) and moment space is the smaller of the two.

``MassOperator`` is shape-agnostic at runtime (batched over an arbitrary leading axis, see
``mass_operator_base.py``), so this is purely a matter of which size to benchmark -- no new kernel.
"""
from _bench import Bench, single_group_operators
from jax_sn.operators.mass_operator import MassStrategy, MassOperator, MassData


def _moment_sized(mgop) -> MassOperator:
    spatial = mgop.data.spatial_data
    strategy = mgop.strategy.single_group_chain_strategy.mass_strategy
    return MassOperator(MassData(spatial.fem_matrices, spatial.precomputed_geometric_data),
                         (mgop.data.n_moments,), strategy)


BENCHES = [
    Bench(name="mass.mv", base=MassStrategy, chain_field="mass_strategy",
          extract=lambda mgop: single_group_operators(mgop)[3]),
    Bench(name="mass.mv.moment", base=MassStrategy, chain_field="mass_strategy", extract=_moment_sized),
]

if __name__ == "__main__":
    from run import run_module
    run_module(__name__)
