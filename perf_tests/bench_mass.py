"""
Mass operator, two axes:

  - ``mass.mv``        : the ``mass_operator`` property as derived (sized like the transport
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
from _bench import Bench
from jax_sn.operators.mass_operator import MassStrategy, MassOperator


def _moment_sized(mgop) -> MassOperator:
    return MassOperator.from_solution_domain_data(
        mgop.setup.spatial_data, (mgop.moment_operator.n_moments,), mgop.mass_strategy)


BENCHES = [
    Bench(name="mass.mv", base=MassStrategy, kwarg="mass_strategy", attr="mass_operator"),
    Bench(name="mass.mv.moment", base=MassStrategy, kwarg="mass_strategy", extract=_moment_sized),
]

if __name__ == "__main__":
    from run import run_module
    run_module(__name__)
