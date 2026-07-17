r'''
Continuous adjoint of the multi-group operator.

The *continuous* adjoint :math:`\mathbf{A}^\dagger` is the discretisation of the adjoint transport
equation: it negates the streaming direction (:math:`\Omega \to -\Omega`) but keeps the moment/discrete
operators :math:`\mathbf{M}, \mathbf{D}` at :math:`+\Omega`, and gathers the *transposed* group scatter.
It is therefore **not** the algebraic transpose :math:`\mathbf{A}^T` (which reverses the whole operator
chain); see :class:`MultiGroupOperator <jax_sn.operators.multi_group_operator.MultiGroupOperator>`.

Because the harmonics ``wk_ylm``/``ylm`` are frozen at :class:`AngularData
<jax_sn.operators.operator_data.AngularData>` construction and only the transport sweep re-reads
``quadrature_set.angles``, negating the angles *after* that freeze flips only the streaming, leaving
:math:`\mathbf{M}, \mathbf{D}` at :math:`+\Omega`. Combined with (a) reversing the sweep ordering to match
the negated direction and (b) transposing the group axes of the scatter tensor (so the forward
``incoming_gather`` computes the *outgoing* sum), the ordinary forward
:class:`MultiGroupOperator <jax_sn.operators.multi_group_operator.MultiGroupOperator>` run on this
transformed data *is* :math:`\mathbf{A}^\dagger`. No new operator/strategy/solver is introduced.

The transformed data is deliberately kept private and consumed immediately inside
:func:`solve_continuous_adjoint`: it is a valid computational object but *internally inconsistent* (its
angles no longer match its frozen harmonics; its scatter tensor is transposed), so it must never escape
into general use.
'''

import equinox as eqx
import jax.numpy as jnp
import lineax as lx

from jax_sn.operators.multi_group_operator import MultiGroupOperator
from jax_sn.operators.operator_data import OperatorData
from jax_sn.solvers.base_solvers import MultiGroupLowerTriangularBlockSolver
from jax_sn.solvers.right_hand_side import solve_multigroup_operator_rhs


def _continuous_adjoint_data(data: OperatorData) -> OperatorData:
    r'''Transform ``data`` into its continuous-adjoint form: negate the streaming angles, reverse the
    sweep ordering to match, and transpose the group axes of the scatter tensor. Private on purpose --
    the result is internally inconsistent (angles vs. frozen harmonics) and must not leave this module.'''
    angles     = data.discretisation_data.angular.quadrature_set.angles
    scattering = data.discretisation_data.spatial.domain.cross_sections_indexed.cross_section_data.scattering

    # Negate streaming + transpose group scatter (sigma_{g'<-g} stored at [..., l, g', g]; swap g'/g).
    data = eqx.tree_at(
        lambda d: (d.discretisation_data.angular.quadrature_set.angles,
                   d.discretisation_data.spatial.domain.cross_sections_indexed.cross_section_data.scattering),
        data,
        (-angles, jnp.swapaxes(scattering, -1, -2)),
    )
    # Reverse the sweep ordering (partner of the angle negation; same concrete SweepData type).
    return eqx.tree_at(
        lambda d: d.non_differentiable_data.sweep_data,
        data,
        data.non_differentiable_data.sweep_data.reverse,
    )


@eqx.filter_jit
def solve_continuous_adjoint(operator: MultiGroupOperator, rhs, solver: lx.AbstractLinearSolver):
    r'''
    Solve :math:`\mathbf{A}^\dagger \psi^\dagger = \mathbf{C}^\dagger[b]` for the continuous adjoint of
    ``operator``, where the source is conditioned by the adjoint chain :math:`\mathbf{C}^\dagger =
    \mathbf{M}\,(\mathbf{T}^\dagger)^{-1}\mathbf{H}\mathbf{D}` (built automatically because the chain
    strategy runs on the adjoint-transformed data).

    The adjoint operator is the ordinary forward :class:`MultiGroupOperator` run on
    :func:`_continuous_adjoint_data`; it is built inside the jit so gradients w.r.t. cross sections /
    geometry still flow through the adjoint solve.

    Parameters
    ----------
    operator : MultiGroupOperator
        The forward multi-group operator whose continuous adjoint is solved.
    rhs : RightHandSide
        The (physical) adjoint source.
    solver : lx.AbstractLinearSolver
        An iterative solver (GMRES / BiCGStab). The direct block solver is rejected: for downscattering
        the continuous adjoint is *upper*-triangular, which that solver does not handle.

    Returns
    -------
    lx.LinearSolveOutput
    '''
    assert not isinstance(solver, MultiGroupLowerTriangularBlockSolver), (
        "The continuous adjoint is upper-triangular for a downscattering problem; the lower-triangular "
        "block solver is not applicable. Use an iterative solver (e.g. lineax GMRES or BiCGStab)."
    )

    adjoint_operator = eqx.tree_at(lambda o: o.data, operator, _continuous_adjoint_data(operator.data))
    chain_strategy   = operator.strategy.single_group_chain_strategy
    return solve_multigroup_operator_rhs(adjoint_operator, rhs, solver, chain_strategy)
