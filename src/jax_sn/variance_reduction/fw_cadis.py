import equinox as eqx
import jax.numpy as jnp
import lineax as lx

from ..operators.multi_group_operator import MultiGroupOperator, ComposedMultiGroupChainStrategy
from ..solvers.right_hand_side import RightHandSide, TotalSource, solve_multigroup_operator_rhs
from .continuous_adjoint import solve_continuous_adjoint


@eqx.filter_jit
def fw_cadis_scalar_flux(operator: MultiGroupOperator, rhs: RightHandSide, solver: lx.AbstractLinearSolver) -> jnp.ndarray:
    r'''
    Runs the FW-CADIS algorithm to compute an importance map for variance reduction.

    First the forward problem is solved to obtain the scalar flux:

    .. math::
        \phi = A^{-1} q

    where :math:`A` is the multi-group operator and :math:`q` is the right-hand side.

    Then, the importance map is computed as:

    .. math::
        \phi^\dagger = (A^\dagger)^{-1} \left( \frac{1}{\phi} \right)

    where :math:`A^\dagger` is the continuous adjoint (see :func:`solve_continuous_adjoint`).

    For stability purposes, the scalar flux is first averaged over each element (due to the conservation
    property of DG, this cannot be zero, while the value at a particular basis function node can be
    extremely small and blow up).

    Parameters
    ----------
    operator : MultiGroupOperator
        The multi-group operator for which to compute the importance map.
    rhs : RightHandSide
        The right-hand side of the equation to solve, typically representing a response function.
    solver : lx.AbstractLinearSolver
        The (iterative) linear solver used for both the forward and adjoint solves.

    Returns
    -------
    jnp.ndarray
        The adjoint scalar flux importance map, shape ``[n_groups, n_moments, n_elements, n_basis]``.
    '''
    assert isinstance(operator, MultiGroupOperator), f"operator must be an instance of MultiGroupOperator, but got {type(operator)}."

    chain_strategy = ComposedMultiGroupChainStrategy.from_composed(operator.strategy)
    forward_result = solve_multigroup_operator_rhs(operator, rhs, solver, chain_strategy)

    integration_coeff = operator.data.spatial_data.fem_matrices.integration_coeff
    forward_average_result = jnp.einsum("gei,i->ge", forward_result.value[:, 0, ...], integration_coeff)  # scalar flux [n_groups, n_elements]

    one_over_phi = 1.0 / forward_average_result

    n_basis = operator.data.elements_basis_shape.shape[-1]
    qdagger_isotropic = jnp.stack([one_over_phi] * n_basis, axis=-1)  # [n_groups, n_elements, n_basis]

    qdagger_rhs = TotalSource.from_operator_and_isotropic_source(operator, qdagger_isotropic)

    adjoint_result = solve_continuous_adjoint(operator, qdagger_rhs, solver)

    return adjoint_result.value
