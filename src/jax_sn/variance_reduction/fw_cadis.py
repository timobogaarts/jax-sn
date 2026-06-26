from ..operators.multi_group_operator import MultiGroupOperator
from ..operators.multi_group_operator.multi_group_operator_imp import _solve_multigroup_operator_rhs_scan
import jax.numpy as jnp
import lineax as lx
import jax
from jax_sn.operators.multi_group_operator.right_hand_side import RightHandSide, solve_multigroup_operator_rhs, TotalSource
from jax_sn.operators.multi_group_operator.adjoint_multigroup_operator import solve_multigroup_operator_rhs_adjoint, AdjointMultiGroupOperator, AdjointMultiGroupOperatorScan
from typing import Type
import equinox as eqx

@eqx.filter_jit
def fw_cadis_scalar_flux(operator : MultiGroupOperator, rhs : RightHandSide, solver : lx.AbstractLinearSolver, adjoint_operator_cls : Type[AdjointMultiGroupOperator] = AdjointMultiGroupOperatorScan) -> jnp.ndarray:
    r'''
    Runs the FW-CADIS algorithm to compute an importance map for variance reduction.

    First the forward problem is solved to obtain the scalar flux:

    .. math::
        \phi = A^{-1} q

    where :math:`A` is the multi-group operator and :math:`q` is the right-hand side.

    Then, the importance map is computed as:

    .. math::
        \phi^\dagger = A^\dagger^{-1} \left( \frac{1}{\phi} \right)

    For stability purposes, the scalar flux is first averaged over each element (due to the conservation property of DG, this cannot be zero, while the value at a particular 
    basis function node can be extremely small).
    
    
    Parameters
    ----------
    operator : MultiGroupOperator
        The multi-group operator for which to compute the importance map.
    rhs : RightHandSide
        The right-hand side of the equation to solve, typically representing a response function.         
    solver : lx.AbstractLinearSolver
        The linear solver to use for solving the linear systems in the FW-CADIS algorithm.
    
    Returns
    -------
    '''
    assert isinstance(operator, MultiGroupOperator), f"operator must be an instance of MultiGroupOperator, but got {type(operator)}."
    assert isinstance(adjoint_operator_cls, type) and issubclass(adjoint_operator_cls, AdjointMultiGroupOperator), f"adjoint_operator_cls must be a subclass of AdjointMultiGroupOperator, but got {adjoint_operator_cls}."
    adjoint_mgop = adjoint_operator_cls.from_multigroup_operator(operator)
    forward_result = solve_multigroup_operator_rhs(operator, rhs, solver)

    integration_coeff = operator.solution_domain_data.fem_matrices.integration_coeff
    forward_average_result = jnp.einsum("gei,i->ge", forward_result.value[:,0,...], integration_coeff) # scalar flux [n_groups, n_elements]

    one_over_phi = 1.0 / forward_average_result

    qdagger_isotropic = jnp.stack([one_over_phi] * adjoint_mgop.n_basis, axis = -1)

    qdagger_rhs = TotalSource.from_operator_and_isotropic_source(adjoint_mgop, qdagger_isotropic)

    adjoint_result = solve_multigroup_operator_rhs_adjoint(adjoint_mgop, qdagger_rhs, solver)

    return adjoint_result.value