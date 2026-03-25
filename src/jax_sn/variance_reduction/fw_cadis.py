from ..operators.multi_group_operator import MultiGroupOperator
from ..operators.multi_group_operator.multi_group_operator_imp import _solve_multigroup_operator_rhs_scan
import jax.numpy as jnp
import lineax as lx
import jax



def run_fw_cadis(operator : MultiGroupOperator, rhs : jnp.ndarray, solver : lx.AbstractLinearSolver):
    r'''
    Runs the FW-CADIS algorithm to compute an importance map for variance reduction.

    We compute the forward solution to obtain a value :math:`\phi_{g,lm,e,i}`.

    This is averaged over each element to obtain a positive definite value for the scalar flux.

    We then discretize this lm bsed value to obtain a psi value. This can be rather unstable:
    the expansion of the coefficients can be very close to zero.

    We use the adjoint of the discrete operator. This is not necessarily equal to the discrete of the adjoint.
    
    Parameters
    ----------
    operator : MultiGroupOperator
        The multi-group operator for which to compute the importance map.
    rhs : jnp.ndarray
        The right-hand side of the equation to solve, typically representing a response function. 
        Must have shape [n_energy_groups, n_angles, n_elements, n_basis] matching the operator structure.
    solver : lx.AbstractLinearSolver
        The linear solver to use for solving the linear systems in the FW-CADIS algorithm.
    
    Returns
    -------
    '''
    forward_result = _solve_multigroup_operator_rhs_scan(operator, rhs, solver)

    integration_coeff = operator.solution_domain_data.fem_matrices.integration_coeff

    forward_average_result = jnp.einsum("glei,i->gle", forward_result.value, integration_coeff)

    def _scan_fn(carry, x):
        operator, forward_average_result = carry
        g = x
        psi_average_g = operator.discrete_operator.mv(forward_average_result[g]) # [l, e] -> [k, e]
        return (operator, forward_average_result), psi_average_g

    _, psi_average = jax.lax.scan(_scan_fn, (operator, forward_average_result), jnp.arange(operator.n_energy_groups))
    
    one_d_psi_average =  1.0 / psi_average # [g, l, e] 

    adjoint_source = jnp.stack([one_d_psi_average] * operator.n_basis, axis=-1)

    
    mgop_adjoint = operator.transpose() # THIS IS NOT THE
