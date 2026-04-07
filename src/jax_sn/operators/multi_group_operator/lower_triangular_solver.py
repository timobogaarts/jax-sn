import jax 
import jax.numpy as jnp
import lineax as lx
from .multi_group_operator_imp import MultiGroupOperator
from typing import Tuple

import equinox as eqx

class SolverDiagnostics(eqx.Module):
    print_sweep : bool 
    print_single_group : bool




class MultiGroupLowerTriangularBlockSolver(lx.AbstractLinearSolver):

    single_group_solver : lx.AbstractLinearSolver    

    solver_diagnostics : SolverDiagnostics

    def init(self, operator : MultiGroupOperator, options ):
        assert isinstance(operator, MultiGroupOperator), f"MultiGroupLowerTriangularSolver can only be used with MultiGroupOperator but got operator of type {type(operator)}."
        return operator
    
    def compute(self, solver_state : MultiGroupOperator, rhs : jnp.ndarray, options ):        
        if solver_state.transpose_bool:
            return solve_upper_triangular_transpose(solver_state, self.single_group_solver, rhs, self.solver_diagnostics)
        else:
            return solve_lower_triangular(solver_state, self.single_group_solver, rhs, self.solver_diagnostics)
        
    
    def assume_full_rank(self) -> bool:
        '''
        The operator is assumed to be full rank, as it is a lower triangular operator with non-singular diagonal blocks (the single group operators).
        '''
        return True    

    def transpose(self, solver_state : MultiGroupOperator, options ):
        '''
        Returns the transposed solver state and options. (the solver_state is simply the operator here, so that is what is transposed).
        '''
        return solver_state.transpose(), options

    def conj(self, solver_state : SolverDiagnostics, options ):
        '''
        Returns the conjugated solver state and options. (the solver_state is simply the operator here, so that is what is conjugated).
        '''
        return solver_state.conj(), options


@eqx.filter_jit
def solve_lower_triangular(mgop : MultiGroupOperator, solver : lx.AbstractLinearSolver, rhs : jnp.ndarray, diagnostic : SolverDiagnostics) -> jnp.ndarray:
    r'''
    Solve the linear system using a given solver, assuming the multi-group operator is lower triangular.
    This is useful for e.g. downscattering only problems. 

    In a downscattering only problem, the multi-group operator has the structure:

    .. math::
        \mathbf{A} = \begin{bmatrix} \mathbf{A}_0 & 0 & 0 & \dots &0 \\ \mathbf{B}_{10} & \mathbf{A}_1 &0 & \dots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ \mathbf{B}_{G0} & \mathbf{B}_{G1} & \mathbf{B}_{G2} & \dots & \mathbf{A}_G \end{bmatrix}

    This can be solved by direct block forward substitution, solving each energy group sequentially using the single-group operator for that group and the already computed higher energy group solutions to form the scattering source.

    The right-hand side is then, for a specific energy group g:

    .. math::
        \text{rhs}_g + \sum_{g' < g} \mathbf{B}_{gg'}[\psi_{g'}] = \text{rhs}_g + \mathbf{M}\mathbf{T}_g^{-1}\mathbf{H}\mathbf{D} \left(\sum_{g' < g} \mathbf{S}_{gg'}[\psi_{g'}] \right)\equiv \text{rhs}_g + \mathbf{M}\mathbf{T}_g^{-1}\mathbf{H}\mathbf{D} \mathbf{U}_g[\psi]

    where we use that for higher energy groups :math:`g' > g`, the scattering operator is zero (since we use the gather scattering of a general :class:`MultiGroupOperator` here, the others are still used, but they are just zero).

    The single group system is then solved with the :class:`SingleGroupOperator` for energy group g.

    *Memory footprint*
    
    Theoretically, it uses (for a discrete angle array :math:`D\in \mathcal{D}\times \mathcal{F}`, and a moment array :math:`M \in \mathcal{<}\times \mathcal{F}` (we don't take into account the batching temporaries from a batched transport operator)):

    .. math::
        \text{In}:  \ &G\times M \\ 
        \text{Out}: \ &G\times M \\ 
        \text{Temporary BiCGSTAB}: \ &5 \times M \\ 
        \text{MV temporary}  :\ &2 \times D \\ 
        \text{Total}: \ &2 \times G \times M + 2 \times D + 5 \times M \\ 

    In practice, due to XLA compilation, this can be somewhat higher. For 15000 triangular elements with 72 angles, 36 moments, and 4 energy groups on an RTX3080, it used 

    .. math::
        2\times G\times M + 2 \times D + 8 \times M


    Parameters
    ----------
    mgop : MultiGroupOperator
        The multi group operator.
    solver : lx.AbstractLinearSolver
        The linear solver to use.
    rhs : jnp.ndarray
        The right-hand side.
    Returns
    -------
    jnp.ndarray
        The solution.   
    '''
    n_groups = mgop.n_energy_groups        
    assert not mgop.transpose_bool, "The multi-group operator must not be transposed to use the lower triangular transpose solver."

    if diagnostic.print_single_group:
        jax.debug.print("Using lower triangular transpose solver for multi-group operator with {n_groups} energy groups.", n_groups = n_groups)

    def body_fun(i, state):                
        solution, n_steps = state
        sgop = mgop.to_single_group_operator(i)        
        
        rhs_i = rhs[i] + mgop.moment_operator.mv(sgop.transport_operator.inv(mgop.mass_operator.mv(mgop.discrete_operator.mv(mgop.gather_scatter_operator.gather_forward(solution, i)))))
        lxsolve = lx.linear_solve(sgop, rhs_i, solver, throw = False, options = {"y0" : rhs_i}) # Solve single group system

        if diagnostic.print_single_group:
            jax.debug.print("Solved energy group {eg} in {n_steps} steps", eg = i, n_steps = lxsolve.stats['num_steps'])
        
        return solution.at[i].set(lxsolve.value), n_steps.at[i].set(lxsolve.stats['num_steps'])
    
    solution = jnp.zeros_like(rhs)
    n_steps = jnp.zeros(n_groups, dtype = mgop.gather_scatter_operator.cross_sections_indexed.indices.dtype)
    
    solution, n_steps =  jax.lax.fori_loop(0, n_groups, body_fun, (solution, n_steps) )

    # Simple check just if the solution converged within max steps for all groups. TODO improve in future? raise runtime in the loop?
    result = lx.RESULTS.where( jnp.all(n_steps < solver.max_steps), lx.RESULTS.successful, lx.RESULTS.max_steps_reached)
    return  solution, result, {'num_steps_per_group' : n_steps}

@eqx.filter_jit
def solve_upper_triangular_transpose(mgop : MultiGroupOperator, solver : lx.AbstractLinearSolver, rhs : jnp.ndarray, diagnostic : SolverDiagnostics) -> jnp.ndarray:
    r'''
    Solve the linear system using a given solver, assuming the multi-group operator is lower triangular.
    This is useful for e.g. downscattering only problems. 

    In a downscattering only problem, the multi-group transpose operator has the structure:

    .. math::
        \mathbf{A} = \begin{bmatrix} \mathbf{A}^T_0 & \mathbf{B}_{10}^T & \mathbf{B}_{20}^T & \dots & \mathbf{B}_{G0}^T \\ 0 & \mathbf{A}^T_1 & \mathbf{B}_{21}^T & \dots & \mathbf{B}_{G1}^T \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \dots & \mathbf{A}^T_G \end{bmatrix}

    This can be solved by direct block forward substitution, solving each energy group sequentially using the single-group operator for that group and the already computed higher energy group solutions to form the scattering source.

    The right-hand side is then, for a specific energy group g and :

    .. math::
        \text{rhs}_g + \sum_{g' > g} \mathbf{B}^T_{g'g}[\phi_{g'}] = \text{rhs}_g + \sum_{g'>g} \mathbf{S}^T_{g'g} \mathbf{D}^T \mathbf{H}^T (\mathbf{T}_{g'}^{-1})^T \mathbf{M}^T [\phi_{g'}] 

    where we use that for lower energy groups :math:`g' < g`, the scattering operator is zero (since we use the gather scattering of a general :class:`MultiGroupOperator <jax_sn.operators.multi_group_operator.MultiGroupOperator>` here, the others are still used, but they are just zero).

    The situation is now similar to the the forward transpose of the :class:`MultiGroupOperator <jax_sn.operators.multi_group_operator.MultiGroupOperator>`. We can use a single sweep per energy group, but we have to not only store the solution :math:`\phi_{g\ell ei}`,
    but also the intermediate values 

    .. math::
        \xi_{g} = \mathbf{D}^T \mathbf{H}^T (\mathbf{T}_{g}^{-1})^T \mathbf{M}^T [\phi_{g}]

    Then, we can reconstruct the source quickly as 

    .. math::
        \text{rhs}_g + \sum_{g' > g} \mathbf{S}^T_{g'g} [\xi_{g'}]

    
    The single group system is then solved with the :class:`SingleGroupOperatorTranspose` for each specific energy group

    *Technical note*
    
    The single group operator is solved using an iterative solver by definition. We do not have a matrix representation of the single group operator transpose, so we cannot use a direct solver.
    The lineax linear solvers do not necessarily check whether the right-hand-side is zero, so we do that here to avoid unnecessary work and to avoid NaNs in the solution. 
    It is an issue here because the differentiating through a forward linear solve involves a solution of the adjoint system, which can have a zero right-hand side in some energy groups (when one of the the objective functions does not depend on that energy group).

    .. math::
        \mathbf{A}^T x = [1, 0,0,0]
    This results in a zero right-hand side for all energy groups except the highest one, which can lead to NaNs in the solution if not handled properly.

    Furthermore, we set the initial guess to the right-hand side. The reason is that if we start with all zeros and the objective function only depends on regions where the inscattering is zero (and thus the adjoint source is one only in those regions), the initial BiCGSTAB update 
    will immediately break (the reason is that in those specific regions, the resulting :math:`p` & :math:`v` will be the same as the initial guess, such that the :math:`s` and :math:`t` vectors become zero in those regions, which makes the updated residual also zero in those regions, leading to <r0, r>=0).
    By setting the initial guess to the right-hand side, we avoid this issue, as the transport immediately creates a much less sparse residual (r0 = Ay0 - b).

    *Memory footprint*
    
    Theoretically, it uses (for a discrete angle array :math:`D\in \mathcal{D}\times \mathcal{F}`, and a moment array :math:`M \in \mathcal{<}\times \mathcal{F}` (we don't take into account the batching temporaries from a batched transport operator)):

    .. math::
        \text{In}:  \ &G\times M \\ 
        \text{Out}: \ &G\times M \\ 
        \text{Temporary BiCGSTAB}: \ &5 \times M \\ 
        \xi_g \text{ temporary} :\  & G \times M \\
        \text{MV temporary} :\ &2 \times D \\ 
        \text{Total}: \ &3 \times G \times M + 2 \times D + 5 \times M \\ 

    In practice, due to XLA compilation, this can be somewhat higher. For 15000 triangular elements with 72 angles, 36 moments, and 4 energy groups on an RTX3080, it used 

    .. math::
        3\times G\times M + 2 \times D + 9 \times M

    Parameters
    ----------
    mgop : MultiGroupOperator
        The multi group operator.
    solver : lx.LinearSolver
        The linear solver to use.
    rhs : jnp.ndarray
        The right-hand side.
    diagnostic : bool, optional
        Whether to print diagnostic information, by default False.
    Returns
    -------
    jnp.ndarray
        The solution.   
    '''
    assert mgop.transpose_bool, "The multi-group operator must be transposed to use the upper triangular transpose solver."
    n_groups = mgop.n_energy_groups
    sgop_t  = mgop.to_single_group_operator(0)  # the sgop transpose doesn't actually transpose its constituents, just has a different mv method that takes care of it.
                                               # so, instead we just transpose the individual members of it and transpose when we need.
    top_t = sgop_t.transport_operator.transpose()
    dop_t = sgop_t.discrete_operator.transpose()
    mop_t = sgop_t.moment_operator.transpose()
    hop_t = sgop_t.mass_operator.transpose()     

    if diagnostic.print_single_group:
        jax.debug.print("Using upper triangular transpose solver for multi-group operator with {n_groups} energy groups.", n_groups = n_groups)

    def body_fun(i, state):                        
        solution, xi_g, n_steps = state        
        
        g_i     = n_groups - i - 1 # Upper triangular, so we go from high to low energy group

        
        sgop_gi = sgop_t.to_energy_group(g_i)

        rhs_i   = rhs[g_i] + mgop.gather_scatter_operator.gather_backward(xi_g, g_i)

        def zero_rhs(rhs_i):
            return jnp.zeros_like(rhs_i), 0
        
        def nonzero_rhs(rhs_i):
            lxsolve = lx.linear_solve(sgop_gi, rhs_i, solver, throw = False, options = {"y0" : rhs_i}) # Solve single group transpose system
            return lxsolve.value, lxsolve.stats['num_steps']
        
        lxsolvevalue, nsteps = jax.lax.cond(jnp.all(rhs_i == 0.0), zero_rhs, nonzero_rhs,  operand = rhs_i)
        
        xi_new       = xi_g.at[g_i].set(dop_t.mv(hop_t.mv(top_t.to_energy_group(g_i).inv(mop_t.mv(lxsolvevalue))))) # D^T H^T (T_g^{-1})^T M^T [phi_g]
        solution_new = solution.at[g_i].set(lxsolvevalue)        
        if diagnostic.print_single_group:
            jax.debug.print("Solved energy group {eg} in {n_steps} steps", eg = g_i, n_steps = nsteps)
        return solution_new, xi_new, n_steps.at[g_i].set(nsteps)
    
    solution          = jnp.zeros_like(rhs)
    xi_g              = jnp.zeros_like(rhs) # Intermediate values to store D^T H^T (T_g^{-1})^T M^T [phi_g]
    n_steps           = jnp.zeros(n_groups, dtype = mgop.gather_scatter_operator.cross_sections_indexed.indices.dtype)    
    solution, _,  n_steps =  jax.lax.fori_loop(0, n_groups, body_fun, (solution, xi_g, n_steps) )    
    # Simple check just if the solution converged within max steps for all groups. TODO improve in future? raise runtime in the loop?
    result = lx.RESULTS.where( jnp.all(n_steps < solver.max_steps), lx.RESULTS.successful, lx.RESULTS.max_steps_reached)
    return  solution, result, {'num_steps_per_group' : n_steps}