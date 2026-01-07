import jax
import jax.numpy as jnp
from jax_sn.operators.base_linear_operator import AbstractLinearOperator_sn
from jax_sn.operators.single_group_operator import SingleGroupOperator
from jax_sn.operators.moment_operator import MomentOperator
from jax_sn.operators.scatter_operator.total_scatter_operator import TotalGroupScatterOperator
from jax_sn.operators.transport_operator import TransportOperatorMultipleAngles
from jax_sn.operators.mass_operator import MassOperator
from jax_sn.operators.scatter_operator.total_scatter_operator import TotalGroupScatterOperator
import lineax as lx 
import equinox as eqx
from abc import abstractmethod

class MultiGroupOperator(AbstractLinearOperator_sn):
    r'''
    Multi energy group linear operator implementation base class.


    Mathematically, this operator maps:

    .. math::
        \mathbf{A} : \mathcal{G} \otimes \mathcal{M} \otimes F  \rightarrow \mathcal{G}\otimes \mathcal{M} \otimes F         

    '''
    moment_operator    : MomentOperator
    transport_operator : TransportOperatorMultipleAngles
    mass_operator      : MassOperator
    scatter_operator   : TotalGroupScatterOperator    
    n_energy_groups : int    

    def __check_init__(self):
        self.transport_operator.solution_domain_data.domain.cross_sections_indexed.cross_section_data.assert_n_energy_groups(self.n_energy_groups)
    
    @abstractmethod
    def mv(self, x):
        ...

    @abstractmethod
    def transpose(self):
        ...

    def in_structure(self):
        return jax.ShapeDtypeStruct(
            shape=(self.n_energy_groups,) + self.moment_operator.out_structure().shape,
            dtype=self.moment_operator.out_structure().dtype
        )
    
    def out_structure(self):
        return jax.ShapeDtypeStruct(
            shape=(self.n_energy_groups,) + self.moment_operator.out_structure().shape,
            dtype  =self.moment_operator.out_structure().dtype            
        )
    
    def to_single_group_operator(self, g_i : int) -> SingleGroupOperator:
        r'''
        Convert to single group operator for energy group g_i.

        Parameters
        ----------
        g_i : int
            Energy group index.
        Returns
        -------
        SingleGroupOperator
            The single group operator for energy group g_i.
        ''' 
        return SingleGroupOperator(self.moment_operator, self.transport_operator.to_energy_group(g_i), self.mass_operator, self.scatter_operator.to_inscatter(g_i))
    
    @property 
    def n_elements(self):
        return self.transport_operator.solution_domain_data.domain.n_elements
    
    @property 
    def n_angles(self):
        return self.transport_operator.n_angles
    
    @property 
    def n_moments(self):
        return self.moment_operator.n_moments
    
    @property 
    def n_basis(self):
        return self.transport_operator.solution_domain_data.n_basis

@lx.is_symmetric.register(MultiGroupOperator)
def _(operator : MultiGroupOperator):
    return False

@lx.linearise.register(MultiGroupOperator)
def _(operator : MultiGroupOperator):
    return operator


@eqx.filter_jit
def _multi_group_operator_scan(n_energy_groups : int, moment_operator    : MomentOperator, transport_operator : TransportOperatorMultipleAngles,  mass_operator      : MassOperator,  scatter_operator   : TotalGroupScatterOperator, x : jnp.ndarray) -> jnp.ndarray:

    in_structure_shape = (n_energy_groups,) + moment_operator.out_structure().shape

    assert x.shape == in_structure_shape, f"Input x shape {x.shape} does not match expected in_structure shape {in_structure_shape}."
    result = jnp.zeros(in_structure_shape, dtype=moment_operator.in_structure().dtype)
        
    def _single_energy_group(_, g_i):
        scatter_source = scatter_operator.to_energy_group(g_i).mv(x)
        y_g = (x[g_i] - moment_operator.mv(transport_operator.to_energy_group(g_i).inv(mass_operator.mv(scatter_source))))
        return None, y_g
    
    _, result = jax.lax.scan(
        _single_energy_group,
        init=None,
        xs=jnp.arange(n_energy_groups),
    )

    return result



class MultiGroupOperatorScan(MultiGroupOperator):
    r'''
    Multi energy group linear operator implementation using fori_loop over energy groups.
    This saves considerable memory compared to vmap implementation.

    '''    
    def mv(self, x : jnp.ndarray) -> jnp.ndarray:
        return _multi_group_operator_scan(self.n_energy_groups, self.moment_operator, self.transport_operator, self.mass_operator, self.scatter_operator, x)
    
    def transpose(self):
        raise ValueError("Transpose not implemented for MultiGroupOperatorScan yet")
   


@eqx.filter_jit
def solve_lower_triangular(mgop : MultiGroupOperator, solver, rhs : jnp.ndarray) -> jnp.ndarray:
    '''
    Solve the linear system using a given solver, assuming the multi-group operator is lower triangular.
    This is useful for e.g. downscattering only problems. 

    Parameters
    ----------
    mgop : MultiGroupOperator
        The multi group operator.
    solver : lx.LinearSolver
        The linear solver to use.
    rhs : jnp.ndarray
        The right-hand side.
    Returns
    -------
    jnp.ndarray
        The solution.   
    '''
    n_groups = mgop.n_energy_groups    
    solution = jnp.zeros_like(rhs)
    def body_fun(i, solution):        
        top_i = mgop.transport_operator.to_energy_group(i)
        sgop = SingleGroupOperator(mgop.moment_operator, mgop.transport_operator.to_energy_group(i), mgop.mass_operator, mgop.scatter_operator.to_inscatter(i))
        rhs_i = rhs[i] + mgop.moment_operator.mv(top_i.inv(mgop.mass_operator.mv(mgop.scatter_operator.to_energy_group(i).mv(solution))))        
        return solution.at[i].set(lx.linear_solve(sgop, rhs_i, solver, throw = False).value)
    return jax.lax.fori_loop(0, n_groups, body_fun, solution) 

@eqx.filter_jit
def solve_total(mgop : MultiGroupOperator, solver, rhs : jnp.ndarray) -> jnp.ndarray:
    r'''
    Solve the linear system using a given solver.

    Parameters
    ----------
    mgop : MultiGroupOperator
        The multi group operator.
    solver : lx.LinearSolver
        The linear solver to use.
    rhs : jnp.ndarray
        The right-hand side.

    Returns
    -------
    jnp.ndarray
        The solution.
    '''    
    return lx.linear_solve(mgop, rhs, solver).value