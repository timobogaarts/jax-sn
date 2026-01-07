import jax 
import jax.numpy as jnp

from dataclasses import dataclass   
from abc import ABC, abstractmethod

from jax_sn.solution_domain import SolutionDomainData
from jax_sn.domain import TopologicalSortedAngle
from jax_sn.source import Source
import equinox as eqx
import lineax as lx
from jax_sn.operators.base_linear_operator import AbstractLinearOperator_sn

class TransportOperatorSolver(lx.AbstractLinearSolver):
    def __init__(self):
        pass

    def init(self, operator : "TransportOperator", options):
        return operator
        
    def compute(self, solver_state, rhs, options):
        return solver_state._invert_internal(rhs), lx.RESULTS.successful, {}

    def allow_dependent_columns(self, operator):
        return False
    
    def allow_dependent_rows(self, operator):
        return False
    
    def transpose(self, solver_state, options):
        return solver_state.transpose(), options
    
    def conj(self, solver_state, options):
        return solver_state.conj(), options


class TransportOperator(AbstractLinearOperator_sn):
    '''
    Base class for transport operators.
    Since the implementation of the transport operator can vary (e.g., vmap-based, batched, Pallas, etc.), this is an abstract base class that defines the interface for transport operators.
        
    '''
    solution_domain_data      : SolutionDomainData 
    topological_sorted_angle  : TopologicalSortedAngle    #[this does not *require* a topological sorted angle for the forward operator. however, since the inverse does require it, we just require it here.]
    energy_group              : int

    def _assert_shape_base(self):
        assert self.topological_sorted_angle.ndim==1, "topological_sorted_angle must be 1D for base TransportOperator"
    
    def mv(self, x : jnp.ndarray) -> jnp.ndarray:
        return self._mv_op_args(*self._forward_args(x))
    
    def _invert_internal(self, b : jnp.ndarray) -> jnp.ndarray:
        return self._inv_op_args(*self._inv_args(b))
        
    @property
    @abstractmethod
    def _mv_op_args(self):
        ...

    @property 
    @abstractmethod
    def _inv_op_args(self):
        ...
    

    def transpose(self):
        return eqx.tree_at(lambda t_operator : t_operator.topological_sorted_angle, self, self.topological_sorted_angle.reverse)
    
    def inv(self, b : jnp.ndarray) -> jnp.ndarray:
        solver = TransportOperatorSolver()
        return lx.linear_solve(self, b, solver, throw=False).value
    
    def as_matrix(self):
        assert False, "Not desired for TransportOperator due to large size"
        return jnp.array([[]])
    
    def conj(self):
        return self  #real operator
    
    def in_structure(self):        
        return self._shape_inout
    
    def out_structure(self):
        return self._shape_inout
    
    @property 
    def _shape_inout(self):
        n_elements = self.solution_domain_data.domain.n_elements
        n_basis = self.solution_domain_data.fem_matrices.n_basis        
        return jax.ShapeDtypeStruct((n_elements, n_basis), self.solution_domain_data.fem_matrices.mass_matrix.dtype)
    
    def _base_forward_args(self, x):
        return (
            *self.solution_domain_data,
            self.solution_domain_data.domain.cross_sections_indexed.map_to_full_array(self.energy_group),
            self.topological_sorted_angle.angle,
            x
        )
    def _base_inv_args(self, x):
        return (
            *self.solution_domain_data,
            self.solution_domain_data.domain.cross_sections_indexed.map_to_full_array(self.energy_group),
            self.topological_sorted_angle,
            x
        )    
    
    def _forward_args(self, x):
        return self._base_forward_args(x)
    
    def _inv_args(self, b):
        return self._base_inv_args(b)
    
    @property
    def _vmap_for_forward_multiple_angles(self):
        return (None, None, None, None, 0,0)
    
    @property
    def _vmap_for_inv_multiple_angles(self):
        return (None, None, None, None, 0,0)
    
    @property 
    def n_angles(self):
        return 1
    

    
@lx.is_symmetric.register(TransportOperator)
def _(operator : TransportOperator):
    return False

@lx.linearise.register(TransportOperator)
def _(operator : TransportOperator):
    return operator

from .transport_operator_imp import transport_inverse_sweep_batched, transport_inverse_sweep_complete_vmap, transport_inverse_sweep_multiple_angles_batched, transport_inverse_sweep_multiple_angles_complete_vmap
from .transport_operator_forward import forward_transport_operator_vmap, forward_transport_operator_multiple_angles_vmap, forward_transport_operator_batch, forward_transport_operator_multiple_angles_batch

class TransportOperatorVmap(TransportOperator):

    @property
    def _mv_op_args(self):
        return forward_transport_operator_vmap
    
    @property
    def _inv_op_args(self):
        return transport_inverse_sweep_complete_vmap

class TransportOperatorBatched(TransportOperator):
    '''
    Transport operator using batched sweeps.

    '''
    batch_size : int = None
    
    @property
    def _mv_op_args(self):
        return forward_transport_operator_batch

    @property 
    def _inv_op_args(self):
        return transport_inverse_sweep_batched

    def _forward_args(self, x):
        return (*self._base_forward_args(x), self.batch_size)
    
    def _inv_args(self, b):
        return (*self._base_inv_args(b), self.batch_size)
    
    @property
    def _vmap_for_forward_multiple_angles(self):
        return (None, None, None, None, 0,0, None)
    
    @property
    def _vmap_for_inv_multiple_angles(self):
        return (None, None, None, None, 0,0, None)

from .transport_operator_pallas import transport_inverse_sweep_batched_pallas, transport_inverse_sweep_complete_vmap_pallas, transport_inverse_sweep_multiple_angles_batched_pallas, transport_inverse_sweep_multiple_angles_complete_vmap_pallas, _next_power_of_two

class TransportOperatorVmapPallas(TransportOperatorVmap):
    '''
    Transport operator using vmap sweeps with Pallas.

    '''  
    @property 
    def _inv_op_args(self):
        return transport_inverse_sweep_complete_vmap_pallas
    

class TransportOperatorBatchedPallas(TransportOperatorBatched):
    '''
    Transport operator using batched sweeps with Pallas.

    '''    
    
    @property 
    def _inv_op_args(self):
        return transport_inverse_sweep_batched_pallas


#===============================================================================================================================================
# Batching angles
#===============================================================================================================================================
@eqx.filter_jit
def vmap_multiple_angles_forward(transport_operator : TransportOperator, x : jnp.ndarray) -> jnp.ndarray:
    args = transport_operator._forward_args(x)
    f_vmap = jax.vmap(transport_operator._mv_op_args, in_axes=transport_operator._vmap_for_forward_multiple_angles)
    return f_vmap(*args)

@eqx.filter_jit
def vmap_multiple_angles_inv(transport_operator : TransportOperator, b : jnp.ndarray) -> jnp.ndarray:
    args = transport_operator._inv_args(b)
    f_vmap = jax.vmap(transport_operator._inv_op_args, in_axes=transport_operator._vmap_for_inv_multiple_angles)
    return f_vmap(*args)

class TransportOperatorMultipleAngles(AbstractLinearOperator_sn):    
    transport_operator : TransportOperator

    def _assert_shape_base(self):
        assert self.transport_operator.topological_sorted_angle.angle.ndim==2, "topological_sorted_angle must be 2D for base TransportOperatorMultipleAngles"

    @abstractmethod
    def mv(self, x : jnp.ndarray) -> jnp.ndarray:
        ...

    @abstractmethod
    def _invert_internal(self, b : jnp.ndarray) -> jnp.ndarray:
        ...

    def inv(self, b : jnp.ndarray) -> jnp.ndarray:
        solver = TransportOperatorSolver()
        return lx.linear_solve(self, b, solver, throw=False).value
    
    def transpose(self):
        return eqx.tree_at(lambda t_operator : t_operator.transport_operator.topological_sorted_angle, self, self.transport_operator.topological_sorted_angle.reverse)    
    
    def as_matrix(self):
        assert False, "Not desired for TransportOperator due to large size"
        return jnp.array([[]])
    
    def conj(self):
        return self  #real operator
    
    def in_structure(self):        
        return self._shape_inout
    
    def out_structure(self):
        return self._shape_inout
    
    def to_energy_group(self, energy_group : int) -> "TransportOperatorMultipleAngles":
        return eqx.tree_at(lambda t_operator : t_operator.transport_operator.energy_group, self, energy_group)
    
    @property
    def solution_domain_data(self):
        return self.transport_operator.solution_domain_data
    
    @property 
    def energy_group(self):
        return self.transport_operator.energy_group

    @property 
    def topological_sorted_angle(self):
        return self.transport_operator.topological_sorted_angle
        
    
    @property 
    def _shape_inout(self):
        self._assert_shape_base()
        n_angles   = self.transport_operator.topological_sorted_angle.angle.shape[0]
        n_elements = self.transport_operator.solution_domain_data.domain.n_elements
        n_basis    = self.transport_operator.solution_domain_data.fem_matrices.n_basis        
        return jax.ShapeDtypeStruct((n_angles, n_elements, n_basis), self.transport_operator.solution_domain_data.fem_matrices.mass_matrix.dtype)
    
    @property
    def n_angles(self):
        return self.transport_operator.topological_sorted_angle.angle.shape[0]

@lx.is_symmetric.register(TransportOperatorMultipleAngles)
def _(operator : TransportOperator):
    return False

@lx.linearise.register(TransportOperatorMultipleAngles)
def _(operator : TransportOperator):
    return operator

class TransportOperatorMultipleAnglesVmap(TransportOperatorMultipleAngles):

    def mv(self, x : jnp.ndarray) -> jnp.ndarray:
        return vmap_multiple_angles_forward(self.transport_operator, x)
    
    def _invert_internal(self, b : jnp.ndarray) -> jnp.ndarray:
        return vmap_multiple_angles_inv(self.transport_operator, b)
    


        