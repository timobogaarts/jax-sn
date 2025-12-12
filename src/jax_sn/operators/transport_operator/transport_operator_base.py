import jax 
import jax.numpy as jnp

from dataclasses import dataclass   
from abc import ABC, abstractmethod

from jax_sn.solution_domain import SolutionDomainData
from jax_sn.domain import TopologicalSortedAngle
from jax_sn.source import Source
import equinox as eqx

class TransportOperator(eqx.Module):
    '''
    Base class for transport operators.
    Since the implementation of the transport operator can vary (e.g., vmap-based, batched, Pallas, etc.), this is an abstract base class that defines the interface for transport operators.

    This class provides methods for applying the transport operator, its inverse, and their adjoints. These adjoint are required for differentiating through the inverse and for adjoint-based methods.
    
    '''
    solution_domain_data      : SolutionDomainData 
    topological_sorted_angles : TopologicalSortedAngle    

    @classmethod 
    def from_solution_domain_data_and_toposort(cls, solution_domain_data : SolutionDomainData, topological_sorted_angles : TopologicalSortedAngle, **kwargs) -> "TransportOperator":
        assert isinstance(solution_domain_data, SolutionDomainData), f"solution_domain_data must be an instance of SolutionDomainData but got {type(solution_domain_data)}"
        assert isinstance(topological_sorted_angles, TopologicalSortedAngle), f"topological_sorted_angles must be an instance of TopologicalSortedAngle but got {type(topological_sorted_angles)}"
        return cls(
            solution_domain_data,
            topological_sorted_angles
        )
    
    @property
    def batched_angles(self) -> bool:
        return self.topological_sorted_angles.angle.ndim == 2            
    
    def __call__(self, x : jnp.ndarray, energy_group : int) -> jnp.ndarray:
        total_forward_args = (*self._base_forward_args(x, energy_group), *self._additional_forward_args(x, energy_group))
        if self.batched_angles:
            return self._multiple_angle_forward_f(*total_forward_args)                
        else:
            return self._single_angle_forward_f(*total_forward_args)

    
    def inv(self, x : jnp.ndarray, energy_group : int) -> jnp.ndarray:
        total_sweep_args = (*self._base_sweep_args(x, energy_group), *self._additional_sweep_args(x, energy_group))
        if self.batched_angles:
            return self._multiple_angle_sweep_f(*total_sweep_args)                
        else:
            return self._single_angle_sweep_f(*total_sweep_args)
        
    def T(self, x : jnp.ndarray, energy_group : int) -> jnp.ndarray:
        total_forward_args_T = (*self._base_forward_args_T(x, energy_group), *self._additional_forward_args(x, energy_group))
        if self.batched_angles:
            return self._multiple_angle_forward_f(*total_forward_args_T)                
        else:
            return self._single_angle_forward_f(*total_forward_args_T)
    
    
    def T_inv(self, x : jnp.ndarray, energy_group : int) -> jnp.ndarray:
        total_sweep_args_T = (*self._base_sweep_args_T(x, energy_group), *self._additional_sweep_args(x, energy_group))
        if self.batched_angles:
            return self._multiple_angle_sweep_f(*total_sweep_args_T)                
        else:
            return self._single_angle_sweep_f(*total_sweep_args_T)
            
    def _base_sweep_args(self, x, energy_group):
        return (
            *self.solution_domain_data,
            self.solution_domain_data.domain.cross_sections_indexed.map_to_full_array(energy_group),
            self.topological_sorted_angles, 
            x
        )
    
    def _base_sweep_args_T(self, x, energy_group):
        return (
            *self.solution_domain_data,
            self.solution_domain_data.domain.cross_sections_indexed.map_to_full_array(energy_group),
            self.topological_sorted_angles.reverse, 
            x
        )
    
    def _base_forward_args_T(self, x, energy_group):
        return (
            *self.solution_domain_data,
            self.solution_domain_data.domain.cross_sections_indexed.map_to_full_array(energy_group),
            - self.topological_sorted_angles.angle,
            x
        )    

    def _additional_sweep_args(self, x, energy_group):
        return ()
    
    def _base_forward_args(self, x, energy_group):
        return (
            *self.solution_domain_data,
            self.solution_domain_data.domain.cross_sections_indexed.map_to_full_array(energy_group),
            self.topological_sorted_angles.angle,
            x
        )
    
    def _additional_forward_args(self, x, energy_group):
        return ()

    @property     
    @abstractmethod    
    def _single_angle_sweep_f(self):
        ...

    @property     
    @abstractmethod    
    def _multiple_angle_sweep_f(self):
        ...

    @property     
    @abstractmethod
    def _single_angle_forward_f(self):
        ...

    @property     
    @abstractmethod
    def _multiple_angle_forward_f(self):
        ...        
    

from .transport_operator_imp import transport_inverse_sweep_batched, transport_inverse_sweep_complete_vmap, transport_inverse_sweep_multiple_angles_batched, transport_inverse_sweep_multiple_angles_complete_vmap
from .transport_operator_forward import forward_transport_operator_vmap, forward_transport_operator_multiple_angles_vmap, forward_transport_operator_batch, forward_transport_operator_multiple_angles_batch

class TransportOperatorVmap(TransportOperator):
    '''
    Transport operator using fully vmap-based sweeps.

    '''
    @property
    def _single_angle_sweep_f(self):
        return transport_inverse_sweep_complete_vmap
    @property
    def _multiple_angle_sweep_f(self):
        return transport_inverse_sweep_multiple_angles_complete_vmap
    
    @property 
    def _single_angle_forward_f(self):
        return forward_transport_operator_vmap
    @property 
    def _multiple_angle_forward_f(self):
        return forward_transport_operator_multiple_angles_vmap

class TransportOperatorBatched(TransportOperator):
    '''
    Transport operator using batched sweeps.

    '''
    batch_size : int = None
    
    @classmethod
    def from_solution_domain_data_and_toposort(cls, solution_domain_data : SolutionDomainData, topological_sorted_angles : TopologicalSortedAngle, **kwargs) -> "TransportOperatorBatched":
        assert isinstance(solution_domain_data, SolutionDomainData), f"solution_domain_data must be an instance of SolutionDomainData but got {type(solution_domain_data)}"
        assert isinstance(topological_sorted_angles, TopologicalSortedAngle), f"topological_sorted_angles must be an instance of TopologicalSortedAngle but got {type(topological_sorted_angles)}"
        assert "batch_size" in kwargs, "batch_size must be provided as a keyword argument for TransportOperatorBatched"
        return cls(
            solution_domain_data      = solution_domain_data,
            topological_sorted_angles = topological_sorted_angles, 
            batch_size = kwargs.get("batch_size")
        )
    @property           
    def _single_angle_sweep_f(self):
        return transport_inverse_sweep_batched
    @property           
    def _multiple_angle_sweep_f(self):
        return transport_inverse_sweep_multiple_angles_batched
    
    @property 
    def _single_angle_forward_f(self):
        return forward_transport_operator_batch
    @property 
    def _multiple_angle_forward_f(self):
        return forward_transport_operator_multiple_angles_batch
    
    def _additional_sweep_args(self, x, energy_group):
        return (self.batch_size,)
    
    def _additional_forward_args(self, x, energy_group):
        return (self.batch_size,)

from .transport_operator_pallas import transport_inverse_sweep_batched_pallas, transport_inverse_sweep_complete_vmap_pallas, transport_inverse_sweep_multiple_angles_batched_pallas, transport_inverse_sweep_multiple_angles_complete_vmap_pallas, _next_power_of_two

class TransportOperatorVmapPallas(TransportOperatorVmap):
    '''
    Transport operator using fully vmap-based sweeps with Pallas.

    '''
    @property
    def _single_angle_sweep_f(self):
        return transport_inverse_sweep_complete_vmap_pallas
    @property
    def _multiple_angle_sweep_f(self):
        return transport_inverse_sweep_multiple_angles_complete_vmap_pallas
        
class TransportOperatorBatchedPallas(TransportOperatorBatched):
    '''
    Transport operator using batched sweeps with Pallas.

    '''
    @property           
    def _single_angle_sweep_f(self):
        return transport_inverse_sweep_batched_pallas
    @property           
    def _multiple_angle_sweep_f(self):
        return transport_inverse_sweep_multiple_angles_batched_pallas    


# def adjoint_inverse_sweep(normal_sweep_function, ):
#     '''
#     Perform an adjoint inverse sweep by reversing the order of the topological sort and negating the angles.

#     Parameters