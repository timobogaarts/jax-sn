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
            
    
    def __call__(self, precomputed_source : Source, energy_group : int) -> jnp.ndarray:
        pass

    @abstractmethod
    def inv(self, precomputed_source : Source, energy_group : int) -> jnp.ndarray:
        ...

    @property
    def batched_angles(self) -> bool:
        return self.topological_sorted_angles.angle.ndim == 2

from .transport_operator_imp import transport_inverse_sweep_batched, transport_inverse_sweep_complete_vmap, transport_inverse_sweep_multiple_angles_batched, transport_inverse_sweep_multiple_angles_complete_vmap

class TransportOperatorVmap(TransportOperator):
    '''
    Transport operator using fully vmap-based sweeps.

    '''
    def inv(self, precomputed_source : Source, energy_group : int) -> jnp.ndarray:
        if self.batched_angles:
            return transport_inverse_sweep_multiple_angles_complete_vmap(
                *self.solution_domain_data,
                self.solution_domain_data.domain.cross_sections_indexed.map_to_full_array(energy_group),
                self.topological_sorted_angles,
                precomputed_source,
                jnp.zeros((self.topological_sorted_angles.angle.shape[0], self.solution_domain_data.domain.n_elements, self.solution_domain_data.n_basis))
            )
        else:
            return transport_inverse_sweep_complete_vmap(
                *self.solution_domain_data,
                self.solution_domain_data.domain.cross_sections_indexed.map_to_full_array(energy_group),
                self.topological_sorted_angles,
                precomputed_source,
                jnp.zeros((self.solution_domain_data.domain.n_elements, self.solution_domain_data.n_basis))
            )

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
    def inv(self, precomputed_source : Source, energy_group : int) -> jnp.ndarray:
        if self.batched_angles:
            return transport_inverse_sweep_multiple_angles_batched(
                *self.solution_domain_data,
                self.solution_domain_data.domain.cross_sections_indexed.map_to_full_array(energy_group),
                self.topological_sorted_angles,
                precomputed_source,
                jnp.zeros((self.topological_sorted_angles.angle.shape[0], self.solution_domain_data.domain.n_elements, self.solution_domain_data.n_basis)),
                self.batch_size
            )
        else:   
            return transport_inverse_sweep_batched(
                *self.solution_domain_data,
                self.solution_domain_data.domain.cross_sections_indexed.map_to_full_array(energy_group),
                self.topological_sorted_angles,
                precomputed_source,
                jnp.zeros((self.solution_domain_data.domain.n_elements, self.solution_domain_data.n_basis)),
                self.batch_size
                )            

from .transport_operator_pallas import transport_inverse_sweep_batched_pallas, transport_inverse_sweep_complete_vmap_pallas, transport_inverse_sweep_multiple_angles_batched_pallas, transport_inverse_sweep_multiple_angles_complete_vmap_pallas, _next_power_of_two

class TransportOperatorVmapPallas(TransportOperator):
    '''
    Transport operator using fully vmap-based sweeps with Pallas.

    '''
    def inv(self, precomputed_source : Source, energy_group : int) -> jnp.ndarray:
        if self.batched_angles:
            return transport_inverse_sweep_multiple_angles_complete_vmap_pallas(
                *self.solution_domain_data,
                self.solution_domain_data.domain.cross_sections_indexed.map_to_full_array(energy_group),
                self.topological_sorted_angles,
                precomputed_source,
                jnp.zeros((self.topological_sorted_angles.angle.shape[0], self.solution_domain_data.domain.n_elements, _next_power_of_two(self.solution_domain_data.n_basis)))
            )
        else:
            return transport_inverse_sweep_complete_vmap_pallas(
                *self.solution_domain_data,
                self.solution_domain_data.domain.cross_sections_indexed.map_to_full_array(energy_group),
                self.topological_sorted_angles,
                precomputed_source,
                jnp.zeros((self.solution_domain_data.domain.n_elements, _next_power_of_two(self.solution_domain_data.n_basis)))
            )
        
class TransportOperatorBatchedPallas(TransportOperator):
    '''
    Transport operator using batched sweeps with Pallas.
    '''
    batch_size : int = None

    @classmethod
    def from_solution_domain_data_and_toposort(cls, solution_domain_data : SolutionDomainData, topological_sorted_angles : TopologicalSortedAngle, **kwargs) -> "TransportOperatorBatchedPallas":
        assert isinstance(solution_domain_data, SolutionDomainData), f"solution_domain_data must be an instance of SolutionDomainData but got {type(solution_domain_data)}"
        assert isinstance(topological_sorted_angles, TopologicalSortedAngle), f"topological_sorted_angles must be an instance of TopologicalSortedAngle but got {type(topological_sorted_angles)}"
        assert "batch_size" in kwargs, "batch_size must be provided as a keyword argument for TransportOperatorBatchedPallas"
        return cls(
            solution_domain_data      = solution_domain_data,
            topological_sorted_angles = topological_sorted_angles, 
            batch_size = kwargs.get("batch_size")
        )
    def inv(self, precomputed_source : Source, energy_group : int) -> jnp.ndarray:
        if self.batched_angles:
            return transport_inverse_sweep_multiple_angles_batched_pallas(
                *self.solution_domain_data,
                self.solution_domain_data.domain.cross_sections_indexed.map_to_full_array(energy_group),
                self.topological_sorted_angles,
                precomputed_source,
                jnp.zeros((self.topological_sorted_angles.angle.shape[0], self.solution_domain_data.domain.n_elements, _next_power_of_two(self.solution_domain_data.n_basis))),
                self.batch_size
            )
        else:   
            return transport_inverse_sweep_batched_pallas(
                *self.solution_domain_data,
                self.solution_domain_data.domain.cross_sections_indexed.map_to_full_array(energy_group),
                self.topological_sorted_angles,
                precomputed_source,
                jnp.zeros((self.solution_domain_data.domain.n_elements, _next_power_of_two(self.solution_domain_data.n_basis))),
                self.batch_size
                )
