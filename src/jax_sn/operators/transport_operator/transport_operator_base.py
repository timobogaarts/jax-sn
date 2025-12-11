import jax 
import jax.numpy as jnp

from dataclasses import dataclass   
from abc import ABC, abstractmethod

from jax_sn.solution_domain import SolutionDomainData
from jax_sn.domain import TopologicalSortedAngle
from jax_sn.source import Source
import equinox as eqx

class InverseTransportOperator(eqx.Module):
    '''
    Base class for transport operators.
    '''
    solution_domain_data      : SolutionDomainData 
    topological_sorted_angles : TopologicalSortedAngle    

    @classmethod 
    def from_solution_domain_data_and_toposort(cls, solution_domain_data : SolutionDomainData, topological_sorted_angles : TopologicalSortedAngle, **kwargs) -> "InverseTransportOperator":
        assert isinstance(solution_domain_data, SolutionDomainData), f"solution_domain_data must be an instance of SolutionDomainData but got {type(solution_domain_data)}"
        assert isinstance(topological_sorted_angles, TopologicalSortedAngle), f"topological_sorted_angles must be an instance of TopologicalSortedAngle but got {type(topological_sorted_angles)}"
        return cls(
            solution_domain_data,
            topological_sorted_angles
        )
            
    @abstractmethod
    def __call__(self, precomputed_source : Source, energy_group : int) -> jnp.ndarray:
        ...

    @property
    def batched_angles(self) -> bool:
        return self.topological_sorted_angles.angle.ndim == 2

from .transport_operator_imp import transport_inverse_sweep_batched, transport_inverse_sweep_complete_vmap, transport_inverse_sweep_multiple_angles_batched, transport_inverse_sweep_multiple_angles_complete_vmap

class InverseTransportOperatorVmap(InverseTransportOperator):
    '''
    Transport operator using fully vmap-based sweeps.

    '''
    def __call__(self, precomputed_source : Source, energy_group : int) -> jnp.ndarray:
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

class InverseTransportOperatorBatched(InverseTransportOperator):
    '''
    Transport operator using batched sweeps.

    '''
    batch_size : int = None
    
    @classmethod
    def from_solution_domain_data_and_toposort(cls, solution_domain_data : SolutionDomainData, topological_sorted_angles : TopologicalSortedAngle, **kwargs) -> "InverseTransportOperatorBatched":
        assert isinstance(solution_domain_data, SolutionDomainData), f"solution_domain_data must be an instance of SolutionDomainData but got {type(solution_domain_data)}"
        assert isinstance(topological_sorted_angles, TopologicalSortedAngle), f"topological_sorted_angles must be an instance of TopologicalSortedAngle but got {type(topological_sorted_angles)}"
        assert "batch_size" in kwargs, "batch_size must be provided as a keyword argument for InverseTransportOperatorBatched"
        return cls(
            solution_domain_data      = solution_domain_data,
            topological_sorted_angles = topological_sorted_angles, 
            batch_size = kwargs.get("batch_size")
        )
    def __call__(self, precomputed_source : Source, energy_group : int) -> jnp.ndarray:
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


