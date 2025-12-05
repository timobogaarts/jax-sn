import jax 
import jax.numpy as jnp 
from dataclasses import dataclass
from .base_fem import Element, PrecomputedGeometricData, FEMMatrices
from typing import List, Tuple
from jax_sn.domain.geometric_domain import GeometricDomain, TopologicalSortedAngle, CrossSectionData

from jax_sn.source import Source
import equinox as eqx
from .base_fem import Element
from jax_sn.domain import Domain
from .transport_operator import _transport_inverse_sweep_complete_vmap_fori_domain

def compute_setup_for_solution_domain(element : Element, domain : GeometricDomain) -> Tuple[FEMMatrices, PrecomputedGeometricData]:
    fem_matrices     = FEMMatrices.from_element(element)
    precomputed_data = element.precompute_geometric_data(domain)
    return fem_matrices, precomputed_data


class SolutionDomain(eqx.Module):
    '''
    Convenience class to bundle all data required to solve the transport equation on a domain
    Is not fully JIT-compatible due to the Element class. Its methods are also available as standalone functions.
    '''
    domain             : Domain        # JIT-compatible (frozen dataclass)
    fem_matrices       : FEMMatrices   # JIT-compatible (frozen dataclass)
    precomputed_data   : PrecomputedGeometricData # JIT-compatible (frozen dataclass)
    element            : Element  = eqx.field(static=True) # NOT JIT-compatible [can involve e.g. Basix element which is non-jax]

    @classmethod
    def from_element_and_domain(cls, element : Element, domain : Domain) -> "SolutionDomain":        
        return cls(domain, 
            *compute_setup_for_solution_domain(element, domain),
            element)
    
    def sweep(self, toposort_angle, source : Source, energy_group : int, sweep_buffer : jnp.ndarray, method : str = "vmap") -> jnp.ndarray:

        if method == "vmap":
            return _transport_inverse_sweep_complete_vmap_fori_domain(
                self.fem_matrices,
                self.domain,
                toposort_angle,                 
                self.precomputed_data,                
                source,                
                sweep_buffer,
                energy_group)
        else:
            raise ValueError(f"Unknown sweep method {method}")
        
    def interpolate(self, points : jnp.ndarray, solution : jnp.ndarray):
        return _interpolate_from_solution_domain(points, self, solution)


def _interpolate_from_solution_domain(points, solution_domain : SolutionDomain, solution : jnp.ndarray):
    '''
    Interpolate the solution at given points from the solution domain
    '''
    indices_elem = solution_domain.domain.containing_cell(points)
    return solution_domain.element.interpolate_from_real_coordinate(points, indices_elem, solution_domain.domain, solution_domain.precomputed_data, solution)
    