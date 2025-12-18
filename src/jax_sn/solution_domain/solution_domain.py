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

def compute_setup_for_solution_domain(element : Element, domain : GeometricDomain) -> Tuple[FEMMatrices, PrecomputedGeometricData]:
    fem_matrices     = FEMMatrices.from_element(element)
    precomputed_data = element.precompute_geometric_data(domain)
    return fem_matrices, precomputed_data

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SolutionDomainData:
    domain                       : Domain        # JIT-compatible (frozen dataclass)
    fem_matrices                 : FEMMatrices   # JIT-compatible (frozen dataclass)
    precomputed_geometric_data   : PrecomputedGeometricData # JIT-compatible (frozen dataclass)


    def __iter__(self):        
        yield self.domain
        yield self.fem_matrices
        yield self.precomputed_geometric_data

    @property
    def n_basis(self) -> int:
        return self.fem_matrices.n_basis


class SolutionDomain(eqx.Module):
    '''
    Convenience class to bundle all data required to solve the transport equation on a domain
    Is not fully JIT-compatible due to the Element class. Its methods are also available as standalone functions.
    '''
    solution_domain_data : SolutionDomainData     
    element              : Element  = eqx.field(static=True) # NOT JIT-compatible [can involve e.g. Basix element which is non-jax]

    @classmethod
    def from_element_and_domain(cls, element : Element, domain : Domain) -> "SolutionDomain":        
        assert isinstance(domain, Domain), f"domain must be an instance of Domain but got {type(domain)}"
        assert isinstance(element, Element), f"element must be an instance of Element but got {type(element)}"
        return cls(SolutionDomainData(domain, *compute_setup_for_solution_domain(element, domain)),
            element)
        
    def interpolate(self, points : jnp.ndarray, solution : jnp.ndarray):
        return _interpolate_from_solution_domain(points, self, solution)
    
    @property
    def domain(self) -> Domain:
        return self.solution_domain_data.domain
    @property   
    def precomputed_geometric_data(self) -> PrecomputedGeometricData:
        return self.solution_domain_data.precomputed_geometric_data
    @property
    def fem_matrices(self) -> FEMMatrices:
        return self.solution_domain_data.fem_matrices
    
    @property 
    def n_basis(self) -> int:
        return self.solution_domain_data.fem_matrices.n_basis
    @property 
    def n_elements(self) -> int:
        return self.solution_domain_data.domain.n_elements
    
    
    def map_cross_sections_to_full_array(self, energy_group : int):
        return self.solution_domain_data.domain.cross_sections_indexed.map_to_full_array(energy_group)
    


def _interpolate_from_solution_domain(points, solution_domain : SolutionDomain, solution : jnp.ndarray):
    '''
    Interpolate the solution at given points from the solution domain
    '''
    indices_elem = solution_domain.domain.containing_cell(points)
    return jnp.where(indices_elem >=0, solution_domain.element.interpolate_from_real_coordinate(points, indices_elem, solution_domain.domain, solution_domain.precomputed_geometric_data, solution), jnp.nan)
    