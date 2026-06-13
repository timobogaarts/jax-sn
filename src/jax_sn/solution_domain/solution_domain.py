import jax 
import jax.numpy as jnp 
from dataclasses import dataclass
from .base_fem import Element, PrecomputedGeometricData, FEMMatrices
from typing import List, Tuple
from jax_sn.domain import GeometricDomain

from jax_sn.source import Source
import equinox as eqx
from .base_fem import Element
from jax_sn.domain.geometric_domain import Domain

def compute_setup_for_solution_domain(element : Element, domain : GeometricDomain) -> Tuple[FEMMatrices, PrecomputedGeometricData]:
    '''
    Computes the FEM matrices and precomputed geometric data for the given element and domain.

    Parameters
    ----------
    element : Element
        The finite element to use.
    domain : GeometricDomain
        The geometric domain to use.    

    Returns
    -------
    fem_matrices : FEMMatrices  
        The finite element matrices for the given element.
    precomputed_data : PrecomputedGeometricData
        The precomputed geometric data for the given element and domain.
    '''
    fem_matrices     = FEMMatrices.from_element(element)
    precomputed_data = element.precompute_geometric_data(domain)
    return fem_matrices, precomputed_data

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SolutionDomainData:
    '''
    Bundles all data required to solve the transport equation on a domain in a JIT-compatible way.
    Is a frozen dataclass to allow JIT-compilation.
    '''

    domain                       : Domain        # JIT-compatible (frozen dataclass)
    '''
        The geometric domain on which to solve the transport equation. 
    '''
    fem_matrices                 : FEMMatrices   # JIT-compatible (frozen dataclass)
    '''
        The finite element matrices for the domain.
    '''
    precomputed_geometric_data   : PrecomputedGeometricData # JIT-compatible (frozen dataclass)
    '''
        The precomputed geometric data for the domain.
    '''

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
        assert points.ndim == 2 and points.shape[1] == self.dimension, f"Points must be of shape (n_points, dimension) but got {points.shape} with domain dimension {self.dimension}"
        assert solution.shape[-2:] == self.elements_basis_shape.shape, f"Solution must be of shape {self.elements_basis_shape} but got {solution.shape}"
        assert solution.ndim in (2,3), f"Solution must be of shape (n_elements, n_basis) or (n_solutions, n_elements, n_basis) but got {solution.shape}"
        if solution.ndim == 2:
            return _interpolate_from_solution_domain(points, self, solution)
        else:
            return _interpolate_from_solution_domain_solution_batch(points, self, solution)
    
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
    @property 
    def dimension(self) -> int:
        return self.solution_domain_data.domain.vertices.shape[-1]
    
    @property 
    def elements_basis_shape(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(
            shape=(self.n_elements, self.n_basis),
            dtype=self.fem_matrices.mass_matrix.dtype
        )
    
    
    def map_cross_sections_to_full_array(self, energy_group : int):
        return self.solution_domain_data.domain.cross_sections_indexed.map_to_full_array(energy_group)
    

def _interpolate_from_solution_domain(points, solution_domain : SolutionDomain, solution : jnp.ndarray):
    '''
    Interpolate the solution at given points from the solution domain
    '''
    indices_elem = solution_domain.domain.containing_cell(points)
    return jnp.where(indices_elem >=0, solution_domain.element.interpolate_from_real_coordinate(points, indices_elem, solution_domain.domain, solution_domain.precomputed_geometric_data, solution), jnp.nan)


def _interpolate_from_solution_domain_solution_batch(points, solution_domain : SolutionDomain, solution : jnp.ndarray):
    '''
    Interpolate the solution at given points from the solution domain for a batch of solutions
    '''
    indices_elem = solution_domain.domain.containing_cell(points)
    
    return jax.vmap(lambda solution_i: jnp.where(indices_elem >=0, solution_domain.element.interpolate_from_real_coordinate(points, indices_elem, solution_domain.domain, solution_domain.precomputed_geometric_data, solution_i), jnp.nan), in_axes=0)(solution)


    