import jax 
import jax.numpy as jnp 
from dataclasses import dataclass
from .base_fem import Element, compute_volume_matrices_linear_jacobian, compute_face_matrices_linear_jacobian
from typing import List, Tuple
from jax_sn.domain.geometric_domain import GeometricDomain, TopologicalSortedAngle, PrecomputedGeometricData, CrossSectionData

from jax_sn.source import Source
import equinox as eqx
from .base_fem import Element


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FEMMatrices:
    mass_matrix       : jnp.ndarray  #[n_basis, n_basis]
    integration_coeff : jnp.ndarray  #[n_basis]
    derivative_matrix : jnp.ndarray  #[n_dim, n_basis, n_basis]
    outflow_matrices  : jnp.ndarray  #[n_faces_per_element, n_basis, n_basis]
    inflow_matrices   : jnp.ndarray  #[n_faces_per_element, n_permutations, n_faces_per_element, n_basis, n_basis]

    @classmethod 
    def from_element(cls, element : Element):
        return cls(*compute_volume_matrices_linear_jacobian(element), *compute_face_matrices_linear_jacobian(element))

def remove_zeros(array):
    import numpy as onp
    a = onp.array(array)
    a[onp.abs(a) < 1e-14] = 0.0
    return a

class SolutionDomain(eqx.Module):
    '''
    Convenience class to bundle all data required to solve the transport equation on a domain
    Is not fully JIT-compatible due to the Element class.
    '''
    fem_matrices       : FEMMatrices
    geometric_domain   : GeometricDomain    
    cross_sections     : CrossSectionData
    precomputed_data   : PrecomputedGeometricData
    element            : Element

    @classmethod
    def from_element_and_domain(cls, element : Element, domain : GeometricDomain, cross_sections : CrossSectionData):
        fem_matrices     = FEMMatrices.from_element(element)
        precomputed_data = element.precompute_geometric_data(domain)
        return cls(
            fem_matrices=fem_matrices,
            geometric_domain=domain,            
            cross_sections=cross_sections,
            precomputed_data=precomputed_data,
            element=element
        )
    @classmethod 
    def from_element_and_mesh(cls, element : Element, mesh : Tuple[jnp.ndarray, jnp.ndarray], cross_sections : CrossSectionData):
        domain = GeometricDomain.from_mesh(mesh, element.face_template)
        element.assert_compatible(domain)
        return cls.from_element_and_domain(element, domain, cross_sections)