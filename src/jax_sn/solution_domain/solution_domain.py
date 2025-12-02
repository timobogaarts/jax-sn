import jax 
import jax.numpy as jnp 
from dataclasses import dataclass
from .base_fem import Element, compute_volume_matrices_linear_jacobian, compute_face_matrices_linear_jacobian
from typing import List, Tuple
from jax_sn.domain.geometric_domain import GeometricDomain, TopologicalSortedAngle, PrecomputedGeometricData, CrossSectionData

from jax_sn.source import Source
import equinox as eqx

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FEMMatrices:
    mass_matrix       : jnp.ndarray 
    integration_coeff : jnp.ndarray
    derivative_matrix : jnp.ndarray
    outflow_matrices  : jnp.ndarray
    inflow_matrices   : jnp.ndarray

    @classmethod 
    def from_element(cls, element : Element):
        return cls(*compute_volume_matrices_linear_jacobian(element), *compute_face_matrices_linear_jacobian(element))


@eqx.filter_jit
def Gij_kernel(
    elem_idx                   : int,
    FEMMatrices                : FEMMatrices,
    geometric_domain           : GeometricDomain,
    topological_sorted_angle   : TopologicalSortedAngle,
    cross_sections             : CrossSectionData,
    precomputed_geometric_data : PrecomputedGeometricData,
    precomputed_source         : Source,    
    inflow                     : jnp.ndarray,
    abs_omega_dot_n_Je         : jnp.ndarray
    ) -> None:
    vol_jacobian = precomputed_geometric_data.vol_jacobians_det[elem_idx]            
    outflow_matrices = jnp.where(inflow[elem_idx][:, None, None], FEMMatrices.outflow_matrices, 0.0)
    # note the b,ab,aij term could be precomputed to a bij term ;)
    Gij_matrix       = FEMMatrices.mass_matrix * vol_jacobian * cross_sections.total[elem_idx]\
                       + jnp.einsum("b, ab, aij-> ij", topological_sorted_angle.angle, precomputed_geometric_data.vol_jacobians_inv[elem_idx], FEMMatrices.derivative_matrix)\
                       + jnp.einsum("f, fij-> ij", abs_omega_dot_n_Je[elem_idx], outflow_matrices)     
    return jnp.linalg.inv(Gij_matrix)


Gij_kernel_vmap = jax.jit(jax.vmap(Gij_kernel, in_axes=(0, None, None, None, None, None, None, None, None)))


def T_inv_sweep(FEMMatrices : FEMMatrices, geometric_domain : GeometricDomain, topological_sorted_angle : TopologicalSortedAngle, cross_sections : CrossSectionData, precomputed_geometric_data : PrecomputedGeometricData, precomputed_source : Source, sweep_result : jnp.ndarray) -> jnp.ndarray:    

    omega_dot_n = jnp.sum(geometric_domain.face_normals * topological_sorted_angle.angle[None, None, :], axis=-1)
    abs_omega_dot_n_Je = jnp.abs(omega_dot_n) * precomputed_geometric_data.face_jacobians_det
    inflow = omega_dot_n < 0.0

    Gijmatrices = Gij_kernel_vmap(
        topological_sorted_angle.levels_argsort,
        FEMMatrices,
        geometric_domain,
        topological_sorted_angle,
        cross_sections,
        precomputed_geometric_data,
        precomputed_source,
        inflow,
        abs_omega_dot_n_Je
    )
     # incorrect ! check with stell-blanket base 
    return Gijmatrices
    
        
        

        
        

        
        
        
        