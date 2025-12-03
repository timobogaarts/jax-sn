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

def remove_zeros(array):
    import numpy as onp
    a = onp.array(array)
    a[onp.abs(a) < 1e-14] = 0.0
    return a

def Gij_kernel(
    elem_idx                   : int,
    FEMMatrices                : FEMMatrices,    
    angle                      : jnp.ndarray ,
    cross_sections             : CrossSectionData,
    precomputed_geometric_data : PrecomputedGeometricData,    
    inflow                     : jnp.ndarray,
    abs_omega_dot_n_Je         : jnp.ndarray
    ) -> None:
    '''
    Compute the Gij matrix for a single element index.

    Computes:
    Gij = σ J ∫H_i H_j dV +  J ∑_{a,b} Ω_{b} (J)⁻¹_{ab} ∫ - ∂H_i/∂q_a H_j dV + ∑_{f outflow} |Ω·n_f| J_f ∫H_i H_j dA

    Parameters
    ----------
    elem_idx : int
        The element index to compute the Gij matrix for.
    FEMMatrices : FEMMatrices
        The finite element method matrices.
    angle : jnp.ndarray
        The direction vector.
    cross_sections : CrossSectionData
        The cross section data.
    precomputed_geometric_data : PrecomputedGeometricData
        The precomputed geometric data (Jacobian determinants and inverses).
    inflow : jnp.ndarray
        Boolean array indicating inflow faces for all alements (indexed by element and face).
    abs_omega_dot_n_Je : jnp.ndarray
        The absolute value of the dot product of omega and face normals multiplied by face Jacobian determinants
        for all elements and faces.
    Returns
    -------
    Gij_matrix : jnp.ndarray
        The computed Gij matrix for the specified element.
    '''
    vol_jacobian     = precomputed_geometric_data.vol_jacobians_det[elem_idx]            
    outflow_matrices = jnp.where(inflow[elem_idx][:, None, None], 0.0, FEMMatrices.outflow_matrices)        
    Aij = vol_jacobian * cross_sections.total[elem_idx] * FEMMatrices.mass_matrix 
    Bij = vol_jacobian * jnp.einsum("b, ab, aij-> ij", angle, precomputed_geometric_data.vol_jacobians_inv[elem_idx], FEMMatrices.derivative_matrix) 
    Cij = jnp.einsum("f, fij-> ij", abs_omega_dot_n_Je[elem_idx], outflow_matrices)    
    Gij_matrix       = Aij + Bij + Cij    
    return Gij_matrix

def QM_i_kernel(
        elem_idx                   : int,
        FEMMatrices                : FEMMatrices,    
        precomputed_geometric_data : PrecomputedGeometricData,
        source                     : Source
    ):
    '''
    Compute the QM_i external source vector for a single element index.    
    '''    
    QM_i_vector = precomputed_geometric_data.vol_jacobians_det[elem_idx] * (FEMMatrices.mass_matrix @ source.source_values[elem_idx])
    return QM_i_vector

def Dij_f_p_fm_kernel(
        elem_idx : int, 
        FEMMatrices                : FEMMatrices,                    
        inflow                     : jnp.ndarray,        
        etof                       : jnp.ndarray,
        face_perm_idx              : jnp.ndarray):
    '''
    Compute the Dij_f_p_fm inflow face matrices for a single element index.
    In other words, compute:
    Dij_f_p_fo = ∫H_i Hᴼ_j dA
    for the specific face (f), permutation (p) and connected element face (fm).
    Uses the precomputed permutated face matrices from the FEMMatrices class.

    Parameters
    ----------
    elem_idx : int
        The element index to compute the Dij_f_p_fm matrices for.
    FEMMatrices :               FEMMatrices
        The finite element method matrices.
    inflow : jnp.ndarray        [n_elements, n_faces_per_element]
        Boolean array indicating inflow faces for all alements (indexed by element and face).
    etof : jnp.ndarray          [n_elements, n_faces_per_element]
        Element-to-face connectivity array.
    face_perm_idx : jnp.ndarray [n_elements, n_faces_per_element]
        Face permutation indices for all elements and faces.
    '''          
    inflow_faces   = inflow[elem_idx]
    face_perms     = jnp.where(inflow_faces, face_perm_idx[elem_idx], -1)    
    face_conns     = jnp.where(inflow_faces, etof[elem_idx],          -1)    
    n_faces_arange = jnp.arange(etof.shape[-1])    
    DijeV          = jnp.where(inflow_faces[:, None, None], FEMMatrices.inflow_matrices[n_faces_arange, face_perms, face_conns, ... ], 0.0)
    return DijeV 



    
Gij_kernel_vmap = jax.jit(jax.vmap(Gij_kernel, in_axes=(0, None, None, None, None, None, None)))


def T_inv_sweep(FEMMatrices : FEMMatrices, geometric_domain : GeometricDomain, topological_sorted_angle : TopologicalSortedAngle, cross_sections : CrossSectionData, precomputed_geometric_data : PrecomputedGeometricData, precomputed_source : Source, sweep_result : jnp.ndarray) -> jnp.ndarray:    

    omega_dot_n = jnp.sum(geometric_domain.face_normals * topological_sorted_angle.angle[None, None, :], axis=-1)
    abs_omega_dot_n_Je = jnp.abs(omega_dot_n) * precomputed_geometric_data.face_jacobians_det
    inflow = omega_dot_n < 0.0

    #Gij = Gij_kernel(28, FEMMatrices, topological_sorted_angle.angle, cross_sections, precomputed_geometric_data, inflow, abs_omega_dot_n_Je)
    #QMi = QM_i_kernel(28, FEMMatrices, precomputed_geometric_data, precomputed_source)
    #QOi  =QO_i_kernel(28, FEMMatrices, precomputed_geometric_data, inflow, abs_omega_dot_n_Je, sweep_result, geometric_domain.etoe, geometric_domain.etof)
    

    Gij    = Gij_kernel(29, FEMMatrices, topological_sorted_angle.angle, cross_sections, precomputed_geometric_data, inflow, abs_omega_dot_n_Je)
    QMi    = QM_i_kernel(29, FEMMatrices, precomputed_geometric_data, precomputed_source)
    Dijev  = Dij_f_p_fm_kernel(29, FEMMatrices, inflow, geometric_domain.etof, geometric_domain.face_perm_idx)
    return Gij, QMi, Dijev

    #print(Gij, QMi)

    # Gijmatrices = Gij_kernel_vmap(
    #     topological_sorted_angle.levels_argsort,
    #     FEMMatrices,
    #     geometric_domain,
    #     topological_sorted_angle,
    #     cross_sections,
    #     precomputed_geometric_data,
    #     precomputed_source,
    #     inflow,
    #     abs_omega_dot_n_Je
    # )
     # incorrect ! check with stell-blanket base 
    #return Gijmatrices
    
        
        

        
        

        
        
        
        