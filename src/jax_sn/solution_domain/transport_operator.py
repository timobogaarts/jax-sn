import jax 
import jax.numpy as jnp 
from jax_sn.domain.geometric_domain import GeometricDomain, TopologicalSortedAngle, CrossSectionDataSingleGroup, Domain
from .base_fem import PrecomputedGeometricData, FEMMatrices
from .solution_domain import FEMMatrices
from jax_sn.source import Source
from functools import partial
from typing import Tuple

import numpy as onp
from dataclasses import dataclass

#==============================================================================================================================================================================
#  Full Transport operator functions
#==============================================================================================================================================================================

# This file contains functions that perform the full transport operator sweep over a domain for a single angle.
# In principle, all these functions only use simple matrix algebra. They could be implemented in custom kernels for performance.
# All setup is done in precomputation functions in base_fem.py and elsewhere.

@partial(jax.jit, static_argnums = (7,))
def transport_inverse_sweep_batched(geometric_domain           : GeometricDomain, 
                                    fem_matrices               : FEMMatrices, 
                                    precomputed_geometric_data : PrecomputedGeometricData,                                      
                                    cross_sections             : CrossSectionDataSingleGroup, 
                                    topological_sorted_angle   : TopologicalSortedAngle,                                     
                                    precomputed_source         : Source, 
                                    sweep_result               : jnp.ndarray, 
                                    batch_size                 : int):
    
    '''
    Perform a complete transport inverse sweep over all elements in the geometric domain using batched vmap and fori_loop.

    vmap is used to compute all Gij_inv, QM_i and Dij_f_p_fm matrices for all elements in a vectorized manner for a fixed batch size.
    Then, a fori_loop is used to iterate over the elements in topological order to compute the sweep result in batches.
    This takes significantly less memory than using the complete vmap version.

    Parameters
    ----------
    geometric_domain : GeometricDomain
        The geometric domain data.
    fem_matrices : FEMMatrices
        The finite element method matrices.
    precomputed_geometric_data : PrecomputedGeometricData
        The precomputed geometric data (Jacobian determinants and inverses).
    cross_sections : CrossSectionDataSingleGroup
        The cross section data.
    topological_sorted_angle : TopologicalSortedAngle
        The topologically sorted angle data.
    precomputed_source : Source
        The source data.
    sweep_result : jnp.ndarray [n_elements, n_basis]
        The initial sweep result array (usually zeros).
    Returns
    -------
    sweep_result : jnp.ndarray [n_elements, n_basis]    
        The computed sweep result array after performing the transport sweep.
    '''
    
    _assert_shape_base(geometric_domain, fem_matrices, precomputed_geometric_data, cross_sections, topological_sorted_angle, precomputed_source, sweep_result)
    
    n_batches                  = jnp.ceil(geometric_domain.n_elements / batch_size).astype(int)        

    def single_batch(batch_i, sweep_result):
        batch_indices                               = jnp.clip(jnp.arange(batch_size) + batch_i * batch_size, 0, geometric_domain.n_elements - 1)    
        batch_elements                              = topological_sorted_angle.levels_argsort[batch_indices]        
        gij_inv_batch, qm_i_batch, dij_f_p_fm_batch, _, abs_omega_dot_n_Je = _setup_elements_idx_in_precomp_with_omega_n_calculation(batch_elements, geometric_domain, fem_matrices,
                                                            precomputed_geometric_data, cross_sections, topological_sorted_angle.angle,
                                                            precomputed_source)        
        return _sweep_elements(batch_elements, sweep_result, gij_inv_batch, qm_i_batch, dij_f_p_fm_batch, geometric_domain, abs_omega_dot_n_Je)    
    return jax.lax.fori_loop(0, n_batches, single_batch, sweep_result)

@jax.jit
def transport_inverse_sweep_complete_vmap_fori(geometric_domain           : GeometricDomain, 
                                               fem_matrices               : FEMMatrices, 
                                               precomputed_geometric_data : PrecomputedGeometricData, 
                                               cross_sections             : CrossSectionDataSingleGroup,                                                
                                               topological_sorted_angle   : TopologicalSortedAngle, 
                                               precomputed_source         : Source, 
                                               sweep_result               : jnp.ndarray) -> jnp.ndarray:    
    '''
    Perform a complete transport inverse sweep over all elements in the geometric domain using vmap and fori_loop.

    vmap is used to compute all Gij_inv, QM_i and Dij_f_p_fm matrices for all elements in a vectorized manner.
    Then, a fori_loop is used to iterate over the elements in topological order to compute the sweep result.
    This takes significant memory due to the large intermediate matrices. 

    Parameters
    ----------
    geometric_domain : GeometricDomain
        The geometric domain data.
    fem_matrices : FEMMatrices
        The finite element method matrices.
    precomputed_geometric_data : PrecomputedGeometricData
        The precomputed geometric data (Jacobian determinants and inverses).
    cross_sections : CrossSectionDataSingleGroup
        The cross section data.
    topological_sorted_angle : TopologicalSortedAngle
        The topologically sorted angle data.
    precomputed_source : Source
        The source data.
    sweep_result : jnp.ndarray [n_elements, n_basis]
        The initial sweep result array (usually zeros).
    Returns
    -------
    sweep_result : jnp.ndarray [n_elements, n_basis]    
        The computed sweep result array after performing the transport sweep.
    '''    
    _assert_shape_base(geometric_domain, fem_matrices, precomputed_geometric_data, cross_sections, topological_sorted_angle, precomputed_source, sweep_result)

    gij_inv_all, qm_i_all, dij_f_p_fm_all, _, abs_omega_dot_n_Je = _setup_elements_idx_in_precomp_with_omega_n_calculation(topological_sorted_angle.levels_argsort, geometric_domain, fem_matrices,
                                                            precomputed_geometric_data, cross_sections, topological_sorted_angle.angle,
                                                            precomputed_source)    
                                                            
    return _sweep_elements(topological_sorted_angle.levels_argsort, sweep_result, gij_inv_all, qm_i_all, dij_f_p_fm_all, geometric_domain, abs_omega_dot_n_Je)


#=============================================================================================================================================================================
# Internal per-element kernels and vmap versions
#=============================================================================================================================================================================
def gij_kernel_idx_in_precomp(
               elem_idx                 : int,
               fem_matrices               : FEMMatrices,
               precomputed_geometric_data : PrecomputedGeometricData,
               cross_sections             : CrossSectionDataSingleGroup,
               angle                      : jnp.ndarray,                
               inflow                     : jnp.ndarray,
               abs_omega_dot_n_Je         : jnp.ndarray):
    '''
    Internal function to compute the Gij matrix for a single element index:
        Gij = σ_t Jv ∫H_i H_j dV  + Jv ∫H_i (Ω·∇H_j) dV + ∑_{f} |Ω·n_f| Je_f ∫H_i H_j dA

    Parameters
    ----------
    elem_idx : int
        The element index to compute the Gij matrix for.
    fem_matrices : FEMMatrices
        The finite element method matrices.
    precomputed_geometric_data : PrecomputedGeometricData
        The precomputed geometric data (Jacobian determinants and inverses).
        Since these are already precomputed, we do not pass a matrix that has ordered data for this specific direction. This would be significantly more memory intensive.
    angle : jnp.ndarray [n_dim]
        The direction vector.
    inflow : jnp.ndarray [n_faces_per_element]
        Boolean array indicating inflow faces for the specified element. This has the advantage of allowing us to batch it by providing it for each element in the batch only (instead of all elements).
        Since we have to compute it separately for each direction anyway, this does not involve additional copying (as it would be for the precomputed jacobian data).
        
    abs_omega_dot_n_Je : jnp.ndarray [n_faces_per_element]
        The absolute value of the dot product of omega and face normals multiplied by face Jacobian determinants
        for the specified element.
        Similarly to inflow, we pass only for this specific element here.
    cross_sections : CrossSectionDataSingleGroup
        The cross section data. total attribute must have [n_elements] shape (indexed in). This is used to get σ_t for the specific element. This is not precomputed per direction to save memory.
    Returns    
    '''
    
    vol_jacobian_det = precomputed_geometric_data.vol_jacobians_det[elem_idx]
    vol_jacobian_inv = precomputed_geometric_data.vol_jacobians_inv[elem_idx]
    outflow_matrices = jnp.where(inflow[:, None, None], 0.0, fem_matrices.outflow_matrices)        
    Aij              = vol_jacobian_det * cross_sections.total[elem_idx] * fem_matrices.mass_matrix 
    # We restrict the angle to the number of spatial dimensions.
    Bij              = vol_jacobian_det  * jnp.einsum("b, ab, aij-> ij", angle[:vol_jacobian_inv.shape[-1]], vol_jacobian_inv, fem_matrices.derivative_matrix) 
    Cij              = jnp.einsum("f, fij-> ij", abs_omega_dot_n_Je, outflow_matrices)    
    Gij_matrix       = Aij + Bij + Cij    
    return Gij_matrix

def gij_inv_kernel_idx_in_precomp( 
            elem_idx                 : int,
            fem_matrices               : FEMMatrices,
            precomputed_geometric_data : PrecomputedGeometricData,
            cross_sections             : CrossSectionDataSingleGroup,
            angle                      : jnp.ndarray,                
            inflow                     : jnp.ndarray,
            abs_omega_dot_n_Je         : jnp.ndarray):
    '''
    Compute the inverse of the Gij matrix for a single element index.
        Gij = σ_t Jv ∫H_i H_j dV  + Jv ∫H_i (Ω·∇H_j) dV + ∑_{f} |Ω·n_f| Je_f ∫H_i H_j dA    

    See also gij_kernel_idx_in_precomp for explanation of the parameters.
    Returns
    -------
    gij_inv : jnp.ndarray [n_basis, n_basis]
        The computed inverse Gij matrix for the specified element.
    '''
    Gij_matrix = gij_kernel_idx_in_precomp(
          elem_idx,
          fem_matrices,
          precomputed_geometric_data,
          cross_sections,
          angle,          
          inflow,
          abs_omega_dot_n_Je)    
    return jnp.linalg.inv(Gij_matrix)

gij_inv_kernel_idx_in_precomp_vmap = jax.jit(jax.vmap(gij_inv_kernel_idx_in_precomp, in_axes=(0, None, None, None, None, 0, 0)))

def qm_i_kernel_idx_in_precomp(
        elem_idx                    : int,
        fem_matrices                : FEMMatrices,    
        precomputed_geometric_data  : PrecomputedGeometricData,
        source                      : Source
    ):
    '''
    Compute the QM_i external source vector for a single element index:
        QM_i = ∑_{j} Jv ∫H_i H_j dV q_j 

    We index directly in the source instead of passing a subsource. This can save significant memory when vmapping over all elements and all directions with the same source independent of direction.

    Parameters
    ----------
    elem_idx : int
        The element index to compute the QM_i vector for.
    fem_matrices : FEMMatrices
        The finite element method matrices.
    precomputed_geometric_data : PrecomputedGeometricData
        The precomputed geometric data (Jacobian determinants and inverses).
    source : Source
        The source data.
    Returns
    -------
    qm_i : jnp.ndarray
        The computed QM_i vector for the specified element.
    '''
    qm_i = precomputed_geometric_data.vol_jacobians_det[elem_idx] * fem_matrices.mass_matrix @ source.source_values[elem_idx]
    return qm_i

qm_i_kernel_idx_in_precomp_vmap = jax.jit(jax.vmap(qm_i_kernel_idx_in_precomp, in_axes=(0, None, None, None)))


def dij_f_p_fm_kernel_idx_in_precomp(
        elem_idx                    : int,
        fem_matrices                : FEMMatrices,    
        inflow                      : jnp.ndarray,
        etof                        : jnp.ndarray,
        face_perm_idx               : jnp.ndarray
    ):
    '''
    Compute the Dij_f_p_fm matrices for a single element index:
        Dij_f_p_fm = |Ω·n_f| J_f ∫ H_i H_j dA

    See gij_kernel_idx_in_precomp for explanation on why we pass inflow as a single array instead of an indexed array.

    Parameters
    ----------
    elem_idx : int
        The element index to compute the Dij_f_p_fm matrices for.
    fem_matrices : FEMMatrices
        The finite element method matrices.
    inflow : jnp.ndarray [n_faces_per_element]
        Boolean array indicating inflow faces for the specified element.
    etof : jnp.ndarray [n_elements, n_faces_per_element]
        The element-to-face connectivity array.
    face_perm_idx : jnp.ndarray [n_elements, n_faces_per_element]
        The face permutation index array.
    Returns
    -------
    dij_f_p_fm : jnp.ndarray
        The computed Dij_f_p_fm matrices for the specified element.
    '''    
    face_perms     = jnp.where(inflow, face_perm_idx[elem_idx] ,          -1)        
    face_conns     = jnp.where(inflow, etof[elem_idx]          ,          -1)        
    n_faces_arange = jnp.arange(etof.shape[-1])    
    dij_f_p_fo     = jnp.where((inflow &  (face_perms >= 0))[:, None, None], fem_matrices.inflow_matrices[n_faces_arange, face_perms, face_conns, ... ], 0.0)
    return dij_f_p_fo 

dij_f_p_fm_kernel_idx_in_precomp_vmap = jax.jit(jax.vmap(dij_f_p_fm_kernel_idx_in_precomp, in_axes=(0, None, 0, None, None)))

def _setup_elements_idx_in_precomp(element_indices : jnp.ndarray, geometric_domain : GeometricDomain, fem_matrices : FEMMatrices, precomputed_geometric_data : PrecomputedGeometricData, cross_sections : CrossSectionDataSingleGroup, angle : jnp.ndarray, precomputed_source : Source, inflow : jnp.ndarray, abs_omega_dot_n_Je : jnp.ndarray):
    '''
    Setup all Gij_inv, QM_i and Dij_f_p_fm matrices for a set of element indices. This is embarassingly parallel and uses vmap.

    Not necessarily *all* elements at the same time. The element indices can be an arbitrary subset of elements.

    Parameters
    ----------
    element_indices : jnp.ndarray [n_elements_setup]
        The element indices to setup the matrices for.
    geometric_domain : GeometricDomain
        The geometric domain data.
    fem_matrices : FEMMatrices
        The finite element method matrices.
    precomputed_geometric_data : PrecomputedGeometricData
        The precomputed geometric data (Jacobian determinants and inverses).
    cross_sections : CrossSectionDataSingleGroup 
        The cross section data. total attribute must have [n_elements] shape (indexed in)
    angle : jnp.ndarray [n_dim]
        The direction vector.
    precomputed_source : Source [n_elements, n_basis]
        The source data.
    inflow : jnp.ndarray [n_elements_setup, n_faces_per_element]
        Boolean array indicating inflow faces for the specified elements (indexed by in the same order, i.e. inflow[0] corresponds to element_indices[0], etc).
    abs_omega_dot_n_Je : jnp.ndarray [n_elements_setup, n_faces_per_element]
        The absolute value of the dot product of omega and face normals multiplied by face Jacobian determinants
        for the specified elements (indexed by in the same order, i.e. abs_omega_dot_n_Je[0] corresponds to element_indices[0], etc).
    Returns
    -------
    gij_inv_all    : jnp.ndarray [n_elements_setup, n_basis, n_basis]
        The computed Gij_inv matrices for all specified elements.
    qm_i_all       : jnp.ndarray [n_elements_setup, n_basis]
        The computed QM_i vectors for all specified elements.
    dij_f_p_fm_all : jnp.ndarray [n_elements_setup, n_faces_per_element, n_basis, n_basis]
        The computed Dij_f_p_fm matrices for all specified elements.
    '''
    gij_inv_all    = gij_inv_kernel_idx_in_precomp_vmap(
                                element_indices,
                                fem_matrices,
                                precomputed_geometric_data,
                                cross_sections,
                                angle,
                                inflow,
                                abs_omega_dot_n_Je)     #[n_elements, n_basis, n_basis]
    qm_i_all       = qm_i_kernel_idx_in_precomp_vmap(
                                element_indices,
                                fem_matrices,
                                precomputed_geometric_data,
                                precomputed_source)     #[n_elements, n_basis]
    dij_f_p_fm_all = dij_f_p_fm_kernel_idx_in_precomp_vmap(
                                element_indices,
                                fem_matrices,
                                inflow,
                                geometric_domain.etof,
                                geometric_domain.face_perm_idx) #[n_elements, n_faces_per_element, n_basis, n_basis]
    return gij_inv_all, qm_i_all, dij_f_p_fm_all


def _setup_elements_idx_in_precomp_with_omega_n_calculation(element_indices : jnp.ndarray, geometric_domain : GeometricDomain, fem_matrices : FEMMatrices, precomputed_geometric_data : PrecomputedGeometricData, cross_sections : CrossSectionDataSingleGroup, angle : jnp.ndarray, precomputed_source : Source):
    '''
    Setup all Gij_inv, QM_i and Dij_f_p_fm matrices for a set of element indices. This is embarassingly parallel and uses vmap.

    Not necessarily *all* elements at the same time. The element indices can be an arbitrary subset of elements.

    This function also computes the |Ω·n_f| J_f and inflow arrays for convenience.

    Parameters
    ----------
    element_indices : jnp.ndarray [n_elements_setup]
        The element indices to setup the matrices for.
    geometric_domain : GeometricDomain
        The geometric domain data.
    fem_matrices : FEMMatrices
        The finite element method matrices.
    precomputed_geometric_data : PrecomputedGeometricData
        The precomputed geometric data (Jacobian determinants and inverses).
    cross_sections : CrossSectionDataSingleGroup 
        The cross section data. total attribute must have [n_elements] shape (indexed in)
    angle : jnp.ndarray [n_dim]
        The direction vector.
    precomputed_source : Source [n_elements, n_basis]
        The source data.
    Returns
    -------
    gij_inv_all    : jnp.ndarray [n_elements_setup, n_basis, n_basis]
        The computed Gij_inv matrices for all specified elements.
    qm_i_all       : jnp.ndarray [n_elements_setup, n_basis]
        The computed QM_i vectors for all specified elements.
    dij_f_p_fm_all : jnp.ndarray [n_elements_setup, n_faces_per_element, n_basis, n_basis]
        The computed Dij_f_p_fm matrices for all specified elements.
    '''
    abs_omega_dot_n_Je, inflow = _compute_omega_dot_n_Je_inflow_vmap(element_indices, geometric_domain, angle, precomputed_geometric_data)        
    return *_setup_elements_idx_in_precomp(element_indices, geometric_domain, fem_matrices, precomputed_geometric_data, cross_sections, angle, precomputed_source, inflow, abs_omega_dot_n_Je), inflow, abs_omega_dot_n_Je
@jax.jit
def _collapse_dij(elem_idx : int, dij_f_p_fm : jnp.ndarray,  sweep_result : jnp.ndarray, etoe : jnp.ndarray, abs_omega_dot_n_Je : jnp.ndarray):  
    '''
    Compute the face source from neighbouring elements by collapsing the Dij_f_p_fm matrices with the sweep result.
        s_i = ∑_{f,p, fm} |Ω·n_f| J_f Dij_f_p_fm ψ_fm = ∑_{f} |Ω·n_f| J_f ∫H_i Hᴼ_j dA ψ_j

    Parameters
    ----------
    elem_idx : int
        The element index to compute the collapsed Dij for.
    dij_f_p_fm : jnp.ndarray
        The Dij_f_p_fm matrices for the specified element.
    sweep_result : jnp.ndarray
        The current sweep result array.
    etoe : jnp.ndarray
        The element-to-element connectivity array.
    abs_omega_dot_n_Je : jnp.ndarray
        The absolute value of the dot product of omega and face normals multiplied by face Jacobian determinants
        for all elements and faces.
    Returns
    -------
    collapsed_dij : jnp.ndarray
        The collapsed Dij vector for the specified element.
    '''  
    values = sweep_result[etoe[elem_idx], :]        
    return jnp.einsum("f, fij, fj->i",abs_omega_dot_n_Je, dij_f_p_fm, values) 

_collapse_dij_vmap = jax.vmap(_collapse_dij, in_axes=(0,0,None, None, 0))
    

def _sweep_elements(elements_topological_sorting, sweep_result, gij_inv_all, qm_i_all, dij_f_p_fm_all, geometric_domain, abs_omega_dot_n_Je):
    '''
    Internal function that performs the sweep over elements in topological order using fori_loop. This function assumes that all necessary matrices have already been computed.
    It can perform part of the sweep for a subset of elements.    
    It computes:  ψ_i     = Gij_inv ( QM_i + ∑_{f < 0 } |Ω·n_f| J_f Dij_f_p_fm ψ_fm )
                          = Gij_inv ( QM_i + s_dj )
                          = Gij_inv ( QM_i + ∑_{f < 0,j} |Ω·n_f| J_f ∫H_i Hᴼ_j dA ψ_j )
    Since this requires ψ_j from neighbouring elements, the elements must be processed in topological order.

    Parameters
    ----------
    elements_topological_sorting : jnp.ndarray [n_elements_subset]
        The element indices in topological order to process.
    sweep_result : jnp.ndarray [n_elements, n_basis]
        The initial sweep result array (usually zeros).
    gij_inv_all : jnp.ndarray [n_elements_subset, n_basis, n_basis]
        The computed Gij_inv matrices for all specified elements.
    qm_i_all : jnp.ndarray [n_elements_subset, n_basis]
        The computed QM_i vectors for all specified elements.
    dij_f_p_fm_all : jnp.ndarray [n_elements_subset, n_faces_per_element, n_basis, n_basis]
        The computed Dij_f_p_fm matrices for all specified elements.
    geometric_domain : GeometricDomain
        The geometric domain data.
    abs_omega_dot_n_Je : jnp.ndarray [n_elements_subset, n_faces_per_element]
        The absolute value of the dot product of omega and face normals multiplied by face Jacobian determinants
        for the specified elements (indexed by in the same order, i.e. abs_omega_dot_n_Je[0] corresponds to element_indices[0], etc).
    Returns
    -------
    sweep_result : jnp.ndarray [n_elements, n_basis]    
        The computed sweep result array after performing the transport sweep for the specified elements.
    '''
    assert elements_topological_sorting.shape[0] == gij_inv_all.shape[0] == qm_i_all.shape[0] == dij_f_p_fm_all.shape[0] == abs_omega_dot_n_Je.shape[0],\
        "Input arrays must have the same number of elements but got: elements_topological_sorting {}, gij_inv_all {}, qm_i_all {}, dij_f_p_fm_all {}, abs_omega_dot_n_Je {}".format(\
            elements_topological_sorting.shape[0], gij_inv_all.shape[0], qm_i_all.shape[0], dij_f_p_fm_all.shape[0], abs_omega_dot_n_Je.shape[0])
    
    def sweep_elements_in_order(elem_i_in_order,sweep_result):  
        elem_idx = elements_topological_sorting[elem_i_in_order]
        s_dj     = _collapse_dij(elem_idx, dij_f_p_fm_all[elem_i_in_order], sweep_result, geometric_domain.etoe, abs_omega_dot_n_Je[elem_i_in_order])
        result   =  gij_inv_all[elem_i_in_order] @ (qm_i_all[elem_i_in_order] + s_dj)                
        return sweep_result.at[elem_idx].set(result)    
    
    return jax.lax.fori_loop(0, elements_topological_sorting.shape[0], sweep_elements_in_order, sweep_result)


def _compute_omega_dot_n_Je_inflow(elem_idx : int, geometric_domain : GeometricDomain, angle : jnp.ndarray, precomputed_geometric_data : PrecomputedGeometricData):
    omega_dot_n        = jnp.sum(geometric_domain.face_normals[elem_idx, :, :] * angle[None, :], axis=-1)
    abs_omega_dot_n_Je = jnp.abs(omega_dot_n) * precomputed_geometric_data.face_jacobians_det[elem_idx, :]
    inflow             = omega_dot_n < 0.0
    return abs_omega_dot_n_Je, inflow

_compute_omega_dot_n_Je_inflow_vmap = jax.jit(jax.vmap(_compute_omega_dot_n_Je_inflow, in_axes=(0, None, None, None)))

def _assert_shape_base( geometric_domain           : GeometricDomain, 
                        fem_matrices               : FEMMatrices, 
                        precomputed_geometric_data : PrecomputedGeometricData, 
                        cross_sections             : CrossSectionDataSingleGroup,                                                
                        topological_sorted_angle   : TopologicalSortedAngle, 
                        precomputed_source         : Source, 
                        sweep_result               : jnp.ndarray):    
    '''
    Function checking the types and shapes of the inputs to the transport sweep functions.
    Raises AssertionError if any of the checks fail.
    '''
    assert isinstance(geometric_domain, GeometricDomain), f"geometric_domain is not of type GeometricDomain but {type(geometric_domain)}"
    assert isinstance(fem_matrices, FEMMatrices), f"fem_matrices is not of type FEMMatrices but {type(fem_matrices)}"
    assert isinstance(precomputed_geometric_data, PrecomputedGeometricData), f"precomputed_geometric_data is not of type PrecomputedGeometricData but {type(precomputed_geometric_data)}"
    assert isinstance(cross_sections, CrossSectionDataSingleGroup), f"cross_sections is not of type CrossSectionDataSingleGroup but {type(cross_sections)}"
    assert isinstance(topological_sorted_angle, TopologicalSortedAngle), f"topological_sorted_angle is not of type TopologicalSortedAngle but {type(topological_sorted_angle)}"
    assert isinstance(precomputed_source, Source), f"precomputed_source is not of type Source but {type(precomputed_source)}"
    
    assert sweep_result.shape == (geometric_domain.n_elements, fem_matrices.mass_matrix.shape[0]), f"sweep_result shape {sweep_result.shape} does not match expected (n_elements, n_basis): {(geometric_domain.n_elements, fem_matrices.mass_matrix.shape[0])}"
    assert cross_sections.total.shape[0] == geometric_domain.n_elements, f"cross_sections shape {cross_sections.total.shape} does not match expected (n_elements, ): {(geometric_domain.n_elements, )}"
    assert precomputed_source.source_values.shape == (geometric_domain.n_elements, fem_matrices.mass_matrix.shape[0]), f"precomputed_source shape {precomputed_source.source_values.shape} does not match expected (n_elements, n_basis): {(geometric_domain.n_elements, fem_matrices.mass_matrix.shape[0])}"


#=============================================================================================================================================================================
# Multiple angle functions
#=============================================================================================================================================================================

def _assert_shape_multiple_angles( geometric_domain : GeometricDomain, 
                                  fem_matrices : FEMMatrices, 
                                  precomputed_geometric_data : PrecomputedGeometricData, 
                                  cross_sections : CrossSectionDataSingleGroup,                                 
                                  topological_sorted_angles : TopologicalSortedAngle, 
                                  precomputed_source : Source, 
                                  sweep_result : jnp.ndarray):  
    assert isinstance(geometric_domain, GeometricDomain), f"geometric_domain is not of type GeometricDomain but {type(geometric_domain)}"
    assert isinstance(fem_matrices, FEMMatrices), f"fem_matrices is not of type FEMMatrices but {type(fem_matrices)}"
    assert isinstance(precomputed_geometric_data, PrecomputedGeometricData), f"precomputed_geometric_data is not of type PrecomputedGeometricData but {type(precomputed_geometric_data)}"
    assert isinstance(cross_sections, CrossSectionDataSingleGroup), f"cross_sections is not of type CrossSectionDataSingleGroup but {type(cross_sections)}"
    assert isinstance(topological_sorted_angles, TopologicalSortedAngle), f"topological_sorted_angles is not of type TopologicalSortedAngle but {type(topological_sorted_angles)}"
    assert isinstance(precomputed_source, Source), f"precomputed_source is not of type Source but {type(precomputed_source)}"   

    assert topological_sorted_angles.angle.ndim == 2, f"topological_sorted_angles.angle should have 2 dimensions [n_angles, n_dim] but has shape {topological_sorted_angles.angle.shape}"
    assert precomputed_source.source_values.ndim == 3, f"precomputed_source.source_values should have 3 dimensions [n_angles, n_elements, n_basis] but has shape {precomputed_source.source_values.shape}"
    assert sweep_result.ndim == 3, f"sweep_result should have 3 dimensions [n_angles, n_elements, n_basis] but has shape {sweep_result.shape}"  
    n_angles = topological_sorted_angles.angle.shape[0]
    assert topological_sorted_angles.levels_argsort.shape[0] == n_angles, f"topological sorted_angles.levels_argsort first dimension {topological_sorted_angles.levels_argsort.shape[0]} does not match number of angles {n_angles}"
    assert precomputed_source.source_values.shape[0] == n_angles, f"precomputed_source.source_values first dimension {precomputed_source.source_values.shape[0]} does not match number of angles {n_angles}"
    assert sweep_result.shape[0] == n_angles, f"sweep_result first dimension {sweep_result.shape[0]} does not match number of angles {n_angles}"
    assert sweep_result.shape[1] == geometric_domain.n_elements, f"sweep_result second dimension {sweep_result.shape[1]} does not match number of elements {geometric_domain.n_elements}"
    assert sweep_result.shape[2] == fem_matrices.mass_matrix.shape[0], f"sweep_result third dimension {sweep_result.shape[2]} does not match number of basis functions {fem_matrices.mass_matrix.shape[0]}" 
    
                                

@partial(jax.jit, static_argnums = (7,))
def transport_inverse_sweep_multiple_angles_batched(geometric_domain           : GeometricDomain,
                                                    fem_matrices               : FEMMatrices,
                                                    precomputed_geometric_data : PrecomputedGeometricData,
                                                    cross_sections             : CrossSectionDataSingleGroup,
                                                    topological_sorted_angles  : TopologicalSortedAngle,                                                                                                         
                                                    precomputed_source         : Source,
                                                    sweep_result               : jnp.ndarray,
                                                    batch_size                 : int):
    '''
    Perform a complete transport inverse sweep over all elements in the geometric domain for multiple angles using batched vmap and fori_loop.

    vmap is used to compute all Gij_inv, QM_i and Dij_f_p_fm matrices for all elements in a vectorized manner for a fixed batch size and multiple angles.
    Then, a fori_loop is used to iterate over the elements in topological order to compute the sweep result in batches.
    This takes significantly less memory than using the complete vmap version.

    Parameters
    ----------
    geometric_domain : GeometricDomain
        The geometric domain data.
    fem_matrices : FEMMatrices
        The finite element method matrices.
    precomputed_geometric_data : PrecomputedGeometricData
        The precomputed geometric data (Jacobian determinants and inverses).
    cross_sections : CrossSectionDataSingleGroup
        The cross section data.
    topological_sorted_angles : TopologicalSortedAngle
        The topologically sorted angle data for multiple angles. (batch dimension is the first dimension)
    precomputed_source : Source [n_angles, n_elements, n_basis]
        The source data. Should also be batched over angles (batch dimension is the first dimension).
    sweep_result : jnp.ndarray [n_angles, n_elements, n_basis]
        The initial sweep result array (usually zeros).
    '''
        
    _assert_shape_multiple_angles(geometric_domain, fem_matrices, precomputed_geometric_data, cross_sections, topological_sorted_angles, precomputed_source, sweep_result)
    f_batch = jax.vmap(transport_inverse_sweep_batched, in_axes=(None, None, None, None, 0, 0, 0, None ))
    return f_batch(geometric_domain, fem_matrices, precomputed_geometric_data, cross_sections, topological_sorted_angles, precomputed_source, sweep_result, batch_size)

@partial(jax.jit, static_argnums = (7,))
def transport_inverse_sweep_multiple_angles_batched_single_f( geometric_domain           : GeometricDomain,
                                                              fem_matrices               : FEMMatrices,
                                                              precomputed_geometric_data : PrecomputedGeometricData,
                                                              cross_sections             : CrossSectionDataSingleGroup,
                                                              topological_sorted_angles  : TopologicalSortedAngle,                                                                                                         
                                                              precomputed_source         : Source,
                                                              sweep_result               : jnp.ndarray,
                                                              batch_size                 : int):  
    
    _assert_shape_multiple_angles(geometric_domain, fem_matrices, precomputed_geometric_data, cross_sections, topological_sorted_angles, precomputed_source, sweep_result)

    n_batches                  = jnp.ceil(geometric_domain.n_elements / batch_size).astype(int)        
    multi_angle_batch_setup    = jax.vmap(_setup_elements_idx_in_precomp_with_omega_n_calculation, in_axes=(0, None, None, None, None, 0, 0 ))
    sweep_result_multi_angle   = jax.vmap(_sweep_elements, in_axes= (0, 0, 0,0,0, None, 0))

    def single_batch(batch_i, sweep_result):
        batch_indices = jnp.clip(jnp.arange(batch_size) + batch_i * batch_size, 0, geometric_domain.n_elements - 1)    
        elements_batch = topological_sorted_angles.levels_argsort[:, batch_indices]
        gij_inv_batch, qm_i_batch, dij_f_p_fm_batch, _, abs_omega_dot_n_Je = multi_angle_batch_setup(elements_batch, geometric_domain, fem_matrices, precomputed_geometric_data, cross_sections, topological_sorted_angles.angle, precomputed_source)
        return sweep_result_multi_angle(elements_batch, sweep_result, gij_inv_batch, qm_i_batch, dij_f_p_fm_batch, geometric_domain, abs_omega_dot_n_Je)
    return jax.lax.fori_loop(0, n_batches, single_batch, sweep_result)


      
    

