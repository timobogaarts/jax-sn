import jax 
import jax.numpy as jnp 
from jax_sn.domain.geometric_domain import GeometricDomain, TopologicalSortedAngle, CrossSectionDataSingleGroup, Domain
from .base_fem import PrecomputedGeometricData, FEMMatrices
from .solution_domain import FEMMatrices
from jax_sn.source import Source

def remove_zeros(array):
    import numpy as onp
    a = onp.array(array)
    a[onp.abs(a) < 1e-14] = 0.0
    return a

def _compute_omega_dot_n_Je_inflow(geometric_domain : GeometricDomain, angle : jnp.ndarray, precomputed_geometric_data : PrecomputedGeometricData):
    '''
    Compute the absolute value of the dot product of omega and face normals multiplied by face Jacobian determinants
    for all elements and faces. Also computes the inflow boolean array.

    Parameters
    ----------
    geometric_domain : GeometricDomain
        The geometric domain data.
    angle : jnp.ndarray [n_dim]
        The direction vector.
    precomputed_geometric_data : PrecomputedGeometricData
        The precomputed geometric data (Jacobian determinants and inverses).
    Returns
    -------
    abs_omega_dot_n_Je : jnp.ndarray [n_elements, n_faces_per_element]
        The absolute value of the dot product of omega and face normals multiplied by face Jacobian determinants
        for all elements and faces.
    inflow             : jnp.ndarray [n_elements, n_faces_per_element]
        Boolean array indicating inflow faces for all alements (indexed by element and face).
    '''
    omega_dot_n = jnp.sum(geometric_domain.face_normals * angle[None, None, :], axis=-1)
    abs_omega_dot_n_Je = jnp.abs(omega_dot_n) * precomputed_geometric_data.face_jacobians_det
    inflow = omega_dot_n < 0.0
    return abs_omega_dot_n_Je, inflow

@jax.jit
def transport_inverse_sweep_complete_vmap_fori(fem_matrices : FEMMatrices, geometric_domain : GeometricDomain, topological_sorted_angle : TopologicalSortedAngle, cross_sections : CrossSectionDataSingleGroup, precomputed_geometric_data : PrecomputedGeometricData, precomputed_source : Source, sweep_result : jnp.ndarray) -> jnp.ndarray:    

    abs_omega_dot_n_Je, inflow = _compute_omega_dot_n_Je_inflow(geometric_domain, topological_sorted_angle.angle, precomputed_geometric_data)    
    gij_inv_all    = gij_inv_kernel_vmap(
                                topological_sorted_angle.levels_argsort,
                                fem_matrices,
                                topological_sorted_angle.angle,
                                cross_sections,
                                precomputed_geometric_data,
                                inflow,
                                abs_omega_dot_n_Je)    
    qm_i_all       = qm_i_kernel_vmap(
                                topological_sorted_angle.levels_argsort,
                                fem_matrices,
                                precomputed_geometric_data,
                                precomputed_source)
    dij_f_p_fm_all = dij_f_p_fm_kernel_vmap(
                                topological_sorted_angle.levels_argsort,
                                fem_matrices,
                                inflow,
                                geometric_domain.etof,
                                geometric_domain.face_perm_idx)
    
    def collapse_i(i,sweep_result):        
        elem_idx     = topological_sorted_angle.levels_argsort[i]       
        s_dj         = _collapse_dij(elem_idx, dij_f_p_fm_all[i], sweep_result, geometric_domain.etoe, abs_omega_dot_n_Je)
        result       = gij_inv_all[i] @ (qm_i_all[i] + s_dj)
        sweep_result = sweep_result.at[elem_idx].set(result)        
        return sweep_result
    
    return jax.lax.fori_loop(0, geometric_domain.n_elements, collapse_i, sweep_result)

@jax.jit
def transport_inverse_sweep_complete_vmap_fori2(fem_matrices : FEMMatrices, geometric_domain : GeometricDomain, topological_sorted_angle : TopologicalSortedAngle, cross_sections : CrossSectionDataSingleGroup, precomputed_geometric_data : PrecomputedGeometricData, precomputed_source : Source, sweep_result : jnp.ndarray) -> jnp.ndarray:    
    # Just a dummy function to test which part of the sweep takes the most time: 
    # (always is the scan to collapse, so there is something to be gained there...)
    # ATM i am thinking about:
    #  - vmapping sweep for different angles 
    #     (compute gij_inv for a batch per angle, then collapse in parallel)
    # - batching over different levels [requires significant level parallelism, but I guess this is somewhat satisfied?]
    #     batching gij_inv up to the level size (cache compilation to a set number or pad the level array?)
    #     then batching the collapse over the different sizes
    abs_omega_dot_n_Je, inflow = _compute_omega_dot_n_Je_inflow(geometric_domain, topological_sorted_angle.angle, precomputed_geometric_data)    
    gij_inv_all    = gij_inv_kernel_vmap(
                                topological_sorted_angle.levels_argsort,
                                fem_matrices,
                                topological_sorted_angle.angle,
                                cross_sections,
                                precomputed_geometric_data,
                                inflow,
                                abs_omega_dot_n_Je)    
    qm_i_all       = qm_i_kernel_vmap(
                                topological_sorted_angle.levels_argsort,
                                fem_matrices,
                                precomputed_geometric_data,
                                precomputed_source)
    dij_f_p_fm_all = dij_f_p_fm_kernel_vmap(
                                topological_sorted_angle.levels_argsort,
                                fem_matrices,
                                inflow,
                                geometric_domain.etof,
                                geometric_domain.face_perm_idx)
    
    return gij_inv_all, qm_i_all, dij_f_p_fm_all

@jax.jit
def _transport_inverse_sweep_complete_vmap_fori_domain(fem_matrices : FEMMatrices, 
                                                       domain : Domain, 
                                                       toposort_angle         : TopologicalSortedAngle,    
                                                       precomputed_data : PrecomputedGeometricData,                                                                    
                                                       source        : Source,
                                                       sweep_buffer : jnp.ndarray,
                                                       energy_group : int) -> jnp.ndarray:  
    assert sweep_buffer.shape == (domain.n_elements, fem_matrices.mass_matrix.shape[0]),\
        f"Expected sweep_result shape {(domain.n_elements, fem_matrices.mass_matrix.shape[0])}, got {sweep_buffer.shape}"

    mapped_xs = domain.cross_sections_indexed.map_to_full_array(energy_group)
    return transport_inverse_sweep_complete_vmap_fori(
        fem_matrices,
        domain,
        toposort_angle, 
        mapped_xs,
        precomputed_data,
        source, sweep_buffer)

def gij_kernel(
    elem_idx                   : int,
    fem_matrices                : FEMMatrices,    
    angle                      : jnp.ndarray ,
    cross_sections             : CrossSectionDataSingleGroup,
    precomputed_geometric_data : PrecomputedGeometricData,    
    inflow                     : jnp.ndarray,
    abs_omega_dot_n_Je         : jnp.ndarray
    ) -> None:
    '''
    Compute the Gij matrix for a single element index.

    Computes:
    Gij = σ J ∫H_i H_j dV +  J ∑_{a,b} Ω_{b} (J)⁻¹_{ab} ∫ - ∂H_i/∂q_a H_j dV + ∑_{f outflow} |Ω·n_f| J_f ∫H_i H_j dA

    vmap version is gij_kernel_vmap

    The cross sections are assumed to be mapped to all elements. Either use a full array or use the map_to_full_array method of CrossSectionIndexed.

    Parameters
    ----------
    elem_idx : int
        The element index to compute the Gij matrix for.
    FEMMatrices : FEMMatrices
        The finite element method matrices.
    angle : jnp.ndarray
        The direction vector.
    cross_sections : CrossSectionDataSingleGroup
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
    outflow_matrices = jnp.where(inflow[elem_idx][:, None, None], 0.0, fem_matrices.outflow_matrices)        
    Aij = vol_jacobian * cross_sections.total[elem_idx] * fem_matrices.mass_matrix 
    # We restrict the angle to the number of spatial dimensions.
    Bij = vol_jacobian * jnp.einsum("b, ab, aij-> ij", angle[:precomputed_geometric_data.vol_jacobians.shape[-1]], precomputed_geometric_data.vol_jacobians_inv[elem_idx], fem_matrices.derivative_matrix) 
    Cij = jnp.einsum("f, fij-> ij", abs_omega_dot_n_Je[elem_idx], outflow_matrices)    
    Gij_matrix       = Aij + Bij + Cij    
    return Gij_matrix

def gij_inv_kernel(
    elem_idx                   : int,
    fem_matrices                : FEMMatrices,    
    angle                      : jnp.ndarray ,
    cross_sections             : CrossSectionDataSingleGroup,
    precomputed_geometric_data : PrecomputedGeometricData,    
    inflow                     : jnp.ndarray,
    abs_omega_dot_n_Je         : jnp.ndarray
    ) -> None:
    '''
    Compute the inverse of the Gij matrix for a single element index.

    vmap version is gij_inv_kernel_vmap

    Parameters
    ----------
    elem_idx : int
        The element index to compute the Gij matrix for.
    FEMMatrices : FEMMatrices
        The finite element method matrices.
    angle : jnp.ndarray
        The direction vector.
    cross_sections : CrossSectionDataSingleGroup
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
    Gij_matrix_inv : jnp.ndarray
        The computed inverse Gij matrix for the specified element.
    '''
    Gij_matrix = gij_kernel(
        elem_idx,
        fem_matrices,
        angle,
        cross_sections,
        precomputed_geometric_data,
        inflow,
        abs_omega_dot_n_Je)    
    return jnp.linalg.inv(Gij_matrix)

gij_inv_kernel_vmap = jax.jit(jax.vmap(gij_inv_kernel, in_axes=(0, None, None, None, None, None, None)))
gij_kernel_vmap     = jax.jit(jax.vmap(gij_kernel, in_axes=(0, None, None, None, None, None, None)))

def qm_i_kernel(
        elem_idx                   : int,
        fem_matrices                : FEMMatrices,    
        precomputed_geometric_data : PrecomputedGeometricData,
        source                     : Source
    ):
    '''
    Compute the QM_i external source vector for a single element index:
        ∑_{j} J ∫H_i H_j dV q_j

    vmap version is qm_i_kernel_vmap

    Parameters
    ----------
    elem_idx : int
        The element index to compute the QM_i vector for.
    FEMMatrices : FEMMatrices
        The finite element method matrices.
    precomputed_geometric_data : PrecomputedGeometricData
        The precomputed geometric data (Jacobian determinants and inverses).
    source : Source
        The source data. (source_values [n_elements, n_basis])
    Returns:    
    -------
    QM_i_vector : jnp.ndarray [n_basis]
        The computed QM_i vector for the specified element. 
    '''    
    QM_i_vector = precomputed_geometric_data.vol_jacobians_det[elem_idx] * (fem_matrices.mass_matrix @ source.source_values[elem_idx])
    return QM_i_vector

qm_i_kernel_vmap = jax.jit(jax.vmap(qm_i_kernel, in_axes=(0, None, None, None)))

def dij_f_p_fm_kernel(
        elem_idx                   : int, 
        fem_matrices               : FEMMatrices,                    
        inflow                     : jnp.ndarray,        
        etof                       : jnp.ndarray,
        face_perm_idx              : jnp.ndarray):
    '''
    Compute the Dij_f_p_fm inflow face matrices for a single element index.
    In other words, compute:
    Dij_f_p_fo = ∫H_i Hᴼ_j dA
    for the specific face (f), permutation (p) and connected element face (fm).
    Uses the precomputed permutated face matrices from the FEMMatrices class.

    vmap version is dij_f_p_fm_kernel_vmap

    Parameters
    ----------
    elem_idx : int
        The element index to compute the Dij_f_p_fo matrices for.
    FEMMatrices :               FEMMatrices
        The finite element method matrices.
    inflow : jnp.ndarray        [n_elements, n_faces_per_element]
        Boolean array indicating inflow faces for all alements (indexed by element and face).
    etof : jnp.ndarray          [n_elements, n_faces_per_element]
        Element-to-face connectivity array.
    face_perm_idx : jnp.ndarray [n_elements, n_faces_per_element]
        Face permutation indices for all elements and faces.
    Returns
    -------
    dij_f_p_fo : jnp.ndarray
        The computed Dij_f_p_fo matrices for the specified element.
    '''          
    inflow_faces   = inflow[elem_idx]
    face_perms     = jnp.where(inflow_faces, face_perm_idx[elem_idx], -1)        
    face_conns     = jnp.where(inflow_faces, etof[elem_idx],          -1)        
    n_faces_arange = jnp.arange(etof.shape[-1])    
    dij_f_p_fo     = jnp.where((inflow_faces &  (face_perms >= 0))[:, None, None], fem_matrices.inflow_matrices[n_faces_arange, face_perms, face_conns, ... ], 0.0)
    return dij_f_p_fo 

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
    return jnp.einsum("f, fij, fj->i",abs_omega_dot_n_Je[elem_idx], dij_f_p_fm, values) 
    
    
    
dij_f_p_fm_kernel_vmap = jax.jit(jax.vmap(dij_f_p_fm_kernel, in_axes=(0, None, None, None, None)))

