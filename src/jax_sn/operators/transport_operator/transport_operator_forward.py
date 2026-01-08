import jax
import jax.numpy as jnp 
from jax_sn.domain.geometric_domain import GeometricDomain, CrossSectionDataSingleGroup, Domain
from jax_sn.solution_domain import PrecomputedGeometricData, FEMMatrices
from jax_sn.source import Source
from .transport_operator_imp import gij_kernel_idx_in_precomp, dij_f_p_fm_kernel_idx_in_precomp, _compute_omega_dot_n_Je_inflow_vmap, _compute_omega_dot_n_Je_inflow
from functools import partial




def _forward_transport_operator_kernel(
                                       elem_idx                   : int,
                                       geometric_domain           : GeometricDomain, 
                                       fem_matrices               : FEMMatrices, 
                                       precomputed_geometric_data : PrecomputedGeometricData,                                      
                                       cross_sections             : CrossSectionDataSingleGroup, 
                                       angle                      : jnp.ndarray,
                                       inflow                     : jnp.ndarray,
                                       abs_omega_dot_n_Je         : jnp.ndarray,
                                       x                          : jnp.ndarray, 
                                    ):
    '''
    Forward transport operator kernel for a single angle and single element.

    Computes (note that Dij is a source term in the inverse; making it a sink term in the forward):

    Gij @ x[elem_idx, :] - Dfij @ x[f_conn, :]

    Parameters
    ----------
    elem_idx                   : int
        Element index to compute the Tx operator for.
    geometric_domain           : GeometricDomain [for all elements]
        Geometric domain (containing element to element connectivity and element to face connectivity).
    fem_matrices               : FEMMatrices     [independent of n_elements]
        Finite element method matrices.
    precomputed_geometric_data : PrecomputedGeometricData [n_elements, ...]
        Precomputed geometric data.
    cross_sections             : CrossSectionDataSingleGroup [n_elements, ...]
        Cross sections for a single energy group
    angle                      : jnp.ndarray [ndim,]
        Angle direction.
    inflow                     : jnp.ndarray [n_faces_per_element]
        Inflow boolean array for the element.
    abs_omega_dot_n_Je         : jnp.ndarray [n_faces_per_element,]
        Absolute value of the dot product between angle and face normals for the element.
    x                          : jnp.ndarray [n_elements, n_basis]
        x vector to apply the transport operator to.
    '''
    gij_matrix        = gij_kernel_idx_in_precomp(elem_idx, fem_matrices, precomputed_geometric_data, cross_sections, angle, inflow, abs_omega_dot_n_Je)    
    dij_f_p_fm_matrix = dij_f_p_fm_kernel_idx_in_precomp(elem_idx, fem_matrices, inflow, geometric_domain.etof, geometric_domain.face_perm_idx) #[n_faces_per_element, n_basis, n_basis]
    gather_solution   = x[geometric_domain.etoe[elem_idx], :]    
    return gij_matrix @ x[elem_idx, :] - jnp.einsum("f, fij, fj -> i", abs_omega_dot_n_Je, dij_f_p_fm_matrix, gather_solution) 

@jax.jit
def forward_transport_operator_vmap(
    geometric_domain           : GeometricDomain, 
    fem_matrices               : FEMMatrices, 
    precomputed_geometric_data : PrecomputedGeometricData,                                      
    cross_sections             : CrossSectionDataSingleGroup, 
    angle                      : jnp.ndarray,
    x                          : jnp.ndarray,  
) -> jnp.ndarray:
    '''
    Forward transport operator using vmap over angles.

    Does not require a topological sort so angle is sufficient.

    Parameters
    ----------
    geometric_domain           : GeometricDomain [for all elements]
        Geometric domain (containing element to element connectivity and element to face connectivity).
    fem_matrices               : FEMMatrices     [independent of n_elements]
        Finite element method matrices.
    precomputed_geometric_data : PrecomputedGeometricData [n_elements, ...]
        Precomputed geometric data.
    cross_sections             : CrossSectionDataSingleGroup [n_elements, ...]
        Cross sections for a single energy group
    angle                      : jnp.ndarray [ndim,]
        Angle direction
    x                          : jnp.ndarray [n_elements, n_basis]
        x vector to apply the transport operator to.

    Returns
    -------
    jnp.ndarray
        Result of applying the forward transport operator. Shape [n_angles, n_elements, n_basis]
    '''    
    elements_all = jnp.arange(geometric_domain.n_elements)
    return _single_compute_forward_vmap(
        elements_all,
        geometric_domain,
        fem_matrices,
        precomputed_geometric_data,
        cross_sections,
        angle,
        x
    )


@partial(jax.jit, static_argnums=(6,))
def forward_transport_operator_batch(
    geometric_domain           : GeometricDomain, 
    fem_matrices               : FEMMatrices, 
    precomputed_geometric_data : PrecomputedGeometricData,                                      
    cross_sections             : CrossSectionDataSingleGroup, 
    angle                      : jnp.ndarray,
    x                          : jnp.ndarray,  
    batch_size                 : int
) -> jnp.ndarray:
    '''
    Forward transport operator using vmap over angles.

    Parameters
    ----------
    geometric_domain           : GeometricDomain [for all elements]
        Geometric domain (containing element to element connectivity and element to face connectivity).
    fem_matrices               : FEMMatrices     [independent of n_elements]
        Finite element method matrices.
    precomputed_geometric_data : PrecomputedGeometricData [n_elements, ...]
        Precomputed geometric data.
    cross_sections             : CrossSectionDataSingleGroup [n_elements, ...]
        Cross sections for a single energy group
    angle                      : jnp.ndarray [ndim,]
        Angle direction.
    x                          : jnp.ndarray [n_elements, n_basis]
        x vector to apply the transport operator to.

    Returns
    -------
    jnp.ndarray
        Result of applying the forward transport operator. Shape [n_angles, n_elements, n_basis]
    '''
    elements_all = jnp.arange(geometric_domain.n_elements)
    f_batch = lambda i : _single_compute_forward(i, geometric_domain, fem_matrices, precomputed_geometric_data, cross_sections, angle, x)
    return jax.lax.map(f_batch, elements_all, batch_size=batch_size)

def _single_compute_forward(
    element_idx                : int,
    geometric_domain           : GeometricDomain, 
    fem_matrices               : FEMMatrices, 
    precomputed_geometric_data : PrecomputedGeometricData,                                      
    cross_sections             : CrossSectionDataSingleGroup, 
    angle                      : jnp.ndarray,
    x                          : jnp.ndarray,      
) -> jnp.ndarray:
    abs_omega_dot_n_Je, inflow = _compute_omega_dot_n_Je_inflow(
        element_idx,
        geometric_domain,
        angle,
        precomputed_geometric_data)        
    return _forward_transport_operator_kernel(
        element_idx,
        geometric_domain,
        fem_matrices,
        precomputed_geometric_data,
        cross_sections,
        angle,
        inflow, 
        abs_omega_dot_n_Je,
        x
    )

_single_compute_forward_vmap = jax.vmap(_single_compute_forward, in_axes=(0, None, None, None, None, None, None))

@partial(jax.jit, static_argnums=(6,))
def forward_transport_operator_multiple_angles_batch(
    geometric_domain           : GeometricDomain, 
    fem_matrices               : FEMMatrices, 
    precomputed_geometric_data : PrecomputedGeometricData,                                      
    cross_sections             : CrossSectionDataSingleGroup, 
    angle                      : jnp.ndarray,
    x                          : jnp.ndarray,  
    batch_size                 : int) -> jnp.ndarray:
    f_vmap = jax.vmap(forward_transport_operator_batch, in_axes=(None, None, None, None, 0, 0, None))
    return f_vmap(geometric_domain, fem_matrices, precomputed_geometric_data, cross_sections, angle, x, batch_size)

forward_transport_operator_multiple_angles_vmap = partial(forward_transport_operator_multiple_angles_batch, batch_size=None)




