import jax 
import jax.numpy as jnp
from jax_sn.solution_domain import PrecomputedGeometricData, FEMMatrices
from jax_sn.source import Source

def mass_kernel_idx_in_precomp(
        elem_idx                    : int,
        fem_matrices                : FEMMatrices,    
        precomputed_geometric_data  : PrecomputedGeometricData,
        values                      : jnp.ndarray
    ):
    '''
    Compute the mass matrix map
        v_i = ∑_{j} Jv ∫H_i H_j dV v_j 

    We index directly in the source instead of passing a subsource. This can save significant memory when vmapping over all elements and all directions with the same source independent of direction.

    Parameters
    ----------
    elem_idx : int
        The element index to compute the values vector for
    fem_matrices : FEMMatrices
        The finite element method matrices.
    precomputed_geometric_data : PrecomputedGeometricData
        The precomputed geometric data (Jacobian determinants and inverses).
    values : jnp.ndarray
        The source values array of shape [n_elements, n_basis]
    Returns
    -------
    values : jnp.ndarray
        The computed value vector for the specified element.
    '''    
    
    return precomputed_geometric_data.vol_jacobians_det[elem_idx] * fem_matrices.mass_matrix @ values[elem_idx]


_mass_kernel_idx_in_precomp_vmap = jax.vmap(mass_kernel_idx_in_precomp, in_axes=(0, None, None, None))

_mass_kernel_idx_in_precomp_vmap_vmap_values = jax.vmap(_mass_kernel_idx_in_precomp_vmap, in_axes=( None, None, None,0))