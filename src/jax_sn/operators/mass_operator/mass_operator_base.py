import jax 
import jax.numpy as jnp
from jax_sn.solution_domain import PrecomputedGeometricData, FEMMatrices
from jax_sn.source import Source
from dataclasses import dataclass

import lineax as lx


class MassOperator(lx.AbstractLinearOperator):
    r'''
    Interface for mass operator. Computes:

    .. math::
        v_i = \sum_{j} Jv \int H_i H_j dV v_j
    '''
    fem_matrices                : FEMMatrices    
    precomputed_geometric_data  : PrecomputedGeometricData

    @classmethod    
    def from_fem_matrices_and_precomputed_geometric_data(cls, fem_matrices : FEMMatrices, precomputed_geometric_data : PrecomputedGeometricData) -> "MassOperator":
        return cls(
            fem_matrices,
            precomputed_geometric_data
        )
    
    def mv(self, x : jnp.ndarray) -> jnp.ndarray:
        return _mass_kernel_idx_in_precomp_vmap(
                jnp.arange(self.precomputed_geometric_data.n_elements),
                self.fem_matrices,
                self.precomputed_geometric_data,
                x)
    
    def mv_batch(self, x : jnp.ndarray) -> jnp.ndarray:
        _assert_x_batch(self, x)
        return _mass_kernel_idx_in_precomp_vmap_vmap_values( 
                jnp.arange(self.precomputed_geometric_data.n_elements),
                self.fem_matrices,
                self.precomputed_geometric_data,
                x)

    
    def transpose(self) -> "MassOperator":
        return self  # Mass matrix is symmetric
    
    def as_matrix(self):
        assert False, "Not desired for MassOperator due to large size"
        return jnp.array([[]])
    
    def conj(self):
        return self  #real operator
    
    def in_structure(self):        
        return self._shape_inout
    
    def out_structure(self):
        return self._shape_inout
    
    @property 
    def _shape_inout(self):
        n_elements = self.precomputed_geometric_data.n_elements
        n_basis = self.fem_matrices.n_basis        
        return jax.ShapeDtypeStruct((n_elements, n_basis), self.fem_matrices.mass_matrix.dtype)
    
    
def _assert_x_batch(massoperator : MassOperator, x):
    assert x.ndim == 3, f"x should have 3 dimensions for batching but got {x.shape}"
    
                
@lx.is_symmetric.register(MassOperator)
def _(operator : MassOperator):
    return True

@lx.linearise.register(MassOperator)
def _(operator : MassOperator):
    return operator

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