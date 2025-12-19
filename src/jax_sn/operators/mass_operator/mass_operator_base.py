import jax 
import jax.numpy as jnp
from jax_sn.solution_domain import PrecomputedGeometricData, FEMMatrices
from jax_sn.source import Source
from dataclasses import dataclass

import lineax as lx
from jax_sn.operators.base_linear_operator import AbstractLinearOperator_sn

from typing import Tuple
from jax_sn.solution_domain import SolutionDomainData
from typing import Union    

from jax_sn.operators.transport_operator.transport_operator_base import TransportOperator, TransportOperatorMultipleAngles
class MassOperator(AbstractLinearOperator_sn):
    r'''
    Interface for mass operator. Computes:

    .. math::
        v_i = \sum_{j} Jv \int H_i H_j dV v_j
    '''
    fem_matrices                : FEMMatrices    
    precomputed_geometric_data  : PrecomputedGeometricData
    addtional_shape             : Tuple[int] = ()

    @classmethod 
    def from_solution_domain_data(
        cls,
        solution_domain_data : SolutionDomainData,
        additional_shape     : Tuple[int] = ()
    ) -> "MassOperator":
        return cls(
            solution_domain_data.fem_matrices,
            solution_domain_data.precomputed_geometric_data,
            additional_shape
        )
    
    @classmethod 
    def from_transport_operator(cls, transport_operator : Union[TransportOperator, TransportOperatorMultipleAngles]) -> "MassOperator":
        in_structure = transport_operator.in_structure()
        solution_domain_data = transport_operator.solution_domain_data

        
        return cls(solution_domain_data.fem_matrices,
                   solution_domain_data.precomputed_geometric_data,
                   in_structure.shape[:-2])  
    

    
    def mv(self, x : jnp.ndarray) -> jnp.ndarray:        
        return total_mass_kernel_idx_in_precomp(
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
        return jax.ShapeDtypeStruct(self.addtional_shape + (n_elements, n_basis), self.fem_matrices.mass_matrix.dtype)
    
    
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
        The source values array of shape [..., n_elements, n_basis]
    Returns
    -------
    values : jnp.ndarray [...., n_basis]
        The computed value vector for the specified element.
    '''        
    return precomputed_geometric_data.vol_jacobians_det[elem_idx] * jnp.einsum("ij, ...j->...i", fem_matrices.mass_matrix , values[..., elem_idx, :])


@jax.jit
def total_mass_kernel_idx_in_precomp(
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
    fem_matrices : FEMMatrices
        The finite element method matrices.
    precomputed_geometric_data : PrecomputedGeometricData
        The precomputed geometric data (Jacobian determinants and inverses).
    values : jnp.ndarray
        The source values array of shape [..., n_elements, n_basis]
    Returns
    -------
    values : jnp.ndarray [...., n_elements, n_basis]
        The computed value vector for all elements.
    '''            
    return jax.vmap(mass_kernel_idx_in_precomp, in_axes=(0, None, None, None), out_axes=-2)(
                jnp.arange(precomputed_geometric_data.n_elements),
                fem_matrices,
                precomputed_geometric_data,
                values)

