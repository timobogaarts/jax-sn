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
    Interface for mass operator. 
    Computes for a given vector :math:`\{ v_j \}` the mass matrix map:

    .. math::
        a_i = \sum_{j} \int H_i H_j dV v_j

    Is arbitrarily batched over additional leading dimensions.

    The non-batched version computes the mass matrix for each element and basis function:

    .. math::
        \mathbf{H}_b : F \to F

    where in this case:

    .. math::
        F : \{ \text{all elements} \} \otimes \{ \text{all basis functions per element} \}
        
    and since all elements are independent, the mass matrix is block-diagonal with each block representing the mass matrix for one element.
    
    Its definition is (:math:`e,p` is  element index, `i,j` basis function index):

    .. math::
        \mathbf{H}_b[\{ v_{ei} \}] = a_{ej} = \sum_{i}  J_{vq} \int_{B} H_i H_j dV v_{ei}

    where :math:`J_{vq}` is the volume Jacobian determinant for element :math:`e`, and :math:`B` is the reference element.
    Note this integral is already computed in :attr:`jax_sn.solution_domain.FEMMatrices.mass_matrix`, and the Jacobian determinants are precomputed and stored  :attr:`jax_sn.solution_domain.PrecomputedGeometricData.vol_jacobians_det`.

    This operator is symmetric since the mass matrix is symmetric.

    Batched versions operate over additional leading dimensions, e.g.: 
    
    .. math::
        &\mathbf{H} :   \mathcal{K} \otimes F \to \mathcal{K} \otimes F\\
        &\mathbf{H} = \mathbf{I}_{\mathcal{K}} \otimes \mathbf{H}_b
    
    where :math:`\mathcal{K}` is an arbitrary additional space (e.g. angles).


    In essence, this just is a batched matrix product (and is implemented like that as well).

    '''

    fem_matrices                : FEMMatrices    
    '''Finite element method matrices containing the mass matrix.'''
    
    precomputed_geometric_data  : PrecomputedGeometricData
    '''Precomputed geometric data containing the Jacobian determinants.'''

    addtional_shape             : Tuple[int] = ()
    '''Additional leading shape to batch over. Used for consistent in/out structure definitions.'''

    @classmethod 
    def from_solution_domain_data(
        cls,
        solution_domain_data : SolutionDomainData,
        additional_shape     : Tuple[int] = ()
    ) -> "MassOperator":
        r'''
        Construct a MassOperator from solution domain data.

        Parameters
        ----------
        solution_domain_data : SolutionDomainData
            The solution domain data to use.
        additional_shape : Tuple[int]
            Additional leading shape to batch over.

        Returns
        -------
        MassOperator
            The constructed MassOperator.
        '''
        return cls(
            solution_domain_data.fem_matrices,
            solution_domain_data.precomputed_geometric_data,
            additional_shape
        )
    
    @classmethod 
    def from_transport_operator(cls, transport_operator : Union[TransportOperator, TransportOperatorMultipleAngles]) -> "MassOperator":
        r'''
        Construct a MassOperator from a transport operator.
        Parameters
        ----------
        transport_operator : Union[TransportOperator, TransportOperatorMultipleAngles]
            The transport operator to use.
        Returns
        ------- 
        MassOperator
            The constructed MassOperator.
        '''
        in_structure = transport_operator.in_structure()
        solution_domain_data = transport_operator.solution_domain_data
        
        return cls(solution_domain_data.fem_matrices,
                   solution_domain_data.precomputed_geometric_data,
                   in_structure.shape[:-2])  
    

    
    def mv(self, x : jnp.ndarray) -> jnp.ndarray:        
        return _total_mass_kernel_idx_in_precomp(
                self.fem_matrices,
                self.precomputed_geometric_data,
                x)
    
    def transpose(self) -> "MassOperator":
        return self  # Mass matrix is symmetric        
    
    def in_structure(self):        
        return self._shape_inout
    
    def out_structure(self):
        return self._shape_inout
    
    @property 
    def _shape_inout(self):
        n_elements = self.precomputed_geometric_data.n_elements
        n_basis = self.fem_matrices.n_basis        
        return jax.ShapeDtypeStruct(self.addtional_shape + (n_elements, n_basis), self.fem_matrices.mass_matrix.dtype)
    
                
@lx.is_symmetric.register(MassOperator)
def _(operator : MassOperator):
    return True

@lx.linearise.register(MassOperator)
def _(operator : MassOperator):
    return operator


def _mass_kernel_idx_in_precomp(
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
def _total_mass_kernel_idx_in_precomp(
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
    return jax.vmap(_mass_kernel_idx_in_precomp, in_axes=(0, None, None, None), out_axes=-2)(
                jnp.arange(precomputed_geometric_data.n_elements),
                fem_matrices,
                precomputed_geometric_data,
                values)

