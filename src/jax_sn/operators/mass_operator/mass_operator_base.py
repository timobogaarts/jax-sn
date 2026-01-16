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
from abc import abstractmethod


class MassOperator(AbstractLinearOperator_sn):
    r'''
    Interface for mass operator. 
    Computes for a given vector :math:`\{ v_j \}` the mass matrix map:

    .. math::
        a_i = \sum_{j} \int H_i H_j dV v_j

    Is arbitrarily batched over additional leading dimensions.

    Mathematical description
    ---------------------------

    The non-batched version computes the mass matrix for each element and basis function:

    .. math::
        \overline{\mathbf{H}} : F \to F

    where in this case:

    .. math::
        F : \{ \text{all elements} \} \otimes \{ \text{all basis functions per element} \}
        
    and since all elements are independent, the mass matrix is block-diagonal with each block representing the mass matrix for one element.
    
    Its definition is (:math:`e,p` is  element index, `i,j` basis function index):

    .. math::
        \overline{\mathbf{H}}[\{ v_{ei} \}] = a_{ej} = \sum_{i}  J_{vq} \int_{B} H_i H_j dV v_{ei}

    where :math:`J_{vq}` is the volume Jacobian determinant for element :math:`e`, and :math:`B` is the reference element.
    Note this integral is already computed in :attr:`jax_sn.solution_domain.FEMMatrices.mass_matrix`, and the Jacobian determinants are precomputed and stored  :attr:`jax_sn.solution_domain.PrecomputedGeometricData.vol_jacobians_det`.

    Transposition
    ---------------------------

    This operator is symmetric since the mass matrix is symmetric. Transposition thus returns itself.

    Structure
    ---------------------------

    Batched versions operate over additional leading dimensions, e.g.: 
    
    .. math::
        &\mathbf{H} :   \mathcal{K} \otimes \mathcal{F} \to \mathcal{K} \otimes \mathcal{F}\\
        &\mathbf{H} = \mathbf{I}_{\mathcal{K}} \otimes \overline{\mathbf{H}}
    
    where :math:`\mathcal{K}` is an arbitrary additional space (e.g. angles).


    In essence, this just is a batched matrix product (and is implemented like that as well).

    '''

    fem_matrices                : FEMMatrices    
    '''Finite element method matrices containing the mass matrix.'''
    
    precomputed_geometric_data  : PrecomputedGeometricData
    '''Precomputed geometric data containing the Jacobian determinants.'''

    additional_shape             : Tuple[int] = ()
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
    

    @abstractmethod
    def mv(self, x : jnp.ndarray) -> jnp.ndarray:        
        ...
        
    
    def transpose(self) -> "MassOperator":
        '''Returns the transpose operator, which is itself since the mass matrix is symmetric.'''
        return self  # Mass matrix is symmetric        
    
    def in_structure(self):     
        '''Input structure. Shape [..., n_elements, n_basis] where ... is the same as :attr:`additional_shape`.'''   
        return self._shape_inout
    
    def out_structure(self):
        '''Output structure. Shape [..., n_elements, n_basis] where ... is the same as :attr:`additional_shape`.'''
        return self._shape_inout
    
    @property 
    def _shape_inout(self):
        n_elements = self.precomputed_geometric_data.n_elements
        n_basis = self.fem_matrices.n_basis        
        return jax.ShapeDtypeStruct(self.additional_shape + (n_elements, n_basis), self.fem_matrices.mass_matrix.dtype)
    
                
@lx.is_symmetric.register(MassOperator)
def _(operator : MassOperator):
    return True

@lx.linearise.register(MassOperator)
def _(operator : MassOperator):
    return operator

class MassOperatorVmap(MassOperator):
    r'''
    Uses a vmap of all elements to compute the mass operator.
    Alone, this creates a temporary. However, in a combined operator (e.g. SingleGroupOperator), this temporary can often be re-used, such that it is not necessarily 
    any worse than :class:`MassOperatorLax` (which does not create temporaries).
    '''

    def mv(self,x : jnp.ndarray) -> jnp.ndarray:
        '''
        Apply the mass operator to the input vector x.

        Parameters
        ----------
        x : jnp.ndarray
            Input array of shape [..., n_elements, n_basis]

        Returns
        -------
        jnp.ndarray
            Output array of shape [..., n_elements, n_basis]
        '''        
        return _total_mass_kernel_idx_in_precomp(
            self.fem_matrices,
            self.precomputed_geometric_data,
            x
        )


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


@jax.jit
def total_mass_kernel_lax(
    fem_matrices: FEMMatrices,
    precomputed_geometric_data: PrecomputedGeometricData,
    values: jnp.ndarray,                     # [..., n_elem, n_dof]
):
    M = fem_matrices.mass_matrix              # [n_dof, n_dof]
    detJ = precomputed_geometric_data.vol_jacobians_det  # [n_elem]

    # Move first axes to to batch axis
    # (..., n_elem, n_dof) -> (batch, n_elem, n_dof)
    batch_shape = values.shape[:-2]
    values_b = values.reshape((-1,) + values.shape[-2:])

    # Batched matmul: (n_dof, n_dof) @ (..., n_elem, n_dof)^T
    out = jax.lax.dot_general(
        values_b,                             # (B, n_elem, n_dof)
        M,                                    # (n_dof, n_dof)
        dimension_numbers=(
            ((2,), (1,)),                     # contract dof
            ((), ())                          # no batch contraction
        ),
    )                                         # (B, n_elem, n_dof)

    # Scale by Jacobian determinant
    out = out * detJ[None, :, None]

    return out.reshape(batch_shape + out.shape[-2:])


class MassOperatorLax(MassOperator):
    '''
    Batched matrix multiplication implementation of the mass operator using jax.lax.
    This implementation does not create temporaries and is not particularly less or more performant than :class:`MassOperatorVmap`.
    '''
    
    def mv(self,x : jnp.ndarray) -> jnp.ndarray:
        
        return total_mass_kernel_lax(
            self.fem_matrices,
            self.precomputed_geometric_data,
            x
        )
