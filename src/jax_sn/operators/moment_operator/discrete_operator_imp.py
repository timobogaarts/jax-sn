import jax 
import jax.numpy as jnp 
from functools import partial
from dataclasses import dataclass
from typing import Callable
from jax_sn.quadrature_set import QuadratureSet
import lineax as lx
from jax_sn.operators.base_linear_operator import AbstractLinearOperator_sn
import equinox as eqx

class DiscreteOperator(AbstractLinearOperator_sn):
    r'''

    Represents a discrete operator that maps moments to angular fluxes. Approximate inverse of :class:`MomentOperator`.

    Implements:

    .. math::
        D  = \overline{\mathbf{D}} \otimes \mathbf{I}_F: \mathcal{M}\otimes \mathcal{F} \rightarrow \mathcal{D}\otimes \mathcal{F}

    with    

    .. math::
        \overline{\mathbf{D}}[\{ b_{lm} \}] = \sum_{lm} b_{lm} Y_{lm}(\Omega_k)
   
    '''

    yklm              : jnp.ndarray           #[n_angles, n_moments]
    r'''Matrix of [n_angles, n_moments] containing :math:`w_k Y_{l,m}(\Omega_k)` values for each angle :math:`k` and moment :math:`(l,m)` (collapsed into a single index).'''

    quadrature_set       : QuadratureSet    
    r'''The quadrature set associated with this moment operator.'''

    elements_basis_shape : jax.ShapeDtypeStruct  #[n_elements, n_basis]
    r'''The shape of the elements and basis functions over which this operator is batched. Identified with :math:`F` in the documentation above.'''

    @classmethod
    def from_quadrature_set(cls, quadrature_set : QuadratureSet, l_max : int, elements_basis_shape : jax.ShapeDtypeStruct) -> "DiscreteOperator":
        r'''
        Create a MomentOperator from a quadrature set and desired moment order. This computes the :math:`w_k Y_{l,m}(\Omega_k)` weights for the specified moment order.
        Prefer to initialise directly if possible.
        
        
        :param quadrature_set: 
            The quadrature set to use for the moment operator.
        :param l_max:
            The maximum moment order L to use.
        :param elements_basis_shape:
            The shape of the elements and basis functions over which this operator is batched. Identified with :math:`F` in the documentation above.
        
        
        '''
        yklm = quadrature_set.ylm_real(l_max)
        return cls(yklm, quadrature_set, elements_basis_shape)
    
    def mv(self, x : jnp.ndarray) -> jnp.ndarray:
        return _moment_to_discrete_mv(self.yklm, x)
    
    def transpose(self) -> "DiscreteOperatorTransposed":
        return DiscreteOperatorTransposed(self.yklm, self.quadrature_set, self.elements_basis_shape)
                    
    def in_structure(self):
        return jax.ShapeDtypeStruct(
            shape=(self.yklm.shape[1],) + self.elements_basis_shape.shape, 
            dtype=self.yklm.dtype)
    
    def out_structure(self):
        return jax.ShapeDtypeStruct(
            shape=(self.yklm.shape[0],) + self.elements_basis_shape.shape, 
            dtype=self.yklm.dtype)
    
    @property 
    def n_moments(self):
        '''Number of moments (= :math:`(L+1)^2` for moment order L)'''
        return self.yklm.shape[1]

@lx.is_symmetric.register(DiscreteOperator)
def _(operator):
    return False

@lx.linearise.register(DiscreteOperator)
def _(operator):
    return operator


class DiscreteOperatorTransposed(DiscreteOperator):
    r'''

    Represents the transpose of the discrete operator that maps angular fluxes to moments. Approximate inverse of :class:`MomentOperatorTransposed`.

    Implements:

    .. math::
        D^T  = \overline{\mathbf{D}}^T \otimes \mathbf{I}_F: \mathcal{D}\otimes \mathcal{F} \rightarrow \mathcal{M}\otimes \mathcal{F}   

    with    

    .. math::
        \overline{\mathbf{D}}^T[s_k] = \sum_{k} s_k Y_{lm}(\Omega_k)
   
    '''

    def mv(self, x : jnp.ndarray) -> jnp.ndarray:
        return _moment_to_discrete_transpose_mv(self.yklm, x)
    def transpose(self) -> "DiscreteOperator":
        return DiscreteOperator(self.yklm, self.quadrature_set, self.elements_basis_shape)
    
    def in_structure(self):
        return DiscreteOperator.out_structure(self)
    def out_structure(self):
        return DiscreteOperator.in_structure(self)


@eqx.filter_jit
def _moment_to_discrete_mv(yklm : jnp.ndarray, x) -> jnp.ndarray:
    r'''
    D operator implementation:

    .. math::
        s_{ke} = \sum_{\ell} Y_{k \ell } \psi_{\ell e}

    Parameters
    ----------
    yklm : jnp.ndarray
        The yklm matrix of shape [n_angles, n_moments].
    x : jnp.ndarray
        The input vector of shape [n_angles, ...].

    Returns
    -------
    jnp.ndarray
        The output vector of shape [n_moments, ...].
    '''
    return jnp.einsum('kl, l... -> k...', yklm, x)

@eqx.filter_jit
def _moment_to_discrete_transpose_mv(yklm : jnp.ndarray, x) -> jnp.ndarray:
    r'''
    D^T operator implementation:

    .. math::
        \psi_{\ell e} = \sum_{k} Y_{k \ell } s_{ke}

    Parameters
    ----------
    yklm : jnp.ndarray
        The yklm matrix of shape [n_angles, n_moments].
    x : jnp.ndarray
        The input vector of shape [n_angles, ...].

    Returns
    -------
    jnp.ndarray
        The output vector of shape [n_moments, ...].
    '''
    return jnp.einsum('kl, k... -> l...', yklm, x)


