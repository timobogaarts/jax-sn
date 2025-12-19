import jax 
import jax.numpy as jnp 
from functools import partial
from dataclasses import dataclass
from typing import Callable
from jax_sn.quadrature_set import QuadratureSet
import lineax as lx
from jax_sn.operators.base_linear_operator import AbstractLinearOperator_sn

class MomentOperator(AbstractLinearOperator_sn):
    r'''
    Represents a moment operator that maps angular fluxes to moments.

    
    This operator batches over the trailing dimensions. Although this is inefficient
    (moment operator sums over the angles, which is the leading dimension and thus not contiguous in memory)
    it is more convenient for the sweep operation, which requires the angles to be the leading dimension (since each angular sweep is independent, we want contiguous memory for the trailing n_elements, n_basis dimensions).

    Attributes:
    ----------
    wk_yklm : jnp.ndarray [n_angles, n_moments]
        The moment operator weights mapping angles to moments for the specific desired order L.
    quadrature_set : QuadratureSet [n_angles]
        The quadrature set associated with this moment operator.
    elements_basis_shape : jax.ShapeDtypeStruct [n_elements, n_basis]
        The shape of the elements and basis functions for the FEM domain.
    '''
    wk_yklm              : jnp.ndarray           #[n_angles, n_moments]
    quadrature_set       : QuadratureSet    
    elements_basis_shape : jax.ShapeDtypeStruct  #[n_elements, n_basis]

    @classmethod
    def from_quadrature_set(cls, quadrature_set : QuadratureSet, l_max : int, elements_basis_shape : jax.ShapeDtypeStruct) -> "MomentOperator":
        wk_yklm = quadrature_set.wk_ylm_real(l_max)        
        return cls(wk_yklm, quadrature_set, elements_basis_shape)
    
    def mv(self, x : jnp.ndarray) -> jnp.ndarray:
        return _call_batched_moment_operator(x, self.wk_yklm)
    
    def transpose(self) -> "MomentOperator":
        raise NotImplementedError("Transpose of MomentOperator is not implemented.")
                    
    def in_structure(self):
        return jax.ShapeDtypeStruct(
            shape=(self.wk_yklm.shape[0],) + self.elements_basis_shape.shape, 
            dtype=self.wk_yklm.dtype)
    
    def out_structure(self):
        return jax.ShapeDtypeStruct(
            shape=(self.wk_yklm.shape[1],) + self.elements_basis_shape.shape, 
            dtype=self.wk_yklm.dtype)


@lx.is_symmetric.register(MomentOperator)
def _(operator):
    return False

@lx.linearise.register(MomentOperator)
def _(operator):
    return operator


def _call_batched_moment_operator(buffer_sweep : jnp.ndarray, wk_yklm : jnp.ndarray) -> jnp.ndarray:
    r'''
    Apply the moment operator to the angular flux buffer to compute the moments.

    Computes:

    .. math::
        \phi_{l,m}^{ei} = \sum_{a=1}^{N_{angles}} \psi_k^{ei} w_{k} Y_{l,m}(\Omega_k)

    with :math:`\phi_{l,m}^{ei}` the moment of order :math:`(l,m)` in element :math:`e` and basis function :math:`i`,
    :math:`\psi_k^{ei}` the angular flux for angle :math:`k` in element :math:`e` and basis function :math:`i`, :math:`w_k` the weight for angle :math:`k` 
    and :math:`Y_{l,m}(\Omega_k)` the spherical harmonic of order :math:`(l,m)` evaluated at direction :math:`\Omega_k`. For the latter definition, 
    see `jax_sn.quadrature_set.QuadratureSet.wk_ylm_real`.


    Parameters:
    ----------
    buffer_sweep : jnp.ndarray [n_angles, n_elements, n_basis]
        The angular flux buffer containing the angular fluxes for each angle and each element and its basis functions
    wk_yklm : jnp.ndarray [n_angles, (L+1)**2]
        The moment operator weights mapping angles to moments for the specific desired order L.

    Returns:
    -------
    moments : jnp.ndarray [n_moments, n_elements, n_basis]
        The computed moments.
    
    '''    
    assert buffer_sweep.shape[0] == wk_yklm.shape[0], f"Number of angles in buffer_sweep ({buffer_sweep.shape[0]}) must match number of angles in wk_yklm ({wk_yklm.shape[0]})."
    return jnp.einsum('k...,kl->l...', buffer_sweep, wk_yklm)