import jax 
import jax.numpy as jnp 
from functools import partial
from dataclasses import dataclass
from typing import Callable
from . import QuadratureSet
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MomentOperator:
    r'''
    Represents a moment operator that maps angular fluxes to moments.

    Attributes:
    ----------
    
    '''
    wk_yklm        : jnp.ndarray #[n_angles, n_moments]
    quadrature_set : QuadratureSet

    @classmethod
    def from_quadrature_set(cls, quadrature_set : QuadratureSet, l_max : int) -> "MomentOperator":
        wk_yklm = quadrature_set.wk_ylm_real(l_max)        
        return cls(wk_yklm, quadrature_set)
    
    def __call__(self, buffer_sweep : jnp.ndarray) -> jnp.ndarray:
        return _call_moment_operator(buffer_sweep, self.wk_yklm)

def _call_moment_operator(buffer_sweep : jnp.ndarray, wk_yklm : jnp.ndarray) -> jnp.ndarray:
    r'''
    Apply the moment operator to the angular flux buffer to compute the moments.

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
    return jnp.einsum('a...,al->...l', buffer_sweep, wk_yklm)