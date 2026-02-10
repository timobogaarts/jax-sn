import jax 
from dataclasses import dataclass
import jax.numpy as jnp
from typing import Literal, Tuple
from functools import partial
import equinox as eqx


class QuadratureSet(eqx.Module):
    r'''
    Represents a quadrature set with angles and weights. This class provides methods to compute spherical harmonics expansions based on the quadrature set.    
    '''
    
    angles : jnp.ndarray
    '''        The angles of the quadrature set in Cartesian coordinates. Shape [n_angles, dimension].'''

    weights : jnp.ndarray
    r'''        The weights associated with each angle. Shape [n_angles]. Should sum to 4pi (such that :math:`\sum_{i} w_i 1 = \int d\Omega = 4 \pi`).'''

    def __check_init__(self):
        assert self.angles.shape[-1] == 3, f"Only three-dimensional quadrature sets are currently supported but got {self.angles.shape[-1]}. The angles should be given in Cartesian coordinates (x,y,z) on the unit sphere."
        assert self.angles.shape[0] == self.weights.shape[0], f"The number of angles and weights must be the same but got {self.angles.shape[0]} angles and {self.weights.shape[0]} weights."

    @property 
    def n_angles(self) -> int:
        return self.angles.shape[0]
    
    def __getitem__(self, idx):
        return self.angles[idx], self.weights[idx]
    
    def wk_ylm_real(self, l_max : int) -> jnp.ndarray:
        r'''
        Compute the weighted real spherical harmonic expansion coefficients :math:`\sqrt{w_k} Y_{lm}(\Omega_k)` for this quadrature set up to degree l_max.
        See :func:`compute_3d_wk_yklm_real_from_quadrature_set` for details.
        '''        
        return compute_3d_wk_yklm_real_from_quadrature_set(self, l_max)
        
    def ylm_real(self, l_max : int) -> jnp.ndarray:
        r'''
        Compute the real spherical harmonic expansion coefficients :math:`Y_{lm}(\Omega_k)` for this quadrature set up to degree l_max.
        See :func:`compute_3d_yklm_real_from_quadrature_set` for details.
        '''        
        return compute_3d_yklm_real_from_quadrature_set(self, l_max)
    

class QuadratureSetReduced(QuadratureSet):
    r'''
    Represents a reduced quadrature set, where only unique angles are stored. The original quadrature set can be reconstructed from the unique angles and the indices to reconstruct the original set from the unique set.
    '''    
    angles_original         : jnp.ndarray
    weights_original        : jnp.ndarray
    original_angles_indices : jnp.ndarray

    def __getitem__(self, idx):
        raise ValueError("Cannot get a single angle from a QuadratureSetReduced!")
        return super().__getitem__(idx)
    
    @classmethod
    def from_quadrature_set(cls, quadrature_set : QuadratureSet, n_dim : Literal[1,2,3], rounding_tolerance : int = 10):
        return cls(*_extract_nD_quadrature_set(quadrature_set, n_dim, rounding_tolerance))

    
    def wk_ylm_real(self, l_max : int) -> jnp.ndarray:        
        wk_yklm_real_original = compute_3d_wk_yklm_real_from_quadrature_set(QuadratureSet(self.angles_original, self.weights_original), l_max)

        wk_yklm_reduced = jnp.zeros((self.n_angles, wk_yklm_real_original.shape[1]))
        wk_yklm_reduced = wk_yklm_reduced.at[self.original_angles_indices].add(wk_yklm_real_original)
        return wk_yklm_reduced

def _extract_nD_quadrature_set(quadrature_set : QuadratureSet, n_dim : Literal[1,2,3], rounding_tolerance : int = 10):        
    unique_angles        = jnp.unique(quadrature_set.angles[:, :n_dim].round(rounding_tolerance), return_index=True, return_inverse= True, axis=0) 
    # angles [n_unique, n_dim], indices of unique angles in original set [n_unique], indices to reconstruct original set from unique set [n_angles]        
    angles_red  = quadrature_set.angles[unique_angles[1]]
    weights_red = quadrature_set.weights[unique_angles[1]]

    #onp.testing.assert_allclose(weights_red[unique_angles[2]], quadrature_set.weights, rtol=1e-10)
    
    weights_red = quadrature_set.weights[unique_angles[1]]
    
    return angles_red, weights_red, quadrature_set.angles,quadrature_set.weights, unique_angles[2]
#===========================================================================================================================================================================
# Functions to compute spherical harmonics expansions from quadrature sets
#===========================================================================================================================================================================
    
@partial(jax.jit, static_argnums= 1)
def compute_3d_yklm_real_from_quadrature_set(quadrature_set : QuadratureSet, l_max : int) -> jnp.ndarray:
    r'''
    Compute the real spherical harmonic expansion coefficients :math:`Y_{lm}` for a given quadrature set up to degree l_max.

    See `_ylm_real` for the definition of the real spherical harmonics used. Computes

    .. math::
        Y_{lm}(\Omega_k)

    for all angles :math:`\Omega_k` in the quadrature set.

    This uses a three-dimensional quadrature set, i.e., the angles are given as (x,y,z) coordinates on the unit sphere.
    For lower-dimensional quadrature sets, the symmetry on the unit sphere could be taken into account to reduce the number of angles.

    :param quadrature_set: The quadrature set containing angles and weights.
    :param l_max: Maximum degree of the spherical harmonics.


    
    '''
    assert quadrature_set.angles.shape[1] == 3, "Quadrature set must be three-dimensional."
    l, m = _lm_real_expansion(l_max)
    theta = jnp.arccos(quadrature_set.angles[:,2])
    phi   = jnp.arctan2(quadrature_set.angles[:,1], quadrature_set.angles[:,0])
    Yklm  = _ylm_real(l, m, theta, phi, l_max)
    return Yklm

@partial(jax.jit, static_argnums= 1)
def compute_3d_wk_yklm_real_from_quadrature_set(quadrature_set : QuadratureSet, l_max : int) -> jnp.ndarray:
    r'''
    Compute the weighted real spherical harmonic expansion coefficients :math:`w_k Y_{lm}(\Omega_k)` for a given quadrature set up to degree l_max.

    See `_ylm_real` for the definition of the real spherical harmonics used.

    Computes

    .. math::
        w_k Y_{lm}(\Omega_k)

    for all angles :math:`\Omega_k` in the quadrature set with weights :math:`w_k`.

    This uses a three-dimensional quadrature set, i.e., the angles are given as (x,y,z) coordinates on the unit sphere.
    For lower-dimensional quadrature sets, the symmetry on the unit sphere can be taken into account to reduce the number of angles.
    
    :param quadrature_set: The quadrature set containing angles and weights.
    :param l_max: Maximum degree of the spherical harmonics.
    '''
    Yklm = compute_3d_yklm_real_from_quadrature_set(quadrature_set, l_max) #[n_angles, n_coefficients]
    wk_yklm = quadrature_set.weights[:, None] * Yklm
    return wk_yklm


#===========================================================================================================================================================================
# Internal Functions for spherical harmonics
#===========================================================================================================================================================================

def _size_from_l(l_max : int) -> int:
    return (l_max + 1)**2
def _l_from_size(n_coefficients : int) -> int:
    return int(n_coefficients**(1/2) - 1)

def _lm_real_expansion(l_max : int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    l = jnp.concatenate([jnp.full(2 * l + 1, l) for l in range(l_max + 1)])
    m = jnp.concatenate([jnp.arange(-l, l + 1) for l in range(l_max + 1)])
    return l, m

@partial(jax.jit, static_argnums= 4)
def _ylm_real(l : jnp.ndarray, m : jnp.ndarray, theta : jnp.ndarray, phi : jnp.ndarray, l_max : int) -> jnp.ndarray:
    r'''
    Compute the spherical harmonic real expansion coefficients :math:`Y_{lm}` for given l and m arrays at angles theta and phi.
    Uses the convention:

    .. math::
        Y_{lm}(\theta, \phi)= \begin{cases} \sqrt{2} \mathcal{I}(Y_l^{|m|}(\theta,\phi)) & m < 0 \\ Y_l^0(\theta, \phi) & m =0 \\ \sqrt{2} \mathcal{R}(Y_l^m(\theta, \phi)) & m >0 \end{cases}
        
    where 
        :math:`Y_{lm}` are the complex spherical harmonics defined as per jax.scipy.special.sph_harm_y (including Condon-Shortley phase which is included in the :math:`P_l^m` definition):

    .. math::
        Y_{l}^m(theta, phi) = \sqrt{\frac{2l+1}{4\pi} \cdot \frac{(l-m)!}{(l+m)!}} * P_l^m(\cos(\theta)) * e^{i m \phi}

    If we assume that :math:`\int d\Omega = 4 \pi`, then the normalization is such that:

    .. math::
        \int Y_{lm}(\theta, \phi) Y_{l'm'}(\theta, \phi) d\Omega = \delta_{ll'} \delta_{mm'}

    Note that the inverse of the expansion is also _ylm_real, i.e.,
    
    .. math::
        f(\Omega_k) = \sum_{lm} f_{lm} Y_{lm}(\Omega_k) \\
        f_{lm} = \int f(\Omega) Y_{lm}(\Omega) d\Omega \approx \sum_k w_k f(\Omega_k) Y_{lm}(\Omega_k)

    Finally, note that the so-called scalar flux :math:`\phi` is defined as the zeroth moment coefficient:

    .. math::
        \phi = \sqrt{4\pi} \psi_{00} 

    Parameters:
    ----------
    l : jnp.ndarray [n_coefficients]
        Array of degree l values.
    m : jnp.ndarray [n_coefficients]
        Array of order m values.
    theta : jnp.ndarray [n_angles]
        Polar angles (in radians) at which to evaluate the harmonics.
    phi : jnp.ndarray [n_angles]
        Azimuthal angles (in radians) at which to evaluate the harmonics.
    l_max : int
        Maximum degree of the spherical harmonics. Used for the jax.scipy.special.sph_harm_y function.

    Returns:
    -------
    Yklm : jnp.ndarray [n_angles, n_coefficients]
        Real spherical harmonics evaluated at the specified angles.
    '''
    def _compute_real_single_lm(l,m, theta, phi):                
        Ylm    = jax.scipy.special.sph_harm_y(l, abs(m),      jnp.full(l.shape, theta), jnp.full(l.shape, phi), n_max=l_max)
        
        return jnp.where(m==0, Ylm.real, jnp.where(m > 0, jnp.sqrt(2) * Ylm.real, jnp.sqrt(2) *  Ylm.imag))        
    
    Yklm = jax.vmap(_compute_real_single_lm, in_axes=(None, None, 0, 0))(l,m, theta, phi) 
    return Yklm

