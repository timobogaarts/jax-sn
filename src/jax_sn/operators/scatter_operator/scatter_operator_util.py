import jax
import jax.numpy as jnp
from jax_sn.operators.moment_operator.moment_operator_imp import MomentOperator
from jax_sn.quadrature_set import QuadratureSet
from jax_sn.quadrature_set.quadrature_set_base import _lm_real_expansion
from jax_sn.domain.geometric_domain import CrossSectionIndexed
from jax_sn.operators.base_linear_operator import AbstractLinearOperator_sn
import lineax as lx

def _moment_to_discrete_kernel(moment_array : jnp.ndarray, yklm : jnp.ndarray) -> jnp.ndarray:
    r'''
    Converts a moment array to discrete angles using the yklm matrix.

    .. math::
        \psi^{(e, g, a)} = \sum_{lm} Y_{klm} \psi_{lm}^{(e, g)}

    Parameters
    ----------
    moment_array : jnp.ndarray [n_moments, n_basis]
        The moment array for a specific element and energy group.
    yklm : jnp.ndarray [n_angles, n_moments]
        The yklm matrix for the quadrature set.

    Returns
    -------
    jnp.ndarray [n_angles, n_basis]
        The discrete angular flux for the specified element and energy group.
    '''
    assert moment_array.shape[0] == yklm.shape[-1], f"Moment array last dimension {moment_array.shape} must match yklm last dimension {yklm.shape}."
    return jnp.einsum('l...,kl->k...', moment_array, yklm)  # [n_moments, ... ] x [n_angles, n_moments] -> [n_angles, ...]

def _moment_to_discrete_kernel_transposed(discrete_array : jnp.ndarray, yklm : jnp.ndarray) -> jnp.ndarray:
    r'''
    Transpose of _moment_to_discrete_kernel: Converts a discrete angle array to moments using the yklm matrix. This is *not* a conversion to moments 
    like in MomentOperator, but the transposed operation of converting moments to discrete angles.

    .. math::
        \psi_{lm}^{(e, g)} = \sum_{k} Y_{klm} \psi^{(e, g, a)}

    Parameters
    ----------
    discrete_array : jnp.ndarray [n_angles, n_basis]
        The discrete angular flux for a specific element and energy group.
    yklm : jnp.ndarray [n_angles, n_moments]
        The yklm matrix for the quadrature set.

    Returns
    -------
    jnp.ndarray [n_moments, n_basis]
        The moment array for the specified element and energy group.
    '''
    assert discrete_array.shape[0] == yklm.shape[0], f"Discrete array first dimension {discrete_array.shape} must match yklm first dimension {yklm.shape}."
    return jnp.einsum('k...,kl->l...', discrete_array, yklm)  # [n_angles, ...] x [n_angles, n_moments] -> [n_moments, ...]

def _scatter_kernel_xs_indexed_kernel(elem_idx : int, g_out : int, g_in : int, cross_section_indexed : CrossSectionIndexed, moments_array : jnp.ndarray) -> jnp.ndarray:
    r'''
    Computes from a given element index, outgoing group, incoming group, indexed cross section object and buffer for the specific incoming group, the corresponding scattering scattering source:

    .. math::
        s^{g_{out}}_{lm}=  \sigma_{l, g_{out}\from g_{in}}^{(e)} \psi_{lm}^{(e, g_{in})}

    Automatically pads or truncates the cross section data to match the number of moments in the moments_array.
        
    Parameters
    ----------
    elem_idx : int
        The element index.
    g_out : int
        The outgoing energy group index.
    g_in : int
        The incoming energy group index.
    cross_section_indexed : CrossSectionIndexed
        The indexed cross section data structure.
    moments_array : jnp.ndarray [n_elements, n_basis, n_moments]
        The moments array containing angular flux moments.
    
    Returns
    -------
    jnp.ndarray [n_basis, n_moments]
        The scattering source moments for the specified element and energy groups.
    '''
    assert moments_array.ndim == 3, f"moments_array must be 2D [n_moments, n_elements, n_basis], but has shape {moments_array.shape}."    
    
    xs_idx = cross_section_indexed.indices[elem_idx]

    sigma_l_gg = cross_section_indexed.cross_section_data.scattering[xs_idx, :, g_out, g_in] # [lmax + 1]
    
    # Since this is a l expansion, not in terms of lm, we expand it locally to the full n_moments                                                                                            
    extended_sigma_lm_gg = sigma_l_gg[_lm_real_expansion(sigma_l_gg.shape[0] - 1)[0]] # [n_moments]


    # now we have to pad/truncate to match moments_array
    n_moments = moments_array.shape[0]
    extended_sigma_lm_gg = extended_sigma_lm_gg[:n_moments]  #  first truncate if needed (if n_moments > extended_sigma_lm_gg.shape[0], this does nothing)
    extended_sigma_lm_gg = jnp.pad(extended_sigma_lm_gg, (0, n_moments - extended_sigma_lm_gg.shape[0]))  # pad if needed    
    return extended_sigma_lm_gg[:, None] * moments_array[:, elem_idx, :]  # [n_moments, None] * [n_moments, n_basis] -> [n_moments, n_basis]


def _scatter_kernel_xs_indexed_discrete(elem_idx : int, g_out : int, g_in : int, cross_section_indexed : CrossSectionIndexed, moments_array : jnp.ndarray, yklm : jnp.ndarray) -> jnp.ndarray:
    r'''
      Computes from a given element index, outgoing group, incoming group, indexed cross section object, buffer for the specific incoming group, and the real expansion coefficients, the corresponding scattering scattering source:

    .. math::
        s^{g_{out}}_{k}= \sum_{lm} \sigma_{l, g_{out}\from g_{in}}^{(e)} \phi_{lm}^{(e, g_{in})} Y_{lm}(\Omega_k)

    Automatically pads or truncates the cross section data to match the number of moments in the moments_array.
        
    Parameters
    ----------
    elem_idx : int
        The element index.
    g_out : int
        The outgoing energy group index.
    g_in : int
        The incoming energy group index.
    cross_section_indexed : CrossSectionIndexed
        The indexed cross section data structure.
    moments_array : jnp.ndarray [n_elements, n_basis, n_moments]
        The moments array containing angular flux moments.
    
    Returns
    -------
    jnp.ndarray [n_basis, n_angles]
        The scattering source moments for the specified element and energy groups.
    '''    
    moment_array = _scatter_kernel_xs_indexed_kernel(elem_idx, g_out, g_in, cross_section_indexed, moments_array)  # [n_basis, n_moments]
    return _moment_to_discrete_kernel(moment_array, yklm)  # [n_basis, n_angles]

def _scatter_kernel_xs_indexed_discrete_transposed(elem_idx : int, g_out : int, g_in : int, cross_section_indexed : CrossSectionIndexed, discrete_array : jnp.ndarray, yklm : jnp.ndarray) -> jnp.ndarray:
    r'''
      Computes from a given element index, outgoing group, incoming group, indexed cross section object, buffer for the specific incoming group in discrete angles, and the real expansion coefficients, the corresponding scattering scattering source in moments:

    .. math::
        s^{g_{out}}_{lm}=  \sigma_{l, g_{out}\from g_{in}}^{(e)} \sum_{k} Y_{klm} \psi^{(e, g_{in}, k)} 
    Automatically pads or truncates the cross section data to match the number of moments in the moments_array. 

    The transpose of _scatter_kernel_xs_indexed_discrete does not matter; it does not have a angle index.

    Parameters
    ----------
    elem_idx : int
        The element index.      
    g_out : int
        The outgoing energy group index.
    g_in : int
        The incoming energy group index.
    cross_section_indexed : CrossSectionIndexed
        The indexed cross section data structure.
    discrete_array : jnp.ndarray [n_basis, n_angles]
        The discrete angular flux for the specified element and energy group.
    yklm : jnp.ndarray [n_angles, n_moments]
        The yklm matrix for the quadrature set.
    '''
    moment_array = _moment_to_discrete_kernel_transposed(discrete_array, yklm)  # [n_moments, n_basis]
    return _scatter_kernel_xs_indexed_kernel(elem_idx, g_out, g_in, cross_section_indexed, moment_array)  # [n_basis, n_moments]

