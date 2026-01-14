import jax
import jax.numpy as jnp

from jax_sn.operators.base_linear_operator import AbstractLinearOperator_sn
from jax_sn.operators.moment_operator import DiscreteOperator, DiscreteOperatorTransposed
from jax_sn.domain.cross_sections import CrossSectionsIndexed
from jax_sn.quadrature_set import QuadratureSet
import lineax as lx

import equinox as eqx
from abc import abstractmethod
from jax_sn.quadrature_set.quadrature_set_base import _lm_real_expansion
from .scatter_operator_imp import ScatterOperator

class GatherScatterOperator(eqx.Module):
    r'''
    GatherScatterOperator represents the gather-scatter operator that maps angular flux moments to discrete angles using indexed cross sections for a specific energy group. This implements two functions: the gather-scatter operation and its transpose.
    This is a module instead of just directly called functions to allow for overriding the specific implementation (e.g. using a scan, vmap or just broadcasting).

    This implements the following:

    Forward
    ============

    .. math::
        \mathbf{U}_{g_{out}} = \bar{\matbhf{U}}_{g_{out}} \otimes \mathbf{I}_{\mathcal{B}} :\mathcal{M} \otimes \mathcal{E}\otimes \mathcal{B} \to \mathcal{M} \otimes \mathcal{E}\otimes \mathcal{B}

    with 

    .. math::
        \mathbf{\mathbf{U}}_{g_{out}} [\psi] = \sum_{g_{in}} \sigma_{\ell g_{out} g_{in},e} \psi_{g_{in}\ell e}

    for some :math:`\psi\in \mathcal{M} \otimes \mathcal{E}` (we use :math:`\ell` to denote :math:`(l,m)` moment index for brevity; for :math:`\sigma_{\ell}` this just equals :math:`\sigma_{l}` for all :math:`m`).  It acts trivially on the basis functions in each element.

    Backward
    ============

    .. math::
        \mathbf{V}_{g_{in}} = \bar{\matbhf{V}}_{g_{out}} \otimes \mathbf{I}_{\mathcal{B}} : \mathcal{M} \otimes \mathcal{E}\otimes \mathcal{B} \to  \mathcal{M} \otimes \mathcal{E}\otimes \mathcal{B}

    with
    
    .. math::
        \mathbf{\mathbf{V}}_{g_{in}} [\psi] = \sum_{g_{out}} \sigma_{\ell g_{out} g_{in},e} \psi_{g_{out}\ell e}
    
    '''
    cross_sections_indexed : CrossSectionsIndexed
    r'''The indexed cross sections to use.'''

    def gather_forward(self, x : jnp.ndarray, energy_group_out : int) -> jnp.ndarray:
        r'''
        Applies the gather-scatter operation for a specific energy group with shape checking.

        Parameters
        ---------------------------
        x : jnp.ndarray
            The input angular flux moments.
        energy_group_out : int
            The energy group index the scatter is going to (and gathered to)

        Returns
        ---------------------------
        jnp.ndarray
            The output discrete angles after applying the gather-scatter operation.
        '''
        self._check_x_shape(x)
        return self._gather_forward(x, energy_group_out)
    
    def gather_backward(self, x : jnp.ndarray, energy_group_in : int) -> jnp.ndarray:
        r'''
        Applies the transpose of the gather-scatter operation for a specific energy group with shape checking.

        Parameters
        ---------------------------
        x : jnp.ndarray
            The input discrete angles.
        energy_group : int
            The energy group index the scatter is coming from (and gathered to)

        Returns
        ---------------------------
        jnp.ndarray
            The output angular flux moments after applying the transpose of the gather-scatter operation.
        '''
        self._check_x_shape(x)
        return self._gather_backward(x, energy_group_in)

    @abstractmethod
    def _gather_forward(self, x : jnp.ndarray, energy_group_out : int) -> jnp.ndarray:
        r'''
        Applies the gather-scatter operation for a specific energy group.

        Parameters
        ---------------------------
        x : jnp.ndarray
            The input moments
        energy_group_out : int
            The energy group index the scatter is going to (and gathered to)

        Returns
        ---------------------------
        jnp.ndarray
            The output moments after applying the gather-scatter operation.
        '''
        ...

    @abstractmethod
    def _gather_backward(self, x : jnp.ndarray, energy_group_in : int) -> jnp.ndarray:
        r'''
        Applies the transpose of the gather-scatter operation for a specific energy group.

        Parameters
        ---------------------------
        x : jnp.ndarray
            The input moments
        energy_group : int
            The energy group index the scatter is coming from (and gathered to)

        Returns
        ---------------------------
        jnp.ndarray
            The output angular flux moments after applying the transpose of the gather-scatter operation.
        '''
        ...

    def _check_x_shape(self, x : jnp.ndarray):
        '''
        Checks shape such that we have the correct number of dimensions and elements.

        Required:

        - x must be 4D: [n_groups, n_moments, n_elements, n_basis]
        - number of energy groups in cross_sections_indexed must match first dimension of x
        - number of elements in x must match number of elements in cross_sections_indexed
        '''
        assert x.ndim == 4, f"Input x must be 4D [n_groups, n_moments, n_elements, n_basis], but has shape {x.shape}."
        self.cross_sections_indexed.cross_section_data.assert_n_energy_groups(x.shape[0])
        assert x.shape[-2] == self.cross_sections_indexed.indices.shape[0], f"Input x shape {x.shape} does not match number of elements in cross_sections_indexed {self.cross_sections_indexed.indices.shape[0]}."

    def to_scatter_operator(self, energy_group_out : int, energy_group_in : int, moments_elements_basis_shape : jax.ShapeDtypeStruct) -> "ScatterOperator":
        return ScatterOperator(self.cross_sections_indexed, moments_elements_basis_shape, energy_group_out, energy_group_in)
    
#====================================================================================================================================================================================================================
# Broadcasted implementation
#====================================================================================================================================================================================================================
class GatherScatterOperatorBroadcasted(GatherScatterOperator):
    r'''
    Implementation of the GatherScatterOperator using direct function calls.

    See :class:`GatherScatterOperator` for details.
    '''

    def _gather_forward(self, x : jnp.ndarray, energy_group_out : int) -> jnp.ndarray:
        return _forward_gathering_bc(energy_group_out, self.cross_sections_indexed, x)

    def _gather_backward(self, x : jnp.ndarray, energy_group_in : int) -> jnp.ndarray:
        return _backward_gathering_bc(energy_group_in, self.cross_sections_indexed, x)
    

def _extend_sigma_lgg_match_moments(sigma_l_gg : jnp.ndarray, n_moments : int) -> jnp.ndarray:
    r'''
    For a given element index and indexed cross section object, extracts the scattering cross section data and extends it to match the number of moments.

    Parameters
    ----------
    sigma_l_gg : jnp.ndarray [lmax + 1, n_energy_groups]
        The scattering cross section data for the specified element.
    n_moments : int
        The number of moments to extend/truncate to.

    Returns
    -------
    jnp.ndarray [n_moments, n_energy_groups, n_energy_groups]
        The extended scattering cross section data for the specified element.
    '''
    # Since this is a l expansion, not in terms of lm, we expand it locally to the full n_moments                                                                                            
    extended_sigma_lm_gg = sigma_l_gg[_lm_real_expansion(sigma_l_gg.shape[0] - 1)[0], :] # [n_moments, n_groups]    
    # now we have to pad/truncate to match moments_array    
    extended_sigma_lm_gg = extended_sigma_lm_gg[:n_moments, :]  #  first truncate if needed (if n_moments > extended_sigma_lm_gg.shape[0], this does nothing)
    extended_sigma_lm_gg = jnp.pad(extended_sigma_lm_gg, ((0, n_moments - extended_sigma_lm_gg.shape[0]), (0,0)))  # pad if needed    #[n_moments, n_groups]
    return extended_sigma_lm_gg  # [n_moments, n_groups]


def _bc_gather_forward_xs_indexed_kernel(elem_idx : int, g_out : int, cross_sections_indexed : CrossSectionsIndexed, moments_array : jnp.ndarray) -> jnp.ndarray:
    r'''
    Computes from a given element index, outgoing group, incoming group, indexed cross section object and buffer for the specific incoming group, the corresponding scattering scattering source:

    .. math::
        s^{g_{out}}_{lm}=  \sum_{g_{in}} \sigma_{l, g_{out}\from g_{in}}^{(e)} \psi_{lm}^{(e, g_{in})}

    Automatically pads or truncates the cross section data to match the number of moments in the moments_array.
        
    Parameters
    ----------
    elem_idx : int
        The element index.    
    g_in : int
        The incoming energy group index.
    cross_sections_indexed : CrossSectionsIndexed
        The indexed cross section data structure.
    moments_array : jnp.ndarray [n_elements, n_basis, n_moments]
        The moments array containing angular flux moments.
    
    Returns
    -------
    jnp.ndarray [n_basis, n_moments]
        The scattering source moments for the specified element and energy groups.
    '''
    assert moments_array.ndim == 4, f"moments_array must be 4D [n_groups, n_moments, n_elements, n_basis], but has shape {moments_array.shape}."    
    assert moments_array.shape[0] == cross_sections_indexed.n_energy_groups, f"moments_array first dimension {moments_array.shape[0]} must match number of energy groups in cross_sections_indexed {cross_sections_indexed.n_energy_groups}."
    
    xs_idx               = cross_sections_indexed.indices[elem_idx]
    sigma_l_gg           = cross_sections_indexed.cross_section_data.scattering[xs_idx, :, g_out, :] # [lmax + 1, n_groups]    
    n_moments            = moments_array.shape[1]    
    extended_sigma_lm_gg = _extend_sigma_lgg_match_moments(sigma_l_gg, n_moments)  # [n_moments, n_groups]    #
    return jnp.einsum("lg, gli->li", extended_sigma_lm_gg, moments_array[:, :, elem_idx, :])  # [n_moments, n_in_groups] * [n_in_groups, n_moments, n_basis] -> [n_moments, n_basis]    

@eqx.filter_jit
def _forward_gathering_bc(        
        g_out                       : int,
        cross_sections_indexed       : CrossSectionsIndexed,
        moments_array               : jnp.ndarray
    ) -> jnp.ndarray:
    r'''
    Computes the total scattering source for a specific outgoing energy group over all elements and basis functions:

    .. math::
        s^{g_{out}}_{lm e} = \sum_{g_{in}} \sigma_{l, g_{out}\from g_{in}}^{(e)} \psi_{lm}^{(e, g_{in})}

    Parameters
    ----------
    g_out : int
        The outgoing energy group index.
    cross_sections_indexed : CrossSectionsIndexed
        The indexed cross section data structure.   
    moments_array : jnp.ndarray [n_groups, n_moments, n_elements, n_basis]
        The moments array containing angular flux moments.
    Returns
    -------
    jnp.ndarray [n_moments, n_elements, n_basis]
        The total scattering source moments for the specified outgoing energy group.    
    '''
    return jax.vmap(
        _bc_gather_forward_xs_indexed_kernel,
        in_axes=(0, None, None, None),
        out_axes=1
    )(jnp.arange(cross_sections_indexed.indices.shape[0]), g_out, cross_sections_indexed, moments_array)


def _bc_gather_backward_xs_indexed_kernel(elem_idx : int, g_in : int, cross_sections_indexed : CrossSectionsIndexed, moments_array : jnp.ndarray) -> jnp.ndarray:
    r'''
    Backwards gathering:

    .. math::
        s_{g_{in}} = \sum_{g_{out}} \sigma_{\ell g_{out} g_{in}} \zeta_{g_{out}\ell e}
    '''
    assert moments_array.ndim == 4, f"moments_array must be 4D [n_groups, n_moments, n_elements, n_basis], but has shape {moments_array.shape}."    
    assert moments_array.shape[0] == cross_sections_indexed.n_energy_groups, f"moments_array first dimension {moments_array.shape[0]} must match number of energy groups in cross_sections_indexed {cross_sections_indexed.n_energy_groups}."
    
    xs_idx     = cross_sections_indexed.indices[elem_idx]
    sigma_l_gg = cross_sections_indexed.cross_section_data.scattering[xs_idx, :, :, g_in] #[lmax+1, n_groups]

    extended_sigma_lm_gg = _extend_sigma_lgg_match_moments(sigma_l_gg, moments_array.shape[1])  # [n_moments, n_groups]
    return jnp.einsum("lg, gli->li", extended_sigma_lm_gg, moments_array[:, :, elem_idx, :])  # [n_moments, n_in_groups] * [n_in_groups, n_moments, n_basis] -> [n_moments, n_basis]

@eqx.filter_jit
def _backward_gathering_bc(        
        g_in                        : int,
        cross_sections_indexed       : CrossSectionsIndexed,
        moments_array               : jnp.ndarray
    ) -> jnp.ndarray:
    r'''
    Backwards gathering:

    .. math::
        s_{g_{in}} = \sum_{g_{out}} \sigma_{\ell g_{out} g_{in}} \zeta_{g_{out}\ell e}
    '''
    return jax.vmap(
        _bc_gather_backward_xs_indexed_kernel,
        in_axes=(0, None, None, None),
        out_axes=1
    )(jnp.arange(cross_sections_indexed.indices.shape[0]), g_in, cross_sections_indexed, moments_array)

