from typing import NamedTuple

import jax
import jax.numpy as jnp
import equinox as eqx

from jax_sn.domain.cross_sections import CrossSectionsIndexed
from jax_sn.operators.base_linear_operator import Strategy
from jax_sn.quadrature_set.quadrature_set_base import _lm_real_expansion


class ScatterData(NamedTuple):
    energy_group_out       : int
    energy_group_in        : int
    cross_sections_indexed : CrossSectionsIndexed


class ScatterStrategy(Strategy):
    '''How to apply the (symmetric-shaped) scatter contraction. Reads ``data``.'''


class IndexedScatter(ScatterStrategy):
    def forward(self, data: ScatterData, x):
        return total_scatter_kernel_xs_indexed(
            data.energy_group_out, data.energy_group_in, data.cross_sections_indexed, x
        )


def _scatter_kernel_xs_indexed_kernel(elem_idx : int, g_out : int, g_in : int, cross_sections_indexed : CrossSectionsIndexed, moments_array : jnp.ndarray) -> jnp.ndarray:
    r'''
    From an element index, outgoing/incoming group, indexed cross sections and the moments buffer for
    the incoming group, the scattering source :math:`s^{g_{out}}_{lm} = \sigma_{l, g_{out}\from g_{in}}^{(e)} \psi_{lm}^{(e, g_{in})}`.
    Pads/truncates the cross-section :math:`l`-expansion to the number of moments in ``moments_array``.
    '''
    assert moments_array.ndim == 3, f"moments_array must be 3D [n_moments, n_elements, n_basis], but has shape {moments_array.shape}."

    xs_idx = cross_sections_indexed.indices[elem_idx]
    sigma_l_gg = cross_sections_indexed.cross_section_data.scattering[xs_idx, :, g_out, g_in]  # [lmax + 1]
    # l-expansion (not lm), expand locally to n_moments
    extended_sigma_lm_gg = sigma_l_gg[_lm_real_expansion(sigma_l_gg.shape[0] - 1)[0]]  # [n_moments]
    n_moments = moments_array.shape[0]
    extended_sigma_lm_gg = extended_sigma_lm_gg[:n_moments]  # truncate if needed
    extended_sigma_lm_gg = jnp.pad(extended_sigma_lm_gg, (0, n_moments - extended_sigma_lm_gg.shape[0]))  # pad if needed
    return extended_sigma_lm_gg[:, None] * moments_array[:, elem_idx, :]  # [n_moments, n_basis]


@eqx.filter_jit
def total_scatter_kernel_xs_indexed(g_out : int, g_in : int, cross_sections_indexed : CrossSectionsIndexed, moments_array : jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(_scatter_kernel_xs_indexed_kernel, in_axes=(0, None, None, None, None), out_axes=1)(jnp.arange(cross_sections_indexed.indices.shape[0]), g_out, g_in, cross_sections_indexed, moments_array)
