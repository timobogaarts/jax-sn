r'''
Concrete gather-scatter strategies, mirroring the transport pattern (per-angle kernel + concrete vmap lift):

- :class:`BroadcastedSingleGroupGather` -- the per-group kernel (a :class:`SingleGroupGatherScatterStrategy`),
  one operation for both directions; ``sum_over_outgoing`` only selects which group axis of
  :math:`\sigma_{g_{out}g_{in}}` is contracted.
- :class:`VmapGatherScatterStrategy` -- a concrete :class:`ComposedGatherScatterStrategy` that lifts the
  single-group block over the group axis by ``vmap`` (the group-axis analogue of ``VmapAnglesForward``).
'''

import jax
import jax.numpy as jnp
import equinox as eqx

from jax_sn.quadrature_set.quadrature_set_base import _lm_real_expansion
from .gather_scatter_operator_base import (
    GatherScatterData, SingleGroupGatherScatterStrategy, ComposedGatherScatterStrategy,
)


# ==== per-group kernel (broadcasted over elements) ===================================================
def _extend_sigma_lgg_match_moments(sigma_l_gg : jnp.ndarray, n_moments : int) -> jnp.ndarray:
    '''Expand an ``[lmax+1, n_groups]`` l-expansion to ``[n_moments, n_groups]`` (lm expansion, pad/truncate).'''
    extended = sigma_l_gg[_lm_real_expansion(sigma_l_gg.shape[0] - 1)[0], :]        # [n_moments_full, n_groups]
    extended = extended[:n_moments, :]                                              # truncate
    return jnp.pad(extended, ((0, n_moments - extended.shape[0]), (0, 0)))          # pad


def _gather_group_element_kernel(elem_idx : int, g : int, sum_over_outgoing : bool, cross_sections_indexed, moments_array : jnp.ndarray) -> jnp.ndarray:
    r'''One element's contribution to the single-group gather for group ``g``: fix ``g``, contract
    :math:`\sigma` and the moments over the other group axis. ``sum_over_outgoing`` selects which axis of
    :math:`\sigma_{g_{out}g_{in}}` is ``g`` (``False``: gather over incoming; ``True``: over outgoing).'''
    assert moments_array.ndim == 4, f"moments_array must be 4D [n_groups, n_moments, n_elements, n_basis], but has shape {moments_array.shape}."
    xs_idx    = cross_sections_indexed.indices[elem_idx]
    scat      = cross_sections_indexed.cross_section_data.scattering[xs_idx]        # [lmax+1, g_out, g_in]
    sigma_l_g = scat[:, :, g] if sum_over_outgoing else scat[:, g, :]               # [lmax+1, n_other_groups]
    extended  = _extend_sigma_lgg_match_moments(sigma_l_g, moments_array.shape[1])  # [n_moments, n_groups]
    return jnp.einsum("lg, gli->li", extended, moments_array[:, :, elem_idx, :])    # [n_moments, n_basis]


@eqx.filter_jit
def _gather_group_bc(g : int, sum_over_outgoing : bool, cross_sections_indexed, moments_array : jnp.ndarray) -> jnp.ndarray:
    '''Single-group gather for group ``g`` over all elements: ``[G,M,E,B] -> [M,E,B]``.'''
    return jax.vmap(_gather_group_element_kernel, in_axes=(0, None, None, None, None), out_axes=1)(
        jnp.arange(cross_sections_indexed.indices.shape[0]), g, sum_over_outgoing, cross_sections_indexed, moments_array)


class BroadcastedSingleGroupGather(SingleGroupGatherScatterStrategy):
    '''Broadcasted (vmap-over-elements) single-group gather kernel.'''

    def gather_group(self, data : GatherScatterData, x : jnp.ndarray, g : int, sum_over_outgoing : bool) -> jnp.ndarray:
        return _gather_group_bc(g, sum_over_outgoing, data.cross_sections_indexed, x)


# ==== concrete full strategy: lift the single-group block over groups by vmap =========================
class VmapGatherScatterStrategy(ComposedGatherScatterStrategy):
    '''Full :math:`\\mathbf{U}` / :math:`\\mathbf{U}^T` by vmapping the single-group block over the group axis.'''

    def forward(self, data : GatherScatterData, x : jnp.ndarray) -> jnp.ndarray:
        return self._vmap_over_groups(self.single_group_strategy.incoming_gather, data, x)

    def transpose(self, data : GatherScatterData, x : jnp.ndarray) -> jnp.ndarray:
        return self._vmap_over_groups(self.single_group_strategy.outgoing_gather, data, x)

    @staticmethod
    def _vmap_over_groups(gather_fn, data : GatherScatterData, x : jnp.ndarray) -> jnp.ndarray:
        n_groups = data.cross_sections_indexed.n_energy_groups
        return jax.vmap(lambda g: gather_fn(data, x, g))(jnp.arange(n_groups))
