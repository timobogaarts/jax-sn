r'''
Strategy objects for the moment and discrete operators.

Both operators are a single contraction of the angle axis against a harmonics matrix
(``wk_yklm`` for the moment operator, ``yklm`` for the discrete operator). The *implementation* of
that contraction is selected by a strategy object rather than by subclassing, mirroring the
transport operator:

- :class:`MomentStrategy`   -- how to contract angles <-> moments with the weighted harmonics.
- :class:`DiscreteStrategy` -- how to contract moments <-> angles with the unweighted harmonics.

Each strategy implements one ``forward`` / ``transpose`` pair. A new implementation (e.g. a
streaming variant that trades compute for memory) is a single small class, not a subclass of the
operator. The kernels live here next to the strategies that dispatch to them, so the operator
modules depend only on the (array-free) strategy classes.
'''

import jax
import jax.numpy as jnp
from abc import abstractmethod
from typing import NamedTuple

import equinox as eqx

from jax_sn.operators.base_linear_operator import Strategy


class MomentData(NamedTuple):
    wk_yklm: jnp.ndarray


class DiscreteData(NamedTuple):
    yklm: jnp.ndarray


# ---------------------------------------------------------------------------
# Moment strategies (angles <-> moments, weighted harmonics ``wk_yklm``)
# ---------------------------------------------------------------------------
class MomentStrategy(Strategy):
    '''How to apply the moment contraction. Reads ``data.wk_yklm``.'''

    @abstractmethod
    def forward(self, data: MomentData, x):
        ...

    @abstractmethod
    def transpose(self, data: MomentData, x):
        ...


class BatchedMoment(MomentStrategy):
    r'''
    Batched-matrix (einsum) implementation. Batches over the trailing dimensions. Although this sums
    over the angles -- the leading, non-contiguous dimension -- it is convenient for the sweep, which
    wants the angles leading and the trailing ``(n_elements, n_basis)`` contiguous.
    '''

    def forward(self, data: MomentData, x):
        return _call_batched_moment_operator(x, data.wk_yklm)

    def transpose(self, data: MomentData, x):
        return _call_batched_moment_operator_transposed(x, data.wk_yklm)


class StreamingMoment(MomentStrategy):
    r'''
    Streaming (scan-over-angles) implementation. Accumulates the angle contributions one angle at a
    time, trading compute for a smaller peak memory footprint than the batched einsum.
    '''

    def forward(self, data: MomentData, x):
        return _call_streaming_moment_operator(x, data.wk_yklm)

    def transpose(self, data: MomentData, x):
        return _call_streaming_moment_operator_transposed(x, data.wk_yklm)


# ---------------------------------------------------------------------------
# Discrete strategies (moments <-> angles, unweighted harmonics ``yklm``)
# ---------------------------------------------------------------------------
class DiscreteStrategy(Strategy):
    '''How to apply the discrete (moment-to-angle) contraction. Reads ``data.yklm``.'''

    @abstractmethod
    def forward(self, data: DiscreteData, x):
        ...

    @abstractmethod
    def transpose(self, data: DiscreteData, x):
        ...


class BatchedDiscrete(DiscreteStrategy):
    '''Batched-matrix (einsum) implementation of the discrete operator.'''

    def forward(self, data: DiscreteData, x):
        return _moment_to_discrete_mv(data.yklm, x)

    def transpose(self, data: DiscreteData, x):
        return _moment_to_discrete_transpose_mv(data.yklm, x)


# ---------------------------------------------------------------------------
# Moment kernels
# ---------------------------------------------------------------------------
def _call_batched_moment_operator(buffer_sweep: jnp.ndarray, wk_yklm: jnp.ndarray) -> jnp.ndarray:
    r'''
    Apply the moment operator to the angular flux buffer to compute the moments.

    Computes:

    .. math::
        \phi_{l,m}^{ei} = \sum_{a=1}^{N_{angles}} \psi_k^{ei} w_{k} Y_{l,m}(\Omega_k)

    with :math:`\phi_{l,m}^{ei}` the moment of order :math:`(l,m)` in element :math:`e` and basis function :math:`i`,
    :math:`\psi_k^{ei}` the angular flux for angle :math:`k` in element :math:`e` and basis function :math:`i`, :math:`w_k` the weight for angle :math:`k`
    and :math:`Y_{l,m}(\Omega_k)` the spherical harmonic of order :math:`(l,m)` evaluated at direction :math:`\Omega_k`. For the latter definition,
    see `jax_sn.quadrature_set.QuadratureSet.wk_ylm_real`.

    Parameters
    ----------
    buffer_sweep : jnp.ndarray [n_angles, n_elements, n_basis]
        The angular flux buffer containing the angular fluxes for each angle and each element and its basis functions
    wk_yklm : jnp.ndarray [n_angles, (L+1)**2]
        The moment operator weights mapping angles to moments for the specific desired order L.

    Returns
    -------
    moments : jnp.ndarray [n_moments, n_elements, n_basis]
        The computed moments.
    '''
    assert buffer_sweep.shape[0] == wk_yklm.shape[0], f"Number of angles in buffer_sweep ({buffer_sweep.shape[0]}) must match number of angles in wk_yklm ({wk_yklm.shape[0]})."
    return jnp.einsum('k...,kl->l...', buffer_sweep, wk_yklm)


def _call_batched_moment_operator_transposed(x: jnp.ndarray, wk_yklm: jnp.ndarray) -> jnp.ndarray:
    r'''
    Apply the transposed moment operator to the moments to compute the angular flux buffer.

    Computes:

    .. math::
        \psi_k^{ei} = \sum_{l,m} x_{l,m}^{ei} w_{k} Y_{l,m}(\Omega_k)

    with :math:`\psi_k^{ei}` the angular flux for angle :math:`k` in element :math:`e` and basis function :math:`i`, :math:`x_{l,m}^{ei}` the moment of order :math:`(l,m)` in element :math:`e` and basis function :math:`i`,
    :math:`w_k` the weight for angle :math:`k` and :math:`Y_{l,m}(\Omega_k)` the spherical harmonic of order :math:`(l,m)` evaluated at direction :math:`\Omega_k`. For the latter definition,
    see `jax_sn.quadrature_set.QuadratureSet.wk_ylm_real`.

    This does *not* invert the transformation performed by the moment operator, since the weights are included here as well.

    Parameters
    ----------
    x : jnp.ndarray [(L+1)**2, n_elements, n_basis]
        The moments for each element and its basis functions.
    wk_yklm : jnp.ndarray [n_angles, (L+1)**2]
        The moment operator weights mapping angles to moments for the specific desired order L.

    Returns
    -------
    moments : jnp.ndarray [n_angles, n_elements, n_basis]
        The computed angular fluxes.
    '''
    assert x.shape[0] == wk_yklm.shape[-1], f"Number of moments in x ({x.shape[0]}) must match number of moments in wk_yklm ({wk_yklm.shape[-1]})."
    return jnp.einsum('l...,kl->k...', x, wk_yklm)


@eqx.filter_jit
def _call_streaming_moment_operator(buffer_sweep: jnp.ndarray, wk_yklm: jnp.ndarray) -> jnp.ndarray:
    r'''
    Streaming (scan-over-angles) variant of :func:`_call_batched_moment_operator`. Accumulates the
    per-angle contributions to the moments one angle at a time, trading compute for memory.

    Parameters
    ----------
    buffer_sweep : jnp.ndarray [n_angles, n_elements, n_basis]
        The angular flux buffer containing the angular fluxes for each angle and each element and its basis functions
    wk_yklm : jnp.ndarray [n_angles, (L+1)**2]
        The moment operator weights mapping angles to moments for the specific desired order L.

    Returns
    -------
    moments : jnp.ndarray [n_moments, n_elements, n_basis]
        The computed moments.
    '''
    out0 = jnp.zeros((wk_yklm.shape[1],) + buffer_sweep.shape[1:], dtype=buffer_sweep.dtype)
    def body(out, k):
        wk_broadcast = wk_yklm[k].reshape((wk_yklm.shape[1],) + (1,) * (buffer_sweep.ndim - 1))
        contrib = wk_broadcast * buffer_sweep[k, None, ...]
        return out + contrib, None

    out, _ = jax.lax.scan(body, out0, jnp.arange(buffer_sweep.shape[0]))
    return out


@eqx.filter_jit
def _call_streaming_moment_operator_transposed(buffer_moments: jnp.ndarray, wk_yklm: jnp.ndarray) -> jnp.ndarray:
    r'''
    Streaming (scan-over-moments) variant of :func:`_call_batched_moment_operator_transposed`.
    Accumulates the per-moment contributions to the angular fluxes one moment at a time.

    Parameters
    ----------
    buffer_moments : jnp.ndarray [(L+1)**2, n_elements, n_basis]
        The moments for each element and its basis functions.
    wk_yklm : jnp.ndarray [n_angles, (L+1)**2]
        The moment operator weights mapping angles to moments for the specific desired order L.

    Returns
    -------
    moments : jnp.ndarray [n_angles, n_elements, n_basis]
        The computed angular fluxes.
    '''
    out0 = jnp.zeros((wk_yklm.shape[0],) + buffer_moments.shape[1:], dtype=buffer_moments.dtype)
    def body(out, l):
        wk_broadcast = wk_yklm[:, l].reshape((wk_yklm.shape[0],) + (1,) * (buffer_moments.ndim - 1))
        contrib = wk_broadcast * buffer_moments[l, None, ...]
        return out + contrib, None

    out, _ = jax.lax.scan(body, out0, jnp.arange(buffer_moments.shape[0]))
    return out


# ---------------------------------------------------------------------------
# Discrete kernels
# ---------------------------------------------------------------------------
@eqx.filter_jit
def _moment_to_discrete_mv(yklm: jnp.ndarray, x) -> jnp.ndarray:
    r'''
    D operator implementation:

    .. math::
        s_{ke} = \sum_{\ell} Y_{k \ell } \psi_{\ell e}

    Parameters
    ----------
    yklm : jnp.ndarray
        The yklm matrix of shape [n_angles, n_moments].
    x : jnp.ndarray
        The input vector of shape [n_moments, ...].

    Returns
    -------
    jnp.ndarray
        The output vector of shape [n_angles, ...].
    '''
    return jnp.einsum('kl, l... -> k...', yklm, x)


@eqx.filter_jit
def _moment_to_discrete_transpose_mv(yklm: jnp.ndarray, x) -> jnp.ndarray:
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
