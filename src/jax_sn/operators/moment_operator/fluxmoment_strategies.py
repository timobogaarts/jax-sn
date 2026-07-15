import jax
import jax.numpy as jnp

import equinox as eqx

from .moment_discrete_operator_base import FluxMomentStrategy, DiscreteMomentData


class BatchedFluxMoment(FluxMomentStrategy):
    '''Batched-matrix (einsum) implementation. Batches over the trailing dimensions.'''

    def contract(self, data: DiscreteMomentData, x, axis: int):
        return _batched_contract(data.matrix, x, axis)


class StreamingFluxMoment(FluxMomentStrategy):
    r'''
    Streaming (scan-over-the-contracted-axis) implementation. Accumulates the contributions one slice
    at a time, trading compute for a smaller peak memory footprint than the batched einsum.
    '''

    def contract(self, data: DiscreteMomentData, x, axis: int):
        return _streaming_contract(data.matrix, x, axis)


# ---------------------------------------------------------------------------
# Kernels
#
# ``matrix`` is [n_angles, n_moments] (indices ``k``, ``l``). ``axis`` selects which of the two the
# contraction consumes, so a single kernel covers both directions of both operators:
#
#   axis=0   [n_angles, ...]  -> [n_moments, ...]   M forward     / D transpose
#   axis=1   [n_moments, ...] -> [n_angles, ...]    M transpose   / D forward
# ---------------------------------------------------------------------------
def _check_contract(matrix: jnp.ndarray, x: jnp.ndarray, axis: int):
    assert axis in (0, 1), f"axis must be 0 (angles) or 1 (moments), but got {axis}."
    assert x.shape[0] == matrix.shape[axis], (
        f"Leading dimension of x ({x.shape[0]}) must match axis {axis} of the harmonics matrix "
        f"({matrix.shape[axis]}).")


@eqx.filter_jit
def _batched_contract(matrix: jnp.ndarray, x: jnp.ndarray, axis: int) -> jnp.ndarray:
    r'''
    Contract ``x`` over ``axis`` of the harmonics matrix in a single einsum.

    For ``axis=0`` this computes

    .. math::
        \phi_{l,m}^{ei} = \sum_{k=1}^{N_{angles}} \psi_k^{ei} Y_{l,m}(\Omega_k)

    and for ``axis=1``

    .. math::
        \psi_k^{ei} = \sum_{l,m} \phi_{l,m}^{ei} Y_{l,m}(\Omega_k)

    where :math:`Y_{l,m}(\Omega_k)` is the harmonics matrix supplied as ``matrix`` -- weighted
    (:math:`w_k Y_{lm}`) for the moment operator, unweighted for the discrete operator. The trailing
    dimensions :math:`ei` (elements, basis functions) are batched over and never coupled.

    Parameters
    ----------
    matrix : jnp.ndarray [n_angles, (L+1)**2]
        The harmonics matrix, see :attr:`DiscreteMomentData.matrix`.
    x : jnp.ndarray [n_angles or n_moments, ...]
        The input, whose leading dimension matches ``axis`` of ``matrix``.
    axis : int
        Axis of ``matrix`` to contract over: 0 = angles, 1 = moments.

    Returns
    -------
    jnp.ndarray [n_moments or n_angles, ...]
        The contracted output, leading dimension ``matrix.shape[1 - axis]``.
    '''
    _check_contract(matrix, x, axis)
    if axis == 0:
        return jnp.einsum('k...,kl->l...', x, matrix)
    return jnp.einsum('l...,kl->k...', x, matrix)


@eqx.filter_jit
def _streaming_contract(matrix: jnp.ndarray, x: jnp.ndarray, axis: int) -> jnp.ndarray:
    r'''
    Streaming variant of :func:`_batched_contract`: scans over the contracted axis, accumulating one
    slice's contribution at a time rather than materialising the full batched product.

    Parameters
    ----------
    matrix : jnp.ndarray [n_angles, (L+1)**2]
        The harmonics matrix, see :attr:`DiscreteMomentData.matrix`.
    x : jnp.ndarray [n_angles or n_moments, ...]
        The input, whose leading dimension matches ``axis`` of ``matrix``.
    axis : int
        Axis of ``matrix`` to contract over: 0 = angles, 1 = moments.

    Returns
    -------
    jnp.ndarray [n_moments or n_angles, ...]
        The contracted output, leading dimension ``matrix.shape[1 - axis]``.
    '''
    _check_contract(matrix, x, axis)
    n_out = matrix.shape[1 - axis]
    out0  = jnp.zeros((n_out,) + x.shape[1:], dtype=x.dtype)

    def body(out, i):
        slice_broadcast = jnp.take(matrix, i, axis=axis).reshape((n_out,) + (1,) * (x.ndim - 1))
        return out + slice_broadcast * x[i, None, ...], None

    out, _ = jax.lax.scan(body, out0, jnp.arange(x.shape[0]))
    return out
