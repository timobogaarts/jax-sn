import jax
import jax.numpy as jnp
from functools import partial
from dataclasses import dataclass
from typing import Callable
from abc import abstractmethod
from jax_sn.quadrature_set import QuadratureSet, QuadratureSetBase
import lineax as lx
from jax_sn.operators.base_linear_operator import AbstractLinearOperator_sn
import equinox as eqx

class MomentOperatorBase(AbstractLinearOperator_sn):
    r'''
    Abstract base class for the moment operator that maps angular fluxes to moments.

    Direction handling
    ---------------------------
    Whether this operator applies the forward action or its transpose is carried by the
    boolean field :attr:`transpose_bool`, *not* by a separate subclass. This keeps the
    transpose axis orthogonal to any future implementation axis (batched matrix, streaming,
    ...): a concrete implementation subclass only needs to provide :meth:`_forward_mv` and
    :meth:`_transpose_mv`, while this base class handles dispatch, transposition (a cheap
    flag flip via :func:`equinox.tree_at`, no reconstruction) and the in/out shaping.

    Mathematical description
    ---------------------------

    Implements the following operation (the tensor dot indicates that it operates on the leading dimension, with no coupling in the trailing dimensions):

    .. math::
        M  = \overline{\mathbf{M}} \otimes \mathbf{I}_F: \mathcal{D}\otimes \mathcal{F} \rightarrow \mathcal{M}\otimes \mathcal{F}

    with

    .. math::
        \overline{\mathbf{M}}[\{a_k\}] = \sum_{k=1}^{N_{angles}} a_k w_k Y_{lm}(\Omega_k)

    where :math:`\mathcal{D}` is the discrete angular space of size :math:`N_{angles}`, :math:`\mathcal{M}` is the moment space of size :math:`(L+1)^2` (for moment order L), :math:`F` is the FEM basis function space of size :math:`N_{elements} \times N_{basis}`,
    :math:`a_k` is the angular flux for angle :math:`k`, :math:`w_k` is the weight for angle :math:`k` and :math:`Y_{lm}(\Omega_k)` is the spherical harmonic of order :math:`(l,m)` evaluated at direction :math:`\Omega_k`. For the latter definition, see :attr:`jax_sn.quadrature_set.QuadratureSet.wk_ylm_real`.

    Transposition
    ---------------------------

    Its transposition is defined as (transpose again does not act on the :math:`F` space):

    .. math::
        M^T  = \overline{\mathbf{M}}^T \otimes \mathbf{I}_F: \mathcal{M}\otimes \mathcal{F} \rightarrow \mathcal{D}\otimes \mathcal{F}

    with

    .. math::
        \overline{\mathbf{M}}^T[\{ b_{lm} \}] = \sum_{lm} b_{lm} w_k Y_{lm}(\Omega_k)

    This is *not*  the same as the discrete-to-moment operator, which is defined without the weights. Mathematically, the transpose of this operator is w.r.t to the standard inner product, while the discrete-to-moment operator is defined w.r.t. a weighted inner product (with weights being the quadrature weights).

    '''

    wk_yklm              : jnp.ndarray           #[n_angles, n_moments]
    r'''Matrix of [n_angles, n_moments] containing :math:`w_k Y_{l,m}(\Omega_k)` values for each angle :math:`k` and moment :math:`(l,m)` (collapsed into a single index).'''

    quadrature_set       : QuadratureSetBase
    r'''The quadrature set associated with this moment operator.'''

    elements_basis_shape : jax.ShapeDtypeStruct  #[n_elements, n_basis]
    r'''The shape of the elements and basis functions over which this operator is batched. Identified with :math:`F` in the documentation above.'''

    transpose_bool       : bool = False
    r'''Whether this operator applies the transposed action (moments to angles) instead of the forward action (angles to moments).'''

    @classmethod
    def from_quadrature_set(cls, quadrature_set : QuadratureSetBase, l_max : int, elements_basis_shape : jax.ShapeDtypeStruct) -> "MomentOperatorBase":
        r'''
        Create a moment operator from a quadrature set and desired moment order. This computes the :math:`w_k Y_{l,m}(\Omega_k)` weights for the specified moment order.
        Prefer to initialise directly if possible.


        :param quadrature_set:
            The quadrature set to use for the moment operator.
        :param l_max:
            The maximum moment order L to use.
        :param elements_basis_shape:
            The shape of the elements and basis functions over which this operator is batched. Identified with :math:`F` in the documentation above.


        '''
        wk_yklm = quadrature_set.wk_ylm_real(l_max)
        return cls(wk_yklm, quadrature_set, elements_basis_shape)

    @abstractmethod
    def _forward_mv(self, x : jnp.ndarray) -> jnp.ndarray:
        '''Apply the forward action (angles to moments). Implemented by concrete subclasses.'''
        ...

    @abstractmethod
    def _transpose_mv(self, x : jnp.ndarray) -> jnp.ndarray:
        '''Apply the transposed action (moments to angles). Implemented by concrete subclasses.'''
        ...

    def mv(self, x : jnp.ndarray) -> jnp.ndarray:
        if self.transpose_bool:
            return self._transpose_mv(x)
        return self._forward_mv(x)

    def transpose(self) -> "MomentOperatorBase":
        '''Returns the transposed operator by flipping :attr:`transpose_bool`. There is no setup or reconstruction, so this operation is essentially free.'''
        return eqx.tree_at(lambda op: op.transpose_bool, self, not self.transpose_bool)

    def _forward_in_structure(self):
        '''Forward-direction input structure representing discrete angles. Shape [n_angles, ...].'''
        return jax.ShapeDtypeStruct(
            shape=(self.wk_yklm.shape[0],) + self.elements_basis_shape.shape,
            dtype=self.wk_yklm.dtype)

    def _forward_out_structure(self):
        '''Forward-direction output structure representing moments. Shape [n_moments, ...].'''
        return jax.ShapeDtypeStruct(
            shape=(self.wk_yklm.shape[1],) + self.elements_basis_shape.shape,
            dtype=self.wk_yklm.dtype)

    def in_structure(self):
        '''Input structure. Discrete angles in the forward direction, moments in the transposed direction.'''
        return self._forward_out_structure() if self.transpose_bool else self._forward_in_structure()

    def out_structure(self):
        '''Output structure. Moments in the forward direction, discrete angles in the transposed direction.'''
        return self._forward_in_structure() if self.transpose_bool else self._forward_out_structure()

    @property
    def n_moments(self):
        '''Number of moments (= :math:`(L+1)^2` for moment order L)'''
        return self.wk_yklm.shape[1]

    @property
    def n_angles(self):
        '''Number of discrete angles'''
        return self.wk_yklm.shape[0]


@lx.is_symmetric.register(MomentOperatorBase)
def _(operator):
    return False

@lx.linearise.register(MomentOperatorBase)
def _(operator):
    return operator


class MomentOperator(MomentOperatorBase):
    r'''
    Batched-matrix implementation of the moment operator (see :class:`MomentOperatorBase`).

    This operator batches over the trailing dimensions. Although this is inefficient
    (the moment operator sums over the angles, which is the leading dimension and thus not contiguous in memory)
    it is more convenient for the sweep operation, which requires the angles to be the leading dimension (since each angular sweep is independent, we want contiguous memory for the trailing n_elements, n_basis dimensions).
    '''

    def _forward_mv(self, x : jnp.ndarray) -> jnp.ndarray:
        return _call_batched_moment_operator(x, self.wk_yklm)

    def _transpose_mv(self, x : jnp.ndarray) -> jnp.ndarray:
        return _call_batched_moment_operator_transposed(x, self.wk_yklm)


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


def _call_batched_moment_operator_transposed(x : jnp.ndarray, wk_yklm : jnp.ndarray) -> jnp.ndarray:
    r'''
    Apply the transposed moment operator to the angles to compute the angular flux buffer.

    Computes:

    .. math::
        \psi_k^{ei} = \sum_{l,m} x_{l,m}^{ei} w_{k} Y_{l,m}(\Omega_k)

    with :math:`\psi_k^{ei}` the angular flux for angle :math:`k` in element :math:`e` and basis function :math:`i`, :math:`x_{l,m}^{ei}` the moment of order :math:`(l,m)` in element :math:`e` and basis function :math:`i`,
    :math:`w_k` the weight for angle :math:`k` and :math:`Y_{l,m}(\Omega_k)` the spherical harmonic of order :math:`(l,m)` evaluated at direction :math:`\Omega_k`. For the latter definition,
    see `jax_sn.quadrature_set.QuadratureSet.wk_ylm_real`.

    This does *not* invert the transformation performed by the moment operator, since the weights are included here as well.

    Parameters:
    ----------
    x : jnp.ndarray [(L+1)**2, n_elements, n_basis]
        The angular flux buffer containing the angular fluxes for each angle and each element and its basis functions
    wk_yklm : jnp.ndarray [n_angles, (L+1)**2]
        The moment operator weights mapping angles to moments for the specific desired order L.

    Returns:
    -------
    moments : jnp.ndarray [n_angles, n_elements, n_basis]
        The computed moments.

    '''
    assert x.shape[0] == wk_yklm.shape[-1], f"Number of moments in x ({x.shape[0]}) must match number of moments in wk_yklm ({wk_yklm.shape[-1]})."
    return jnp.einsum('l...,kl->k...', x, wk_yklm)


@eqx.filter_jit
def _call_streaming_moment_operator(buffer_sweep : jnp.ndarray, wk_yklm : jnp.ndarray):
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
    out0 = jnp.zeros((wk_yklm.shape[1],) + buffer_sweep.shape[1:], dtype=buffer_sweep.dtype)
    def body(out, k):
        wk_broadcast = wk_yklm[k].reshape((wk_yklm.shape[1],) + (1,) * (buffer_sweep.ndim - 1))
        contrib = wk_broadcast * buffer_sweep[k, None, ...]
        return out + contrib, None

    out, _ = jax.lax.scan(body, out0, jnp.arange(buffer_sweep.shape[0]))
    return out

@eqx.filter_jit
def _call_streaming_moment_operator_transposed(buffer_moments : jnp.ndarray, wk_yklm : jnp.ndarray):
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
    out0 = jnp.zeros((wk_yklm.shape[0],) + buffer_moments.shape[1:], dtype=buffer_moments.dtype)
    def body(out, l):
        wk_broadcast = wk_yklm[:,l].reshape((wk_yklm.shape[0],) + (1,) * (buffer_moments.ndim - 1))
        contrib = wk_broadcast * buffer_moments[l, None, ...]
        return out + contrib, None

    out, _ = jax.lax.scan(body, out0, jnp.arange(buffer_moments.shape[0]))
    return out
