import jax
import jax.numpy as jnp
from jax_sn.quadrature_set import QuadratureSetBase
from jax_sn.operators.base_linear_operator import TransposableStrategy, TransposeBoolDataStrategyLinearOperator
import equinox as eqx
from abc import abstractmethod
from typing import ClassVar


class DiscreteMomentData(eqx.Module):
    r'''
    A real-spherical-harmonics matrix of shape ``[n_angles, n_moments]``, plus the axis its forward
    action consumes. The matrix layout is the same for the moment and discrete operators; only
    :attr:`in_axis` distinguishes them, which is what lets both share one strategy hierarchy.
    '''

    in_axis : eqx.AbstractClassVar[int]
    '''Axis of :attr:`matrix` the forward action contracts over: 0 = angles, 1 = moments.'''

    @property
    @abstractmethod
    def matrix(self) -> jnp.ndarray:
        '''The harmonics matrix [n_angles, n_moments].'''
        ...

    @classmethod
    @abstractmethod
    def from_quadrature_set(cls, quadrature_set : QuadratureSetBase, l_max : int):
        r'''
        Computes wk_yklm or yklm from the quadrature set. Prefer to initialise directly.
        '''
        ...

    @property
    def n_angles(self):
        '''Number of discrete angles.'''
        return self.matrix.shape[0]

    @property
    def n_moments(self):
        r'''Number of moments (= :math:`(L+1)^2` for moment order L).'''
        return self.matrix.shape[1]


class MomentData(DiscreteMomentData):
    r'''Weighted harmonics :math:`w_k Y_{lm}(\Omega_k)`. Applied angles :math:`\to` moments.'''

    wk_yklm : jnp.ndarray

    in_axis : ClassVar[int] = 0

    @property
    def matrix(self):
        return self.wk_yklm

    @classmethod
    def from_quadrature_set(cls, quadrature_set : QuadratureSetBase, l_max : int):
        return cls(quadrature_set.wk_ylm_real(l_max))


class DiscreteData(DiscreteMomentData):
    r'''Unweighted harmonics :math:`Y_{lm}(\Omega_k)`. Applied moments :math:`\to` angles.'''

    yklm : jnp.ndarray

    in_axis : ClassVar[int] = 1

    @property
    def matrix(self):
        return self.yklm

    @classmethod
    def from_quadrature_set(cls, quadrature_set : QuadratureSetBase, l_max : int):
        return cls(quadrature_set.ylm_real(l_max))


class FluxMomentStrategy(TransposableStrategy):    
    @abstractmethod
    def contract(self, data : DiscreteMomentData, x, axis : int):
        '''Contract ``x`` over ``axis`` of ``data.matrix`` (0 = angles, 1 = moments).'''
        ...

    def forward(self, data : DiscreteMomentData, x):
        return self.contract(data, x, data.in_axis)

    def transpose(self, data : DiscreteMomentData, x):
        return self.contract(data, x, 1 - data.in_axis)


class FluxMomentOperator(TransposeBoolDataStrategyLinearOperator):
    r'''
    Shared base for the moment and discrete operators: a harmonics contraction on the leading axis,
    batched over :attr:`elements_basis_shape`. The direction is carried by :attr:`data`.
    '''

    data                 : eqx.AbstractVar[DiscreteMomentData]
    elements_basis_shape : eqx.AbstractVar[jax.ShapeDtypeStruct]  #[n_elements, n_basis]
    strategy             : eqx.AbstractVar[FluxMomentStrategy]
    transpose_bool       : eqx.AbstractVar[bool]

    def _axis_structure(self, axis : int):
        return jax.ShapeDtypeStruct(
            shape=(self.data.matrix.shape[axis],) + self.elements_basis_shape.shape,
            dtype=self.data.matrix.dtype)

    def _in_structure(self):
        '''Forward-direction input structure. Shape [n_in, ...].'''
        return self._axis_structure(self.data.in_axis)

    def _out_structure(self):
        '''Forward-direction output structure. Shape [n_out, ...].'''
        return self._axis_structure(1 - self.data.in_axis)

    @property
    def n_moments(self):
        r'''Number of moments (= :math:`(L+1)^2` for moment order L)'''
        return self.data.n_moments

    @property
    def n_angles(self):
        '''Number of discrete angles'''
        return self.data.n_angles


class MomentOperator(FluxMomentOperator):
    r'''
    Moment operator that maps angular fluxes to moments.


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

    data                : MomentData
    r'''MomentData containing the :math:`w_k Y_{lm}(\Omega_k)` weights for the moment operator [differentiable]'''

    elements_basis_shape : jax.ShapeDtypeStruct  #[n_elements, n_basis]
    r'''The shape of the elements and basis functions over which this operator is batched. Identified with :math:`F` in the documentation above.'''

    strategy             : FluxMomentStrategy
    r'''How to apply the contraction. See :mod:`~jax_sn.operators.moment_operator.fluxmoment_strategies`.'''

    transpose_bool       : bool = False
    r'''Whether this operator applies the transposed action (moments to angles) instead of the forward action (angles to moments).'''


class DiscreteOperator(FluxMomentOperator):
    r'''
    Discrete operator that maps moments to angular fluxes. Approximate inverse of
    :class:`jax_sn.operators.moment_operator.MomentOperator`.

    Mathematical description
    ---------------------------
    Implements:

    .. math::
        D  = \overline{\mathbf{D}} \otimes \mathbf{I}_F: \mathcal{M}\otimes \mathcal{F} \rightarrow \mathcal{D}\otimes \mathcal{F}

    with

    .. math::
        \overline{\mathbf{D}}[\{ b_{lm} \}] = \sum_{lm} b_{lm} Y_{lm}(\Omega_k)

    where :math:`\mathcal{D}` is the space of discrete angles, :math:`\mathcal{M}` the space of moments, and :math:`\mathcal{F}` the space of elements and basis functions.

    Transposition
    ---------------------------
    The transposed action (selected via :attr:`transpose_bool`) implements:

    .. math::
        D^T  = \overline{\mathbf{D}}^T \otimes \mathbf{I}_F: \mathcal{D}\otimes \mathcal{F} \rightarrow \mathcal{M}\otimes \mathcal{F}

    with

    .. math::
        \overline{\mathbf{D}}^T[s_k] = \sum_{k} s_k Y_{lm}(\Omega_k)

    This is not the same as the moment operator, since the weights :math:`w_k` are not included here.


    '''

    data                : DiscreteData
    r'''Matrix of [n_angles, n_moments] containing :math:`Y_{l,m}(\Omega_k)` values for each angle :math:`k` and moment :math:`(l,m)` (collapsed into a single index).'''

    elements_basis_shape : jax.ShapeDtypeStruct  #[n_elements, n_basis]
    r'''The shape of the elements and basis functions over which this operator is batched. Identified with :math:`F` in the documentation above.'''

    strategy             : FluxMomentStrategy
    r'''How to apply the contraction. See :mod:`~jax_sn.operators.moment_operator.fluxmoment_strategies`.'''

    transpose_bool       : bool = False
    r'''Whether this operator applies the transposed action (angles to moments) instead of the forward action (moments to angles).'''
