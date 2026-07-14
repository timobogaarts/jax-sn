import jax
import jax.numpy as jnp
from jax_sn.quadrature_set import QuadratureSetBase
from jax_sn.operators.base_linear_operator import TransposeBoolDataStrategyLinearOperator
import equinox as eqx

from .moment_operator_strategies import DiscreteStrategy, BatchedDiscrete, DiscreteData


class DiscreteOperator(TransposeBoolDataStrategyLinearOperator):
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

    data : DiscreteData
    r'''Matrix of [n_angles, n_moments] containing :math:`Y_{l,m}(\Omega_k)` values for each angle :math:`k` and moment :math:`(l,m)` (collapsed into a single index).'''

    elements_basis_shape : jax.ShapeDtypeStruct  #[n_elements, n_basis]
    r'''The shape of the elements and basis functions over which this operator is batched. Identified with :math:`F` in the documentation above.'''

    strategy             : DiscreteStrategy
    r'''How to apply the contraction. See :mod:`~jax_sn.operators.moment_operator.moment_operator_strategies`.'''

    transpose_bool       : bool = False
    r'''Whether this operator applies the transposed action (angles to moments) instead of the forward action (moments to angles).'''

    @classmethod
    def from_quadrature_set(cls, quadrature_set : QuadratureSetBase, l_max : int, elements_basis_shape : jax.ShapeDtypeStruct, strategy : DiscreteStrategy) -> "DiscreteOperator":
        r'''
        Create a discrete operator from a quadrature set and desired moment order. This computes the :math:`Y_{l,m}(\Omega_k)` weights for the specified moment order.
        Prefer to initialise directly if possible.
        '''
        yklm = quadrature_set.ylm_real(l_max)
        return cls(DiscreteData(yklm), elements_basis_shape, strategy)    

    def _in_structure(self):
        '''Forward-direction input structure representing moments. Shape [n_moments, ...].'''
        return jax.ShapeDtypeStruct(
            shape=(self.yklm.shape[1],) + self.elements_basis_shape.shape,
            dtype=self.yklm.dtype)

    def _out_structure(self):
        '''Forward-direction output structure representing discrete angles. Shape [n_angles, ...].'''
        return jax.ShapeDtypeStruct(
            shape=(self.yklm.shape[0],) + self.elements_basis_shape.shape,
            dtype=self.yklm.dtype)

    @property
    def n_moments(self):
        '''Number of moments (= :math:`(L+1)^2` for moment order L)'''
        return self.yklm.shape[1]

    @property
    def n_angles(self):
        '''Number of discrete angles'''
        return self.yklm.shape[0]
