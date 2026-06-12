import jax
import jax.numpy as jnp
from jax_sn.quadrature_set import QuadratureSetBase
import lineax as lx
from jax_sn.operators.base_linear_operator import AbstractLinearOperator_sn
import equinox as eqx

from .moment_operator_strategies import DiscreteStrategy, BatchedDiscrete


class DiscreteOperator(AbstractLinearOperator_sn):
    r'''
    Discrete operator that maps moments to angular fluxes. Approximate inverse of
    :class:`jax_sn.operators.moment_operator.MomentOperator`.

    Implementation handling
    ---------------------------
    The contraction implementation is selected by the :attr:`forward_strategy`
    :class:`~.moment_operator_strategies.DiscreteStrategy` field, *not* by a subclass. See
    :mod:`~jax_sn.operators.moment_operator.moment_operator_strategies`.

    Direction handling
    ---------------------------
    Whether this operator applies the forward action or its transpose is carried by the
    boolean field :attr:`transpose_bool`, *not* by a separate subclass. Transposition is a cheap
    flag flip via :func:`equinox.tree_at` (no setup or reconstruction).

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

    yklm                 : jnp.ndarray           #[n_angles, n_moments]
    r'''Matrix of [n_angles, n_moments] containing :math:`Y_{l,m}(\Omega_k)` values for each angle :math:`k` and moment :math:`(l,m)` (collapsed into a single index).'''

    elements_basis_shape : jax.ShapeDtypeStruct  #[n_elements, n_basis]
    r'''The shape of the elements and basis functions over which this operator is batched. Identified with :math:`F` in the documentation above.'''

    forward_strategy     : DiscreteStrategy = eqx.field(default_factory=BatchedDiscrete)
    r'''How to apply the contraction. See :mod:`~jax_sn.operators.moment_operator.moment_operator_strategies`.'''

    transpose_bool       : bool = False
    r'''Whether this operator applies the transposed action (angles to moments) instead of the forward action (moments to angles).'''

    @classmethod
    def from_quadrature_set(cls, quadrature_set : QuadratureSetBase, l_max : int, elements_basis_shape : jax.ShapeDtypeStruct, forward_strategy : DiscreteStrategy = BatchedDiscrete()) -> "DiscreteOperator":
        r'''
        Create a discrete operator from a quadrature set and desired moment order. This computes the :math:`Y_{l,m}(\Omega_k)` weights for the specified moment order.
        Prefer to initialise directly if possible.


        :param quadrature_set:
            The quadrature set to use for the discrete operator.
        :param l_max:
            The maximum moment order L to use.
        :param elements_basis_shape:
            The shape of the elements and basis functions over which this operator is batched. Identified with :math:`F` in the documentation above.
        :param forward_strategy:
            The contraction strategy to use (default :class:`~.moment_operator_strategies.BatchedDiscrete`).


        '''
        yklm = quadrature_set.ylm_real(l_max)
        return cls(yklm, elements_basis_shape, forward_strategy)

    def mv(self, x : jnp.ndarray) -> jnp.ndarray:
        if self.transpose_bool:
            return self.forward_strategy.transpose(self, x)
        return self.forward_strategy.forward(self, x)

    def transpose(self) -> "DiscreteOperator":
        '''Returns the transposed operator by flipping :attr:`transpose_bool`. There is no setup or reconstruction, so this operation is essentially free.'''
        return eqx.tree_at(lambda op: op.transpose_bool, self, not self.transpose_bool)

    def _forward_in_structure(self):
        '''Forward-direction input structure representing moments. Shape [n_moments, ...].'''
        return jax.ShapeDtypeStruct(
            shape=(self.yklm.shape[1],) + self.elements_basis_shape.shape,
            dtype=self.yklm.dtype)

    def _forward_out_structure(self):
        '''Forward-direction output structure representing discrete angles. Shape [n_angles, ...].'''
        return jax.ShapeDtypeStruct(
            shape=(self.yklm.shape[0],) + self.elements_basis_shape.shape,
            dtype=self.yklm.dtype)

    def in_structure(self):
        '''Input structure. Moments in the forward direction, discrete angles in the transposed direction.'''
        return self._forward_out_structure() if self.transpose_bool else self._forward_in_structure()

    def out_structure(self):
        '''Output structure. Discrete angles in the forward direction, moments in the transposed direction.'''
        return self._forward_in_structure() if self.transpose_bool else self._forward_out_structure()

    @property
    def n_moments(self):
        '''Number of moments (= :math:`(L+1)^2` for moment order L)'''
        return self.yklm.shape[1]

    @property
    def n_angles(self):
        '''Number of discrete angles'''
        return self.yklm.shape[0]


@lx.is_symmetric.register(DiscreteOperator)
def _(operator):
    return False

@lx.linearise.register(DiscreteOperator)
def _(operator):
    return operator
