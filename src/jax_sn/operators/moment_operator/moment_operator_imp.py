import jax
import jax.numpy as jnp
from jax_sn.quadrature_set import QuadratureSetBase
import lineax as lx
from jax_sn.operators.base_linear_operator import AbstractLinearOperator_sn
import equinox as eqx

from .moment_operator_strategies import MomentStrategy, BatchedMoment


class MomentOperator(AbstractLinearOperator_sn):
    r'''
    Moment operator that maps angular fluxes to moments.

    Implementation handling
    ---------------------------
    The contraction implementation (batched einsum, streaming scan, ...) is selected by the
    :attr:`strategy` :class:`~.moment_operator_strategies.MomentStrategy` field, *not* by a
    subclass. A new implementation is a new strategy object. See
    :mod:`~jax_sn.operators.moment_operator.moment_operator_strategies`.

    Direction handling
    ---------------------------
    Whether this operator applies the forward action or its transpose is carried by the
    boolean field :attr:`transpose_bool`, *not* by a separate subclass. Transposition is a cheap
    flag flip via :func:`equinox.tree_at` (no setup or reconstruction). The transpose axis is
    therefore orthogonal to the implementation axis.

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

    elements_basis_shape : jax.ShapeDtypeStruct  #[n_elements, n_basis]
    r'''The shape of the elements and basis functions over which this operator is batched. Identified with :math:`F` in the documentation above.'''

    strategy             : MomentStrategy = eqx.field(default_factory=BatchedMoment)
    r'''How to apply the contraction. See :mod:`~jax_sn.operators.moment_operator.moment_operator_strategies`.'''

    transpose_bool       : bool = False
    r'''Whether this operator applies the transposed action (moments to angles) instead of the forward action (angles to moments).'''

    @classmethod
    def from_quadrature_set(cls, quadrature_set : QuadratureSetBase, l_max : int, elements_basis_shape : jax.ShapeDtypeStruct, strategy : MomentStrategy = BatchedMoment()) -> "MomentOperator":
        r'''
        Create a moment operator from a quadrature set and desired moment order. This computes the :math:`w_k Y_{l,m}(\Omega_k)` weights for the specified moment order.
        Prefer to initialise directly if possible.


        :param quadrature_set:
            The quadrature set to use for the moment operator.
        :param l_max:
            The maximum moment order L to use.
        :param elements_basis_shape:
            The shape of the elements and basis functions over which this operator is batched. Identified with :math:`F` in the documentation above.
        :param strategy:
            The contraction strategy to use (default :class:`~.moment_operator_strategies.BatchedMoment`).


        '''
        wk_yklm = quadrature_set.wk_ylm_real(l_max)
        return cls(wk_yklm, elements_basis_shape, strategy)

    def mv(self, x : jnp.ndarray) -> jnp.ndarray:
        if self.transpose_bool:
            return self.strategy.transpose(self, x)
        return self.strategy.forward(self, x)

    def transpose(self) -> "MomentOperator":
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


@lx.is_symmetric.register(MomentOperator)
def _(operator):
    return False

@lx.linearise.register(MomentOperator)
def _(operator):
    return operator
