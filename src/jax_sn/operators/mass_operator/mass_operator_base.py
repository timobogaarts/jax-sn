import jax
import jax.numpy as jnp
from jax_sn.solution_domain import PrecomputedGeometricData, FEMMatrices

import lineax as lx
import equinox as eqx
from jax_sn.operators.base_linear_operator import SymmetricDataStrategyLinearOperator

from typing import Tuple
from jax_sn.solution_domain import SpatialData
from typing import Union

from jax_sn.operators.transport_operator import TransportOperatorSingleAngle, TransportOperatorMultiAngle

from .mass_operator_strategies import MassStrategy, VmapMass, MassData


class MassOperator(SymmetricDataStrategyLinearOperator):
    r'''
    Mass operator.
    Computes for a given vector :math:`\{ v_j \}` the mass matrix map:

    .. math::
        a_i = \sum_{j} \int H_i H_j dV v_j

    Is arbitrarily batched over additional leading dimensions.

    Implementation handling
    ---------------------------
    The contraction implementation (vmap over elements, batched ``lax.dot_general``, ...) is selected
    by the :attr:`strategy` :class:`~.mass_operator_strategies.MassStrategy` field, *not* by a
    subclass. A new implementation is a new strategy object. See
    :mod:`~jax_sn.operators.mass_operator.mass_operator_strategies`.

    Mathematical description
    ---------------------------

    The non-batched version computes the mass matrix for each element and basis function:

    .. math::
        \overline{\mathbf{H}} : F \to F

    where in this case:

    .. math::
        F : \{ \text{all elements} \} \otimes \{ \text{all basis functions per element} \}

    and since all elements are independent, the mass matrix is block-diagonal with each block representing the mass matrix for one element.

    Its definition is (:math:`e,p` is  element index, `i,j` basis function index):

    .. math::
        \overline{\mathbf{H}}[\{ v_{ei} \}] = a_{ej} = \sum_{i}  J_{vq} \int_{B} H_i H_j dV v_{ei}

    where :math:`J_{vq}` is the volume Jacobian determinant for element :math:`e`, and :math:`B` is the reference element.
    Note this integral is already computed in :attr:`jax_sn.solution_domain.FEMMatrices.mass_matrix`, and the Jacobian determinants are precomputed and stored  :attr:`jax_sn.solution_domain.PrecomputedGeometricData.vol_jacobians_det`.

    Transposition
    ---------------------------

    This operator is symmetric since the mass matrix is symmetric. Transposition thus returns itself.
    The strategy therefore implements a single :meth:`~.mass_operator_strategies.MassStrategy.apply`
    (forward == transpose); there is no ``transpose_bool``.

    Structure
    ---------------------------

    Batched versions operate over additional leading dimensions, e.g.:

    .. math::
        &\mathbf{H} :   \mathcal{K} \otimes \mathcal{F} \to \mathcal{K} \otimes \mathcal{F}\\
        &\mathbf{H} = \mathbf{I}_{\mathcal{K}} \otimes \overline{\mathbf{H}}

    where :math:`\mathcal{K}` is an arbitrary additional space (e.g. angles).


    In essence, this just is a batched matrix product (and is implemented like that as well).

    '''
    data                         : MassData
    '''MassData containing FEM matrices and precomputed geometric data. [differentiable]'''

    additional_shape             : Tuple[int] = ()
    '''Additional leading shape to batch over. Used for consistent in/out structure definitions.'''

    strategy                     : MassStrategy
    '''How to apply the mass contraction. See :mod:`~jax_sn.operators.mass_operator.mass_operator_strategies`.'''

    def _structure(self):
        n_elements = self.precomputed_geometric_data.n_elements
        n_basis = self.fem_matrices.n_basis
        return jax.ShapeDtypeStruct(self.additional_shape + (n_elements, n_basis), self.fem_matrices.mass_matrix.dtype)


@lx.is_symmetric.register(MassOperator)
def _(operator : MassOperator):
    return True
