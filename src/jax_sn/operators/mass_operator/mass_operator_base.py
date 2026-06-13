import jax
import jax.numpy as jnp
from jax_sn.solution_domain import PrecomputedGeometricData, FEMMatrices

import lineax as lx
import equinox as eqx
from jax_sn.operators.base_linear_operator import AbstractLinearOperator_sn

from typing import Tuple
from jax_sn.solution_domain import SolutionDomainData
from typing import Union

from jax_sn.operators.transport_operator.transport_operator_base import TransportOperator, TransportOperatorMultipleAngles

from .mass_operator_strategies import MassStrategy, VmapMass


class MassOperator(AbstractLinearOperator_sn):
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

    fem_matrices                : FEMMatrices
    '''Finite element method matrices containing the mass matrix.'''

    precomputed_geometric_data  : PrecomputedGeometricData
    '''Precomputed geometric data containing the Jacobian determinants.'''

    additional_shape             : Tuple[int] = ()
    '''Additional leading shape to batch over. Used for consistent in/out structure definitions.'''

    strategy                     : MassStrategy = eqx.field(default_factory=VmapMass)
    '''How to apply the mass contraction. See :mod:`~jax_sn.operators.mass_operator.mass_operator_strategies`.'''

    @classmethod
    def from_solution_domain_data(
        cls,
        solution_domain_data : SolutionDomainData,
        additional_shape     : Tuple[int] = (),
        strategy             : MassStrategy = VmapMass()
    ) -> "MassOperator":
        r'''
        Construct a MassOperator from solution domain data.

        Parameters
        ----------
        solution_domain_data : SolutionDomainData
            The solution domain data to use.
        additional_shape : Tuple[int]
            Additional leading shape to batch over.
        strategy : MassStrategy
            The contraction strategy to use (default :class:`~.mass_operator_strategies.VmapMass`).

        Returns
        -------
        MassOperator
            The constructed MassOperator.
        '''
        return cls(
            solution_domain_data.fem_matrices,
            solution_domain_data.precomputed_geometric_data,
            additional_shape,
            strategy
        )

    @classmethod
    def from_transport_operator(cls, transport_operator : Union[TransportOperator, TransportOperatorMultipleAngles], strategy : MassStrategy = VmapMass()) -> "MassOperator":
        r'''
        Construct a MassOperator from a transport operator.

        Parameters
        ----------
        transport_operator : Union[TransportOperator, TransportOperatorMultipleAngles]
            The transport operator to use.
        strategy : MassStrategy
            The contraction strategy to use (default :class:`~.mass_operator_strategies.VmapMass`).
        Returns
        -------
        MassOperator
            The constructed MassOperator.
        '''
        in_structure = transport_operator.in_structure()
        solution_domain_data = transport_operator.solution_domain_data

        return cls(solution_domain_data.fem_matrices,
                   solution_domain_data.precomputed_geometric_data,
                   in_structure.shape[:-2],
                   strategy)


    def mv(self, x : jnp.ndarray) -> jnp.ndarray:
        '''
        Apply the mass operator to the input vector x.

        Parameters
        ----------
        x : jnp.ndarray
            Input array of shape [..., n_elements, n_basis]

        Returns
        -------
        jnp.ndarray
            Output array of shape [..., n_elements, n_basis]
        '''
        return self.strategy.forward(self, x)


    def transpose(self) -> "MassOperator":
        '''Returns the transpose operator, which is itself since the mass matrix is symmetric.'''
        return self  # Mass matrix is symmetric

    def in_structure(self):
        '''Input structure. Shape [..., n_elements, n_basis] where ... is the same as :attr:`additional_shape`.'''
        return self._shape_inout

    def out_structure(self):
        '''Output structure. Shape [..., n_elements, n_basis] where ... is the same as :attr:`additional_shape`.'''
        return self._shape_inout

    @property
    def _shape_inout(self):
        n_elements = self.precomputed_geometric_data.n_elements
        n_basis = self.fem_matrices.n_basis
        return jax.ShapeDtypeStruct(self.additional_shape + (n_elements, n_basis), self.fem_matrices.mass_matrix.dtype)


@lx.is_symmetric.register(MassOperator)
def _(operator : MassOperator):
    return True

@lx.linearise.register(MassOperator)
def _(operator : MassOperator):
    return operator
