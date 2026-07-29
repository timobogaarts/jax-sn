from abc import abstractmethod

import equinox as eqx

from jax_sn.operators.base_linear_operator import TransposeBoolDataStrategyLinearOperator, TransposableStrategy


from ..operator_data import OperatorData
from ..base_linear_operator import BaseOperatorData
import jax

class SingleGroupData(BaseOperatorData):
    operator_data : OperatorData
    energy_group  : int

class SingleGroupStrategyBase(TransposableStrategy):
    r'''
    Base class for single-group strategies, which are linear operators that act on a single energy group.
    '''
    @abstractmethod 
    def forward(self, single_group_data : SingleGroupData, x):
        ...

    @abstractmethod 
    def transpose(self, single_group_data : SingleGroupData, x):
        ...

    @abstractmethod
    def default_sweep_data(self):
        ...

class SingleGroupStrategy(SingleGroupStrategyBase):
    r'''
    Apply the full :math:`\mathbf{I} - \mathbf{M}\mathbf{T}_g^{-1}\mathbf{H}\mathbf{D}\mathbf{S}_{gg}` operator, either forward or backward.

    Abstract chain and source operators: a SingleGroupStrategy can have a very logical chain and source operator (e.g. if composed of sub-operators), 
    which it can then yield (or just raise an error if this does not exist).
    '''


class SingleGroupChainStrategy(SingleGroupStrategyBase):
    r'''
    Apply the chain :math:`\mathbf{C}_g = \mathbf{M}\mathbf{T}_g^{-1}\mathbf{H}\mathbf{D}` operator, either forward or backward.
    '''

class SingleGroupSourceStrategy(SingleGroupStrategyBase):
    r'''
    Apply the source :math:`\mathbf{M}\mathbf{T}_g^{-1}\mathbf{H}` operator, either forward or backward.
    '''




def single_group_moment_structure(data: SingleGroupData) -> jax.ShapeDtypeStruct:
    operator_data = data.operator_data
    return jax.ShapeDtypeStruct((operator_data.n_moments,) + operator_data.elements_basis_shape.shape, operator_data.angular_data.wk_ylm.dtype)

def single_group_angle_structure(data: SingleGroupData) -> jax.ShapeDtypeStruct:
    operator_data = data.operator_data
    return jax.ShapeDtypeStruct((operator_data.n_angles,) + operator_data.elements_basis_shape.shape, operator_data.angular_data.wk_ylm.dtype)

class SingleGroupOperatorBase(TransposeBoolDataStrategyLinearOperator):
    '''
    Base operator acting on a single group.
    '''

    data           : SingleGroupData
    strategy       : SingleGroupStrategyBase   # concrete real field (narrowed per concrete operator); declared before transpose_bool so field order stays data, strategy, transpose_bool
    transpose_bool : bool = False

    def to_energy_group(self, energy_group) -> "SingleGroupOperatorBase":
        return eqx.tree_at(lambda op: op.data.energy_group, self, energy_group)


class SingleGroupOperator(SingleGroupOperatorBase):
    r'''
    Single energy group linear operator.

    This implements the total, single-group linear operator:

    .. math::
        \mathbf{A}_g = \mathbf{I} - \mathbf{M}\mathbf{T}_g^{-1} \mathbf{H} \mathbf{D}\mathbf{S}_{gg}

    where :math:`\mathbf{T}` is the (multi-angle) transport operator (see :class:`TransportOperatorMultiAngle`), :math:`\mathbf{H}` the batched (over angles) mass operator (see :class:`MassOperator`), :math:`\mathbf{M}` the batched (over element and basis function)
    moment operator (see :class:`MomentOperator`),  :math:`\mathbf{S}_{gg}` the inscattering operator (see :class:`ScatterOperator`).

    Transposition
    ---------------------------

    The transpose follows from transposing the individual operators:

    .. math::
        \mathbf{A}_g^T = \mathbf{I} - \mathbf{S}_{gg}^T \mathbf{D}^T \mathbf{H}^T (\mathbf{T}_g^{-1})^T \mathbf{M}^T

    where the individual transposes are defined in their respective classes or directly in the fused operator.

    Solution
    ---------------------------
    Since this is a iteratively defined linear operator, the standard way to solve systems involving this operator is to use an iterative solver (e.g., GMRES, CG, etc.) with the mv() method defined here. This can
    be accomplished using e.g. lineax solvers. This allows for the propagation of the derivatives through the solver as well.

    '''
    strategy : SingleGroupStrategy

    def _in_structure(self):
        return single_group_moment_structure(self.data)

    def _out_structure(self):
        return single_group_moment_structure(self.data)


class SingleGroupChainOperator(SingleGroupOperatorBase):
    r'''The chain operator :math:`\mathbf{C}_g = \mathbf{M}\mathbf{T}_g^{-1}\mathbf{H}\mathbf{D}`, built from ``operator_data`` and the energy group.'''
    strategy : SingleGroupChainStrategy

    def _in_structure(self):
        return single_group_moment_structure(self.data)

    def _out_structure(self):
        return single_group_moment_structure(self.data)

class SingleGroupSourceOperator(SingleGroupOperatorBase):
    r'''The source operator :math:`\mathbf{M}\mathbf{T}_g^{-1}\mathbf{H}`, built from ``operator_data`` and the energy group.'''
    strategy : SingleGroupSourceStrategy

    def _in_structure(self):
        return single_group_angle_structure(self.data)

    def _out_structure(self):
        return single_group_moment_structure(self.data)
    




