from abc import abstractmethod

import equinox as eqx

from jax_sn.operators.base_linear_operator import AbstractLinearOperator_sn
from jax_sn.operators.scatter_operator import ScatterOperator
from ..operator_data import OperatorData
import jax

class SingleGroupStrategyBase(eqx.Module):

    @abstractmethod
    def mv_forward(self, operator_data: OperatorData, energy_group: int, x):
        '''Apply a single group operation forward'''
        ...
    @abstractmethod
    def mv_transpose(self, operator_data: OperatorData, energy_group: int, x):
        '''Apply a single group operation backward'''
        ...

class SingleGroupStrategy(SingleGroupStrategyBase):
    r'''
    Apply the full :math:`\mathbf{I} - \mathbf{M}\mathbf{T}_g^{-1}\mathbf{H}\mathbf{D}\mathbf{S}_{gg}` operator, either forward or backward.

    Abstract chain and source operators: a SingleGroupStrategy can have a very logical chain and source operator (e.g. if composed of sub-operators), 
    which it can then yield (or just raise an error if this does not exist).
    '''

    @abstractmethod
    def chain_strategy(self) -> "SingleGroupChainStrategy":
        ...

    @abstractmethod
    def source_strategy(self) -> "SingleGroupSourceStrategy":
        ...

class SingleGroupChainStrategy(SingleGroupStrategyBase):
    r'''
    Apply the chain :math:`\mathbf{C}_g = \mathbf{M}\mathbf{T}_g^{-1}\mathbf{H}\mathbf{D}` operator, either forward or backward.
    '''

class SingleGroupSourceStrategy(SingleGroupStrategyBase):
    r'''
    Apply the source :math:`\mathbf{M}\mathbf{T}_g^{-1}\mathbf{H}` operator, either forward or backward.
    '''

def single_group_moment_structure(operator_data: OperatorData) -> jax.ShapeDtypeStruct:        
    return jax.ShapeDtypeStruct((operator_data.n_moments,) + operator_data.elements_basis_shape.shape, operator_data.angular_data.wk_ylm.dtype)

def single_group_angle_structure(operator_data: OperatorData) -> jax.ShapeDtypeStruct:    
    return jax.ShapeDtypeStruct((operator_data.n_angles,) + operator_data.elements_basis_shape.shape, operator_data.angular_data.wk_ylm.dtype)

class SingleGroupOperatorBase(AbstractLinearOperator_sn):

    operator_data  : OperatorData
    energy_group   : int
    strategy       : SingleGroupStrategyBase   # narrowed per concrete operator; declared here to fix field order
    transpose_bool : bool = False

    @abstractmethod
    def _in_structure(self):
        '''Forward-direction input structure.'''
        ...

    @abstractmethod
    def _out_structure(self):
        '''Forward-direction output structure.'''
        ...

    def in_structure(self):
        return self._out_structure() if self.transpose_bool else self._in_structure()

    def out_structure(self):
        return self._in_structure() if self.transpose_bool else self._out_structure()

    def transpose(self) -> "SingleGroupOperatorBase":
        return eqx.tree_at(lambda op: op.transpose_bool, self, not self.transpose_bool)

    def to_energy_group(self, energy_group) -> "SingleGroupOperatorBase":
        return eqx.tree_at(lambda op: op.energy_group, self, energy_group)

    @eqx.filter_jit
    def mv(self, x):
        if self.transpose_bool:
            return self.strategy.mv_transpose(self.operator_data, self.energy_group, x)
        return self.strategy.mv_forward(self.operator_data, self.energy_group, x)

    def __check_init__(self):
        # Each concrete operator narrows ``strategy`` to its matching strategy type via the field
        # annotation; here we only assert the shared base contract.
        assert isinstance(self.operator_data, OperatorData), f"operator_data must be of type OperatorData but is {type(self.operator_data)}"
        assert isinstance(self.strategy, SingleGroupStrategyBase), f"strategy must be a SingleGroupStrategyBase but is {type(self.strategy)}"

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
        return single_group_moment_structure(self.operator_data)

    def _out_structure(self):
        return single_group_moment_structure(self.operator_data)

    def chain_operator(self) -> "SingleGroupChainOperator":
        r'''The chain :math:`\mathbf{C}_g = \mathbf{M}\mathbf{T}_g^{-1}\mathbf{H}\mathbf{D}` as its own
        (forward) operator, sharing this operator's data and group. Callers transpose it explicitly.'''
        return SingleGroupChainOperator(self.operator_data, self.energy_group, self.strategy.chain_strategy())

    def source_operator(self) -> "SingleGroupSourceOperator":
        r'''The fixed-source sweep :math:`\mathbf{M}\mathbf{T}_g^{-1}\mathbf{H}` as its own (forward)
        operator, used to build right-hand sides.'''
        return SingleGroupSourceOperator(self.operator_data, self.energy_group, self.strategy.source_strategy())


class SingleGroupChainOperator(SingleGroupOperatorBase):
    r'''The chain operator :math:`\mathbf{C}_g = \mathbf{M}\mathbf{T}_g^{-1}\mathbf{H}\mathbf{D}`, built from ``operator_data`` and the energy group.'''
    strategy : SingleGroupChainStrategy

    def _in_structure(self):
        return single_group_moment_structure(self.operator_data)

    def _out_structure(self):
        return single_group_moment_structure(self.operator_data)

class SingleGroupSourceOperator(SingleGroupOperatorBase):
    r'''The source operator :math:`\mathbf{M}\mathbf{T}_g^{-1}\mathbf{H}`, built from ``operator_data`` and the energy group.'''
    strategy : SingleGroupSourceStrategy

    def _in_structure(self):
        return single_group_angle_structure(self.operator_data)

    def _out_structure(self):
        return single_group_moment_structure(self.operator_data)
    




