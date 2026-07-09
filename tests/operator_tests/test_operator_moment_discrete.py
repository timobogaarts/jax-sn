import pytest
import equinox as eqx

from jax_sn.operators.moment_operator import BatchedMoment, StreamingMoment, BatchedDiscrete

from .operator_util import BaseOperatorProperties, Spec, parametrize_strategies

_DEFAULT_MOMENT = BatchedMoment   # pure-JAX reference (pivot) every variant is compared against

MOMENT_STRATEGIES = {BatchedMoment: Spec(), 
                     StreamingMoment: Spec()
                    }  

@parametrize_strategies(MOMENT_STRATEGIES)
class TestMomentOperator(BaseOperatorProperties):

    @pytest.fixture
    def operator(self, bb_mgop, strategy_cls, kwargs):
        '''The MomentOperator with this strategy on the shared BB problem.'''
        return eqx.tree_at(lambda op: op.strategy, bb_mgop[0].moment_operator, strategy_cls(**kwargs))

    @pytest.fixture
    def build_reference(self, bb_mgop):
        return lambda: eqx.tree_at(lambda op: op.strategy, bb_mgop[0].moment_operator, _DEFAULT_MOMENT())


_DEFAULT_DISCRETE = BatchedDiscrete   # pure-JAX reference (pivot) every variant is compared against

DISCRETE_STRATEGIES = {BatchedDiscrete: Spec()} # only one variant in this case!

@parametrize_strategies(DISCRETE_STRATEGIES)
class TestDiscreteOperator(BaseOperatorProperties):

    @pytest.fixture
    def operator(self, bb_mgop, strategy_cls, kwargs):
        '''The DiscreteOperator with this strategy on the shared BB problem.'''
        return eqx.tree_at(lambda op: op.strategy, bb_mgop[0].discrete_operator, strategy_cls(**kwargs))

    @pytest.fixture
    def build_reference(self, bb_mgop):
        return lambda: eqx.tree_at(lambda op: op.strategy, bb_mgop[0].discrete_operator, _DEFAULT_DISCRETE())


