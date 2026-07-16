import pytest
import equinox as eqx

from jax_sn.operators.moment_operator import EinsumFluxMoment, StreamingFluxMoment

from .operator_util import BaseOperatorProperties, Spec, parametrize_strategies

_DEFAULT_MOMENT = EinsumFluxMoment   # pure-JAX reference (pivot) every variant is compared against

MOMENT_STRATEGIES = {EinsumFluxMoment: Spec(),
                     StreamingFluxMoment: Spec()
                    }

@parametrize_strategies(MOMENT_STRATEGIES)
class TestMomentOperator(BaseOperatorProperties):

    @pytest.fixture
    def operator(self, bb_mgop, strategy_cls, kwargs):
        '''The MomentOperator with this strategy on the shared BB problem.'''
        from bb_setup import moment_operator_from_mgop
        return eqx.tree_at(lambda op: op.strategy, moment_operator_from_mgop(bb_mgop[0]), strategy_cls(**kwargs))

    @pytest.fixture
    def build_reference(self, bb_mgop):
        from bb_setup import moment_operator_from_mgop
        return lambda: eqx.tree_at(lambda op: op.strategy, moment_operator_from_mgop(bb_mgop[0]), _DEFAULT_MOMENT())


_DEFAULT_DISCRETE = EinsumFluxMoment   # pure-JAX reference (pivot) every variant is compared against

# Same strategies as the moment operator: the direction lives in the data, not the strategy.
DISCRETE_STRATEGIES = {EinsumFluxMoment: Spec(),
                       StreamingFluxMoment: Spec()
                      }

@parametrize_strategies(DISCRETE_STRATEGIES)
class TestDiscreteOperator(BaseOperatorProperties):

    @pytest.fixture
    def operator(self, bb_mgop, strategy_cls, kwargs):
        '''The DiscreteOperator with this strategy on the shared BB problem.'''
        from bb_setup import discrete_operator_from_mgop
        return eqx.tree_at(lambda op: op.strategy, discrete_operator_from_mgop(bb_mgop[0]), strategy_cls(**kwargs))

    @pytest.fixture
    def build_reference(self, bb_mgop):
        from bb_setup import discrete_operator_from_mgop
        return lambda: eqx.tree_at(lambda op: op.strategy, discrete_operator_from_mgop(bb_mgop[0]), _DEFAULT_DISCRETE())


