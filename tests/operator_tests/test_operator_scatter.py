import pytest

from jax_sn.operators.scatter_operator import ScatterOperator

from .operator_util import BaseOperatorProperties, Spec, parametrize_strategies

SCATTER_STRATEGIES = {ScatterOperator: Spec()} # no different scatter operation

_G_OUT, _G_IN = 1, 0   # a single group->group block; (in=0 -> out=1) is a non-trivial downscatter in BB


@parametrize_strategies(SCATTER_STRATEGIES)
class TestScatterOperator(BaseOperatorProperties):

    @pytest.fixture
    def operator(self, bb_mgop, strategy_cls, kwargs):        
        mgop = bb_mgop[0]
        shape = mgop.moment_operator.out_structure()   # (n_moments, n_elem, n_basis)
        return mgop.gather_scatter_operator.to_scatter_operator(_G_OUT, _G_IN, shape)

    @pytest.fixture
    def build_reference(self, bb_mgop):        
        mgop = bb_mgop[0]
        shape = mgop.moment_operator.out_structure()
        return lambda: mgop.gather_scatter_operator.to_scatter_operator(_G_OUT, _G_IN, shape)
