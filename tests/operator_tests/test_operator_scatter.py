import pytest

from jax_sn.operators.scatter_operator import ScatterOperator

from .operator_util import BaseOperatorProperties, Spec, parametrize_strategies

# The scatter block operator has no strategy variants yet -- nothing to swap, so operator == reference and
# the consistency checks pass trivially (band-aid; they start doing real work once variants / alternative
# scatter operators land, at which point this registry gains entries and the fixture swaps like the others).
SCATTER_STRATEGIES = {ScatterOperator: Spec()}

_G_OUT, _G_IN = 1, 0   # a single group->group block; (in=0 -> out=1) is a non-trivial downscatter in BB


@parametrize_strategies(SCATTER_STRATEGIES)
class TestScatterOperator(BaseOperatorProperties):

    @pytest.fixture
    def operator(self, bb_mgop, strategy_cls, kwargs):
        '''A single group->group ScatterOperator block on the shared BB problem.'''
        mgop = bb_mgop[0]
        shape = mgop.moment_operator.out_structure()   # (n_moments, n_elem, n_basis)
        return mgop.gather_scatter_operator.to_scatter_operator(_G_OUT, _G_IN, shape)

    @pytest.fixture
    def build_reference(self, bb_mgop):
        # No strategy to swap yet, so the reference is the operator itself (see module note).
        mgop = bb_mgop[0]
        shape = mgop.moment_operator.out_structure()
        return lambda: mgop.gather_scatter_operator.to_scatter_operator(_G_OUT, _G_IN, shape)
