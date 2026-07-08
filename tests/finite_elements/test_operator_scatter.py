import pytest

from jax_sn.operators.scatter_operator import ScatterOperator

from .operator_util import BaseOperatorProperties, Spec, valid_combos

# The scatter block operator has no strategy variants yet -- nothing to swap, so operator == reference and
# the consistency checks pass trivially (band-aid; they start doing real work once variants / alternative
# scatter operators land, at which point this registry gains entries and the fixture swaps like the others).
SCATTER_STRATEGIES = {ScatterOperator: Spec()}

_G_OUT, _G_IN = 1, 0   # a single group->group block; (in=0 -> out=1) is a non-trivial downscatter in BB


@pytest.mark.parametrize(
    "bb_mgop_dim_degree, strategy_cls, kwargs",
    valid_combos(SCATTER_STRATEGIES),
    indirect=["bb_mgop_dim_degree"],   # (dim, degree) drives the cached bb_mgop fixture
)
class TestScatterOperator(BaseOperatorProperties):

    @pytest.fixture
    def operator_and_reference(self, bb_mgop, strategy_cls, kwargs):
        '''A single group->group ScatterOperator block on the shared BB problem. No strategy to swap yet,
        so the reference is the operator itself (see module note).'''
        mgop = bb_mgop[0]
        shape = mgop.moment_operator.out_structure()   # (n_moments, n_elem, n_basis)
        operator = mgop.gather_scatter_operator.to_scatter_operator(_G_OUT, _G_IN, shape)
        return operator, operator
