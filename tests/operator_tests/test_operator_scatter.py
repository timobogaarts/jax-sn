import pytest
import equinox as eqx

from jax_sn.operators.scatter_operator import ScatterOperator, ScatterData, IndexedScatter

from .operator_util import BaseOperatorProperties, Spec, parametrize_strategies

_DEFAULT_SCATTER = IndexedScatter   # pure-JAX reference (pivot) every variant is compared against

SCATTER_STRATEGIES = {IndexedScatter: Spec()}

_G_OUT, _G_IN = 1, 0   # a single group->group block; (in=0 -> out=1) is a non-trivial downscatter in BB


def _scatter_operator(mgop, strategy):
    '''The S_{g_out g_in} block on the shared BB problem, with this strategy.'''
    from bb_setup import moment_operator_from_mgop
    return ScatterOperator(
        ScatterData(_G_OUT, _G_IN, mgop.data.domain.cross_sections_indexed),
        strategy,
        moment_operator_from_mgop(mgop).out_structure(),   # (n_moments, n_elem, n_basis)
    )


@parametrize_strategies(SCATTER_STRATEGIES)
class TestScatterOperator(BaseOperatorProperties):

    @pytest.fixture
    def operator(self, bb_mgop, strategy_cls, kwargs):
        return _scatter_operator(bb_mgop[0], strategy_cls(**kwargs))

    @pytest.fixture
    def build_reference(self, bb_mgop):
        return lambda: _scatter_operator(bb_mgop[0], _DEFAULT_SCATTER())
