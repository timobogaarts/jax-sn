'''Session-scoped fixtures for the breeding-blanket problem.
'''
import pytest

from finite_elements.bb_setup import _setup_bb_args, _get_bb_args
from finite_elements.operator_util import DIM_DEGREES


@pytest.fixture(params=DIM_DEGREES, scope="session")
def bb_mgop_dim_degree(request):
    return request.param

from functools import lru_cache


@lru_cache(maxsize=None)
def _build_bb_mgop(dim, degree):
    '''Session-wide memoised BB build, keyed by (dim, degree).

    Cached using lru to avoid pytest limitations (will not be too many dim,degree pairs anyway)
    '''
    args = _get_bb_args(dim, degree)    
    return (*_setup_bb_args(*args), args)

@pytest.fixture(scope='session')
def bb_mgop(bb_mgop_dim_degree):
    '''(MultiGroupOperatorScan, source_total, solution_domain, bb_args); built once per (dim, degree).'''
    return _build_bb_mgop(*bb_mgop_dim_degree)

@pytest.fixture(scope='session')
def bb_sgop(bb_mgop):
    '''(SingleGroupOperator[group 0], source_total, solution_domain, bb_args), sharing bb_mgop.'''
    mgop, source_total, solution_domain, args = bb_mgop
    return mgop.to_single_group_operator(0), source_total, solution_domain, args

@pytest.fixture(scope='session')
def strategy_setup():
    '''Shared, cached BB setup (per FEM degree) for the discovery-driven operator-strategy harness --
    consumed by both test_operator_properties.py and test_operator_regression.py so the per-degree build
    and toposort happen once for the whole session. See operator_strategy_harness.StrategySetup.'''
    from finite_elements.operator_strategy_harness import StrategySetup
    return StrategySetup()
