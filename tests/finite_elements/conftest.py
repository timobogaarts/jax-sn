'''Session-scoped fixtures for the breeding-blanket problem.
'''
import pytest

from finite_elements.bb_setup import _setup_bb_args, _get_bb_args


@pytest.fixture(params=[1, 2, 3], scope='session')
def bb_mgop(request):
    '''(MultiGroupOperatorScan, source_total, solution_domain, bb_args) for dim in {1,2,3}.'''
    args = _get_bb_args(request.param, 2)
    return (*_setup_bb_args(*args), args)


@pytest.fixture(scope='session')
def bb_sgop(bb_mgop):
    '''(SingleGroupOperator[group 0], source_total, solution_domain, bb_args), sharing bb_mgop.'''
    mgop, source_total, solution_domain, args = bb_mgop
    return mgop.to_single_group_operator(0), source_total, solution_domain, args
