'''Session-scoped fixtures for the breeding-blanket problem.

Building the SolutionDomain + MultiGroupOperatorScan is the dominant compile cost in this
package, so we do it once per dimension and share it across every test that needs it:

  * ``bb_mgop`` -- the full 4-group ``MultiGroupOperatorScan`` (params: dim 1/2/3).
  * ``bb_sgop`` -- the group-0 ``SingleGroupOperator`` derived from it (source-iteration /
    single-group adjoint tests).

Tests that need *perturbed* operators (finite-difference derivative checks) can't share a
frozen instance -- they import the builders from ``bb_setup`` and rebuild directly.
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
