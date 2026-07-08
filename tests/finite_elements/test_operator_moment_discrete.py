import pytest
import equinox as eqx

from jax_sn.operators import mass_operator
from jax_sn.operators.moment_operator import BatchedMoment, StreamingMoment, BatchedDiscrete

from .operator_util import BaseOperatorProperties, Spec, valid_combos, simplex_n_basis
from jax_sn.operators.mass_operator.mass_operator_warp import WarpMass

_DEFAULT_MOMENT = BatchedMoment   # pure-JAX reference (pivot) every variant is compared against

# Warp supports only 4 basis functions. linear tetrahedron or cubic interval are only checked.
MOMENT_STRATEGIES = {BatchedMoment: Spec(), 
                     StreamingMoment: Spec()
                    }  

@pytest.mark.parametrize(
    "bb_mgop_dim_degree, strategy_cls, kwargs",
    valid_combos(MOMENT_STRATEGIES),
    indirect=["bb_mgop_dim_degree"],   # (dim, degree) drives the cached bb_mgop fixture
)
class TestMomentOperator(BaseOperatorProperties):

    @pytest.fixture
    def operator_and_reference(self, bb_mgop, strategy_cls, kwargs):
        '''(MomentOperator with this strategy, reference MomentOperator) on the shared BB problem.'''
        moment_operator = bb_mgop[0].moment_operator
        operator  = eqx.tree_at(lambda op: op.strategy, moment_operator, strategy_cls(**kwargs))
        reference = eqx.tree_at(lambda op: op.strategy, moment_operator, _DEFAULT_MOMENT())
        return operator, reference


_DEFAULT_DISCRETE = BatchedDiscrete   # pure-JAX reference (pivot) every variant is compared against

DISCRETE_STRATEGIES = {BatchedDiscrete: Spec()} # only one variant in this case!

@pytest.mark.parametrize(
    "bb_mgop_dim_degree, strategy_cls, kwargs",
    valid_combos(DISCRETE_STRATEGIES),
    indirect=["bb_mgop_dim_degree"],   # (dim, degree) drives the cached bb_mgop fixture
)
class TestDiscreteOperator(BaseOperatorProperties):

    @pytest.fixture
    def operator_and_reference(self, bb_mgop, strategy_cls, kwargs):
        '''(DiscreteOperator with this strategy, reference DiscreteOperator) on the shared BB problem.'''
        discrete_operator = bb_mgop[0].discrete_operator
        operator  = eqx.tree_at(lambda op: op.strategy, discrete_operator, strategy_cls(**kwargs))
        reference = eqx.tree_at(lambda op: op.strategy, discrete_operator, _DEFAULT_DISCRETE())
        return operator, reference
