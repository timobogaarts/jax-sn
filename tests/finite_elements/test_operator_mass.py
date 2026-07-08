import pytest
import equinox as eqx

from jax_sn.operators.mass_operator import VmapMass, LaxMass

from .operator_util import BaseOperatorProperties, Spec, valid_combos, simplex_n_basis
from jax_sn.operators.mass_operator.mass_operator_warp import WarpMass

_DEFAULT_MASS = VmapMass   # pure-JAX reference (pivot) every variant is compared against

# Warp supports only 4 basis functions. linear tetrahedron or cubic interval are only checked.
MASS_STRATEGIES = {VmapMass: Spec(), 
                   LaxMass: Spec(), 
                   WarpMass: Spec(supports= lambda dim, degree : simplex_n_basis(dim, degree) == 4)}  

@pytest.mark.parametrize(
    "bb_mgop_dim_degree, strategy_cls, kwargs",
    valid_combos(MASS_STRATEGIES),
    indirect=["bb_mgop_dim_degree"],   # (dim, degree) drives the cached bb_mgop fixture
)
class TestMassOperator(BaseOperatorProperties):

    @pytest.fixture
    def operator_and_reference(self, bb_mgop, strategy_cls, kwargs):
        '''(MassOperator with this strategy, reference MassOperator) on the shared BB problem.'''
        mass_operator = bb_mgop[0].mass_operator
        operator  = eqx.tree_at(lambda op: op.strategy, mass_operator, strategy_cls(**kwargs))
        reference = eqx.tree_at(lambda op: op.strategy, mass_operator, _DEFAULT_MASS())
        return operator, reference
