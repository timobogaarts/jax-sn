import pytest
import equinox as eqx

from jax_sn.operators.mass_operator import VmapMass, LaxMass

from .operator_util import BaseOperatorProperties, Spec, parametrize_strategies, simplex_n_basis
from jax_sn.operators.mass_operator.mass_operator_warp import WarpMass

_DEFAULT_MASS = VmapMass   # pure-JAX reference (pivot) every variant is compared against

# Warp supports only 4 basis functions. linear tetrahedron or cubic interval are only checked.
MASS_STRATEGIES = {VmapMass: Spec(), 
                   LaxMass: Spec(), 
                   WarpMass: Spec(supports= lambda dim, degree : simplex_n_basis(dim, degree) == 4)}  

@parametrize_strategies(MASS_STRATEGIES)
class TestMassOperator(BaseOperatorProperties):

    @pytest.fixture
    def operator(self, bb_mgop, strategy_cls, kwargs):
        '''The MassOperator with this strategy on the shared BB problem.'''
        return eqx.tree_at(lambda op: op.strategy, bb_mgop[0].mass_operator, strategy_cls(**kwargs))

    @pytest.fixture
    def build_reference(self, bb_mgop):
        return lambda: eqx.tree_at(lambda op: op.strategy, bb_mgop[0].mass_operator, _DEFAULT_MASS())
