import pytest
import equinox as eqx

from jax_sn.operators.mass_operator import VmapMass, LaxMass

from .operator_util import BaseOperatorProperties, Spec, parametrize_strategies, warp_supported
from jax_sn.operators.mass_operator.mass_operator_warp import WarpMass

_DEFAULT_MASS = VmapMass   # pure-JAX reference (pivot) every variant is compared against

# Warp is parametrised on n_basis but capped at 4 by its fixed-size vector types; see warp_supported.
MASS_STRATEGIES = {VmapMass: Spec(), 
                   LaxMass: Spec(), 
                   WarpMass: Spec(supports=warp_supported)}  

@parametrize_strategies(MASS_STRATEGIES)
class TestMassOperator(BaseOperatorProperties):

    @pytest.fixture
    def operator(self, bb_mgop, strategy_cls, kwargs):
        '''The MassOperator with this strategy on the shared BB problem.'''
        from bb_setup import mass_operator_from_mgop
        return eqx.tree_at(lambda op: op.strategy, mass_operator_from_mgop(bb_mgop[0]), strategy_cls(**kwargs))

    @pytest.fixture
    def build_reference(self, bb_mgop):
        from bb_setup import mass_operator_from_mgop
        return lambda: eqx.tree_at(lambda op: op.strategy, mass_operator_from_mgop(bb_mgop[0]), _DEFAULT_MASS())
