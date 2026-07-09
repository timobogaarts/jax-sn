import pytest
import equinox as eqx

from jax_sn.operators.transport_operator import VmapForward, BatchedForward

from .operator_util import BaseOperatorProperties, Spec, parametrize_strategies

_ANGLE_I = 0   # single-angle transport tests pick this angle from the multi-angle operator

_DEFAULT_FORWARD = VmapForward   # pure-JAX reference (pivot) every variant is compared against

# Forward strategies only differ in the forward apply (mv). They all run on the shared BasicSweepData the
# BB operator is built with, so the variant is a cheap strategy swap on the single-angle operator (no
# sweep-data rebuild) -- exactly the mass/moment pattern. The batched variant is checked at a batch size.
FORWARD_STRATEGIES = {VmapForward:   Spec(),
                      BatchedForward: Spec(kwargs=dict(batch_size=64))}


@parametrize_strategies(FORWARD_STRATEGIES)
class TestTransportForward(BaseOperatorProperties):

    rtol_exact, atol_exact = 1e-10, 1e-10
    rtol, atol = 1e-10, 1e-10

    @pytest.fixture
    def operator(self, bb_mgop, strategy_cls, kwargs):
        '''The single-angle TransportOperator with this forward strategy on the shared BB problem.'''
        top = bb_mgop[0].transport_operator[_ANGLE_I]
        return eqx.tree_at(lambda o: o.forward_strategy, top, strategy_cls(**kwargs))

    @pytest.fixture
    def build_reference(self, bb_mgop):
        top = bb_mgop[0].transport_operator[_ANGLE_I]
        return lambda: eqx.tree_at(lambda o: o.forward_strategy, top, _DEFAULT_FORWARD())
