import pytest
import numpy as onp
import equinox as eqx

from jax_sn.operators.transport_operator import (
    VmapSweep, BatchedSweep, VmapPallasSweep, BatchedPallasSweep, VmapAngles, ScanAngles,
)
from jax_sn.operators.transport_operator.implementations.transport_operator_warp import (
    WarpSequentialSweep, WarpLevelSweep, WarpSequentialAngles, WarpLevelAngles,
)
from tests.conftest import bb_mgop

from .operator_util import (
    BaseOperatorProperties, Spec, parametrize_strategies, parametrize_instances,
    linear_tet_only, _rand,
)

_ANGLE_I = 0   # single-angle inversion tests pick this angle from the multi-angle operator


INVERSION_STRATEGIES = {
    VmapSweep:          Spec(),
    BatchedSweep:       Spec(kwargs=dict(batch_size=64)),
    VmapPallasSweep:    Spec(marks=(pytest.mark.pallas,)),
    BatchedPallasSweep: Spec(kwargs=dict(batch_size=64), marks=(pytest.mark.pallas,)),
    WarpSequentialSweep: Spec(supports=linear_tet_only),
    WarpLevelSweep:      Spec(supports=linear_tet_only),
}

MULTI_ANGLE_STRATEGIES = {
    VmapAngles(VmapSweep()):          Spec(),   
    ScanAngles(VmapSweep()):          Spec(),   
    VmapAngles(BatchedSweep(64)):     Spec(),   
    WarpSequentialAngles():           Spec(supports=linear_tet_only),  
    WarpLevelAngles():                Spec(supports=linear_tet_only),  
}

_SWEEP_DATA_CACHE = {}

def _dim_degree(bb_mgop):
    return bb_mgop[3][0], bb_mgop[3][1]


def _angle_sample(n_angles):    
    return sorted({0, n_angles // 3, (2 * n_angles) // 3, n_angles - 1})


def _sweep_data(bb_mgop, sweep_data_type):
    dim, degree = _dim_degree(bb_mgop)
    key = (dim, degree, sweep_data_type.__name__)
    if key not in _SWEEP_DATA_CACHE:
        base_sd = bb_mgop[0].transport_operator.sweep_data
        if isinstance(base_sd, sweep_data_type):
            _SWEEP_DATA_CACHE[key] = base_sd
        else:
            solution_domain, quadrature_set = bb_mgop[2], bb_mgop[0].quadrature_set
            _SWEEP_DATA_CACHE[key] = sweep_data_type.from_domain_quadrature_set(solution_domain.domain, quadrature_set)
    return _SWEEP_DATA_CACHE[key]




class _TransportInversionBase(BaseOperatorProperties):
    rtol_exact, atol_exact = 1e-9, 1e-9
    rtol, atol = 1e-9, 1e-9
    rtol_grad, atol_grad = 1e-7, 1e-7

    def apply(self, operator, x):
        return operator.inv(x)


@parametrize_strategies(INVERSION_STRATEGIES)
class TestTransportInversionSingleAngle(_TransportInversionBase):

    @pytest.fixture
    def operator(self, bb_mgop, strategy_cls, kwargs):
        strategy = strategy_cls(**kwargs)
        top = bb_mgop[0].transport_operator[_ANGLE_I]
        sweep_data = _sweep_data(bb_mgop, strategy.default_sweep_data_type())[_ANGLE_I]
        return eqx.tree_at(lambda t: (t.sweep_data, t.inversion_strategy), top, (sweep_data, strategy))

    @pytest.fixture
    def build_reference(self, bb_mgop):
        return lambda: bb_mgop[0].transport_operator[_ANGLE_I]

    def test_forward_backward_identity(self, operator):
        '''T^-1(T x) == x and its transpose -- a self-contained round-trip property of the inverse.'''
        x = _rand(operator.in_structure().shape, 8)
        onp.testing.assert_allclose(operator.inv(operator.mv(x)), x, rtol=self.rtol, atol=self.atol)
        onp.testing.assert_allclose(operator.transpose().inv(operator.transpose().mv(x)), x, rtol=self.rtol, atol=self.atol)


@parametrize_instances(MULTI_ANGLE_STRATEGIES)
class TestTransportInversionMultiAngle(_TransportInversionBase):    

    @pytest.fixture
    def operator(self, bb_mgop, strategy):
        top = bb_mgop[0].transport_operator
        sweep_data = _sweep_data(bb_mgop, strategy.default_sweep_data_type())
        return eqx.tree_at(lambda t: (t.sweep_data, t.multi_angle_inversion_strategy), top, (sweep_data, strategy))

    @pytest.fixture
    def build_reference(self, bb_mgop):
        return lambda: bb_mgop[0].transport_operator

    def test_indexing_consistency(self, operator):        
        x = _rand(operator.in_structure().shape, 10)
        b = _rand(operator.in_structure().shape, 11)
        mv_all, inv_all = operator.mv(x), operator.inv(b)
        for i in _angle_sample(operator.in_structure().shape[0]):
            onp.testing.assert_allclose(operator[i].mv(x[i]),  mv_all[i],  rtol=self.rtol, atol=self.atol)
            onp.testing.assert_allclose(operator[i].inv(b[i]), inv_all[i], rtol=self.rtol, atol=self.atol)


