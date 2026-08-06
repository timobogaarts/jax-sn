import pytest
import numpy as onp
import equinox as eqx

from jax_sn.operators.transport_operator import (
    VmapSweep, BatchedSweep, VmapPallasSweep, BatchedPallasSweep, VmapAnglesInversion, ScanAnglesInversion,
)
from jax_sn.operators.transport_operator.implementations.transport_operator_warp import (
    WarpSequentialSweep, WarpLevelSweep, WarpSequentialAngles, WarpLevelAngles,
)

from jax_sn.operators.transport_operator.implementations.simplex_warp_kernel import (
    build_kernels, SUPPORTED,
)

from .operator_util import (
    BaseOperatorProperties, Spec, parametrize_strategies, parametrize_instances,
    warp_supported, simplex_n_basis, _rand,
)

_ANGLE_I = 0   # single-angle inversion tests pick this angle from the multi-angle operator


INVERSION_STRATEGIES = {
    VmapSweep:          Spec(),
    BatchedSweep:       Spec(kwargs=dict(batch_size=64)),
    VmapPallasSweep:    Spec(marks=(pytest.mark.pallas,)),
    BatchedPallasSweep: Spec(kwargs=dict(batch_size=64), marks=(pytest.mark.pallas,)),
    WarpSequentialSweep: Spec(supports=warp_supported),
    WarpLevelSweep:      Spec(supports=warp_supported),
}

MULTI_ANGLE_STRATEGIES = {
    VmapAnglesInversion(VmapSweep()):          Spec(),
    ScanAnglesInversion(VmapSweep()):          Spec(),
    VmapAnglesInversion(BatchedSweep(64)):     Spec(),
    WarpSequentialAngles():           Spec(supports=warp_supported),  
    WarpLevelAngles():                Spec(supports=warp_supported),  
}

_SWEEP_DATA_CACHE = {}

def _dim_degree(bb_mgop):
    return bb_mgop[3][0], bb_mgop[3][1]


def _angle_sample(n_angles):    
    return sorted({0, n_angles // 3, (2 * n_angles) // 3, n_angles - 1})


def _sweep_data(bb_mgop, strategy):
    '''The sweep data ``strategy`` needs for this operator: the one the operator already carries if
    that serves (which is the strategy's own ``accepts``), else the one the strategy builds itself.'''
    base_sd = bb_mgop[0].data.sweep_data
    if strategy.accepts(base_sd):
        return base_sd
    dim, degree = _dim_degree(bb_mgop)
    key = (dim, degree, type(strategy).__name__)
    if key not in _SWEEP_DATA_CACHE:
        solution_domain, quadrature_set = bb_mgop[2], bb_mgop[0].data.angular_data.quadrature_set
        _SWEEP_DATA_CACHE[key] = strategy.build_default_sweep_data(solution_domain.domain, quadrature_set.angles)
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
        from bb_setup import transport_operator_from_mgop
        strategy = strategy_cls(**kwargs)
        top = transport_operator_from_mgop(bb_mgop[0])[_ANGLE_I]
        sweep_data = _sweep_data(bb_mgop, strategy)[_ANGLE_I]
        return eqx.tree_at(lambda t: (t.data.sweep_data, t.inversion_strategy), top, (sweep_data, strategy))

    @pytest.fixture
    def build_reference(self, bb_mgop):
        from bb_setup import transport_operator_from_mgop
        return lambda: transport_operator_from_mgop(bb_mgop[0])[_ANGLE_I]

    def test_forward_backward_identity(self, operator):
        '''T^-1(T x) == x and its transpose -- a self-contained round-trip property of the inverse.'''
        x = _rand(operator.in_structure().shape, 8)
        onp.testing.assert_allclose(operator.inv(operator.mv(x)), x, rtol=self.rtol, atol=self.atol)
        onp.testing.assert_allclose(operator.transpose().inv(operator.transpose().mv(x)), x, rtol=self.rtol, atol=self.atol)


@parametrize_instances(MULTI_ANGLE_STRATEGIES)
class TestTransportInversionMultiAngle(_TransportInversionBase):    

    @pytest.fixture
    def operator(self, bb_mgop, strategy):
        from bb_setup import transport_operator_from_mgop
        top = transport_operator_from_mgop(bb_mgop[0])
        sweep_data = _sweep_data(bb_mgop, strategy)
        return eqx.tree_at(lambda t: (t.data.sweep_data, t.inversion_strategy), top, (sweep_data, strategy))

    @pytest.fixture
    def build_reference(self, bb_mgop):
        from bb_setup import transport_operator_from_mgop
        return lambda: transport_operator_from_mgop(bb_mgop[0])

    def test_indexing_consistency(self, operator):        
        x = _rand(operator.in_structure().shape, 10)
        b = _rand(operator.in_structure().shape, 11)
        mv_all, inv_all = operator.mv(x), operator.inv(b)
        for i in _angle_sample(operator.in_structure().shape[0]):
            onp.testing.assert_allclose(operator[i].mv(x[i]),  mv_all[i],  rtol=self.rtol, atol=self.atol)
            onp.testing.assert_allclose(operator[i].inv(b[i]), inv_all[i], rtol=self.rtol, atol=self.atol)




# ---------------------------------------------------------------------------------
# The (n_basis, n_faces) parametrisation itself.
#
# The kernels are compiled per element shape, so the set of shapes they cover is the thing that can
# silently drift -- from the test gate above, or from a wrong Warp type table. These pin both ends.
# ---------------------------------------------------------------------------------
class TestWarpElementParametrisation:

    def test_supported_matches_simplex_families(self):
        '''``SUPPORTED`` and the test gate ``warp_supported`` are written independently (one from Warp
        type widths, one from the (dim, degree) grid); they must describe the same set of elements.'''
        reached = {(simplex_n_basis(dim, degree), dim + 1)
                   for dim in (1, 2, 3) for degree in range(1, 6)
                   if warp_supported(dim, degree)}
        assert reached == set(SUPPORTED)

    def test_every_supported_shape_compiles(self):
        '''Every advertised shape must actually build -- including the ones no (dim, degree) in
        DIM_DEGREES exercises.'''
        for n_basis, n_faces in SUPPORTED:
            kernels = build_kernels(n_basis, n_faces)
            assert (kernels.n_basis, kernels.n_faces) == (n_basis, n_faces)

    def test_build_is_memoised(self):
        '''A rebuild must return the same compiled set rather than recompiling the Warp module.'''
        assert build_kernels(*SUPPORTED[0]) is build_kernels(*SUPPORTED[0])

    @pytest.mark.parametrize("n_basis, n_faces", [(6, 3), (10, 4), (5, 2), (4, 5)])
    def test_rejects_unsupported_shape(self, n_basis, n_faces):
        '''Warp has no vec5d/mat55d, so an element that does not fit must fail loudly at build time
        rather than compiling something subtly wrong.'''
        with pytest.raises(ValueError, match="n_basis"):
            build_kernels(n_basis, n_faces)

    @pytest.mark.parametrize("bb_mgop_dim_degree", [(2, 2)], indirect=True)
    def test_operator_rejects_unsupported_element(self, bb_mgop):
        '''A quadratic triangle (n_basis == 6) exceeds the Warp types; the strategy\'s compatibility
        check must reject it, which is what lets perf_tests self-classify it as a skip.

        Constructed directly rather than via ``eqx.tree_at``: tree_at rebuilds through
        ``tree_unflatten`` and does *not* re-run ``__check_init__``, so it would silently accept the
        incompatible strategy.'''
        from bb_setup import transport_operator_from_mgop
        top = transport_operator_from_mgop(bb_mgop[0])
        with pytest.raises(AssertionError, match="n_basis"):
            type(top)(top.data, top.strategy, WarpSequentialAngles())
