import pytest
import numpy as onp
import jax.numpy as jnp

from jax_sn.operators.transport_operator import VmapSweep, BatchedSweep, VmapPallasSweep, BatchedPallasSweep
from jax_sn.operators.transport_operator.implementations.transport_operator_warp import WarpSequentialSweep, WarpLevelSweep
from jax_sn.operators.multi_group_operator.multi_group_operator_creation import create_multi_group_operator

from .operator_util import (
    BaseOperatorProperties, Spec, parametrize_strategies, linear_tet_only, _rand, _inexact_leaves,
)

_ANGLE_I = 0   # single-angle inversion tests pick this angle from the multi-angle operator

_DEFAULT_INVERSION = VmapSweep   # pure-JAX reference (pivot) every variant is compared against

# Unlike the mass/moment/forward axes, an inversion strategy also fixes the *sweep-data toposort* it needs
# (WarpLevelSweep -> LeveledSweepData, everything else -> BasicSweepData) and the multi-angle *lifting*
# (fused warp lifts vs VmapAngles). create_multi_group_operator resolves both from the strategy, so the
# variant is built through the factory rather than a bare tree_at swap. The warp kernels are the 3-D
# linear tetrahedron only (linear_tet_only). The level-*padded* sweep (a build-parameter, orthogonal to
# the strategy axis) stays in test_warp_linear_tetrahedron.py.
INVERSION_STRATEGIES = {
    VmapSweep:          Spec(),
    BatchedSweep:       Spec(kwargs=dict(batch_size=64)),
    VmapPallasSweep:    Spec(marks=(pytest.mark.pallas,)),
    BatchedPallasSweep: Spec(kwargs=dict(batch_size=64), marks=(pytest.mark.pallas,)),
    WarpSequentialSweep: Spec(supports=linear_tet_only),
    WarpLevelSweep:      Spec(supports=linear_tet_only),
}


# ---------------------------------------------------------------------------
# Per-(dim, degree, strategy) mgop build, cached: the six property methods per parametrised variant would
# otherwise each rebuild the (expensive) sweep-data toposort. Keyed by a string label so heterogeneous
# strategy kwargs stay hashable. The pivot (all-defaults) mgop is cached the same way.
# ---------------------------------------------------------------------------
_MGOP_CACHE = {}


def _sd_qs(bb_mgop):
    return bb_mgop[2], bb_mgop[0].quadrature_set   # (solution_domain, quadrature_set)


def _dim_degree(bb_mgop):
    return bb_mgop[3][0], bb_mgop[3][1]


def _variant_mgop(bb_mgop, strategy_cls, kwargs):
    dim, degree = _dim_degree(bb_mgop)
    key = (dim, degree, f"{strategy_cls.__name__}-{sorted(kwargs.items())}")
    if key not in _MGOP_CACHE:
        sd, qs = _sd_qs(bb_mgop)
        _MGOP_CACHE[key] = create_multi_group_operator(sd, qs, inversion_strategy=strategy_cls(**kwargs))
    return _MGOP_CACHE[key]


def _pivot_mgop(bb_mgop):
    dim, degree = _dim_degree(bb_mgop)
    key = (dim, degree, "__pivot__")
    if key not in _MGOP_CACHE:
        sd, qs = _sd_qs(bb_mgop)
        _MGOP_CACHE[key] = create_multi_group_operator(sd, qs, inversion_strategy=_DEFAULT_INVERSION())
    return _MGOP_CACHE[key]


class _TransportInversionBase(BaseOperatorProperties):
    '''Shared config for the transport inversion axis: the battery runs against ``op.inv`` (via the
    ``apply`` override), tolerances are relaxed to the iterative sweep's accuracy, and the gradient
    comparison is projected onto ``solution_domain_data`` because a WarpLevelSweep variant carries a
    different ``sweep_data`` type than the BasicSweepData pivot (so a full-tree leaf compare would not
    even line up).'''

    rtol_exact, atol_exact = 1e-9, 1e-9
    rtol, atol = 1e-9, 1e-9
    rtol_grad, atol_grad = 1e-7, 1e-7

    def apply(self, operator, x):
        return operator.inv(x)

    def _grad_compare_leaves(self, grad_tree):
        return _inexact_leaves(grad_tree.solution_domain_data)


@parametrize_strategies(INVERSION_STRATEGIES)
class TestTransportInversionSingleAngle(_TransportInversionBase):

    @pytest.fixture
    def operator(self, bb_mgop, strategy_cls, kwargs):
        '''The single-angle inverse sweep with this strategy on the shared BB problem.'''
        return _variant_mgop(bb_mgop, strategy_cls, kwargs).transport_operator[_ANGLE_I]

    @pytest.fixture
    def build_reference(self, bb_mgop):
        return lambda: _pivot_mgop(bb_mgop).transport_operator[_ANGLE_I]

    def test_forward_backward_identity(self, operator):
        '''T^-1(T x) == x and its transpose -- a self-contained round-trip property of the inverse.'''
        x = _rand(operator.in_structure().shape, 8)
        onp.testing.assert_allclose(operator.inv(operator.mv(x)), x, rtol=self.rtol, atol=self.atol)
        onp.testing.assert_allclose(operator.transpose().inv(operator.transpose().mv(x)), x, rtol=self.rtol, atol=self.atol)


@parametrize_strategies(INVERSION_STRATEGIES)
class TestTransportInversionMultiAngle(_TransportInversionBase):
    '''The multi-angle inverse: each inversion uses its own default angle lifting (a fused warp lift for
    the warp kernels, VmapAngles otherwise), so the consistency check is exactly the fused-vs-stacked
    batching-correctness test.'''

    @pytest.fixture
    def operator(self, bb_mgop, strategy_cls, kwargs):
        return _variant_mgop(bb_mgop, strategy_cls, kwargs).transport_operator

    @pytest.fixture
    def build_reference(self, bb_mgop):
        return lambda: _pivot_mgop(bb_mgop).transport_operator


