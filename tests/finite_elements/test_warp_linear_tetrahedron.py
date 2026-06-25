import jax

from jax_sn.tools.memory_analysis import assert_pytree_allclose
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as onp
import lineax as lx
import equinox as eqx
import pytest

from jax_sn.tools.setup_test_mgop import make_xs_test, setup_layered_slab_mgop

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_base_mgop():
    """Build the reference mgop with default (Vmap) operators. Expensive — call once."""
    xs = [make_xs_test(m, n_groups=2, l_order=1) for m in ["fuel", "moderator", "reflector"]]
    return setup_layered_slab_mgop(
        dim=3,
        degree=1,
        tn_quadrature_set=3,
        region_sizes=[1.0, 4.0, 2.0],
        region_resolutions=[2, 1, 3],
        region_xs=xs,
        region_sources=[1.0, 0.5, 0.2],
        transverse_specs=[(2.0, 2), (1.523, 3)],
    )

# ---------------------------------------------------------------------------
# Operator factories
# Add new single-angle or multi-angle operator classes here.
# ---------------------------------------------------------------------------
def make_sequential_warp_single_angle(base_mgop, angle_i):
    from jax_sn.operators.transport_operator import TransportOperator, BatchedForward
    from jax_sn.operators.transport_operator.implementations.transport_operator_warp import WarpSequentialSweep
    return TransportOperator(
        base_mgop.solution_domain_data,
        base_mgop.transport_operator[angle_i].sweep_data,
        base_mgop.transport_operator[angle_i].angle,
        0,
        BatchedForward(100),
        WarpSequentialSweep(),
    )

def make_level_warp_single_angle(base_mgop, angle_i):
    from jax_sn.operators.transport_operator import TransportOperator, BatchedForward, LeveledSweepData
    from jax_sn.operators.transport_operator.implementations.transport_operator_warp import WarpLevelSweep
    level_toposort = LeveledSweepData.from_basic_sweep_data(base_mgop.transport_operator[angle_i].sweep_data)
    return TransportOperator(
        base_mgop.solution_domain_data,
        level_toposort,
        base_mgop.transport_operator[angle_i].angle,
        0,
        BatchedForward(100),
        WarpLevelSweep(),
    )

def make_level_warp_single_angle_pad(base_mgop, angle_i):
    from jax_sn.operators.transport_operator import TransportOperator, BatchedForward, LeveledSweepData
    from jax_sn.operators.transport_operator.implementations.transport_operator_warp import WarpLevelSweep
    level_toposort = LeveledSweepData.from_basic_sweep_data(base_mgop.transport_operator[angle_i].sweep_data, 400)
    return TransportOperator(
        base_mgop.solution_domain_data,
        level_toposort,
        base_mgop.transport_operator[angle_i].angle,
        0,
        BatchedForward(100),
        WarpLevelSweep(),
    )

def make_sequential_warp_multi_angle(base_mgop):
    from jax_sn.operators.transport_operator import TransportOperator, TransportOperatorMultipleAngles, BatchedForward
    from jax_sn.operators.transport_operator.implementations.transport_operator_warp import WarpSequentialSweep, WarpSequentialAngles
    top_sa = TransportOperator(
        base_mgop.solution_domain_data,
        base_mgop.transport_operator.sweep_data,
        base_mgop.transport_operator.angle,
        0,
        BatchedForward(100),
        WarpSequentialSweep(),
    )
    return TransportOperatorMultipleAngles(top_sa, WarpSequentialAngles())


def make_levels_warp_multi_angle(base_mgop):
    from jax_sn.operators.transport_operator import TransportOperator, TransportOperatorMultipleAngles, BatchedForward, LeveledSweepData
    from jax_sn.operators.transport_operator.implementations.transport_operator_warp import WarpLevelSweep, WarpLevelAngles
    level_toposort = LeveledSweepData.from_basic_sweep_data(base_mgop.transport_operator.sweep_data)
    top_sa = TransportOperator(
        base_mgop.solution_domain_data,
        level_toposort,
        base_mgop.transport_operator.angle,
        0,
        BatchedForward(100),
        WarpLevelSweep(),
    )
    return TransportOperatorMultipleAngles(top_sa, WarpLevelAngles())

def make_levels_warp_multi_angle_pad(base_mgop):
    from jax_sn.operators.transport_operator import TransportOperator, TransportOperatorMultipleAngles, BatchedForward, LeveledSweepData
    from jax_sn.operators.transport_operator.implementations.transport_operator_warp import WarpLevelSweep, WarpLevelAngles
    level_toposort = LeveledSweepData.from_basic_sweep_data(base_mgop.transport_operator.sweep_data, 400)
    top_sa = TransportOperator(
        base_mgop.solution_domain_data,
        level_toposort,
        base_mgop.transport_operator.angle,
        0,
        BatchedForward(100),
        WarpLevelSweep(),
    )
    return TransportOperatorMultipleAngles(top_sa, WarpLevelAngles())

# Parametrise over these factories to add more operator classes to test.
SINGLE_ANGLE_OP_FACTORIES = [
    pytest.param(make_sequential_warp_single_angle, id="SequentialWarpSingleAngle"),
    pytest.param(make_level_warp_single_angle, id="LevelsWarpSingleAngle"),
    pytest.param(make_level_warp_single_angle_pad, id="LevelsWarpSingleAnglePadded"),
]

MULTI_ANGLE_OP_FACTORIES = [
    pytest.param(make_sequential_warp_multi_angle, id="SequentialWarpMultipleAngles"),
    pytest.param(make_levels_warp_multi_angle, id="LevelsWarpMultipleAngles"),
    pytest.param(make_levels_warp_multi_angle_pad, id="LevelsWarpMultipleAnglesPadded"),
]

@eqx.filter_jit
def solve_and_sum(op, b):
    return jnp.sum(op.inv(b))

grad_solve_and_sum = eqx.filter_jit(eqx.filter_grad(solve_and_sum))

# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------
from jax_sn.tools import assert_pytree_allclose

class TestWarpTransportOperator:

    @pytest.fixture(scope="class", autouse=True)
    def setup_class_fixtures(self, request):
        """Build the reference mgop and its solution once for the entire class."""

        onp.random.seed(2042026)
        angle_i = 4

        base_mgop = _build_base_mgop()

        b_apply = onp.random.random_sample(base_mgop.transport_operator.out_structure().shape)
        request.cls.base_mgop      = base_mgop
        request.cls.angle_i          = angle_i
        request.cls.xs_0 = base_mgop.domain.cross_sections_indexed.map_to_full_array(0).total
        request.cls.b_apply = jnp.array(b_apply)
        request.cls.result_reference = base_mgop.transport_operator.inv(b_apply)

        request.cls.result_reference_t = base_mgop.transport_operator.transpose().inv(b_apply)
        request.cls.solve_and_sum_ref_sa = grad_solve_and_sum(base_mgop.transport_operator[angle_i], b_apply[angle_i]).solution_domain_data

        request.cls.solve_and_sum_ref_ma = grad_solve_and_sum(base_mgop.transport_operator, b_apply).solution_domain_data
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("op_factory", SINGLE_ANGLE_OP_FACTORIES)
    def test_single_angle_solve(self, op_factory):
        """Test the forward solve for a single-angle operator."""
        angle_i = self.angle_i
        top = op_factory(self.base_mgop, angle_i)
        onp.testing.assert_allclose(top.inv(self.b_apply[angle_i]), self.result_reference[angle_i], rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("op_factory", SINGLE_ANGLE_OP_FACTORIES)
    def test_single_angle_solve_transposed(self, op_factory):
        """Test the forward solve for a single-angle operator in transposed mode."""
        angle_i = self.angle_i
        top = op_factory(self.base_mgop, angle_i)
        onp.testing.assert_allclose(top.transpose().inv(self.b_apply[angle_i]), self.result_reference_t[angle_i], rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("op_factory", SINGLE_ANGLE_OP_FACTORIES)
    def test_single_angle_solve_grad(self, op_factory):
        """Test the gradient of the forward solve for a single-angle operator."""
        angle_i = self.angle_i
        top = op_factory(self.base_mgop, angle_i)

        grad_result = grad_solve_and_sum(top, self.b_apply[angle_i])        
        
        assert_pytree_allclose(grad_result.solution_domain_data, self.solve_and_sum_ref_sa, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("op_factory", MULTI_ANGLE_OP_FACTORIES)
    def test_multi_angle_solve(self, op_factory):
        """Test the forward solve for a multi-angle operator."""
        top = op_factory(self.base_mgop)
        onp.testing.assert_allclose(top.inv(self.b_apply), self.result_reference, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("op_factory", MULTI_ANGLE_OP_FACTORIES)
    def test_multi_angle_solve_transposed(self, op_factory):
        """Test the forward solve for a multi-angle operator in transposed mode."""
        top = op_factory(self.base_mgop)
        onp.testing.assert_allclose(top.transpose().inv(self.b_apply), self.result_reference_t, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("op_factory", MULTI_ANGLE_OP_FACTORIES)
    def test_multi_angle_solve_grad(self, op_factory):
        """Test the gradient of the forward solve for a multi-angle operator."""
        top = op_factory(self.base_mgop)
        grad_result = grad_solve_and_sum(top, self.b_apply)
        from jax_sn.tools import assert_pytree_allclose
        assert_pytree_allclose(grad_result.solution_domain_data, self.solve_and_sum_ref_ma, rtol=1e-10, atol=1e-10)
