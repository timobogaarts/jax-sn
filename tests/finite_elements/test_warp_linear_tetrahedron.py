import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as onp
import lineax as lx
import equinox as eqx
import pytest

from jax_sn.operators.transport_operator import (
    TransportOperatorVmap,
    TransportOperatorBatched,
    TransportOperatorMultipleAnglesVmap,
)
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

class TestWarpTransportOperator:

    @pytest.fixture(scope="class", autouse=True)
    def setup_class_fixtures(self, request):

        """Build the reference mgop and its solution once for the entire class."""

        onp.random.seed(2042026)

        base_mgop = _build_base_mgop()

        b_apply = onp.random.random_sample(base_mgop.transport_operator.out_structure().shape)
        request.cls.base_mgop      = base_mgop

        request.cls.xs_0 = self.base_mgop.domain.cross_sections_indexed.map_to_full_array(0).total
        
        request.cls.b_apply = jnp.array(b_apply)
        request.cls.result_reference = base_mgop.transport_operator.inv(b_apply)
        
    # -----------------------------------------------------------------------

    def test_sequential_single_angle_solve(self):
        """Test the sequential solve for a single angle."""
        angle_i = 4        
        from jax_sn.operators.transport_operator.linear_tetrahedron_warp_kernel import TransportOperatorSequentialWarpSingleAngle
        top_2 = TransportOperatorSequentialWarpSingleAngle(self.base_mgop.solution_domain_data, self.base_mgop.transport_operator[angle_i].topological_sorted_angle, 0)
        onp.testing.assert_allclose(top_2.inv(self.b_apply[angle_i]), self.result_reference[angle_i], rtol=1e-10, atol=1e-10)

    def test_sequential_single_angle_solve_grad(self):

        """Test the gradient of the sequential solve for a single angle."""
        angle_i = 4        
        from jax_sn.operators.transport_operator.linear_tetrahedron_warp_kernel import TransportOperatorSequentialWarpSingleAngle
        top_2 = TransportOperatorSequentialWarpSingleAngle(self.base_mgop.solution_domain_data, self.base_mgop.transport_operator[angle_i].topological_sorted_angle, 0)

        @eqx.filter_jit
        def solve_and_sum(top_2 : TransportOperatorSequentialWarpSingleAngle):
            return jnp.sum(top_2.inv(self.b_apply[angle_i]))
        
        grad_f = eqx.filter_jit(eqx.filter_grad(solve_and_sum))
        grad_result = grad_f(top_2)
        grad_reference = grad_f(self.base_mgop.transport_operator[angle_i])
        from jax_sn.tools import assert_pytree_allclose
        assert_pytree_allclose(grad_result.solution_domain_data, grad_reference.solution_domain_data, rtol=1e-5, atol=1e-5)
   
    
    def test_sequential_multi_angle_solve(self):
        from jax_sn.operators.transport_operator.linear_tetrahedron_warp_kernel import TransportOperatorSequentialWarpSingleAngle, TransportOperatorSequentialWarpMultipleAngles

        top_sa = TransportOperatorSequentialWarpSingleAngle(self.base_mgop.solution_domain_data, self.base_mgop.transport_operator.topological_sorted_angle, 0)
        top_ma = TransportOperatorSequentialWarpMultipleAngles(top_sa)

        result_warp = top_ma.inv(self.b_apply)

        onp.testing.assert_allclose(result_warp, self.result_reference, rtol=1e-10, atol=1e-10)

    def test_sequential_multi_angle_solve_grad(self):
        from jax_sn.operators.transport_operator.linear_tetrahedron_warp_kernel import TransportOperatorSequentialWarpSingleAngle, TransportOperatorSequentialWarpMultipleAngles

        top_sa = TransportOperatorSequentialWarpSingleAngle(self.base_mgop.solution_domain_data, self.base_mgop.transport_operator.topological_sorted_angle, 0)
        top_ma = TransportOperatorSequentialWarpMultipleAngles(top_sa)

        @eqx.filter_jit
        def solve_and_sum(top_ma : TransportOperatorSequentialWarpMultipleAngles):
            return jnp.sum(top_ma.inv(self.b_apply))
        
        grad_f = eqx.filter_jit(eqx.filter_grad(solve_and_sum))
        grad_result = grad_f(top_ma)
        grad_reference = grad_f(self.base_mgop.transport_operator)
        from jax_sn.tools import assert_pytree_allclose
        assert_pytree_allclose(grad_result.solution_domain_data, grad_reference.solution_domain_data, rtol=1e-5, atol=1e-5)