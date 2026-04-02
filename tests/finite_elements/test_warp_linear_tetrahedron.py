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
# Operator variants to compare.
# Each entry: (id, single_cls, multi_cls, single_kwargs, multi_kwargs)
# ---------------------------------------------------------------------------

OPERATOR_CONFIGS = [
    pytest.param(
        TransportOperatorVmap, TransportOperatorMultipleAnglesVmap, {}, {},
        id="vmap",
    ),
    pytest.param(
        TransportOperatorBatched, TransportOperatorMultipleAnglesVmap, {}, {},
        id="batched",
    ),
    # pytest.param(
    #     TransportOperatorWarp, TransportOperatorMultipleAnglesVmap, {}, {},
    #     id="warp",
    # ),
]


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
        from jax_sn.operators.transport_operator.linear_tetrahedron_warp_kernel import WarpSweep
        """Build the reference mgop and its solution once for the entire class."""

        onp.random.seed(2042026)



        base_mgop = _build_base_mgop()

        b_apply = onp.random.random_sample(base_mgop.transport_operator.out_structure().shape)
        request.cls.base_mgop      = base_mgop

        request.cls.xs_0 = self.base_mgop.domain.cross_sections_indexed.map_to_full_array(0).total
        request.cls.warpsweep = WarpSweep.from_transport_operator(self.base_mgop.transport_operator)

        request.cls.b_apply = jnp.array(b_apply)
        request.cls.result_reference = base_mgop.transport_operator.inv(b_apply)
        
    # -----------------------------------------------------------------------

    def test_sequential_single_angle_solve(self):
        """Test the sequential solve for a single angle."""

        angle_i = 4

        result_sa = self.warpsweep.compute_single_angle(self.xs_0, angle_i = angle_i, x_apply = self.b_apply[angle_i])

        onp.testing.assert_allclose(result_sa, self.result_reference[ angle_i], rtol=1e-10, atol=1e-10)

        
        

    
    def test_sequential_solve(self):    
        result = self.warpsweep.compute_all_angles(self.xs_0, self.b_apply)        
        onp.testing.assert_allclose(result, self.result_reference, rtol=1e-10, atol=1e-10)

    
        

    