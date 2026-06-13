# %%
import jax
import jax.numpy as jnp
import numpy as onp

import jax_sn

jax.config.update("jax_enable_x64", True)
import equinox as eqx
import warp as wp 
from jax_sn.tools.setup_test_mgop import setup_slab_mgop, make_xs_test, setup_layered_slab_mgop
from jax_sn.operators.transport_operator import TransportOperatorBatchedPallas


# %%
N_PER_DIM = 35

def _build_base_mgop():
    """Build the reference mgop with default (Vmap) operators. Expensive — call once."""
    xs = [make_xs_test(m, n_groups=2, l_order=1) for m in ["fuel", "moderator", "reflector"]]
    return setup_layered_slab_mgop(
        dim=3,
        degree=1,
        tn_quadrature_set=3,
        region_sizes=[1.0, 4.0, 2.0],
        region_resolutions=[N_PER_DIM - 3, 2, 1],
        region_xs=xs,
        region_sources=[1.0, 0.5, 0.2],
        transverse_specs=[(2.0, N_PER_DIM), (1.523, N_PER_DIM)],
        use_numpy=True,
        single_angle_transport_operator_cls = TransportOperatorBatchedPallas, single_angle_transport_operator_cls_kwargs = {"batch_size": 100}
    )

# %%
onp.random.seed(2042026)


base_mgop = _build_base_mgop()
b_apply = jnp.array(onp.random.random_sample(base_mgop.transport_operator.out_structure().shape))
xs = base_mgop.domain.cross_sections_indexed.map_to_full_array(0)
# %%
from jax_sn.operators.transport_operator.transport_operator_warp import TransportOperatorSequentialWarpSingleAngle, TransportOperatorSequentialWarpMultipleAngles, TransportOperatorLevelsWarpSingleAngle, TransportOperatorLevelsWarpMultipleAngles
top_sa = TransportOperatorSequentialWarpSingleAngle(base_mgop.solution_domain_data, base_mgop.transport_operator[0].sweep_data, 0, batch_size = 100)
top_ma = TransportOperatorSequentialWarpMultipleAngles(TransportOperatorSequentialWarpSingleAngle(base_mgop.solution_domain_data, base_mgop.transport_operator.sweep_data, 0, batch_size = 100))

top_sa_levels = TransportOperatorLevelsWarpSingleAngle(base_mgop.solution_domain_data, base_mgop.transport_operator.sweep_data[0],0, batch_size=100, max_n_levels= None)
top_ma_levels = TransportOperatorLevelsWarpMultipleAngles(TransportOperatorLevelsWarpSingleAngle(base_mgop.solution_domain_data, base_mgop.transport_operator.sweep_data, 0, batch_size = 100, max_n_levels=None))

# %%
import warp  as wp 

wp.init()

# Warmup

top_sa.inv(b_apply[0]).block_until_ready()
top_sa_levels.inv(b_apply[0]).block_until_ready()
top_ma.inv(b_apply).block_until_ready()
top_ma_levels.inv(b_apply).block_until_ready()


import timeit
def _time_sweep(op, b, name, number = 5):    
    duration = timeit.timeit(lambda  : op.inv(b).block_until_ready(), number = number) / number
    print(name, f"{duration * 1000:.2f}", "ms")
print("For a transport operator with ")
print("\tn_elements =", base_mgop.domain.n_elements)
print("\tn_angles   =", base_mgop.transport_operator.n_angles)  

print("Single angle")
print("---------------")
_time_sweep(base_mgop.transport_operator, b_apply, "Reference (Pallas)           ", number = 5)
_time_sweep(top_sa, b_apply[0],                    "Single angle sequential      ", number = 5)
_time_sweep(top_sa_levels, b_apply[0],             "Single angle level-based     ", number = 100)
print("\nMultiple angles")
print("---------------")
_time_sweep(base_mgop.transport_operator, b_apply, "Reference (Pallas)           ", number = 5)
_time_sweep(top_ma, b_apply,                       "Multiple angle sequential    ", number = 3)
_time_sweep(top_ma_levels, b_apply,                "Multiple angle level-based   ", number = 100)


# Example output on RTX 3080
# Warp 1.12.0 initialized:
#    CUDA Toolkit 12.9, Driver 13.1
#    Devices:
#      "cpu"      : "x86_64"
#      "cuda:0"   : "NVIDIA GeForce RTX 3080" (10 GiB, sm_86, mempool enabled)
#    Kernel cache:
#      /home/tbogaarts/.cache/warp/1.12.0
# Module jax_sn.operators.transport_operator.linear_tetrahedron_warp_kernel 03b08e1 load on device 'cuda:0' took 1.31 ms  (cached)
# For a transport operator with
#         n_elements = 162000
#         n_angles   = 72
# Single angle
# ---------------
# Reference (Pallas)            615.94 ms
# Single angle sequential       739.43 ms
# Single angle level-based      10.38 ms

# Multiple angles
# ---------------
# Reference (Pallas)            469.79 ms
# Multiple angle sequential     2002.46 ms
# Multiple angle level-based    34.03 ms