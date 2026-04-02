# %%
import os

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from breeding_blanket_3d import get_blanket_base_case

from jax_sbgeom.jax_utils import mesh_to_pyvista_mesh
import jax_sn
from jax_sn.operators.transport_operator import TransportOperatorBatchedPallas
import jax_sbgeom as jsb

# %%
discrete_blanket, flux_surface,  data, nt_data, d_physical = get_blanket_base_case()

# %%
jax.devices()[0]

# %%
mesh_base = discrete_blanket.volume_mesh(flux_surface)
element = jax_sn.solution_domain.BasixLagrangianSimplex(1,3)
cross_sections = jax_sn.domain.cross_sections.CrossSectionsIndexed(discrete_blanket.layer_array, data)
domain = jax_sn.domain.Domain.from_mesh_and_cross_sections(mesh_base, element.face_template, cross_sections)
quadrature_set = jax_sn.quadrature_set.create_tn_quadrature_set(3)
solution_domain = jax_sn.solution_domain.SolutionDomain.from_element_and_domain(element, domain)

# %%
from breeding_blanket_3d import get_source

source = jnp.stack([get_source(discrete_blanket)] * solution_domain.n_basis, axis=-1)

# %%
mgop = jax_sn.operators.multi_group_operator.multi_group_operator_creation.create_multi_group_operator(solution_domain, quadrature_set,
                                                                                                 single_angle_transport_operator_cls=TransportOperatorBatchedPallas,
                                                                                                 single_angle_transport_operator_cls_kwargs={"batch_size" : 100})

# %%
from warp_kernel_v2 import WarpSweep, LevelBasedParallelismSetup

lsetup = LevelBasedParallelismSetup.from_transport_operator(mgop.transport_operator)

# %%
lsetup.n_levels

# %%
xapply = jnp.ones_like(mgop.transport_operator.out_structure())

result = jnp.zeros_like(xapply)

# %%
result_orig = mgop.transport_operator.inv(xapply)

# %%
import warp as wp
wp.init()


# %%
xs_wp = wp.from_jax(mgop.domain.cross_sections_indexed.map_to_full_array(0).total)

# %%
warpsweep = WarpSweep.from_transport_operator(mgop.transport_operator, xs_wp)

# %%

result_2 = warpsweep.multi_angle_levels_prealloc(xs_wp, xapply, lsetup, result)

# %%
import timeit 
time = timeit.timeit(lambda: warpsweep.multi_angle_levels_prealloc(xs_wp, xapply, lsetup, result), number=5)

print("Time Warp", time / 5)

# %%
time_jax = timeit.timeit(lambda: mgop.transport_operator.inv(xapply).block_until_ready(), number=5)
print("Time JAX", time_jax / 5)

# %%
result_final = wp.to_jax(warpsweep.multi_angle_levels_prealloc(xs_wp, xapply, lsetup, result))

# %%
import numpy as onp
onp.testing.assert_allclose(result_final, result_orig, rtol=1e-7, atol=1e-12)


