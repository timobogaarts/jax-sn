# %%
import jax_sn as jsn

import jax_sbgeom

import jax.numpy as jnp
import numpy as onp
from typing import Tuple
import jax
import numpy as onp
from functools import partial
jax.config.update("jax_enable_x64", True)
from jax_sn.operators.transport_operator import TransportOperatorBatched, TransportOperatorMultipleAnglesVmap, TransportOperatorVmap, TransportOperatorBatchedPallas, TransportOperatorMultipleAngles
from jax_sn.domain.geometric_domain import Domain, TopologicalSortedAngle
from jax_sn.domain.cross_sections import CrossSectionData
from jax_sn.solution_domain.basix_fem import BasixLagrangianSimplex
from jax_sn.operators.mass_operator import MassOperatorVmap
from jax_sn.solution_domain.solution_domain import SolutionDomain
from jax_sn.operators.moment_operator import MomentOperator, DiscreteOperator
from jax_sn.operators.scatter_operator import GatherScatterOperatorBroadcasted
from jax_sn.operators.multi_group_operator import MultiGroupOperatorScan
from jax_sn.operators.multi_group_operator import MultiGroupLowerTriangularBlockSolver
from jax_sn.tools.memory_analysis import MultiGroupOperatorSigmaAllParametrised, _filter_jit_f_to_memory, OperatorAnalysis, pytree_nbytes, MultiGroupOperatorEverythingParametrised

from jax_sn.domain.cross_sections import CrossSectionsIndexed
import equinox as eqx
from typing import List
from jax_sn.tools.mesh_tools import _pad_last_dimension
import lineax as lx
import matplotlib.pyplot as plt
from dataclasses import dataclass
onp.set_printoptions(linewidth = 200)


# %%
from jax_sn.domain.domain_generation import create_cartesian_setup
from jax_sn.solution_domain import BasixLagrangianSimplex
from jax_sn.solution_domain import SolutionDomain
from jax_sn.domain import CrossSectionData, Domain
from jax_sn.source import Source
from jax_sn.tools.memory_analysis import _filter_jit_f_to_memory
import lineax as lx
import jax_sn
# %%
from breeding_blanket_setup import bb_data_dict, base_region_sizes, base_resolutions

def _get_bb_verts_conn_xs_sources(dim, d_elem_regions, n_elem_regions, d_other_dim, n_other_dim):    
    data_dictionary = bb_data_dict() # dictionary with matrices
    xs_data_dict = {name : CrossSectionData(total = jnp.array(data_dictionary[name][0]), scattering = jnp.array(data_dictionary[name][1])) for name in data_dictionary.keys()}    
    other_dims   = [[(d_other_dim, n_other_dim)]]*(dim - 1)
    region_names = ['Plasma', 'TungstenArmor', 'FirstWall', 'BreedingZone','BackSupportingStructure', 'VVWallFront', 'VVShield', 'VVWallBack']
    xs_regions = [xs_data_dict[i] for i in region_names]
    region_size_res = [(size, n_elem) for size, n_elem in zip(d_elem_regions, n_elem_regions)]    
    source_eg0 = [1 if i ==0 else 0 for i in range(len(region_names))]

    verts, conn, cross_sections, sources = create_cartesian_setup(dim, [region_size_res] + other_dims, xs_regions, source_eg0, 0)
    return verts, conn, cross_sections, sources

def setup_from_verts(verts, conn, cross_sections, sources,  degree, tn_order):        
    element_type                         = BasixLagrangianSimplex(degree, verts.shape[-1])
    domain                               = Domain.from_mesh_and_cross_sections((verts, conn), element_type.face_template, cross_sections)    
    solution_domain                      = SolutionDomain.from_element_and_domain(element_type, domain)
    # isotropic flat source
    sources = jnp.repeat(sources, repeats = solution_domain.n_basis, axis = -1).reshape(-1, solution_domain.n_basis)

    return solution_domain, sources, jax_sn.quadrature_set.create_tn_quadrature_set(tn_order)

def _setup_operators(solution_domain: SolutionDomain, sources: jnp.ndarray, quadrature_set: jax_sn.quadrature_set.QuadratureSet, batch_size = 50) -> Tuple[MultiGroupOperatorScan, jnp.ndarray, SolutionDomain]:        
    n_energy_groups           = solution_domain.domain.cross_sections_indexed.n_energy_groups
    lmax                      = solution_domain.domain.cross_sections_indexed.l_max

    topological_sorted_angles = TopologicalSortedAngle.from_domain_quadrature_set(solution_domain.domain, quadrature_set, n_iterations_per_loop = 100)
    transport_operator        = TransportOperatorMultipleAnglesVmap(TransportOperatorBatched(solution_domain.solution_domain_data, topological_sorted_angles, 0, batch_size))
    mass_operator             = MassOperatorVmap.from_transport_operator(transport_operator)
    moment_operator           = MomentOperator.from_quadrature_set(quadrature_set, l_max = lmax, elements_basis_shape=solution_domain.elements_basis_shape)    
    discrete_operator         = DiscreteOperator.from_quadrature_set(quadrature_set, l_max = lmax, elements_basis_shape=solution_domain.elements_basis_shape)
    gather_scatter_operator   = GatherScatterOperatorBroadcasted(cross_sections_indexed=solution_domain.domain.cross_sections_indexed)
    source_total              = jnp.broadcast_to(sources[None, :, :], (topological_sorted_angles.angle.shape[0],) + sources.shape) # source for all angles
    Aop                       = MultiGroupOperatorScan(moment_operator, discrete_operator, transport_operator, mass_operator, gather_scatter_operator, n_energy_groups)
    return Aop.convert_to_lower_block_triangular(), source_total, solution_domain

def modify_setup_mgop(dim, degree, modify_verts_xs):
    verts, conn, cross_sections, sources = _get_bb_verts_conn_xs_sources(dim, base_region_sizes(), base_resolutions(), 5.0, 10)
    verts, cross_sections = modify_verts_xs(verts, cross_sections)

    solution_domain, sources, quadrature_set = setup_from_verts(verts, conn, cross_sections, sources, degree, 3)
    
    return _setup_operators(solution_domain, sources, quadrature_set)    

def get_rhs(MGOP, source_eg0):
    rhs_eg0 = MGOP.moment_operator.mv(MGOP.transport_operator.inv(MGOP.mass_operator.mv(source_eg0)))

    rhs_total = jnp.stack([rhs_eg0] + [jnp.zeros_like(rhs_eg0) for _ in range(MGOP.n_energy_groups -1)], axis=0)
    return rhs_total

dim, degree = 2,2
bicgstab_solver = lx.BiCGStab(1e-12, 1e-12, max_steps = 160)
lower_triangular_solver = MultiGroupLowerTriangularBlockSolver(bicgstab_solver)
MGOP, source_eg0, _ = modify_setup_mgop(dim, degree, lambda v,c : (v,c))
rhs = get_rhs(MGOP, source_eg0)

# %%
# def set_bs_to_1(MGOP):
#     return eqx.tree_at(lambda op : op.transport_operator.transport_operator.batch_size, MGOP, replace = 1)


# MGOP1 = set_bs_to_1(MGOP)


# %%
@eqx.filter_jit
def solve_and_sum(MGOP, rhs, solver):
    solution = lx.linear_solve(MGOP, rhs, solver)
    return jnp.sum(solution.value)



@eqx.filter_jit
def solve_result_everything(theta, MGOP, rhs_base, ltri_solver):    
    MGOP_new = MultiGroupOperatorEverythingParametrised.reconstruct_operator(MGOP, theta)
    solution_value =  lx.linear_solve(MGOP_new, rhs_base, solver=ltri_solver).value
    return jnp.sum(solution_value)

theta_everything_base = MultiGroupOperatorEverythingParametrised.theta_base(MGOP)
grad_theta_everything = eqx.filter_jit(jax.grad(solve_result_everything, argnums=0))

# %%
from jax_sn.operators.multi_group_operator.multi_group_operator_imp import MultiGroupOperatorScanCheckpoint
MGOP_cp = MultiGroupOperatorScanCheckpoint(MGOP.moment_operator, MGOP.discrete_operator, MGOP.transport_operator, MGOP.mass_operator, MGOP.gather_scatter_operator, MGOP.n_energy_groups, MGOP.transpose_bool)

# %%

theta_everything_base.vertices.device

# %%
from jax_sn.tools.memory_analysis import custom_timeit
def print_memory_parametrisation(theta, MGOP, attribute_name, attribute_method):
    
    
    if attribute_name is None:
        op = MGOP
    else:
        op = getattr(MGOP, attribute_name)

    rhs = jnp.ones_like(op.in_structure())

    @eqx.filter_jit
    def sum_attr(mgop, theta, rhs, attribute_name):
        mgop1 = MultiGroupOperatorEverythingParametrised.reconstruct_operator(mgop, theta)
        if attribute_name is None:
            op = mgop1
        else:
            op = getattr(mgop1, attribute_name)

    
        solution = getattr(op, attribute_method)(rhs)
        
        return jnp.sum(solution)
    
    f_grad = eqx.filter_jit(jax.value_and_grad(sum_attr, argnums=1))
    
    print(f"{attribute_name}.{attribute_method} memory usage:\n\t Direct {_filter_jit_f_to_memory(sum_attr, MGOP, theta, rhs, attribute_name)} MB", f"\n\t Grad {_filter_jit_f_to_memory(f_grad, MGOP, theta, rhs, attribute_name)} MB")

    def sum_attr_p():
        return sum_attr(MGOP, theta, rhs, attribute_name)
    
    def cu():
        return f_grad(MGOP, theta, rhs, attribute_name)
    
    print("  Timing:\n\t","Direct", custom_timeit(sum_attr_p, number = 3, ignore_n=1)[0], "\n\t", "Grad", custom_timeit(cu, number = 3, ignore_n=1)[0])


print("No checkpointing:")
print("==============================")
print_memory_parametrisation(theta_everything_base, MGOP, None, "mv")
print_memory_parametrisation(theta_everything_base, MGOP, "moment_operator", "mv")
print_memory_parametrisation(theta_everything_base, MGOP, "transport_operator", "inv")
print_memory_parametrisation(theta_everything_base, MGOP, "transport_operator", "mv")
print_memory_parametrisation(theta_everything_base, MGOP, "mass_operator", "mv")
print("==============================\n\n")
print("With checkpointing:")
print("==============================")

print_memory_parametrisation(theta_everything_base, MGOP_cp, None, "mv")
print_memory_parametrisation(theta_everything_base, MGOP_cp, "moment_operator", "mv")
print_memory_parametrisation(theta_everything_base, MGOP_cp, "transport_operator", "inv")
print_memory_parametrisation(theta_everything_base, MGOP_cp, "transport_operator", "mv")
print_memory_parametrisation(theta_everything_base, MGOP_cp, "mass_operator", "mv")
print("==============================")

# %%
@eqx.filter_jit
def solve_fwd(mgop, rhs, solver):
    return lx.linear_solve(mgop, rhs, solver=solver).value


@eqx.filter_jit
def solve_lower_tri(mgop, rhs, solver):
    ltri_solver=  MultiGroupLowerTriangularBlockSolver(solver)
    return lx.linear_solve(mgop, rhs, solver=ltri_solver).value

@eqx.filter_jit
def solve_bwd(mgop, rhs, solver):    
    return lx.linear_solve(mgop.transpose(), rhs, solver=solver).value

@eqx.filter_jit
def solve_bwd_lower_tri(mgop, rhs, solver):    
    ltri_solver=  MultiGroupLowerTriangularBlockSolver(solver)
    return lx.linear_solve(mgop.transpose(), rhs, solver=ltri_solver).value

def print_costs_solves(MGOP, rhs_base, bicgstab_solver):
    sgop = MGOP.to_single_group_operator(0)
    rhs_base = jnp.ones_like(MGOP.in_structure())
    bicgstab_solver = lx.BiCGStab(1e-12, 1e-12, max_steps = 160)
    print("Transport Operator")
    print(OperatorAnalysis.analyze_memory(sgop.transport_operator, "inv"))
    print("Single group operator")
    print(OperatorAnalysis.analyze_memory(sgop, "mv"))
    print("Multi group operator")
    print(OperatorAnalysis.analyze_memory(MGOP, "mv"))
    print("Forward solve")
    print(_filter_jit_f_to_memory(solve_fwd, MGOP, rhs_base, bicgstab_solver ))
    print("Forward tri solve")
    print(_filter_jit_f_to_memory(solve_lower_tri, MGOP, rhs_base, bicgstab_solver ))

    print("Backward solve")
    print(_filter_jit_f_to_memory(solve_bwd, MGOP, rhs_base, bicgstab_solver ))
    print("Backward tri solve")
    print(_filter_jit_f_to_memory(solve_bwd_lower_tri, MGOP, rhs_base, bicgstab_solver ))
print("No checkpointing:")
print("==============================")
print_costs_solves(MGOP, rhs, bicgstab_solver)
print("==============================\n\n")
print("With checkpointing:")
print("==============================")
print_costs_solves(MGOP_cp, rhs, bicgstab_solver)
print("==============================")

# %%
ltri_solver = MultiGroupLowerTriangularBlockSolver(bicgstab_solver)

# %%
from jax_sn.tools.memory_analysis import OperatorThetaParametrised
from jax_sn.solution_domain import PrecomputedGeometricData
from jax_sn.domain.cross_sections import CrossSectionData
from jax_sn.domain.geometric_domain import compute_face_normal_vectors_from_3d_simplices
from jax_sn.solution_domain.basix_fem import _precompute_jacobians_simplex

theta_everything_base = MultiGroupOperatorEverythingParametrised.theta_base(MGOP)

@eqx.filter_jit
def solve_result_everything(theta, MGOP, rhs_base, ltri_solver):    
    MGOP_new = MultiGroupOperatorEverythingParametrised.reconstruct_operator(MGOP, theta)
    solution_value =  lx.linear_solve(MGOP_new, rhs_base, solver=ltri_solver).value
    return jnp.sum(solution_value)

grad_theta_everything = eqx.filter_jit(jax.grad(solve_result_everything, argnums=0))


def print_mem_timing(theta, mgop, rhs_base, ltri_solver):
    @eqx.filter_jit
    def f(theta):
        mgop_new = MultiGroupOperatorEverythingParametrised.reconstruct_operator(mgop, theta)
        solution_value =  lx.linear_solve(mgop_new, rhs_base, solver=ltri_solver).value
        return jnp.sum(solution_value)
    
    f_grad = eqx.filter_jit(jax.value_and_grad(f, argnums=0))
    
    print(f"Solve memory usage:\n\t Direct {_filter_jit_f_to_memory(f, theta)} MB", f"\n\t Grad {_filter_jit_f_to_memory(f_grad, theta)} MB")

    def sum_attr_p():
        return f(theta)
    
    def cu():
        return f_grad(theta)
    
    print("  Timing:\n\t","Direct", custom_timeit(sum_attr_p, number = 1, ignore_n=0)[0], "\n\t", "Grad", custom_timeit(cu, number = 1, ignore_n=0)[0])

# %%
print("Everything parametrised")
theta_everything_base = MultiGroupOperatorEverythingParametrised.theta_base(MGOP)

print("Without checkpointing Lower Triangular")
print("======================================")
print_mem_timing(theta_everything_base, MGOP, rhs, ltri_solver)
print("======================================\n\n")
print("With checkpointing Lower Triangular")
print("======================================")
print_mem_timing(theta_everything_base, MGOP_cp, rhs, ltri_solver)
print("======================================")



# No checkpointing:
# ==============================
# None.mv memory usage:
#          Direct {'peak': 14.758445739746094, 'argument': 3.95343017578125, 'output': 3.0517578125e-05, 'temp': 10.828300476074219} MB
#          Grad {'peak': 37.15237522125244, 'argument': 3.95343017578125, 'output': 0.01052093505859375, 'temp': 33.20463562011719} MB
#   Timing:
#          Direct 0.07041335105895996
#          Grad 0.14512646198272705
# moment_operator.mv memory usage:
#          Direct {'peak': 4.159461975097656, 'argument': 1.667724609375, 'output': 3.0517578125e-05, 'temp': 2.4919586181640625} MB
#          Grad {'peak': 4.169952392578125, 'argument': 1.667724609375, 'output': 0.01052093505859375, 'temp': 2.4919815063476562} MB
#   Timing:
#          Direct 0.0005929470062255859
#          Grad 0.0005670785903930664
# transport_operator.inv memory usage:
#          Direct {'peak': 8.126632690429688, 'argument': 2.265655517578125, 'output': 3.0517578125e-05, 'temp': 5.864631652832031} MB
#          Grad {'peak': 11.812332153320312, 'argument': 2.265655517578125, 'output': 0.01052093505859375, 'temp': 9.547286987304688} MB
#   Timing:
#          Direct 0.041327714920043945
#          Grad 0.06586384773254395
# transport_operator.mv memory usage:
#          Direct {'peak': 4.622779846191406, 'argument': 1.7104339599609375, 'output': 3.0517578125e-05, 'temp': 2.9154129028320312} MB
#          Grad {'peak': 10.690963745117188, 'argument': 1.7104339599609375, 'output': 0.01052093505859375, 'temp': 8.977218627929688} MB
#   Timing:
#          Direct 0.006909847259521484
#          Grad 0.016488194465637207
# mass_operator.mv memory usage:
#          Direct {'peak': 6.608192443847656, 'argument': 1.664031982421875, 'output': 3.0517578125e-05, 'temp': 4.9445953369140625} MB
#          Grad {'peak': 6.618682861328125, 'argument': 1.664031982421875, 'output': 0.01052093505859375, 'temp': 4.944862365722656} MB
#   Timing:
#          Direct 0.0026912689208984375
#          Grad 0.0028263330459594727
# ==============================


# With checkpointing:
# ==============================
# None.mv memory usage:
#          Direct {'peak': 14.758445739746094, 'argument': 3.95343017578125, 'output': 3.0517578125e-05, 'temp': 10.828300476074219} MB
#          Grad {'peak': 18.551666259765625, 'argument': 3.95343017578125, 'output': 0.01052093505859375, 'temp': 14.606475830078125} MB
#   Timing:
#          Direct 0.07059133052825928
#          Grad 0.22380352020263672
# moment_operator.mv memory usage:
#          Direct {'peak': 4.159461975097656, 'argument': 1.667724609375, 'output': 3.0517578125e-05, 'temp': 2.4919586181640625} MB
#          Grad {'peak': 4.169952392578125, 'argument': 1.667724609375, 'output': 0.01052093505859375, 'temp': 2.4919815063476562} MB
#   Timing:
#          Direct 0.0005769729614257812
#          Grad 0.0005109310150146484
# transport_operator.inv memory usage:
#          Direct {'peak': 8.126632690429688, 'argument': 2.265655517578125, 'output': 3.0517578125e-05, 'temp': 5.864631652832031} MB
#          Grad {'peak': 11.812332153320312, 'argument': 2.265655517578125, 'output': 0.01052093505859375, 'temp': 9.547286987304688} MB
#   Timing:
#          Direct 0.037072062492370605
#          Grad 0.06548380851745605
# transport_operator.mv memory usage:
#          Direct {'peak': 4.622779846191406, 'argument': 1.7104339599609375, 'output': 3.0517578125e-05, 'temp': 2.9154129028320312} MB
#          Grad {'peak': 10.690963745117188, 'argument': 1.7104339599609375, 'output': 0.01052093505859375, 'temp': 8.977218627929688} MB
#   Timing:
#          Direct 0.0074062347412109375
#          Grad 0.02523648738861084
# mass_operator.mv memory usage:
#          Direct {'peak': 6.608192443847656, 'argument': 1.664031982421875, 'output': 3.0517578125e-05, 'temp': 4.9445953369140625} MB
#          Grad {'peak': 6.618682861328125, 'argument': 1.664031982421875, 'output': 0.01052093505859375, 'temp': 4.944862365722656} MB
#   Timing:
#          Direct 0.002141714096069336
#          Grad 0.004360795021057129
# ==============================
# No checkpointing:
# ==============================
# Transport Operator
# {'peak': 8.141891479492188, 'argument': 2.34576416015625, 'output': 1.6479721069335938, 'temp': 4.150764465332031}
# Single group operator
# {'peak': 9.760894775390625, 'argument': 1.575103759765625, 'output': 0.8239974975585938, 'temp': 7.364875793457031}
# Multi group operator
# {'peak': 14.787193298339844, 'argument': 4.047027587890625, 'output': 3.2959213256835938, 'temp': 7.466499328613281}
# Forward solve
# {'peak': 34.61592197418213, 'argument': 4.0999908447265625, 'output': 3.2959213256835938, 'temp': 29.676513671875}
# Forward tri solve
# {'peak': 20.745513916015625, 'argument': 4.0999908447265625, 'output': 3.2959213256835938, 'temp': 14.849861145019531}
# Backward solve
# {'peak': 41.48410129547119, 'argument': 4.0999908447265625, 'output': 3.2959213256835938, 'temp': 34.102783203125}
# Backward tri solve
# {'peak': 26.108474731445312, 'argument': 4.0999908447265625, 'output': 3.2959213256835938, 'temp': 19.38526153564453}
# ==============================


# With checkpointing:
# ==============================
# Transport Operator
# {'peak': 8.141891479492188, 'argument': 2.34576416015625, 'output': 1.6479721069335938, 'temp': 4.150764465332031}
# Single group operator
# {'peak': 9.760894775390625, 'argument': 1.575103759765625, 'output': 0.8239974975585938, 'temp': 7.364875793457031}
# Multi group operator
# {'peak': 14.787193298339844, 'argument': 4.047027587890625, 'output': 3.2959213256835938, 'temp': 7.466499328613281}
# Forward solve
# {'peak': 34.61592197418213, 'argument': 4.0999908447265625, 'output': 3.2959213256835938, 'temp': 29.676513671875}
# Forward tri solve
# {'peak': 20.745513916015625, 'argument': 4.0999908447265625, 'output': 3.2959213256835938, 'temp': 14.849861145019531}
# Backward solve
# {'peak': 41.48410129547119, 'argument': 4.0999908447265625, 'output': 3.2959213256835938, 'temp': 34.102783203125}
# Backward tri solve
# {'peak': 26.108474731445312, 'argument': 4.0999908447265625, 'output': 3.2959213256835938, 'temp': 19.38526153564453}
# ==============================
# Everything parametrised
# Without checkpointing Lower Triangular
# ======================================
# Solve memory usage:
#          Direct {'peak': 21.39806365966797, 'argument': 0.010467529296875, 'output': 3.0517578125e-05, 'temp': 17.877822875976562} MB
#          Grad {'peak': 38.07039737701416, 'argument': 0.010467529296875, 'output': 0.01052093505859375, 'temp': 31.93964385986328} MB
#   Timing:
#          Direct 6.132092475891113
#          Grad 13.027214288711548
# ======================================


# With checkpointing Lower Triangular
# ======================================
# Solve memory usage:
#          Direct {'peak': 21.39806365966797, 'argument': 0.010467529296875, 'output': 3.0517578125e-05, 'temp': 17.877822875976562} MB
#          Grad {'peak': 34.65501403808594, 'argument': 0.010467529296875, 'output': 0.01052093505859375, 'temp': 28.947959899902344} MB
#   Timing:
#          Direct 6.0999157428741455
#          Grad 13.124618291854858
# ======================================