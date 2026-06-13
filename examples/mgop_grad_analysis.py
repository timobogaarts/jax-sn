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
from jax_sn.domain import Domain
from jax_sn.operators.transport_operator import BasicSweepData
from jax_sn.domain.cross_sections import CrossSectionData
from jax_sn.solution_domain.basix_fem import BasixLagrangianSimplex
from jax_sn.operators.mass_operator import MassOperator
from jax_sn.solution_domain.solution_domain import SolutionDomain
from jax_sn.operators.moment_operator import MomentOperator, DiscreteOperator
from jax_sn.operators.scatter_operator import GatherScatterOperatorBroadcasted
from jax_sn.operators.multi_group_operator import MultiGroupOperatorScan
from jax_sn.operators.multi_group_operator import MultiGroupLowerTriangularBlockSolver
from jax_sn.tools.memory_analysis import _filter_jit_f_to_memory, OperatorAnalysis, pytree_nbytes
from jax_sn.parametrisation import MultigroupOperatorParametrisedVerticesXS

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

def _setup_operators(solution_domain: SolutionDomain, sources: jnp.ndarray, quadrature_set: jax_sn.quadrature_set.QuadratureSet, batch_size = 100) -> Tuple[MultiGroupOperatorScan, jnp.ndarray, SolutionDomain]:        
    n_energy_groups           = solution_domain.domain.cross_sections_indexed.n_energy_groups
    lmax                      = solution_domain.domain.cross_sections_indexed.l_max

    sweep_data = BasicSweepData.from_domain_quadrature_set(solution_domain.domain, quadrature_set, n_iterations_per_loop = 100)
    transport_operator        = TransportOperatorMultipleAnglesVmap(TransportOperatorBatched(solution_domain.solution_domain_data, sweep_data, 0, batch_size))
    mass_operator             = MassOperator.from_transport_operator(transport_operator)
    moment_operator           = MomentOperator.from_quadrature_set(quadrature_set, l_max = lmax, elements_basis_shape=solution_domain.elements_basis_shape)    
    discrete_operator         = DiscreteOperator.from_quadrature_set(quadrature_set, l_max = lmax, elements_basis_shape=solution_domain.elements_basis_shape)
    gather_scatter_operator   = GatherScatterOperatorBroadcasted(cross_sections_indexed=solution_domain.domain.cross_sections_indexed)
    source_total              = jnp.broadcast_to(sources[None, :, :], (sweep_data.angle.shape[0],) + sources.shape) # source for all angles
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

dim, degree = 3,2
bicgstab_solver = lx.BiCGStab(1e-12, 1e-12, max_steps = 160)
lower_triangular_solver = MultiGroupLowerTriangularBlockSolver(bicgstab_solver)

MGOP, source_eg0, _ = modify_setup_mgop(dim, degree, lambda v,c : (v,c))
print("===================")
print("MultiGroupOperatorScan")
print("With:")
print("\t Dimension:", dim)
print("\t Degree:", degree)
print("\t Number of energy groups:", MGOP.n_energy_groups)
print("\t Number of angles:       ", MGOP.n_angles)
print("\t Number of elements:     ", MGOP.n_elements)
print("\t Number of moments       ", MGOP.n_moments)
print("\t Number of basis         ", MGOP.n_basis)
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
    MGOP_new = MultigroupOperatorParametrisedVerticesXS.reconstruct_operator(MGOP, theta)
    solution_value =  lx.linear_solve(MGOP_new, rhs_base, solver=ltri_solver).value
    return jnp.sum(solution_value)

theta_everything_base = MultigroupOperatorParametrisedVerticesXS.theta_base(MGOP)
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
        mgop1 = MultigroupOperatorParametrisedVerticesXS.reconstruct_operator(mgop, theta)
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

from jax_sn.solution_domain import PrecomputedGeometricData
from jax_sn.domain.cross_sections import CrossSectionData
from jax_sn.domain.geometric_domain import compute_face_normal_vectors_from_3d_simplices
from jax_sn.solution_domain.basix_fem import _precompute_jacobians_simplex

theta_everything_base = MultigroupOperatorParametrisedVerticesXS.theta_base(MGOP)

@eqx.filter_jit
def solve_result_everything(theta, MGOP, rhs_base, ltri_solver):    
    MGOP_new = MultigroupOperatorParametrisedVerticesXS.reconstruct_operator(MGOP, theta)
    solution_value =  lx.linear_solve(MGOP_new, rhs_base, solver=ltri_solver).value
    return jnp.sum(solution_value)

grad_theta_everything = eqx.filter_jit(jax.grad(solve_result_everything, argnums=0))


def print_mem_timing(theta, mgop, rhs_base, ltri_solver):
    @eqx.filter_jit
    def f(theta):
        mgop_new = MultigroupOperatorParametrisedVerticesXS.reconstruct_operator(mgop, theta)
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
theta_everything_base = MultigroupOperatorParametrisedVerticesXS.theta_base(MGOP)

print("Without checkpointing Lower Triangular")
print("======================================")
print_mem_timing(theta_everything_base, MGOP, rhs, ltri_solver)
print("======================================\n\n")
print("With checkpointing Lower Triangular")
print("======================================")
print_mem_timing(theta_everything_base, MGOP_cp, rhs, ltri_solver)
print("======================================")


# RTx 3080
# ===================
# MultiGroupOperatorScan
# With:
#          Dimension: 3
#          Degree: 2
#          Number of energy groups: 4
#          Number of angles:        72
#          Number of elements:      15000
#          Number of moments        36
#          Number of basis          10
# No checkpointing:
# ==============================
# None.mv memory usage:
#          Direct {'peak': 543.8956604003906, 'argument': 183.41949462890625, 'output': 3.0517578125e-05, 'temp': 360.4821090698242} MB
#          Grad {'peak': 1624.0409088134766, 'argument': 183.41949462890625, 'output': 0.07816314697265625, 'temp': 1440.5605926513672} MB
#   Timing:
#          Direct 1.9877829551696777
#          Grad 4.17750871181488
# moment_operator.mv memory usage:
#          Direct {'peak': 127.61600494384766, 'argument': 82.417236328125, 'output': 3.0517578125e-05, 'temp': 45.19898986816406} MB
#          Grad {'peak': 127.69413757324219, 'argument': 82.417236328125, 'output': 0.07816314697265625, 'temp': 45.199012756347656} MB
#   Timing:
#          Direct 0.005095124244689941
#          Grad 0.005202889442443848
# transport_operator.inv memory usage:
#          Direct {'peak': 214.2653045654297, 'argument': 100.98171997070312, 'output': 3.0517578125e-05, 'temp': 113.28772735595703} MB
#          Grad {'peak': 401.4707794189453, 'argument': 100.98171997070312, 'output': 0.07816314697265625, 'temp': 300.4242401123047} MB
#   Timing:
#          Direct 0.5078434944152832
#          Grad 1.0253645181655884
# transport_operator.mv memory usage:
#          Direct {'peak': 176.79979705810547, 'argument': 84.49629211425781, 'output': 3.0517578125e-05, 'temp': 92.30725860595703} MB
#          Grad {'peak': 221.48089599609375, 'argument': 84.49629211425781, 'output': 0.07816314697265625, 'temp': 136.9152069091797} MB
#   Timing:
#          Direct 0.035828590393066406
#          Grad 0.17823994159698486
# mass_operator.mv memory usage:
#          Direct {'peak': 251.72295379638672, 'argument': 82.92799377441406, 'output': 3.0517578125e-05, 'temp': 168.79518127441406} MB
#          Grad {'peak': 251.80108642578125, 'argument': 82.92799377441406, 'output': 0.07816314697265625, 'temp': 168.79569244384766} MB
#   Timing:
#          Direct 0.003974437713623047
#          Grad 0.005253314971923828
# ==============================
#
# With checkpointing:
# ==============================
# None.mv memory usage:
#          Direct {'peak': 543.8956604003906, 'argument': 183.41949462890625, 'output': 3.0517578125e-05, 'temp': 360.4821090698242} MB
#          Grad {'peak': 954.3883743286133, 'argument': 183.41949462890625, 'output': 0.07816314697265625, 'temp': 770.9099426269531} MB
#   Timing:
#          Direct 1.963196039199829
#          Grad 6.137986779212952
# moment_operator.mv memory usage:
#          Direct {'peak': 127.61600494384766, 'argument': 82.417236328125, 'output': 3.0517578125e-05, 'temp': 45.19898986816406} MB
#          Grad {'peak': 127.69413757324219, 'argument': 82.417236328125, 'output': 0.07816314697265625, 'temp': 45.199012756347656} MB
#   Timing:
#          Direct 0.005663275718688965
#          Grad 0.005121707916259766
# transport_operator.inv memory usage:
#          Direct {'peak': 214.2653045654297, 'argument': 100.98171997070312, 'output': 3.0517578125e-05, 'temp': 113.28772735595703} MB
#          Grad {'peak': 401.4707794189453, 'argument': 100.98171997070312, 'output': 0.07816314697265625, 'temp': 300.4242401123047} MB
#   Timing:
#          Direct 0.48498594760894775
#          Grad 1.0300859212875366
# transport_operator.mv memory usage:
#          Direct {'peak': 176.79979705810547, 'argument': 84.49629211425781, 'output': 3.0517578125e-05, 'temp': 92.30725860595703} MB
#          Grad {'peak': 221.48089599609375, 'argument': 84.49629211425781, 'output': 0.07816314697265625, 'temp': 136.9152069091797} MB
#   Timing:
#          Direct 0.08780908584594727
#          Grad 0.17808616161346436
# mass_operator.mv memory usage:
#          Direct {'peak': 251.72295379638672, 'argument': 82.92799377441406, 'output': 3.0517578125e-05, 'temp': 168.79518127441406} MB
#          Grad {'peak': 251.80108642578125, 'argument': 82.92799377441406, 'output': 0.07816314697265625, 'temp': 168.79569244384766} MB
#   Timing:
#          Direct 0.004551887512207031
#          Grad 0.005142331123352051
# ==============================
# No checkpointing:
# ==============================
# Transport Operator
# {'peak': 215.2952651977539, 'argument': 104.98715209960938, 'output': 82.3974838256836, 'temp': 27.91297149658203}
# Single group operator
# {'peak': 281.5982894897461, 'argument': 64.0634765625, 'output': 41.198753356933594, 'temp': 176.33899688720703}
# Multi group operator
# {'peak': 545.1603698730469, 'argument': 187.65966796875, 'output': 164.7949447631836, 'temp': 192.70990753173828}
# Forward solve
# {'peak': 1673.4162607192993, 'argument': 190.2606658935547, 'output': 164.7949447631836, 'temp': 1318.37255859375}
# Forward tri solve
# {'peak': 898.8746118545532, 'argument': 190.2606658935547, 'output': 164.7949447631836, 'temp': 622.2316970825195}
# Backward solve
# {'peak': 1874.3625802993774, 'argument': 190.2606658935547, 'output': 164.7949447631836, 'temp': 1519.321533203125}
# Backward tri solve
# {'peak': 1099.9423141479492, 'argument': 190.2606658935547, 'output': 164.7949447631836, 'temp': 758.4199295043945}
# ==============================
#
# With checkpointing:
# ==============================
# Transport Operator
# {'peak': 215.2952651977539, 'argument': 104.98715209960938, 'output': 82.3974838256836, 'temp': 27.91297149658203}
# Single group operator
# {'peak': 281.5982894897461, 'argument': 64.0634765625, 'output': 41.198753356933594, 'temp': 176.33899688720703}
# Multi group operator
# {'peak': 545.1603698730469, 'argument': 187.65966796875, 'output': 164.7949447631836, 'temp': 192.70990753173828}
# Forward solve
# {'peak': 1673.4162607192993, 'argument': 190.2606658935547, 'output': 164.7949447631836, 'temp': 1318.37255859375}
# Forward tri solve
# {'peak': 898.8746118545532, 'argument': 190.2606658935547, 'output': 164.7949447631836, 'temp': 622.2316970825195}
# Backward solve
# {'peak': 1874.3625802993774, 'argument': 190.2606658935547, 'output': 164.7949447631836, 'temp': 1519.321533203125}
# Backward tri solve
# {'peak': 1099.9423141479492, 'argument': 190.2606658935547, 'output': 164.7949447631836, 'temp': 758.4199295043945}
# ==============================
# Everything parametrised
# Without checkpointing Lower Triangular
# ======================================
# Solve memory usage:
#          Direct {'peak': 916.5773239135742, 'argument': 0.0781097412109375, 'output': 3.0517578125e-05, 'temp': 790.0023345947266} MB
#          Grad {'peak': 1705.7125244140625, 'argument': 0.0781097412109375, 'output': 0.07816314697265625, 'temp': 1440.3554153442383} MB
#   Timing:
#          Direct 157.4159665107727
#          Grad 364.88139247894287
# ======================================
# With checkpointing Lower Triangular
# ======================================
# Solve memory usage:
#          Direct {'peak': 916.5773239135742, 'argument': 0.0781097412109375, 'output': 3.0517578125e-05, 'temp': 790.0023345947266} MB
#          Grad {'peak': 1494.7709197998047, 'argument': 0.0781097412109375, 'output': 0.07816314697265625, 'temp': 1240.6984481811523} MB
#   Timing:
#          Direct 160.66807341575623
#          Grad 369.60521483421326
# ======================================