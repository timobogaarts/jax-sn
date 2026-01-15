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
from jax_sn.operators.mass_operator import MassOperator
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
from finite_elements.test_moment_scatter_operator import _setup_all_sb
from functions_for_testing import solve_and_plot

# %%
from jax_sn.domain.domain_generation import create_cartesian_setup
from jax_sn.solution_domain import BasixLagrangianSimplex
from jax_sn.solution_domain import SolutionDomain
from jax_sn.domain import CrossSectionData, Domain
from jax_sn.source import Source
from jax_sn.tools.memory_analysis import _filter_jit_f_to_memory
import lineax as lx

# %%
from finite_elements.test_multigroup_derivative   import modify_setup_mgop,get_rhs, _get_bb_verts_conn_xs_sources, base_region_sizes, base_resolutions, setup_from_verts, _setup_operators
def modify_setup_mgop(dim, degree, modify_verts_xs):
    verts, conn, cross_sections, sources = _get_bb_verts_conn_xs_sources(dim, base_region_sizes(), base_resolutions(), 5.0, 10)
    verts, cross_sections = modify_verts_xs(verts, cross_sections)

    solution_domain, sources, quadrature_set = setup_from_verts(verts, conn, cross_sections, sources, degree, 3)
    
    return _setup_operators(solution_domain, sources, quadrature_set)    

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



