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
from jax_sn.operators.transport_operator import TransportOperator, TransportOperatorMultipleAngles, VmapForward, VmapSweep, BatchedForward, BatchedSweep, VmapAngles, TransportOperatorBatchedPallas

def _vmap_op(*args):
    return TransportOperator(*args, VmapForward(), VmapSweep())

def _batched_op(*args, batch_size=None):
    return TransportOperator(*args, BatchedForward(batch_size), BatchedSweep(batch_size))
import equinox as eqx
from typing import List
from jax_sn.tools.mesh_tools import _pad_last_dimension
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

# %%
from breeding_blanket_setup import bb_data_dict, base_region_sizes, base_resolutions,  _get_bb_args, _setup_bb_args, _get_bb_args

def n_timings(dim):
    if dim ==2 or dim == 1:
        return {"n_samples" : 30, "ignore_n" : 8}
    elif dim ==3:
        return {"n_samples" : 3, "ignore_n" : 1}
    
def n_timings_solve(dim):
    if dim ==2 or dim == 1:
        return {"number" : 3, "ignore_n" : 1}
    elif dim ==3:
        return {"number" : 1, "ignore_n" : 0}
    
def run_all_for_particular_operator(dim, degree, operator_cls, **kwargs):

    def setup_mgop(dim, degree):
        MGOP, source_eg0, solution_domain = _setup_bb_args(dim,degree, base_region_sizes(), base_resolutions(), 5.0, 10, 3)
        transport_operator                = TransportOperatorMultipleAngles(operator_cls(MGOP.transport_operator.solution_domain_data, MGOP.transport_operator.sweep_data, 0, **kwargs), VmapAngles())
        MGOP                              = eqx.tree_at(lambda t : t.transport_operator, MGOP, transport_operator)
        return MGOP, source_eg0, solution_domain

    
    MGOP, source_eg0, solution_domain = setup_mgop(dim,degree)

    # %%
    from jax_sn.tools.memory_analysis import OperatorAnalysis
    from jax_sn.parametrisation import SingleGroupOperatorSigmaAllParametrised

    # %%
    print("================================")
    print(f"Analyzing operator: {operator_cls.__name__} with params {kwargs}")
    print(f" and problem_size:")
    print(f" n_energy_groups    : {MGOP.n_energy_groups}")    
    print(f" n_angles           : {MGOP.n_angles}")
    print(f" n_moments          : {MGOP.n_moments}")
    print(f" n_elements         : {MGOP.n_elements}")
    print(f" n_basis            : {MGOP.n_basis}")
    print(f" dim                : {dim}")
    print(f" degree             : {degree}")
    print(f" Total dofs         :", MGOP.n_energy_groups * MGOP.n_moments * MGOP.n_elements * MGOP.n_basis)
    print("================================")
    
    def _pprint_memory_analysis(memory_dict):
        # structure is like:
        # {'peak': 215.2952651977539, 'argument': 104.98715209960938, 'output': 82.3974838256836, 'temp': 27.91297149658203}
        print(f"peak           : {memory_dict["peak"]     :.2f} MB")
        print(f"argument       : {memory_dict["argument"] :.2f} MB")
        print(f"output         : {memory_dict["output"]   :.2f} MB")
        print(f"temp           : {memory_dict["temp"]     :.2f} MB")
        

    # %%
    def analyse_operators(MGOP):
        print("TransportOperator")
        _pprint_memory_analysis(OperatorAnalysis.analyze_memory(MGOP.transport_operator, "inv"))
        print("MassOperator")
        _pprint_memory_analysis(OperatorAnalysis.analyze_memory(MGOP.mass_operator, "mv"))
        print("MomentOperator")
        _pprint_memory_analysis(OperatorAnalysis.analyze_memory(MGOP.moment_operator, "mv"))
        print("DiscreteOperator")
        _pprint_memory_analysis(OperatorAnalysis.analyze_memory(MGOP.discrete_operator, "mv"))
        print("Single Group Operator")
        sgop = MGOP.to_single_group_operator(0)
        _pprint_memory_analysis(OperatorAnalysis.analyze_memory(sgop, "mv"))
        print("Multi Group Operator")
        _pprint_memory_analysis(OperatorAnalysis.analyze_memory(MGOP, "mv"))

    print("Memory Analysis for MV of Operators")
    analyse_operators(MGOP)

    # %%
    rhs_eg0 = MGOP.moment_operator.mv(MGOP.transport_operator.to_energy_group(0).inv(MGOP.mass_operator.mv(source_eg0)))
    rhs_total = jnp.stack([rhs_eg0] + [jnp.zeros_like(rhs_eg0) for _ in range(MGOP.n_energy_groups -1)], axis =0)

    # %%
    from jax_sn.operators.multi_group_operator import solve_lower_triangular

    import lineax as lx
    bicgstab_solver = lx.BiCGStab(1e-10, 1e-10, max_steps = 5)

    @eqx.filter_jit
    def solve(sgop, rhs):
        return lx.linear_solve(sgop, rhs, solver = bicgstab_solver, throw = False).value

    def analyse_solve_memory(MGOP):

        print("Single Group Operator Solve")
        sgop = MGOP.to_single_group_operator(0)
        _pprint_memory_analysis(_filter_jit_f_to_memory(solve, sgop, rhs_total[0]))
        print("Multi Group Operator Total Solve")
        _pprint_memory_analysis(_filter_jit_f_to_memory(solve, MGOP, rhs_total))
        print("Multi Group Operator Group-wise Solve")
        _pprint_memory_analysis(_filter_jit_f_to_memory(solve_lower_triangular, MGOP, bicgstab_solver, rhs_total))
        
    analyse_solve_memory(MGOP)
        

    # %%
    def _pprint_timing_analysis(timing_dict):
        print(f"\tmv time      : {timing_dict      * 1000:.2f} ms")

    def analyse_timings(MGOP):
        print("TransportOperator")
        _pprint_timing_analysis(OperatorAnalysis.analyze_timing(MGOP.transport_operator, "inv", **n_timings(dim)))
        print("MassOperator")
        _pprint_timing_analysis(OperatorAnalysis.analyze_timing(MGOP.mass_operator, "mv",        **n_timings(dim)))
        print("MomentOperator")
        _pprint_timing_analysis(OperatorAnalysis.analyze_timing(MGOP.moment_operator, "mv",      **n_timings(dim)))
        print("ScatterOperator")
        _pprint_timing_analysis(OperatorAnalysis.analyze_timing(MGOP.discrete_operator, "mv",     **n_timings(dim)))
        print("Single Group Operator")
        sgop = MGOP.to_single_group_operator(0)
        _pprint_timing_analysis(OperatorAnalysis.analyze_timing(sgop, "mv", **n_timings(dim)))
        print("Multi Group Operator")
        _pprint_timing_analysis(OperatorAnalysis.analyze_timing(MGOP, "mv",**n_timings(dim)))

    print("Timing Analysis for MV of Operators")
    analyse_timings(MGOP)

    # %%
    bicgstab_solver = lx.BiCGStab(1e-10, 1e-10, max_steps = 5)

    #solution, stats = solve_lower_triangular_with_stats(MGOP, bicgstab_solver, rhs_total)
    # I know it is not going to converge in 5, so I can just /5 / 2 to get the time for a single MV expected / commented out for this reason

    from jax_sn.tools.memory_analysis import custom_timeit
    def analyse_solve_timing(mgop):
        def solve_lower_triangular_part():
            return solve_lower_triangular(mgop, bicgstab_solver, rhs_total)[0]
        time, std_time = custom_timeit(solve_lower_triangular_part, **n_timings_solve(dim))
        
        print(f"Lower triangular solve time: {time     * 1000:.2f} ms ")
        time_d_n_mv = time / (mgop.n_energy_groups * (bicgstab_solver.max_steps * 2 + 1))  # each step has 2 MVs + 1 initial MV
        print(f"\t Total time divided by number of single group MV: {time_d_n_mv * 1000:.2f} ms ")

    analyse_solve_timing(MGOP)


    # %% [markdown]
    # 


run_all_for_particular_operator(3,2, _batched_op, batch_size=100)
run_all_for_particular_operator(3,2, TransportOperatorBatchedPallas, batch_size=100)

run_all_for_particular_operator(2,2, _batched_op, batch_size=100)
run_all_for_particular_operator(2,2, TransportOperatorBatchedPallas, batch_size=100)

# Sample Output RTx3080:
# ================================
# Analyzing operator: TransportOperatorBatched with params {'batch_size': 100}
#  and problem_size:
#  n_energy_groups    : 4
#  n_angles           : 72
#  n_moments          : 36
#  n_elements         : 15000
#  n_basis            : 10
#  dim                : 3
#  degree             : 2
#  Total dofs         : 21600000
# ================================
# Memory Analysis for MV of Operators
# TransportOperator
# peak           : 215.30 MB
# argument       : 104.99 MB
# output         : 82.40 MB
# temp           : 27.91 MB
# MassOperator
# peak           : 251.31 MB
# argument       : 82.51 MB
# output         : 82.40 MB
# temp           : 86.40 MB
# MomentOperator
# peak           : 127.62 MB
# argument       : 82.42 MB
# output         : 41.20 MB
# temp           : 4.00 MB
# DiscreteOperator
# peak           : 127.62 MB
# argument       : 41.22 MB
# output         : 82.40 MB
# temp           : 4.00 MB
# Single Group Operator
# peak           : 281.60 MB
# argument       : 64.06 MB
# output         : 41.20 MB
# temp           : 176.34 MB
# Multi Group Operator
# peak           : 545.16 MB
# argument       : 187.66 MB
# output         : 164.79 MB
# temp           : 192.71 MB
# Single Group Operator Solve
# peak           : 610.48 MB
# argument       : 66.66 MB
# output         : 41.20 MB
# temp           : 581.03 MB
# Multi Group Operator Total Solve
# peak           : 1673.42 MB
# argument       : 190.26 MB
# output         : 164.79 MB
# temp           : 1318.37 MB
# Multi Group Operator Group-wise Solve
# peak           : 898.87 MB
# argument       : 190.26 MB
# output         : 164.79 MB
# temp           : 622.23 MB
# Timing Analysis for MV of Operators
# TransportOperator
#         mv time      : 525.97 ms
# MassOperator
#         mv time      : 4.48 ms
# MomentOperator
#         mv time      : 5.13 ms
# ScatterOperator
#         mv time      : 3.88 ms
# Single Group Operator
#         mv time      : 531.66 ms
# Multi Group Operator
#         mv time      : 2109.50 ms
# Lower triangular solve time: 24656.19 ms
#          Total time divided by number of single group MV: 560.37 ms
# ================================
# Analyzing operator: TransportOperatorBatchedPallas with params {'batch_size': 100}
#  and problem_size:
#  n_energy_groups    : 4
#  n_angles           : 72
#  n_moments          : 36
#  n_elements         : 15000
#  n_basis            : 10
#  dim                : 3
#  degree             : 2
#  Total dofs         : 21600000
# ================================
# Memory Analysis for MV of Operators
# TransportOperator
# peak           : 339.17 MB
# argument       : 104.99 MB
# output         : 82.40 MB
# temp           : 151.78 MB
# MassOperator
# peak           : 251.31 MB
# argument       : 82.51 MB
# output         : 82.40 MB
# temp           : 86.40 MB
# MomentOperator
# peak           : 127.62 MB
# argument       : 82.42 MB
# output         : 41.20 MB
# temp           : 4.00 MB
# DiscreteOperator
# peak           : 127.62 MB
# argument       : 41.22 MB
# output         : 82.40 MB
# temp           : 4.00 MB
# Single Group Operator
# peak           : 387.42 MB
# argument       : 64.06 MB
# output         : 41.20 MB
# temp           : 282.16 MB
# Multi Group Operator
# peak           : 638.38 MB
# argument       : 187.66 MB
# output         : 164.79 MB
# temp           : 285.93 MB
# Single Group Operator Solve
# peak           : 709.36 MB
# argument       : 66.66 MB
# output         : 41.20 MB
# temp           : 679.90 MB
# Multi Group Operator Total Solve
# peak           : 1673.42 MB
# argument       : 190.26 MB
# output         : 164.79 MB
# temp           : 1318.37 MB
# Multi Group Operator Group-wise Solve
# peak           : 997.75 MB
# argument       : 190.26 MB
# output         : 164.79 MB
# temp           : 721.11 MB
# Timing Analysis for MV of Operators
# TransportOperator
#         mv time      : 195.94 ms
# MassOperator
#         mv time      : 4.49 ms
# MomentOperator
#         mv time      : 5.11 ms
# ScatterOperator
#         mv time      : 4.06 ms
# Single Group Operator
#         mv time      : 205.93 ms
# Multi Group Operator
#         mv time      : 822.07 ms
# Lower triangular solve time: 9945.69 ms
#          Total time divided by number of single group MV: 226.04 ms
# ================================
# Analyzing operator: TransportOperatorBatched with params {'batch_size': 100}
#  and problem_size:
#  n_energy_groups    : 4
#  n_angles           : 72
#  n_moments          : 36
#  n_elements         : 500
#  n_basis            : 6
#  dim                : 2
#  degree             : 2
#  Total dofs         : 432000
# ================================
# Memory Analysis for MV of Operators
# TransportOperator
# peak           : 12.21 MB
# argument       : 2.35 MB
# output         : 1.65 MB
# temp           : 8.22 MB
# MassOperator
# peak           : 6.60 MB
# argument       : 1.65 MB
# output         : 1.65 MB
# temp           : 3.30 MB
# MomentOperator
# peak           : 4.16 MB
# argument       : 1.67 MB
# output         : 0.82 MB
# temp           : 1.67 MB
# DiscreteOperator
# peak           : 3.34 MB
# argument       : 0.84 MB
# output         : 1.65 MB
# temp           : 0.84 MB
# Single Group Operator
# peak           : 13.74 MB
# argument       : 1.58 MB
# output         : 0.82 MB
# temp           : 11.35 MB
# Multi Group Operator
# peak           : 18.87 MB
# argument       : 4.05 MB
# output         : 3.30 MB
# temp           : 11.53 MB
# Single Group Operator Solve
# peak           : 19.04 MB
# argument       : 1.63 MB
# output         : 0.82 MB
# temp           : 17.13 MB
# Multi Group Operator Total Solve
# peak           : 38.68 MB
# argument       : 4.10 MB
# output         : 3.30 MB
# temp           : 32.31 MB
# Multi Group Operator Group-wise Solve
# peak           : 24.81 MB
# argument       : 4.10 MB
# output         : 3.30 MB
# temp           : 17.43 MB
# Timing Analysis for MV of Operators
# TransportOperator
#         mv time      : 20.10 ms
# MassOperator
#         mv time      : 0.58 ms
# MomentOperator
#         mv time      : 0.53 ms
# ScatterOperator
#         mv time      : 0.59 ms
# Single Group Operator
#         mv time      : 19.77 ms
# Multi Group Operator
#         mv time      : 76.01 ms
# Lower triangular solve time: 885.46 ms
#          Total time divided by number of single group MV: 20.12 ms
# ================================
# Analyzing operator: TransportOperatorBatchedPallas with params {'batch_size': 100}
#  and problem_size:
#  n_energy_groups    : 4
#  n_angles           : 72
#  n_moments          : 36
#  n_elements         : 500
#  n_basis            : 6
#  dim                : 2
#  degree             : 2
#  Total dofs         : 432000
# ================================
# Memory Analysis for MV of Operators
# TransportOperator
# peak           : 24.06 MB
# argument       : 2.35 MB
# output         : 1.65 MB
# temp           : 20.07 MB
# MassOperator
# peak           : 6.60 MB
# argument       : 1.65 MB
# output         : 1.65 MB
# temp           : 3.30 MB
# MomentOperator
# peak           : 4.16 MB
# argument       : 1.67 MB
# output         : 0.82 MB
# temp           : 1.67 MB
# DiscreteOperator
# peak           : 3.34 MB
# argument       : 0.84 MB
# output         : 1.65 MB
# temp           : 0.84 MB
# Single Group Operator
# peak           : 24.12 MB
# argument       : 1.58 MB
# output         : 0.82 MB
# temp           : 21.72 MB
# Multi Group Operator
# peak           : 29.60 MB
# argument       : 4.05 MB
# output         : 3.30 MB
# temp           : 22.26 MB
# Single Group Operator Solve
# peak           : 30.40 MB
# argument       : 1.63 MB
# output         : 0.82 MB
# temp           : 27.96 MB
# Multi Group Operator Total Solve
# peak           : 49.42 MB
# argument       : 4.10 MB
# output         : 3.30 MB
# temp           : 43.96 MB
# Multi Group Operator Group-wise Solve
# peak           : 36.17 MB
# argument       : 4.10 MB
# output         : 3.30 MB
# temp           : 28.79 MB
# Timing Analysis for MV of Operators
# TransportOperator
#         mv time      : 10.97 ms
# MassOperator
#         mv time      : 0.70 ms
# MomentOperator
#         mv time      : 0.62 ms
# ScatterOperator
#         mv time      : 0.56 ms
# Single Group Operator
#         mv time      : 3.69 ms
# Multi Group Operator
#         mv time      : 12.01 ms
# Lower triangular solve time: 129.16 ms
#          Total time divided by number of single group MV: 2.94 ms