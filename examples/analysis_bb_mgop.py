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

def run_all_for_particular_operator(dim, degree, operator_cls, **kwargs):

    def setup_mgop(dim, degree):
        MGOP, source_eg0, solution_domain = _setup_bb_args(dim,degree, base_region_sizes(), base_resolutions(), 5.0, 10, 3)
        transport_operator                = TransportOperatorMultipleAnglesVmap(operator_cls(MGOP.transport_operator.solution_domain_data, MGOP.transport_operator.topological_sorted_angle, 0, **kwargs))
        MGOP                              = eqx.tree_at(lambda t : t.transport_operator, MGOP, transport_operator)
        return MGOP, source_eg0, solution_domain

    
    MGOP, source_eg0, solution_domain = setup_mgop(dim,degree)

    # %%
    from jax_sn.tools.memory_analysis import OperatorAnalysis, TotalOperatorSigmaParametrised

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
        print("ScatterOperator")
        _pprint_memory_analysis(OperatorAnalysis.analyze_memory(MGOP.scatter_operator, "mv"))
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
    from jax_sn.operators.multi_group_operator.multi_group_operator_imp import solve_lower_triangular, solve_total

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
        _pprint_timing_analysis(OperatorAnalysis.analyze_timing(MGOP.transport_operator, "inv", n_samples=3, ignore_n=1))
        print("MassOperator")
        _pprint_timing_analysis(OperatorAnalysis.analyze_timing(MGOP.mass_operator, "mv", n_samples=3, ignore_n=1))
        print("MomentOperator")
        _pprint_timing_analysis(OperatorAnalysis.analyze_timing(MGOP.moment_operator, "mv", n_samples=3, ignore_n=1))
        print("ScatterOperator")
        _pprint_timing_analysis(OperatorAnalysis.analyze_timing(MGOP.scatter_operator, "mv", n_samples=3, ignore_n=1))
        print("Single Group Operator")
        sgop = MGOP.to_single_group_operator(0)
        _pprint_timing_analysis(OperatorAnalysis.analyze_timing(sgop, "mv", n_samples=3, ignore_n=1))
        print("Multi Group Operator")
        _pprint_timing_analysis(OperatorAnalysis.analyze_timing(MGOP, "mv", n_samples=3, ignore_n=1))

    print("Timing Analysis for MV of Operators")
    analyse_timings(MGOP)

    # %%
    bicgstab_solver = lx.BiCGStab(1e-10, 1e-10, max_steps = 5)

    def solve_lower_triangular_with_stats(mgop, solver, rhs : jnp.ndarray) -> jnp.ndarray:
        '''
        Solve the linear system using a given solver, assuming the multi-group operator is lower triangular.
        This is useful for e.g. downscattering only problems. 

        Parameters
        ----------
        mgop : MultiGroupOperator
            The multi group operator.
        solver : lx.LinearSolver
            The linear solver to use.
        rhs : jnp.ndarray
            The right-hand side.
        Returns
        -------
        jnp.ndarray
            The solution.   
        '''
        n_groups = mgop.n_energy_groups    
        solution = jnp.zeros_like(rhs)
        stats = [] 

        for i in range(n_groups):
            
            sgop = mgop.to_single_group_operator(i)
            rhs_i = rhs[i] + mgop.moment_operator.mv(sgop.transport_operator.inv(mgop.mass_operator.mv(mgop.scatter_operator.to_energy_group(i).mv(solution))))        
            solution_i = lx.linear_solve(sgop, rhs_i, solver, throw = False)
            solution = solution.at[i].set(solution_i.value)
            print(solution_i.stats)
            
            stats.append(solution_i.stats)
        return solution, stats

    #solution, stats = solve_lower_triangular_with_stats(MGOP, bicgstab_solver, rhs_total)
    # I know it is not going to converge in 5, so I can just /5 / 2 to get the time for a single MV expected / commented out for this reason

    from jax_sn.tools.memory_analysis import custom_timeit
    def analyse_solve_timing(mgop):
        def solve_lower_triangular_part():
            return solve_lower_triangular(mgop, bicgstab_solver, rhs_total)
        time, std_time = custom_timeit(solve_lower_triangular_part, number = 1)
        
        print(f"Lower triangular solve time: {time     * 1000:.2f} ms ")
        time_d_n_mv = time / (mgop.n_energy_groups * bicgstab_solver.max_steps * 2)
        print(f"\t Total time divided by number of single group MV: {time_d_n_mv * 1000:.2f} ms ")

    analyse_solve_timing(MGOP)


    # %% [markdown]
    # 


run_all_for_particular_operator(3,2, TransportOperatorBatched, batch_size=100)
run_all_for_particular_operator(3,2, TransportOperatorBatchedPallas, batch_size=100)

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
# ScatterOperator
# peak           : 292.53 MB
# argument       : 164.93 MB
# output         : 82.40 MB
# temp           : 45.20 MB
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
# peak           : 1673.41 MB
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
#         mv time      : 494.56 ms
# MassOperator
#         mv time      : 3.87 ms
# MomentOperator
#         mv time      : 5.04 ms
# ScatterOperator
#         mv time      : 4.93 ms
# Single Group Operator
#         mv time      : 496.37 ms
# Multi Group Operator
#         mv time      : 1938.63 ms
# Lower triangular solve time: 22974.68 ms
#          Total time divided by number of single group MV: 574.37 ms
#
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
# ScatterOperator
# peak           : 292.53 MB
# argument       : 164.93 MB
# output         : 82.40 MB
# temp           : 45.20 MB
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
# peak           : 1673.41 MB
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
#         mv time      : 186.12 ms
# MassOperator
#         mv time      : 3.86 ms
# MomentOperator
#         mv time      : 4.90 ms
# ScatterOperator
#         mv time      : 4.91 ms
# Single Group Operator
#         mv time      : 195.77 ms
# Multi Group Operator
#         mv time      : 783.78 ms
# Lower triangular solve time: 9585.49 ms
#          Total time divided by number of single group MV: 239.64 ms