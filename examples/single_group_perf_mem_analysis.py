import jax.numpy as jnp
import jax
import equinox as eqx

import jax_sn
from jax_sn.tools.memory_analysis import TotalOperatorSigmaParametrised

jax.config.update("jax_enable_x64", True)

def _get_anisotropic_args(dim, degree):
    n_elem_regions = [4, 2, 10, 4, 2, 2, 2]
    n_elem_regions = [6, 3, 76, 6, 3, 3, 3] # total 100
    d_other_dim = 5.0
    n_other_dim = 10
    tn_order = 3
    return (dim, degree, n_elem_regions, d_other_dim, n_other_dim, tn_order)

def setup_anisotropic(dim, degree, n_elem_regions, d_other_dim, n_other_dim, tn_order):

    from jax_sn.domain.geometric_domain import CrossSectionData
    from jax_sn.domain.domain_generation import create_cartesian_setup
    from jax_sn.domain import Domain
    from jax_sn.solution_domain import SolutionDomain, BasixLagrangianSimplex
    
    from jax_sn.quadrature_set import create_tn_quadrature_set

    water = CrossSectionData(jnp.array([15.2090, 24.4140]), jnp.array([[[5.5069, 0.0], 
                                                                        [6.6227, 11.4080]], 
                                                                        [[1.158567, 0.0],
                                                                         [1.1234767, 2.4733]]]))
    
    iron = CrossSectionData(jnp.array([30.3110, 26.8760]), jnp.array([[[18.25, 0.0], 
                                                                       [6.45294, 21.3570]], 
                                                                      [[4.363, 0.0],
                                                                       [-0.003508, 2.3457]]]))
    void = CrossSectionData.constant(1e-10, 2, 2)
    other_dims   = [[(d_other_dim, n_other_dim)]]*(dim - 1)

    region_sizes = [0.2, 0.1,     0.5,   0.2,  0.1,  0.1,    0.1]
    xs_regions   = [water, water, water, iron, void, water , void]

    region_size_res = [(size, n_elem) for size, n_elem in zip(region_sizes, n_elem_regions)]

    source_eg0 = [0.0, 9.7702e4, 0, 0,0,0,0]

    verts, conn, cross_sections, sources = create_cartesian_setup(dim, [region_size_res] + other_dims, xs_regions, source_eg0, 0)
    element_type                         = BasixLagrangianSimplex(degree, dim)
    domain                               = Domain.from_mesh_and_cross_sections((verts, conn), element_type.face_template, cross_sections)    
    solution_domain                      = SolutionDomain.from_element_and_domain(element_type, domain)     
    sources     = jnp.repeat(sources, repeats = solution_domain.n_basis, axis = -1).reshape(-1, solution_domain.n_basis)
    sources_eg0 = sources 
    sources_eg1 = jnp.where(sources_eg0 > 0, 4.5441e5, 0.0)
    return solution_domain, (sources_eg0, sources_eg1), create_tn_quadrature_set(tn_order)

def _setup_anisotropic_scattering_args(dim, degree, n_elem_regions, d_other_dim, n_other_dim, tn_order):
    from jax_sn.operators.scatter_operator import ScatterOperator
    from jax_sn.operators.transport_operator import TransportOperatorMultipleAnglesVmap, TransportOperatorVmap
    from jax_sn.operators.mass_operator import MassOperator
    from jax_sn.operators.moment_operator import MomentOperator, DiscreteOperator
    from jax_sn.domain.geometric_domain import TopologicalSortedAngle
    from jax_sn.operators.single_group_operator import SingleGroupOperator    
            
    solution_domain, sources, quadrature_set = setup_anisotropic(dim, degree, n_elem_regions, d_other_dim, n_other_dim, tn_order) 

    topological_sorted_angles = TopologicalSortedAngle.from_domain_quadrature_set(solution_domain.domain, quadrature_set, n_iterations_per_loop = 100)
    transport_operator        = TransportOperatorMultipleAnglesVmap(TransportOperatorVmap(solution_domain.solution_domain_data, topological_sorted_angles, 0))
    mass_operator             = MassOperator.from_transport_operator(transport_operator)
    moment_operator           = MomentOperator.from_quadrature_set(quadrature_set, l_max = 1, elements_basis_shape=solution_domain.elements_basis_shape)
    scatter_op                = ScatterOperator(solution_domain.domain.cross_sections_indexed, moment_operator.out_structure(), energy_group_in=0, energy_group_out=0)
    source_total              = jnp.broadcast_to(sources[0][None, :, :], (topological_sorted_angles.angle.shape[0],) + sources[0].shape) # source for all angles
    discrete_operator         = DiscreteOperator.from_quadrature_set(quadrature_set, l_max = 1, elements_basis_shape=solution_domain.elements_basis_shape)
    Aop = SingleGroupOperator(moment_operator, transport_operator, mass_operator, discrete_operator,  scatter_op)
    return Aop, source_total, solution_domain


def _run_multi_angle_operators(dim, degree, pallas = False):
    from jax_sn.operators.transport_operator.transport_operator_base import TransportOperatorBatched, TransportOperatorBatchedPallas, TransportOperatorMultipleAnglesVmap    
    from jax_sn.domain.geometric_domain import TopologicalSortedAngle
    from jax_sn.tools.memory_analysis import TransportOperatorSigmaParametrised
    from jax_sn.operators.single_group_operator import SingleGroupOperator    

    transportoperators_ma = {    
        TransportOperatorBatched : {"batch_size" : 100},             
    }
    if pallas:
        transportoperators_ma[TransportOperatorBatchedPallas] = {"batch_size" : 100}
    
    Aop, source_total, solution_domain = _setup_anisotropic_scattering_args(*_get_anisotropic_args(dim, degree))
    
    angle_i  = jax_sn.quadrature_set.create_tn_quadrature_set(3)
    n_angles = angle_i.n_angles
    toposorts = [TopologicalSortedAngle.from_domain_angle(solution_domain.domain, angle_i.angles[i], 100) for i in range(n_angles)]
    total_toposorts = jax.tree.map(lambda * xs : jnp.stack(xs), *toposorts)
    for operator_cls in transportoperators_ma.keys():        
        operator_instance = TransportOperatorMultipleAnglesVmap(operator_cls(solution_domain.solution_domain_data, total_toposorts, 0, **transportoperators_ma[operator_cls]))
        mem_mv, mem_mv_grad   = TransportOperatorSigmaParametrised.analyze_grad_memory(operator_instance, method="mv")    
        mem_inv, mem_inv_grad = TransportOperatorSigmaParametrised.analyze_grad_memory(operator_instance, method="inv")    
        print(f"{operator_cls.__name__} multi angle for a dimension {dim} degree {degree} operator with {solution_domain.domain.n_elements} elements and {n_angles} angles:")
        print(f"\tmv memory      : {mem_mv}")
        print(f"\tgrad mv memory : {mem_mv_grad}")
        print(f"\tinv memory     : {mem_inv}")
        print(f"\tgrad inv memory: {mem_inv_grad}")           
        print("\tTotal Operator:")
        anew = SingleGroupOperator(Aop.moment_operator, operator_instance, Aop.mass_operator, Aop.discrete_operator, Aop.inscatter_operator)
        mem_mv, mem_mv_grad   = TotalOperatorSigmaParametrised.analyze_grad_memory(anew, method="mv")            
        print(f"\t\tmv memory      : {mem_mv}")
        print(f"\t\tgrad mv memory : {mem_mv_grad}")        
        
    for operator_cls in transportoperators_ma.keys():        
        operator_instance = TransportOperatorMultipleAnglesVmap(operator_cls(solution_domain.solution_domain_data, total_toposorts, 0, **transportoperators_ma[operator_cls]))
        time_mv, time_mv_grad   = TransportOperatorSigmaParametrised.analyze_grad_timing(operator_instance, method="mv", n_samples = 3, ignore_n = 1)    
        time_inv, time_inv_grad = TransportOperatorSigmaParametrised.analyze_grad_timing(operator_instance, method="inv", n_samples= 3,  ignore_n = 1)    
        print(f"{operator_cls.__name__} multi angle for a dimension {dim} degree {degree} operator with {solution_domain.domain.n_elements} elements and {n_angles} angles:")        
        print(f"\tmv time      : {time_mv       * 1000:.2f} ms")
        print(f"\tgrad mv time : {time_mv_grad  * 1000:.2f} ms")
        print(f"\tinv time     : {time_inv      * 1000:.2f} ms")
        print(f"\tgrad inv time: {time_inv_grad * 1000:.2f} ms")

        anew = SingleGroupOperator(Aop.moment_operator, operator_instance, Aop.mass_operator,  Aop.discrete_operator,Aop.inscatter_operator)
        time_mv, time_mv_grad = TotalOperatorSigmaParametrised.analyze_grad_timing(anew, method="mv", n_samples = 3, ignore_n = 1)    

        print(f"\tTotal operator")        
        print(f"\t\tmv time      : {time_mv       * 1000:.2f} ms")
        print(f"\t\tgrad mv time : {time_mv_grad  * 1000:.2f} ms")
        



import lineax as lx
@eqx.filter_jit
def solve_bicgstab(Aop, rhs):
    bicgstab_solver = lx.BiCGStab(1e-10, 1e-10, max_steps = 20)
    solution = lx.linear_solve(Aop, rhs, bicgstab_solver)
    return solution.value

@eqx.filter_jit
def solve_bicgstab_stat(Aop, rhs):
    bicgstab_solver = lx.BiCGStab(1e-10, 1e-10, max_steps = 20)
    solution = lx.linear_solve(Aop, rhs, bicgstab_solver)
    return solution



def _solve_single_group(dim, degree, pallas = True):
    from jax_sn.operators.transport_operator.transport_operator_base import TransportOperatorBatched, TransportOperatorBatchedPallas, TransportOperatorMultipleAnglesVmap    
    from jax_sn.domain.geometric_domain import TopologicalSortedAngle
    from jax_sn.tools.memory_analysis import TransportOperatorSigmaParametrised
    from jax_sn.operators.single_group_operator import SingleGroupOperator    

    transportoperators_ma = {    
        TransportOperatorBatched : {"batch_size" : 100},             
    }
    if pallas:
        transportoperators_ma = {
            TransportOperatorBatchedPallas : {"batch_size" : 100}
        }
        #transportoperators_ma[TransportOperatorBatchedPallas] = {"batch_size" : 100}
    
    Aop, source_total, solution_domain = _setup_anisotropic_scattering_args(*_get_anisotropic_args(dim, degree))
    
    angle_i         = jax_sn.quadrature_set.create_tn_quadrature_set(3)
    n_angles        = angle_i.n_angles
    toposorts       = [TopologicalSortedAngle.from_domain_angle(solution_domain.domain, angle_i.angles[i], 100) for i in range(n_angles)]
    total_toposorts = jax.tree.map(lambda * xs : jnp.stack(xs), *toposorts)
    for operator_cls in transportoperators_ma.keys():        
        operator_instance     = TransportOperatorMultipleAnglesVmap(operator_cls(solution_domain.solution_domain_data, total_toposorts, 0, **transportoperators_ma[operator_cls]))

        rhs = Aop.moment_operator.mv(operator_instance.inv(source_total))
        
        anew                  = SingleGroupOperator(Aop.moment_operator, operator_instance, Aop.mass_operator, Aop.discrete_operator, Aop.inscatter_operator)
        
        mem_solve , mem_grad_solve = TotalOperatorSigmaParametrised.analyze_grad_memory(anew, method=solve_bicgstab)
        
        time_solve, time_grad_solve = TotalOperatorSigmaParametrised.analyze_grad_timing(anew, method=solve_bicgstab, n_samples = 3, ignore_n = 1)

        print("Total operator solve times:")
        print(f"{operator_cls.__name__} single angle for a dimension {dim} degree {degree} operator with {solution_domain.domain.n_elements} elements and {n_angles} angles:")
        print(f" with stats:", solve_bicgstab_stat(anew,rhs).stats)
        print(f"\tSolve memory      : {mem_solve       }")
        print(f"\tGrad solve memory : {mem_grad_solve  }")
        print(f"\tSolve time      : {time_solve       * 1000:.2f} ms")
        print(f"\tGrad solve time : {time_grad_solve  * 1000:.2f} ms")
        
_run_multi_angle_operators(3,2, True)
_solve_single_group(3,2,True)

# Example output [RTx 3080]:
# Note that these numbers make sense:
# Solve time is ~28 times inv time (~14 BiCGStab iterations, each requiring two MV and thus two inv): 28 * 743 ms = 20804 ms close to 17606 ms measured
# For grad solve, we expect it to be ~2 times grad inv time per MV (one for the forward pass, one for the backward pass), thus ~28 * 2 = 56 times inv time: 56 * 743 ms = 41608 ms close to 36551 ms measured
# Similarly for memory, we expect roughly double the grad inv memory per MV for the grad solve: 2 * 2041 MB = 4082 MB close to 3382 MB measured
# ===============
# TransportOperatorBatched multi angle for a dimension 3 degree 2 operator with 60000 elements and 72 angles:
#         mv memory      : {'peak': 684.7307968139648, 'argument': 347.5237274169922, 'output': 3.0517578125e-05, 'temp': 337.20886993408203}
#         grad mv memory : {'peak': 1338.8416290283203, 'argument': 347.5237274169922, 'output': 0.00014495849609375, 'temp': 991.3197631835938}
#         inv memory     : {'peak': 777.5167617797852, 'argument': 419.67293548583984, 'output': 3.0517578125e-05, 'temp': 357.84632110595703}
#         grad inv memory: {'peak': 1469.6572036743164, 'argument': 419.67293548583984, 'output': 0.00014495849609375, 'temp': 1049.990608215332}
#         Total Operator:
#                 mv memory      : {'peak': 796.7484359741211, 'argument': 109.31475067138672, 'output': 3.0517578125e-05, 'temp': 687.436897277832}
#                 grad mv memory : {'peak': 1432.1332702636719, 'argument': 109.31475067138672, 'output': 0.000579833984375, 'temp': 1648.414924621582}
# TransportOperatorBatchedPallas multi angle for a dimension 3 degree 2 operator with 60000 elements and 72 angles:
#         mv memory      : {'peak': 684.7307968139648, 'argument': 347.5237274169922, 'output': 3.0517578125e-05, 'temp': 337.20886993408203}
#         grad mv memory : {'peak': 1338.8416290283203, 'argument': 347.5237274169922, 'output': 0.00014495849609375, 'temp': 991.3197631835938}
#         inv memory     : {'peak': 1018.9413146972656, 'argument': 419.67293548583984, 'output': 3.0517578125e-05, 'temp': 599.270393371582}
#         grad inv memory: {'peak': 1908.8356552124023, 'argument': 419.67293548583984, 'output': 0.00014495849609375, 'temp': 1489.168830871582}
#         Total Operator:
#                 mv memory      : {'peak': 1038.1729888916016, 'argument': 109.31475067138672, 'output': 3.0517578125e-05, 'temp': 928.860969543457}
#                 grad mv memory : {'peak': 1823.6471710205078, 'argument': 109.31475067138672, 'output': 0.000579833984375, 'temp': 1846.168586730957}
# TransportOperatorBatched multi angle for a dimension 3 degree 2 operator with 60000 elements and 72 angles:
#         mv time      : 126.09 ms
#         grad mv time : 438.62 ms
#         inv time     : 1924.20 ms
#         grad inv time: 3881.54 ms
#         Total operator
#                 mv time      : 1919.63 ms
#                 grad mv time : 3822.78 ms
# TransportOperatorBatchedPallas multi angle for a dimension 3 degree 2 operator with 60000 elements and 72 angles:
#         mv time      : 120.86 ms
#         grad mv time : 438.67 ms
#         inv time     : 735.39 ms
#         grad inv time: 1520.03 ms
#         Total operator
#                 mv time      : 759.39 ms
#                 grad mv time : 1549.29 ms
# Total operator solve times:
# TransportOperatorBatchedPallas single angle for a dimension 3 degree 2 operator with 60000 elements and 72 angles:
#  with stats: {'max_steps': 20, 'num_steps': Array(14, dtype=int64)}
#         Solve memory      : {'peak': 1931.1783227920532, 'argument': 119.46871185302734, 'output': 3.0517578125e-05, 'temp': 2137.311279296875}
#         Grad solve memory : {'peak': 3124.2888984680176, 'argument': 119.46871185302734, 'output': 0.000579833984375, 'temp': 3136.17138671875}
#         Solve time      : 17520.24 ms
#         Grad solve time : 36461.75 ms
# ===============
