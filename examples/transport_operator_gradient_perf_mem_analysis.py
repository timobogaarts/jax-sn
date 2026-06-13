# %%
import jax 
import jax.numpy as jnp 

from functools import partial

jax.config.update("jax_enable_x64", True)

# %%
import jax_sn
from jax_sn.domain import CrossSectionData, Domain
from jax_sn.operators.transport_operator import BasicSweepData
from jax_sn.solution_domain import BasixLagrangianSimplex, SolutionDomain
from jax_sn.domain.domain_generation import create_cartesian_setup

from jax_sn.operators.transport_operator import TransportOperatorBatched, transport_inverse_sweep_batched, TransportOperatorVmap, TransportOperatorBatchedPallas, TransportOperatorVmapPallas
from jax_sn.operators.transport_operator.transport_operator_base import TransportOperatorMultipleAnglesVmap
import pyvista as pv
import equinox as eqx
import numpy as onp

#============================================================================================================
#  
# This file analyses the performance and memory usage of gradients of various transport operator implementations.
# It compares single angle and multi-angle operators, as well as different batching and vectorisation strategies.
# It uses a simple cartesian setup with a fixed number of elements and cross-sections.
# Example output is provided at the end of the file (RTx3080)
#============================================================================================================

def setup_base_case(dim, degree):
    n_elem       = 100
    n_other_dim  = 10
    other_dims   = [[(1.0, n_other_dim)]]*(dim - 1)
    xs0          = CrossSectionData.constant(0.2, 1, 1)    
    xs1          = CrossSectionData.constant(0.4, 1, 1)    
    verts, conn, cross_sections, sources = create_cartesian_setup(dim, [[(0.5, n_elem // 2), (0.5 , n_elem // 2)]] + other_dims, [xs0, xs1], [1.0 , 1.0], 0)
    element_type                         = BasixLagrangianSimplex(degree, dim)
    domain                               = Domain.from_mesh_and_cross_sections((verts, conn), element_type.face_template, cross_sections)    
    solution_domain                      = SolutionDomain.from_element_and_domain(element_type, domain)
    sources = jnp.repeat(sources, repeats = solution_domain.n_basis, axis = -1).reshape(-1, solution_domain.n_basis)
    return solution_domain, sources



from jax_sn.tools.memory_analysis import pytree_nbytes, _array_bytes, analyze_grad_memory, analyze_grad_timing
from jax_sn.parametrisation import TransportOperatorSigmaParametrised


def _run_single_angle_operators(dim, degree):
    transportoperators_sa = {    
        TransportOperatorVmap : {}, 
        TransportOperatorBatched : {"batch_size" : 1000}, 
        TransportOperatorVmapPallas : {},
        TransportOperatorBatchedPallas: {"batch_size" : 1000}, 
    }
    solution_domain, sources = setup_base_case(3,2)
    i_tn     = 50
    angle_i  = jax_sn.quadrature_set.create_tn_quadrature_set(4)
    toposort = BasicSweepData.from_domain_angle(solution_domain.domain, angle_i[i_tn][0], 100)
    for operator_cls in transportoperators_sa.keys():        
        operator_instance = operator_cls(solution_domain.solution_domain_data, toposort, 0, **transportoperators_sa[operator_cls])
        mem_mv, mem_mv_grad   = analyze_grad_memory(TransportOperatorSigmaParametrised, operator_instance, method="mv")    
        mem_inv, mem_inv_grad = analyze_grad_memory(TransportOperatorSigmaParametrised, operator_instance, method="inv")    
        print(f"{operator_cls.__name__} single angle for a dimension {dim} degree {degree} operator with {solution_domain.domain.n_elements} elements:")
        print(f"\tmv memory      : {mem_mv}")
        print(f"\tgrad mv memory : {mem_mv_grad}")
        print(f"\tinv memory     : {mem_inv}")
        print(f"\tgrad inv memory: {mem_inv_grad}")           
    for operator_cls in transportoperators_sa.keys():        
        operator_instance = operator_cls(solution_domain.solution_domain_data, toposort, 0, **transportoperators_sa[operator_cls])
        time_mv, time_mv_grad   = analyze_grad_timing(TransportOperatorSigmaParametrised, operator_instance, method="mv", n_samples = 10, ignore_n = 2)    
        time_inv, time_inv_grad = analyze_grad_timing(TransportOperatorSigmaParametrised, operator_instance, method="inv", n_samples= 7,  ignore_n = 2)    
        print(f"{operator_cls.__name__} single angle for a dimension {dim} degree {degree} operator with {solution_domain.domain.n_elements} elements:")        
        print(f"\tmv time      : {time_mv       * 1000:.2f} ms")
        print(f"\tgrad mv time : {time_mv_grad  * 1000:.2f} ms")
        print(f"\tinv time     : {time_inv      * 1000:.2f} ms")
        print(f"\tgrad inv time: {time_inv_grad * 1000:.2f} ms")


def _run_multi_angle_operators(dim, degree):
    transportoperators_ma = {    
        TransportOperatorBatched : {"batch_size" : 100},             
        TransportOperatorBatchedPallas: {"batch_size" : 100}}
    
    solution_domain, sources = setup_base_case(3,2)    
    
    angle_i  = jax_sn.quadrature_set.create_tn_quadrature_set(3)
    n_angles = angle_i.n_angles
    toposorts = [BasicSweepData.from_domain_angle(solution_domain.domain, angle_i.angles[i], 100) for i in range(n_angles)]
    total_toposorts = jax.tree.map(lambda * xs : jnp.stack(xs), *toposorts)
    for operator_cls in transportoperators_ma.keys():        
        operator_instance = TransportOperatorMultipleAnglesVmap(operator_cls(solution_domain.solution_domain_data, total_toposorts, 0, **transportoperators_ma[operator_cls]))
        mem_mv, mem_mv_grad   = analyze_grad_memory(TransportOperatorSigmaParametrised, operator_instance, method="mv")    
        mem_inv, mem_inv_grad = analyze_grad_memory(TransportOperatorSigmaParametrised, operator_instance, method="inv")    
        print(f"{operator_cls.__name__} multi angle for a dimension {dim} degree {degree} operator with {solution_domain.domain.n_elements} elements and {n_angles} angles:")
        print(f"\tmv memory      : {mem_mv}")
        print(f"\tgrad mv memory : {mem_mv_grad}")
        print(f"\tinv memory     : {mem_inv}")
        print(f"\tgrad inv memory: {mem_inv_grad}")           
    for operator_cls in transportoperators_ma.keys():        
        operator_instance = TransportOperatorMultipleAnglesVmap(operator_cls(solution_domain.solution_domain_data, total_toposorts, 0, **transportoperators_ma[operator_cls]))
        time_mv, time_mv_grad   = analyze_grad_timing(TransportOperatorSigmaParametrised, operator_instance, method="mv", n_samples = 3, ignore_n = 1)    
        time_inv, time_inv_grad = analyze_grad_timing(TransportOperatorSigmaParametrised, operator_instance, method="inv", n_samples= 3,  ignore_n = 1)    
        print(f"{operator_cls.__name__} multi angle for a dimension {dim} degree {degree} operator with {solution_domain.domain.n_elements} elements and {n_angles} angles:")        
        print(f"\tmv time      : {time_mv       * 1000:.2f} ms")
        print(f"\tgrad mv time : {time_mv_grad  * 1000:.2f} ms")
        print(f"\tinv time     : {time_inv      * 1000:.2f} ms")
        print(f"\tgrad inv time: {time_inv_grad * 1000:.2f} ms")

# %%
_run_single_angle_operators(3,2)
_run_multi_angle_operators(3,2)


# TransportOperatorVmap single angle for a dimension 3 degree 2 operator with 60000 elements:
#         mv memory      : {'peak': 122.30461120605469, 'argument': 22.509803771972656, 'output': 3.0517578125e-05, 'temp': 98.8801498413086}
#         grad mv memory : {'peak': 122.30463409423828, 'argument': 22.509803771972656, 'output': 5.340576171875e-05, 'temp': 98.8801498413086}
#         inv memory     : {'peak': 264.9477996826172, 'argument': 29.656158447265625, 'output': 3.0517578125e-05, 'temp': 235.29498291015625}
#         grad inv memory: {'peak': 506.19004821777344, 'argument': 29.656158447265625, 'output': 5.340576171875e-05, 'temp': 476.31109619140625}
# TransportOperatorBatched single angle for a dimension 3 degree 2 operator with 60000 elements:
#         mv memory      : {'peak': 28.927688598632812, 'argument': 22.509803771972656, 'output': 3.0517578125e-05, 'temp': 6.419563293457031}
#         grad mv memory : {'peak': 40.574745178222656, 'argument': 22.509803771972656, 'output': 5.340576171875e-05, 'temp': 18.06707763671875}
#         inv memory     : {'peak': 38.811492919921875, 'argument': 29.656158447265625, 'output': 3.0517578125e-05, 'temp': 9.159065246582031}
#         grad inv memory: {'peak': 50.157249450683594, 'argument': 29.656158447265625, 'output': 5.340576171875e-05, 'temp': 20.50965118408203}
# TransportOperatorVmapPallas single angle for a dimension 3 degree 2 operator with 60000 elements:
#         mv memory      : {'peak': 122.30461120605469, 'argument': 22.509803771972656, 'output': 3.0517578125e-05, 'temp': 98.8801498413086}
#         grad mv memory : {'peak': 122.30463409423828, 'argument': 22.509803771972656, 'output': 5.340576171875e-05, 'temp': 98.8801498413086}
#         inv memory     : {'peak': 632.0731887817383, 'argument': 29.656158447265625, 'output': 3.0517578125e-05, 'temp': 602.4199447631836}
#         grad inv memory: {'peak': 1157.12890625, 'argument': 29.656158447265625, 'output': 5.340576171875e-05, 'temp': 1126.5640029907227}
# TransportOperatorBatchedPallas single angle for a dimension 3 degree 2 operator with 60000 elements:
#         mv memory      : {'peak': 28.927688598632812, 'argument': 22.509803771972656, 'output': 3.0517578125e-05, 'temp': 6.419563293457031}
#         grad mv memory : {'peak': 40.574745178222656, 'argument': 22.509803771972656, 'output': 5.340576171875e-05, 'temp': 18.06707763671875}
#         inv memory     : {'peak': 48.005653381347656, 'argument': 29.656158447265625, 'output': 3.0517578125e-05, 'temp': 18.35315704345703}
#         grad inv memory: {'peak': 63.111900329589844, 'argument': 29.656158447265625, 'output': 5.340576171875e-05, 'temp': 33.46350860595703}
# TransportOperatorVmap single angle for a dimension 3 degree 2 operator with 60000 elements:
#         mv time      : 24.56 ms
#         grad mv time : 2.70 ms
#         inv time     : 591.33 ms
#         grad inv time: 1244.57 ms
# TransportOperatorBatched single angle for a dimension 3 degree 2 operator with 60000 elements:
#         mv time      : 4.62 ms
#         grad mv time : 7.04 ms
#         inv time     : 583.50 ms
#         grad inv time: 1242.98 ms
# TransportOperatorVmapPallas single angle for a dimension 3 degree 2 operator with 60000 elements:
#         mv time      : 3.34 ms
#         grad mv time : 2.79 ms
#         inv time     : 151.49 ms
#         grad inv time: 285.37 ms
# TransportOperatorBatchedPallas single angle for a dimension 3 degree 2 operator with 60000 elements:
#         mv time      : 4.15 ms
#         grad mv time : 6.72 ms
#         inv time     : 153.50 ms
#         grad inv time: 306.05 ms
# TransportOperatorBatched multi angle for a dimension 3 degree 2 operator with 60000 elements and 72 angles:
#         mv memory      : {'peak': 744.2499008178711, 'argument': 347.5236358642578, 'output': 3.0517578125e-05, 'temp': 396.72864532470703}
#         grad mv memory : {'peak': 1517.5012664794922, 'argument': 347.5236358642578, 'output': 5.340576171875e-05, 'temp': 1169.9804077148438}
#         inv memory     : {'peak': 1078.8521728515625, 'argument': 419.6724319458008, 'output': 3.0517578125e-05, 'temp': 659.183479309082}
#         grad inv memory: {'peak': 1830.685691833496, 'argument': 419.6724319458008, 'output': 5.340576171875e-05, 'temp': 1411.021369934082}
# TransportOperatorBatchedPallas multi angle for a dimension 3 degree 2 operator with 60000 elements and 72 angles:
#         mv memory      : {'peak': 744.2499008178711, 'argument': 347.5236358642578, 'output': 3.0517578125e-05, 'temp': 396.72864532470703}
#         grad mv memory : {'peak': 1517.5012664794922, 'argument': 347.5236358642578, 'output': 5.340576171875e-05, 'temp': 1169.9804077148438}
#         inv memory     : {'peak': 1708.2779693603516, 'argument': 419.6724319458008, 'output': 3.0517578125e-05, 'temp': 1288.609016418457}
#         grad inv memory: {'peak': 2730.923683166504, 'argument': 419.6724319458008, 'output': 5.340576171875e-05, 'temp': 2311.258674621582}
# TransportOperatorBatched multi angle for a dimension 3 degree 2 operator with 60000 elements and 72 angles:
#         mv time      : 115.66 ms
#         grad mv time : 654.03 ms
#         inv time     : 1858.62 ms
#         grad inv time: 4016.33 ms
# TransportOperatorBatchedPallas multi angle for a dimension 3 degree 2 operator with 60000 elements and 72 angles:
#         mv time      : 106.39 ms
#         grad mv time : 649.72 ms
#         inv time     : 657.13 ms
#         grad inv time: 1328.08 ms