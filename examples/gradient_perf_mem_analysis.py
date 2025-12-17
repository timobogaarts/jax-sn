# %%
import jax 
import jax.numpy as jnp 

from functools import partial

jax.config.update("jax_enable_x64", True)

# %%
import jax_sn
from jax_sn.domain import CrossSectionData, Domain, TopologicalSortedAngle
from jax_sn.solution_domain import BasixLagrangianSimplex, SolutionDomain
from jax_sn.domain.domain_generation import create_cartesian_setup
from jax_sn.operators.mass_operator import MassOperator
from jax_sn.operators.mass_operator.mass_operator_base import _mass_kernel_idx_in_precomp_vmap
from jax_sn.operators.transport_operator import TransportOperatorBatched, transport_inverse_sweep_batched, TransportOperatorVmap, TransportOperatorBatchedPallas, TransportOperatorVmapPallas
from jax_sn.operators.transport_operator.transport_operator_base import TransportOperatorMultipleAnglesVmap
import pyvista as pv
import equinox as eqx
import numpy as onp

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



from jax_sn.tools.memory_analysis import pytree_nbytes, array_bytes, TransportOperatorSigmaParametrised


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
    toposort = TopologicalSortedAngle.from_domain_angle(solution_domain.domain, angle_i[i_tn][0], 100)
    for operator_cls in transportoperators_sa.keys():        
        operator_instance = operator_cls(solution_domain.solution_domain_data, toposort, 0, **transportoperators_sa[operator_cls])
        mem_mv, mem_mv_grad   = TransportOperatorSigmaParametrised.analyze_grad_memory(operator_instance, method="mv")    
        mem_inv, mem_inv_grad = TransportOperatorSigmaParametrised.analyze_grad_memory(operator_instance, method="inv")    
        print(f"{operator_cls.__name__} single angle for a dimension {dim} degree {degree} operator with {solution_domain.domain.n_elements} elements:")
        print(f"\tmv memory      : {mem_mv}")
        print(f"\tgrad mv memory : {mem_mv_grad}")
        print(f"\tinv memory     : {mem_inv}")
        print(f"\tgrad inv memory: {mem_inv_grad}")           
    for operator_cls in transportoperators_sa.keys():        
        operator_instance = operator_cls(solution_domain.solution_domain_data, toposort, 0, **transportoperators_sa[operator_cls])
        time_mv, time_mv_grad   = TransportOperatorSigmaParametrised.analyze_grad_timing(operator_instance, method="mv", n_samples = 10, ignore_n = 2)    
        time_inv, time_inv_grad = TransportOperatorSigmaParametrised.analyze_grad_timing(operator_instance, method="inv", n_samples= 7,  ignore_n = 2)    
        print(f"{operator_cls.__name__} single angle for a dimension {dim} degree {degree} operator with {solution_domain.domain.n_elements} elements:")
        
        print(f"\tmv time      : {time_mv       * 1000:.2f} ms")
        print(f"\tgrad mv time : {time_mv_grad  * 1000:.2f} ms")
        print(f"\tinv time     : {time_inv      * 1000:.2f} ms")
        print(f"\tgrad inv time: {time_inv_grad * 1000:.2f} ms")


def _run_multi_angle_operators(dim, degree):
    transportoperators_ma = {    
        TransportOperatorBatched : {"batch_size" : 1000},             
        TransportOperatorBatchedPallas: {"batch_size" : 1000}}
    
    solution_domain, sources = setup_base_case(3,2)
    i_tn     = 50
    
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
    for operator_cls in transportoperators_ma.keys():        
        operator_instance = TransportOperatorMultipleAnglesVmap(operator_cls(solution_domain.solution_domain_data, total_toposorts, 0, **transportoperators_ma[operator_cls]))
        time_mv, time_mv_grad   = TransportOperatorSigmaParametrised.analyze_grad_timing(operator_instance, method="mv", n_samples = 3, ignore_n = 1)    
        time_inv, time_inv_grad = TransportOperatorSigmaParametrised.analyze_grad_timing(operator_instance, method="inv", n_samples= 3,  ignore_n = 1)    
        print(f"{operator_cls.__name__} multi angle for a dimension {dim} degree {degree} operator with {solution_domain.domain.n_elements} elements and {n_angles} angles:")
        print(time_mv)
        print(f"\tmv time      : {time_mv       * 1000:.2f} ms")
        print(f"\tgrad mv time : {time_mv_grad  * 1000:.2f} ms")
        print(f"\tinv time     : {time_inv      * 1000:.2f} ms")
        print(f"\tgrad inv time: {time_inv_grad * 1000:.2f} ms")

# %%
_run_single_angle_operators(3,2)
_run_multi_angle_operators(3,2)


