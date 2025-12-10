import jax_sn as jsn

import jax_sbgeom

import jax.numpy as jnp
import numpy as onp
from typing import Tuple
import jax
import numpy as onp
from functools import partial
jax.config.update("jax_enable_x64", True)
from jax_sn.domain.geometric_domain import GeometricDomain, TopologicalSortedAngle
from jax_sn.solution_domain.solution_domain import FEMMatrices, PrecomputedGeometricData
from jax_sn.domain.geometric_domain import CrossSectionData, CrossSectionDataSingleGroup
from jax_sn.source import Source
from typing import List
from jax_sn.tools.mesh_tools import _pad_last_dimension
import matplotlib.pyplot as plt
from dataclasses import dataclass
onp.set_printoptions(linewidth = 200)
from jax_sn.domain.domain_generation import create_cartesian_setup
from jax_sn.solution_domain import BasixLagrangianSimplex
from jax_sn.solution_domain import SolutionDomain
from jax_sn.domain import CrossSectionData, Domain

from jax_sn.solution_domain import compute_setup_for_solution_domain, transport_inverse_sweep_complete_vmap
from jax_sn.solution_domain.transport_operator import transport_inverse_sweep_multiple_angles_batched, transport_inverse_sweep_multiple_angles_batched_single_f
from jax_sn.solution_domain.transport_operator_pallas import transport_inverse_sweep_batched_pallas, transport_inverse_sweep_complete_vmap_pallas, _next_power_of_two, _setup_elements_pad_args, transport_inverse_sweep_multiple_angles_batched_pallas
from jax_sn.solution_domain.transport_operator import transport_inverse_sweep_batched, transport_inverse_sweep_complete_vmap, _setup_elements_from_args
import timeit


sigma      = 5
xs_0       = CrossSectionData.constant(sigma, 1, 1)
q          = 5
dim        = 2
batch_size = 1000
n_elem     = 100
other_dims = [[(0.2, 10)]]*(dim - 1)
verts, conn, cross_sections, sources = create_cartesian_setup(dim, [[(1.0, n_elem)]] + other_dims, cross_sections_regions=[xs_0], source_regions=[q], inhomogeneity_direction=0)
face_template                        = BasixLagrangianSimplex(0, dim).face_template
domain                               = Domain.from_mesh_and_cross_sections((verts, conn), face_template, cross_sections_indexed=cross_sections)    


quad_set = jsn.quadrature_set.create_tn_quadrature_set(4)
toposort = TopologicalSortedAngle.from_domain_angle(domain, quad_set[0][0], 100) 


def time_executions(degree: int):
    element_type                         = BasixLagrangianSimplex(degree, dim)    

    solution_domain                      = SolutionDomain.from_element_and_domain(element_type, domain)
    source_mapped                        = Source(jnp.repeat(sources, element_type.n_basis, axis=-1).reshape(-1, element_type.n_basis))

    n_basis_pow2                         = _next_power_of_two(element_type.n_basis)
    buffer_sweep                         = jnp.zeros((domain.n_elements, element_type.n_basis))
    buffer_sweep_pl                      = jnp.zeros((domain.n_elements, n_basis_pow2))

    base_parameters                      = (*solution_domain.solution_domain_data, solution_domain.map_cross_sections_to_full_array(0), toposort, source_mapped)

    def run_setup():
        return _setup_elements_from_args(*base_parameters, buffer_sweep)[0].block_until_ready()
    
    def run_setup_pl():
        return _setup_elements_pad_args(*base_parameters, buffer_sweep_pl)[0].block_until_ready()

    def run_vmap():
        return transport_inverse_sweep_complete_vmap(*base_parameters, buffer_sweep).block_until_ready()
    
    def run_batch():
        return transport_inverse_sweep_batched(*base_parameters, buffer_sweep, batch_size=batch_size).block_until_ready()
    
    def run_vmap_pl():        
        return transport_inverse_sweep_complete_vmap_pallas(*base_parameters, buffer_sweep_pl).block_until_ready()
    
    def run_batch_pl():
        return transport_inverse_sweep_batched_pallas(*base_parameters, buffer_sweep_pl, batch_size=batch_size).block_until_ready()
    element_str = ["interval", "triangle", "tetrahedron"][dim - 1]
    print("Timings for a degree", degree, element_str, "domain with", domain.n_elements, "elements and batch size", batch_size)
    run_setup() # warmup jit
    duration = timeit.timeit(run_setup, number = 5) / 5 
    print(" Setup elements:                                      ", f"{duration * 1000:.2f}", "ms")
    run_setup_pl() # warmup jit
    duration = timeit.timeit(run_setup_pl, number = 5) / 5 
    print(" Setup elements padded to power of two:               ", f"{duration * 1000:.2f}", "ms")
    run_vmap() # warmup jit
    duration = timeit.timeit(run_vmap, number = 5) / 5 
    print(" transport_inverse_sweep_complete_vmap:          ", f"{duration * 1000:.2f}", "ms")
    run_batch() # warmup jit
    duration = timeit.timeit(run_batch, number = 5) / 5 
    print(" transport_inverse_sweep_batched:                     ", f"{duration * 1000:.2f}", "ms")
    run_vmap_pl() # warmup jit
    duration = timeit.timeit(run_vmap_pl, number = 5) / 5 
    print(" transport_inverse_sweep_complete_vmap_pallas:        ", f"{duration * 1000:.2f}", "ms")
    run_batch_pl() # warmup jit
    duration = timeit.timeit(run_batch_pl, number = 5) / 5 
    print(" transport_inverse_sweep_batched_pallas:              ", f"{duration * 1000:.2f}", "ms")
    print("")

time_executions(degree=0)
time_executions(degree=1)
time_executions(degree=2)
time_executions(degree=3)

def test_multiple_angles():
    quad_set = jsn.quadrature_set.create_tn_quadrature_set(3)
    toposorts = [TopologicalSortedAngle.from_domain_angle(domain, angle, 100) for angle in quad_set.angles]
    total_toposorts = jax.tree.map(lambda * xs : jnp.stack(xs), *toposorts)
    multiple_angles_batch_size = 100
    def time_degree(degree: int):
        element_type                         = BasixLagrangianSimplex(degree, dim)    

        solution_domain                      = SolutionDomain.from_element_and_domain(element_type, domain)
        source_mapped                        = Source(jnp.repeat(sources, element_type.n_basis, axis=-1).reshape(-1, element_type.n_basis))
        source_total = jax.tree.map(lambda * xs : jnp.stack(xs), *[source_mapped for _ in toposorts])

        n_basis_pow2                         = _next_power_of_two(element_type.n_basis)

        base_parameters                      = (*solution_domain.solution_domain_data, solution_domain.map_cross_sections_to_full_array(0), total_toposorts, source_total)

        def run_batch():
            buffer_sweep                         = jnp.zeros((len(toposorts), domain.n_elements, element_type.n_basis))
            return transport_inverse_sweep_multiple_angles_batched(*base_parameters, buffer_sweep, batch_size=multiple_angles_batch_size).block_until_ready()

        def run_batch_pl():
            buffer_sweep_pl                      = jnp.zeros((len(toposorts), domain.n_elements, n_basis_pow2))
            return transport_inverse_sweep_multiple_angles_batched_pallas(*base_parameters, buffer_sweep_pl, batch_size=multiple_angles_batch_size).block_until_ready()

        element_str = ["interval", "triangle", "tetrahedron"][dim - 1]
        print("Timings for a degree", degree, element_str, "domain with", domain.n_elements, "elements and batch size", multiple_angles_batch_size, "and", len(toposorts), "angles")
        run_batch() # warmup jit
        duration = timeit.timeit(run_batch, number = 5) / 5 
        print(" transport_inverse_sweep_multiple_angles_batched:          ", f"{duration * 1000:.2f}", "ms")
        run_batch_pl() # warmup jit
        duration = timeit.timeit(run_batch_pl, number = 5) / 5 
        print(" transport_inverse_sweep_multiple_angles_batched_pallas:   ", f"{duration * 1000:.2f}", "ms")
        print("")
    time_degree(degree=1)
    time_degree(degree=2)
    time_degree(degree=3)


test_multiple_angles()        
    

    