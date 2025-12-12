import jax 
import jax.numpy as jnp
from jax_sn.operators.transport_operator import transport_inverse_sweep_complete_vmap
import jax_sn

import numpy as onp
from .test_finite_elements import create_permutation_dimension
import pytest
def check_transport_inverse_sweep_complete_vmap(element, degree, 
                                                _vertices_conn_function, 
                                                _angle_function,  
                                                _expected_values_function, 
                                                _source_function, 
                                                _cross_section_function):
    
    
    lagrangian_element, domain, angle, toposort, femmatrices, precomputed_data, sweep_array, cross_sections, source, expected_values = _setup_all(element, degree, _vertices_conn_function, _angle_function, _expected_values_function, _source_function, _cross_section_function)

    mass_operator = jax_sn.operators.mass_operator.MassOperator.from_fem_matrices_and_precomputed_geometric_data(femmatrices, precomputed_data)

    source_mass_op = mass_operator(source.source_values)

    mapped_array = cross_sections.map_to_full_array(0)

    sweep_array = transport_inverse_sweep_complete_vmap(
        domain, 
        femmatrices,         
        precomputed_data,
        mapped_array, 
        toposort,                
        source_mass_op
    )
    
    onp.testing.assert_allclose(sweep_array, expected_values, atol=1e-12)

    
def _setup_domain(element, degree, _vertices_conn_function):
    vertices, connectivity = _vertices_conn_function(element, degree)
    lagrangian_element     = jax_sn.solution_domain.basix_fem.BasixLagrangianSimplex(degree=degree, dimension=element)    
    domain           = jax_sn.domain.GeometricDomain.from_mesh((jnp.array(vertices)[..., :element], jnp.array(connectivity)), jnp.array(lagrangian_element.face_template))
    return lagrangian_element, domain

def _setup_all(element, degree, _vertices_conn_function, _angle_function, _expected_values_function, _source_function, _cross_section_function):
    lagrangian_element, domain = _setup_domain(element, degree, _vertices_conn_function)
    angle                      = _angle_function()    
    toposort         = jax_sn.domain.geometric_domain.TopologicalSortedAngle.from_domain_angle(domain, angle)    
    femmatrices      = jax_sn.solution_domain.FEMMatrices.from_element(lagrangian_element)    
    precomputed_data = lagrangian_element.precompute_geometric_data(domain)    
    sweep_array      = jnp.zeros((domain.n_elements, lagrangian_element.n_basis))
    cross_sections   = _cross_section_function(element, degree)
    source           = _source_function(element, degree)
    expected_values  = _expected_values_function(element, degree)
    return lagrangian_element, domain, angle, toposort, femmatrices, precomputed_data, sweep_array, cross_sections, source, expected_values

#===========================================================================================================================
#                                      StellBlanket comparison functions
#===========================================================================================================================
def _switch_element_type(dimension):
    import StellBlanket as SB
    if dimension == 1:
        return SB.Lines.Line_Element
    elif dimension == 2:
        return SB.Triangles.Triangle_Element
    elif dimension == 3:
        return SB.Tetrahedrons.Tetrahedron_Element
    else:
        raise ValueError("Dimension must be 1, 2 or 3.")
    
def _test_domain_from_sb(dimension, degree):
    import StellBlanket as SB
    material_0 = SB.Generic_Elements.Material(SB.Generic_Elements.Constant_Cross_Section(0.673))
    material_1 = SB.Generic_Elements.Material(SB.Generic_Elements.Constant_Cross_Section(2.5))
    region_sizes=  [0.8, 0.2]
    resolutions  = [4,3]
    size_other_dim = 7.54
    resolutions_other_dim = 3
    element = _switch_element_type(dimension)
    cartesian_domain = SB.Generic_Elements.Create_Slab(element, region_sizes, resolutions, [material_0, material_1], size_other_dim, resolutions_other_dim)    
    return cartesian_domain

def _arbitrary_source_values(n_elements):
    return onp.linspace(1.0, 10.0, n_elements)

def tn_order_index():
    return 4, 3

def _angle_function_indexed(tn_order, index):
    import StellBlanket as SB
    quad_set = SB.Generic_Elements.TN_Quadrature_Set(tn_order)
    return quad_set.Directions()[index:index+1,:]
    
def _matfree_nte_from_sb(dimension, degree, tn_order, angle_index):
    import StellBlanket as SB
    domain   = _test_domain_from_sb(dimension, degree)
    quad_set = SB.Generic_Elements.Quadrature_Set(onp.array(_angle_function_indexed(tn_order, angle_index)), onp.array([1.0]))

    energy_set = SB.Generic_Elements.Energy_Set([0,1])
    matfreente = SB.Generic_Elements.Create_RAM_Simulation(quad_set, energy_set, domain, degree, 0)

    source_values = _arbitrary_source_values(domain.connectivity.shape[0])
    sources = [SB.Generic_Elements.Constant_Source(_switch_element_type(dimension), val, energy_set[0]) for val in source_values]
    total_source = SB.Generic_Elements.Per_Element_Source(sources, domain)
    return matfreente, total_source
    
def _verts_conn_from_sb(dimension, degree):
    import StellBlanket as SB
    domain   = _test_domain_from_sb(dimension, degree)
    return jnp.array(domain.vertices), jnp.array(domain.connectivity)

def _source_from_sb(dimension, degree):
    nte, total_source = _matfree_nte_from_sb(dimension, degree, *tn_order_index()) # angle doesn't matter for this
    source_values = _arbitrary_source_values(nte.geometric_domain.connectivity.shape[0])
    n_values = int(nte.solution_domain.total_size / nte.geometric_domain.connectivity.shape[0])
    return jax_sn.source.Source(jnp.stack([source_values] * n_values, axis=-1))

def _cross_sections_from_sb(dimension, degree):
    nte, total_source = _matfree_nte_from_sb(dimension, degree, *tn_order_index()) # angle doesn't matter for this
    n_elements = nte.geometric_domain.connectivity.shape[0]
    cross_sections = jnp.array([nte.geometric_domain[i].Material().Cross_Section().Value_Total(nte.energy_set[0]) for i in range(nte.geometric_domain.connectivity.shape[0])])

    cross_sections = cross_sections.reshape((n_elements, 1)) # [n_elements, n_energy_groups]
    
    vals, inverse = jnp.unique(cross_sections[:,0], return_inverse=True)
    
    scattering = jnp.zeros((vals.shape[0], 1, 1,1)) # Scattering is irrelevant for transport operator test [n_elements, n_L, n_energy_groups, n_energy_groups]
    total      = jnp.array(vals).reshape((vals.shape[0], 1))
    xsdata =  jax_sn.domain.cross_sections.CrossSectionData(total, scattering)
    xsdata.assert_valid_batched()
    
    return jax_sn.domain.geometric_domain.CrossSectionIndexed(inverse, xsdata)

def _angle_function_tn():
    return _angle_function_indexed(*tn_order_index())[0]

def _n_degree_old_code(dimension):
    return [None, 3, 3, 2][dimension]


def _expected_values_from_sb(dimension, degree):
    import StellBlanket as SB
    nte, total_source = _matfree_nte_from_sb(dimension, degree, *tn_order_index())
    
    nte.simulation.Sweep(nte.energy_set[0], nte.quadrature_set[0], total_source)
    values = nte.global_solution.All_Solution_Values(nte.energy_set[0], nte.quadrature_set[0]).reshape(nte.geometric_domain.connectivity.shape[0], -1   )

    permutation = create_permutation_dimension(dimension, degree)
    
    return values[:, permutation]
    
@pytest.mark.slow
@pytest.mark.parametrize("dimension", [1,2,3])
def test_sweep_complete_vmap(dimension):
    degree = _n_degree_old_code(dimension)
    for i in range(degree + 1):
        check_transport_inverse_sweep_complete_vmap(dimension, degree, _verts_conn_from_sb, _angle_function_tn, _expected_values_from_sb, _source_from_sb, _cross_sections_from_sb)

def setup_all_sb(element, degree):
    return _setup_all(element, degree, _verts_conn_from_sb, _angle_function_tn, _expected_values_from_sb, _source_from_sb, _cross_sections_from_sb)


@pytest.mark.slow
@pytest.mark.parametrize("dimension", [1,2,3])
def test_local_real_inversion(dimension):
    # Tests BVH containing & local/real coordinate conversion for simplexes
    lagrangian_element, domain = _setup_domain(dimension, 1, _verts_conn_from_sb)
    aabb_min = jnp.min(domain.vertices, axis=0)
    aabb_max = jnp.max(domain.vertices, axis=0)
    points = jnp.linspace(aabb_min, aabb_max, 37)
    element_indices = domain.containing_cell(points)
    precomputed_data= lagrangian_element.precompute_geometric_data(domain)
    local_coords = lagrangian_element.local_coordinates(points, element_indices, domain, precomputed_data)
    
    atol = 1e-12
    onp.testing.assert_array_less(0 - atol, local_coords)
    onp.testing.assert_array_less(local_coords.sum(axis=-1), 1.0 + atol)
    
    real_coords  = lagrangian_element.real_coordinates(local_coords, element_indices, domain, precomputed_data)
    onp.testing.assert_allclose(points, real_coords, atol=1e-12)



@pytest.mark.slow
@pytest.mark.parametrize("dimension", [1,2,3])
def test_different_sweeps(dimension):
    degree = 2

    lagrangian_element, domain, angle, toposort, femmatrices, precomputed_data, sweep_array, cross_sections, source, expected_values = _setup_all(dimension, degree, _verts_conn_from_sb, _angle_function_tn, _expected_values_from_sb, _source_from_sb, _cross_sections_from_sb)
    mapped_xs = cross_sections.map_to_full_array(0)
    
    mass_operator = jax_sn.operators.mass_operator.MassOperator.from_fem_matrices_and_precomputed_geometric_data(femmatrices, precomputed_data)
    source_mass_op = mass_operator(source.source_values)
    base_parameters = (domain, femmatrices, precomputed_data, mapped_xs, toposort, source_mass_op)
    
    xs_domain = jax_sn.domain.Domain.from_geometric_domain_and_cross_sections(domain, cross_sections)

    complete_vmap_sweep = jax_sn.operators.transport_operator.transport_inverse_sweep_complete_vmap(*base_parameters)
    batched_sweep = jax_sn.operators.transport_operator.transport_inverse_sweep_batched(*base_parameters, batch_size=11)

    onp.testing.assert_allclose(complete_vmap_sweep, batched_sweep, atol=1e-12)


    topolevel = jax_sn.operators.transport_operator.transport_operator_levels.TopologicalSortedAnglePrecomputedLevels.from_topological_sorted_angle(toposort)
    max_level = int(jnp.max(topolevel.level_log2_size))


    base_parameters_levels = (domain, femmatrices, precomputed_data, mapped_xs, topolevel, source_mass_op)
    sweep_levels = jax_sn.operators.transport_operator.transport_operator_levels.transport_inverse_sweep_complete_vmap_levels_max(*base_parameters_levels, max_level)

    onp.testing.assert_allclose(complete_vmap_sweep, sweep_levels[:-1], atol=1e-12)

    



@pytest.mark.slow
@pytest.mark.parametrize("dimension", [1,2,3])
def test_multi_angle_batch_vs_loop(dimension):    
    degree    = 2
    lagrangian_element, domain, angle, toposort, femmatrices, precomputed_data, sweep_array, cross_sections, source, expected_values = _setup_all(dimension, degree, _verts_conn_from_sb, _angle_function_tn, _expected_values_from_sb, _source_from_sb, _cross_sections_from_sb)
    mapped_xs = cross_sections.map_to_full_array(0)

    mass_operator = jax_sn.operators.mass_operator.MassOperator.from_fem_matrices_and_precomputed_geometric_data(femmatrices, precomputed_data)

    quad_set = jax_sn.quadrature_set.create_tn_quadrature_set(2)
    toposorts = [jax_sn.domain.geometric_domain.TopologicalSortedAngle.from_domain_angle(domain, angle, 100) for angle in quad_set.angles]
    total_toposorts = jax.tree.map(lambda * xs : jnp.stack(xs), *toposorts)

    source_angle = jax.tree.map(lambda ts : jnp.repeat(ts, total_toposorts.angle.shape[0], axis=0).reshape((total_toposorts.angle.shape[0],)+ts.shape), source)
    source_x = mass_operator(source_angle.source_values)    

    sweep_results = [] 

    for i in range(total_toposorts.angle.shape[0]):
        single_toposort = jax.tree.map(lambda x : x[i], total_toposorts)
        
        sweep_result = jax_sn.operators.transport_operator.transport_inverse_sweep_complete_vmap(
            domain, 
            femmatrices,         
            precomputed_data,
            mapped_xs, 
            single_toposort,                
            source_x[i]                
        )
        sweep_results.append(sweep_result)

    sweep_result = jnp.stack(sweep_results, axis=0)
    
    sweep_result_multi_angle = jax_sn.operators.transport_operator.transport_inverse_sweep_multiple_angles_batched(domain, femmatrices, precomputed_data, mapped_xs, total_toposorts, source_x, batch_size=7)
    onp.testing.assert_allclose(sweep_result_multi_angle, sweep_result, atol=1e-12)

    
@pytest.mark.pallas
@pytest.mark.slow
@pytest.mark.parametrize("dimension", [1,2,3])
def test_pallas_sweeps(dimension):
    
    for degree in range(0, _n_degree_old_code(dimension)):
        lagrangian_element, domain, angle, toposort, femmatrices, precomputed_data, sweep_array, cross_sections, source, expected_values = _setup_all(dimension, degree, _verts_conn_from_sb, _angle_function_tn, _expected_values_from_sb, _source_from_sb, _cross_sections_from_sb)
        mapped_xs = cross_sections.map_to_full_array(0)
        
        mass_operator = jax_sn.operators.mass_operator.MassOperator.from_fem_matrices_and_precomputed_geometric_data(femmatrices, precomputed_data)

        source_x = mass_operator(source.source_values)
        base_parameters = (domain, femmatrices, precomputed_data, mapped_xs, toposort, source_x)

        
        xs_domain = jax_sn.domain.Domain.from_geometric_domain_and_cross_sections(domain, cross_sections)
        complete_vmap_sweep = jax_sn.operators.transport_operator.transport_inverse_sweep_complete_vmap(*base_parameters)
        from jax_sn.operators.transport_operator.transport_operator_pallas import transport_inverse_sweep_complete_vmap_pallas, transport_inverse_sweep_batched_pallas

        
        pallas_complete_vmap = transport_inverse_sweep_complete_vmap_pallas(*base_parameters)
        onp.testing.assert_allclose(complete_vmap_sweep, pallas_complete_vmap, atol=1e-12)

        pallas_batched = transport_inverse_sweep_batched_pallas(*base_parameters, batch_size=13)
        onp.testing.assert_allclose(complete_vmap_sweep, pallas_batched, atol=1e-12)

        solution_domain = jax_sn.solution_domain.SolutionDomain.from_element_and_domain(lagrangian_element, xs_domain)

        pallas_vmap_op = jax_sn.operators.transport_operator.TransportOperatorVmapPallas.from_solution_domain_data_and_toposort(solution_domain.solution_domain_data, toposort)
        pallas_batched_op = jax_sn.operators.transport_operator.TransportOperatorBatchedPallas.from_solution_domain_data_and_toposort(solution_domain.solution_domain_data, toposort, batch_size=13)
        pallas_vmap_sweep = pallas_vmap_op.inv(source_x, 0)
        onp.testing.assert_allclose(complete_vmap_sweep, pallas_vmap_sweep, atol=1e-12)
        pallas_batched_sweep = pallas_batched_op.inv(source_x, 0)
        onp.testing.assert_allclose(complete_vmap_sweep, pallas_batched_sweep, atol=1e-12)

@pytest.mark.pallas
@pytest.mark.slow
@pytest.mark.parametrize("dimension", [1,2,3])
def test_multi_angle_batch_vs_loop_pallas(dimension):    
    degree    = 2
    lagrangian_element, domain, angle, toposort, femmatrices, precomputed_data, sweep_array, cross_sections, source, expected_values = _setup_all(dimension, degree, _verts_conn_from_sb, _angle_function_tn, _expected_values_from_sb, _source_from_sb, _cross_sections_from_sb)
    mapped_xs = cross_sections.map_to_full_array(0)

    quad_set = jax_sn.quadrature_set.create_tn_quadrature_set(2)
    toposorts = [jax_sn.domain.geometric_domain.TopologicalSortedAngle.from_domain_angle(domain, angle, 100) for angle in quad_set.angles]
    total_toposorts = jax.tree.map(lambda * xs : jnp.stack(xs), *toposorts)

    source_angle = jax.tree.map(lambda ts : jnp.repeat(ts, total_toposorts.angle.shape[0], axis=0).reshape((total_toposorts.angle.shape[0],)+ts.shape), source)

    mass_operator = jax_sn.operators.mass_operator.MassOperator.from_fem_matrices_and_precomputed_geometric_data(femmatrices, precomputed_data)

    source_x = mass_operator(source_angle.source_values)
    

    sweep_results = [] 

    for i in range(total_toposorts.angle.shape[0]):
        single_toposort = jax.tree.map(lambda x : x[i], total_toposorts)
        single_source  = jax.tree.map(lambda x : x[i], source_angle)
        sweep_result = jax_sn.operators.transport_operator.transport_inverse_sweep_complete_vmap(
            domain, 
            femmatrices,         
            precomputed_data,
            mapped_xs, 
            single_toposort,                
            source_x[i]
        )
        sweep_results.append(sweep_result)

    sweep_result = jnp.stack(sweep_results, axis=0)        

    
    from jax_sn.operators.transport_operator.transport_operator_pallas import transport_inverse_sweep_multiple_angles_batched_pallas
    sweep_result_multi_angle_pl = transport_inverse_sweep_multiple_angles_batched_pallas(domain, femmatrices, precomputed_data, mapped_xs, total_toposorts, source_x, batch_size=7)
    onp.testing.assert_allclose(sweep_result_multi_angle_pl, sweep_result, atol=1e-12)
    
    solution_domain = jax_sn.solution_domain.SolutionDomain.from_element_and_domain(lagrangian_element, jax_sn.domain.Domain.from_geometric_domain_and_cross_sections(domain, cross_sections))
    vmap_angles = jax_sn.operators.transport_operator.TransportOperatorVmapPallas.from_solution_domain_data_and_toposort(solution_domain.solution_domain_data, total_toposorts)
    batched_angles = jax_sn.operators.transport_operator.TransportOperatorBatchedPallas.from_solution_domain_data_and_toposort(solution_domain.solution_domain_data, total_toposorts, batch_size=7)

    sweep_result_vmap_op = vmap_angles.inv(source_x, 0)
    sweep_result_batched_op = batched_angles.inv(source_x, 0)

    onp.testing.assert_allclose(sweep_result_multi_angle_pl, sweep_result_vmap_op, atol=1e-12)
    onp.testing.assert_allclose(sweep_result_multi_angle_pl, sweep_result_batched_op, atol=1e-12)


#@pytest.mark.slow
@pytest.mark.parametrize("dimension", [1,2,3])
def test_multi_angle_transport_operator(dimension):    
    # This tests the TransportOperatorVmap and TransportOperatorBatched interfaces for multiple angles and single angles.
    
    degree    = 2
    lagrangian_element, domain, angle, toposort, femmatrices, precomputed_data, sweep_array, cross_sections, source, expected_values = _setup_all(dimension, degree, _verts_conn_from_sb, _angle_function_tn, _expected_values_from_sb, _source_from_sb, _cross_sections_from_sb)
    mapped_xs = cross_sections.map_to_full_array(0)

    quad_set = jax_sn.quadrature_set.create_tn_quadrature_set(2)
    toposorts = [jax_sn.domain.geometric_domain.TopologicalSortedAngle.from_domain_angle(domain, angle, 100) for angle in quad_set.angles]
    total_toposorts = jax.tree.map(lambda * xs : jnp.stack(xs), *toposorts)

    source_angle = jax.tree.map(lambda ts : jnp.repeat(ts, total_toposorts.angle.shape[0], axis=0).reshape((total_toposorts.angle.shape[0],)+ts.shape), source)

    mass_operator = jax_sn.operators.mass_operator.MassOperator.from_fem_matrices_and_precomputed_geometric_data(femmatrices, precomputed_data)

    source_x  = mass_operator(source_angle.source_values)

        
    sweep_result_multi_angle = jax_sn.operators.transport_operator.transport_inverse_sweep_multiple_angles_batched(domain, femmatrices, precomputed_data, mapped_xs, total_toposorts, source_x, batch_size=7)
    # This was already tested to work. 
    # now we are going to test the different TransportOperator interface

    solution_domain = jax_sn.solution_domain.SolutionDomain.from_element_and_domain(lagrangian_element, jax_sn.domain.Domain.from_geometric_domain_and_cross_sections(domain, cross_sections))


    from jax_sn.operators.transport_operator import TransportOperatorVmap, TransportOperatorBatched
    tvmap_tot  = TransportOperatorVmap.from_solution_domain_data_and_toposort(solution_domain.solution_domain_data, total_toposorts)
    tbatch_tot = TransportOperatorBatched.from_solution_domain_data_and_toposort(solution_domain.solution_domain_data, total_toposorts, batch_size=7)

    sweep_result_vmap_op = tvmap_tot.inv(source_x, 0)
    sweep_result_batched_op = tbatch_tot.inv(source_x, 0)

    onp.testing.assert_allclose(sweep_result_multi_angle, sweep_result_vmap_op, atol=1e-12)
    onp.testing.assert_allclose(sweep_result_multi_angle, sweep_result_batched_op, atol=1e-12)

    tvmap_sa = TransportOperatorVmap.from_solution_domain_data_and_toposort(solution_domain.solution_domain_data, total_toposorts[0])
    tbatch_sa = TransportOperatorBatched.from_solution_domain_data_and_toposort(solution_domain.solution_domain_data, total_toposorts[0], batch_size=7)

    sweep_result_vmap_op_sa = tvmap_sa.inv(source_x[0], 0)
    sweep_result_batched_op_sa = tbatch_sa.inv(source_x[0], 0)

    onp.testing.assert_allclose(sweep_result_multi_angle[0], sweep_result_vmap_op_sa, atol=1e-12)
    onp.testing.assert_allclose(sweep_result_multi_angle[0], sweep_result_batched_op_sa, atol=1e-12)




@pytest.mark.parametrize("dimension", [1,2,3])
def test_forward_transport_operator(dimension):
    degree = 2
    lagrangian_element, domain, angle, toposort, femmatrices, precomputed_data, sweep_array, cross_sections, source, expected_values = _setup_all(dimension, degree, _verts_conn_from_sb, _angle_function_tn, _expected_values_from_sb, _source_from_sb, _cross_sections_from_sb)
    mapped_xs = cross_sections.map_to_full_array(0)

    from jax_sn.operators.transport_operator.transport_operator_forward import forward_transport_operator_vmap, forward_transport_operator_batch
    
    mass_operator = jax_sn.operators.mass_operator.MassOperator.from_fem_matrices_and_precomputed_geometric_data(femmatrices, precomputed_data)
    source_x = mass_operator(source.source_values)

    inv_result = jax_sn.operators.transport_operator.transport_inverse_sweep_complete_vmap(
        domain, 
        femmatrices,         
        precomputed_data,
        mapped_xs, 
        toposort,                
        source_x
    )

    forward_result = forward_transport_operator_vmap(
        domain,
        femmatrices,
        precomputed_data,
        mapped_xs,
        toposort,
        inv_result
    )
    forward_result_batch = forward_transport_operator_batch(
        domain,
        femmatrices,
        precomputed_data,
        mapped_xs,
        toposort,
        inv_result,
        7
    )
    onp.testing.assert_allclose(forward_result, forward_result_batch, atol=1e-12)
    onp.testing.assert_allclose(forward_result, source_x, atol=1e-12)

    quad_set = jax_sn.quadrature_set.create_tn_quadrature_set(2)
    toposorts = [jax_sn.domain.geometric_domain.TopologicalSortedAngle.from_domain_angle(domain, angle, 100) for angle in quad_set.angles]
    total_toposorts = jax.tree.map(lambda * xs : jnp.stack(xs), *toposorts) 

    source_angle = jax.tree.map(lambda ts : jnp.repeat(ts, total_toposorts.angle.shape[0], axis=0).reshape((total_toposorts.angle.shape[0],)+ts.shape), source)    
    mass_operator = jax_sn.operators.mass_operator.MassOperator.from_fem_matrices_and_precomputed_geometric_data(femmatrices, precomputed_data)
    source_x_ma = mass_operator(source_angle.source_values)

    sweep_result_multi_angle = jax_sn.operators.transport_operator.transport_inverse_sweep_multiple_angles_batched(domain, femmatrices, precomputed_data, mapped_xs, total_toposorts, source_x_ma, batch_size=7)
    solution_domain = jax_sn.solution_domain.SolutionDomain.from_element_and_domain(lagrangian_element, jax_sn.domain.Domain.from_geometric_domain_and_cross_sections(domain, cross_sections))    

    forward_result_multi_angle_batch = jax_sn.operators.transport_operator.transport_operator_forward.forward_transport_operator_multiple_angles_batch(
        domain,
        femmatrices,
        precomputed_data,
        mapped_xs,
        total_toposorts,
        sweep_result_multi_angle,
        7
    )
    forward_result_multi_angle_vmap = jax_sn.operators.transport_operator.transport_operator_forward.forward_transport_operator_multiple_angles_vmap(
        domain,
        femmatrices,
        precomputed_data,
        mapped_xs,
        total_toposorts,
        sweep_result_multi_angle
    )

    onp.testing.assert_allclose(forward_result_multi_angle_batch, source_x_ma, atol=1e-12)
    onp.testing.assert_allclose(forward_result_multi_angle_batch, forward_result_multi_angle_vmap, atol=1e-12)
