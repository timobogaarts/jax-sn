import jax_sn 
import jax 
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as onp
import pytest
import jax_sbgeom

def create_networkx_graph(geometric_domain, angle : jnp.ndarray) -> "networkx.DiGraph":
    import networkx as nx
    inflows                      = jax_sn.domain.graph_generation.compute_face_inflow(geometric_domain, angle) #[n_elements, n_faces_per_element]    
    
    # Mask out inflows on domainboundary faces
    inflows_connected_to_element = (geometric_domain.etof != -1) & inflows # [n_elements, n_faces_per_element]     
    
    dependency_matrix = jnp.where(inflows_connected_to_element, geometric_domain.etoe, -1)    
    edges = onp.where(dependency_matrix != -1)
    
    edges_conn  = dependency_matrix[edges[0], edges[1]]
    total_edges = onp.stack([edges_conn, edges[0]], axis=0).T
    n_elements = geometric_domain.n_elements
    G = nx.DiGraph()    
    G.add_nodes_from(onp.arange(geometric_domain.n_elements))
    G.add_edges_from(total_edges)
    return G

def n_theta_n_phi():
    return 20, 30

def _get_torus():
    torus = jax_sbgeom.flux_surfaces.FluxSurface.from_rmnc_zmns_settings(Rmnc=jnp.array(
                                                                        [[5.0, 0.0],
                                                                        [5.0, 1.0]]), 
                                                                        Zmns=jnp.array(
                                                                            [[0.0, 0.0],
                                                                            [0.0, 1.0]]), 
                                                                        settings = jax_sbgeom.flux_surfaces.FluxSurfaceSettings(1, 0, 1))
    return torus

def get_tetrahedral_domain():
    torus = _get_torus()
    mesh = jax_sbgeom.flux_surfaces.flux_surface_meshing.mesh_tetrahedra(torus, jnp.linspace(0,1,10), jax_sbgeom.flux_surfaces.ToroidalExtent.half_module(torus), *n_theta_n_phi())
    face_template = jax_sn.tools.mesh_tools._face_template(mesh[1].shape[1])
    geom_domain = jax_sn.domain.GeometricDomain.from_mesh(mesh, face_template)
    return geom_domain

def get_triangular_domain():
    torus = _get_torus()
    pol_mesh = jax_sbgeom.flux_surfaces.flux_surface_meshing.mesh_poloidal_plane(torus, jnp.linspace(0,1,10)**2, 0.52,  50)
    tri_face_template = jax_sn.tools.mesh_tools._face_template(pol_mesh[1].shape[1])
    pol_domain = jax_sn.domain.GeometricDomain.from_mesh(pol_mesh, tri_face_template)
    return pol_domain


def get_line_domain():
    torus = _get_torus()
    n_points = 11
    connectivity = jnp.stack([
            jnp.arange(n_points - 1),
            jnp.arange(1, n_points)
        ], axis=-1)
    points = torus.cartesian_position(jnp.linspace(0,1,n_points), 0.52, 0.2)
    tri_face_template = jax_sn.tools.mesh_tools._face_template(connectivity.shape[1])
    pol_domain = jax_sn.domain.GeometricDomain.from_mesh((points, connectivity), tri_face_template)
    return pol_domain

def _get_domain(domain_type):
    if domain_type == "tetrahedral":
        return get_tetrahedral_domain()
    elif domain_type == "triangular":
        return get_triangular_domain()
    elif domain_type == "line":
        return get_line_domain()
    else:
        raise ValueError(f"Unknown domain type: {domain_type}")
    

@pytest.mark.slow
@pytest.mark.parametrize("domain", ["tetrahedral", "triangular", "line"])
def test_levels(domain):
    import networkx as nx
    from jax_sbgeom.flux_surfaces import ToroidalExtent
    from jax_sn.domain.geometric_domain import GeometricDomain
    
    domain = _get_domain(domain)

    angle = jax_sn.quadrature_set.tn.create_tn_quadrature_set(4)[100][0]
    
    networks_graph = create_networkx_graph(domain, angle)
    generations = jax_sn.domain.graph_generation.compute_levels_domain(domain, angle, 15000, while_loop=True)

    generations_2 = jax_sn.domain.graph_generation.compute_levels_domain(domain, angle, 1500, while_loop=False)
    generations_3 = jax_sn.domain.graph_generation.compute_levels_domain_safe(domain , angle, n_iterations_per_loop=50)
    generations_4 = jax_sn.domain.graph_generation.compute_levels_domain_safe_jax(domain , angle, n_iterations_per_loop=50)
    onp.testing.assert_array_equal(generations, generations_2)
    onp.testing.assert_array_equal(generations, generations_3)
    onp.testing.assert_array_equal(generations, generations_4)
    
    for i, gen in enumerate(nx.algorithms.topological_generations(networks_graph)):                
        gen_nx = onp.sort(gen)
        gen_jsn = onp.sort(onp.where(generations == i)[0])
        assert onp.all(gen_nx == gen_jsn)
        
