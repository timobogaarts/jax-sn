import jax_sn 
import jax 
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as onp
import pytest
import jax_sbgeom

def create_networkx_graph(geometric_domain, angle : jnp.ndarray) -> "networkx.DiGraph":
    import networkx as nx
    outflows                      = jax_sn.domain.graph_generation.compute_face_outflow(geometric_domain, angle) #[n_elements, n_faces_per_element]    
    
    # Mask out outflows on domainboundary faces
    outflows_connected_to_element = (geometric_domain.etof != -1) & outflows # [n_elements, n_faces_per_element]     

    # matrix that for each element i and each face k gives the element connected to element i through face k if that face is an outflow face, else -1    
    outflow_face_to_element_matrix = jnp.where(outflows_connected_to_element, geometric_domain.etoe, -1)    

    outflow_face_to_element_matrix = jnp.where(outflows_connected_to_element, geometric_domain.etoe, -1)    
    edges = onp.where(outflow_face_to_element_matrix != -1)

    
    edges_conn  = outflow_face_to_element_matrix[edges[0], edges[1]]
    total_edges = onp.stack([edges_conn, edges[0]], axis=0).T
    n_elements = geometric_domain.n_elements
    G = nx.DiGraph()    
    G.add_nodes_from(onp.arange(geometric_domain.n_elements))
    G.add_edges_from(total_edges)
    return G

def n_theta_n_phi():
    return 20, 30

def test_tetrahedral_levels():
    import networkx as nx
    from jax_sbgeom.flux_surfaces import ToroidalExtent
    from jax_sn.domain.geometric_domain import GeometricDomain
    
    torus = jax_sbgeom.flux_surfaces.FluxSurface.from_rmnc_zmns_settings(Rmnc=jnp.array(
                                                                        [[5.0, 0.0],
                                                                        [5.0, 1.0]]), 
                                                                        Zmns=jnp.array(
                                                                            [[0.0, 0.0],
                                                                            [0.0, 1.0]]), 
                                                                        settings = jax_sbgeom.flux_surfaces.FluxSurfaceSettings(1, 0, 1))
    mesh = jax_sbgeom.flux_surfaces.flux_surface_meshing.mesh_tetrahedra(torus, jnp.linspace(0,1,10), ToroidalExtent(0.0, 2 * jnp.pi / 10), *n_theta_n_phi())
    geom_domain = GeometricDomain.from_mesh(mesh, jax_sn.tools.mesh_tools._face_template(mesh[1].shape[1]))
    angle = jax_sn.quadrature_set.tn.create_tn_quadrature_set(4)[100][0]
    
    networks_graph = create_networkx_graph(geom_domain, angle)
    generations = jax_sn.domain.compute_levels_domain(geom_domain, angle, 15000, while_loop=True)
    for i, gen in enumerate(nx.algorithms.topological_generations(networks_graph)):                
        gen_nx = onp.sort(gen)
        gen_jsn = onp.sort(onp.where(generations == i)[0])
        assert onp.all(gen_nx == gen_jsn)
        