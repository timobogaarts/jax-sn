import jax_sn 
import jax 
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as onp
import pytest
def n_theta_n_phi():
    return 10, 20


@pytest.mark.Parametrise("elem", ["lines", "triangles", "tetrahedra"])
def test_mesh_graph(elem):
    import jax_sbgeom
    if elem == "lines":
        connectivity = jnp.stack([
            jnp.arange(10),
            jnp.arange(1, 11)
        ], axis=-1)
    elif elem == "triangles":
        connectivity = jax_sbgeom.flux_surfaces.flux_surface_meshing._mesh_surface_connectivity(*n_theta_n_phi(), True, True)
    elif elem == "tetrahedra":
        connectivity = jax_sbgeom.flux_surfaces.flux_surface_meshing._mesh_tetrahedra_connectivity(10, True, False, *n_theta_n_phi())

    face_template = jax_sn.tools.mesh_tools._face_template(connectivity.shape[1])
    
    etoe, etof = jax_sn.tools.mesh_tools._compute_adjacency(connectivity, face_template)    

    connectivity_i = connectivity[:, face_template] # [n_elements, n_faces_per_element, n_vertices_per_face]
    connectivity_i_sort = onp.sort(connectivity_i, axis=-1)

    connectivity_j = onp.take_along_axis(connectivity[etoe],  face_template[etof], axis=-1) # [n_elements, n_faces_per_element, n_vertices_per_face]
    connectivity_j_sort = onp.sort(connectivity_j, axis=-1)

    onp.where(etoe != -1,
              lambda: onp.testing.assert_equal(connectivity_i_sort, connectivity_j_sort),
              lambda: None)


def test_surface_mesh_graph():
    import jax_sbgeom 

    torus = jax_sbgeom.flux_surfaces.FluxSurface.from_rmnc_zmns_settings(Rmnc=jnp.array([5.0, 1.0])[None, :], Zmns=jnp.array([0.0, 1.0])[None, :], settings = jax_sbgeom.flux_surfaces.FluxSurfaceSettings(1, 0, 1))
    mesh  = jax_sbgeom.flux_surfaces.mesh_surface(torus, 1, jax_sbgeom.flux_surfaces.ToroidalExtent.half_module(torus), *n_theta_n_phi())

    import trimesh 
    
    tmesh = trimesh.Trimesh(vertices=onp.array(mesh[0]), faces=onp.array(mesh[1]))
    adjacency = tmesh.face_adjacency

    connectivity = mesh[1]
    face_template = jax_sn.tools.mesh_tools._face_template(connectivity.shape[1])
    etoe, etof = jax_sn.tools.mesh_tools._compute_adjacency(connectivity, face_template)

    for i in range(adjacency.shape[0]):
        assert adjacency[i,1] in etoe[adjacency[i,0]]
        

def get_normal_case(plot = False):
    '''
    The expected values in the test below where generated using this function with plot = False
    You  can run this function with plot = True to visualize the normals on the mesh.
    '''
    import jax_sbgeom
    from jax_sbgeom.flux_surfaces import ToroidalExtent
    import jax_sn as jsn
    flux_surface = jax_sbgeom.flux_surfaces.FluxSurface.from_rmnc_zmns_settings(Rmnc=jnp.array(
    [[5.0, 0.0],
    [5.0, 1.0]]), Zmns=jnp.array(
        [[0.0, 0.0],
        [0.0, 1.0]]), settings = jax_sbgeom.flux_surfaces.FluxSurfaceSettings(1, 0, 1))
    mesh_tet = jax_sbgeom.flux_surfaces.mesh_tetrahedra(flux_surface,  jnp.linspace(0,1,10), ToroidalExtent.half_module(flux_surface), 50, 60)
    mesh_surf = jax_sbgeom.flux_surfaces.mesh_surface(flux_surface, 1, ToroidalExtent.half_module(flux_surface), 50, 60)
    mesh_line  = (flux_surface.cartesian_position(jnp.linspace(0,1,10), 0.2, 0.5), jnp.array([[i,i+1] for i in range(9)]))
    def get_normal_centroids(mesh, i_checks):
        values = jsn.tools.mesh_tools.extract_elements_from_mesh(mesh, i_checks)

        connectivity_check = mesh[1][i_checks, :]
        positions_simplices = mesh[0][connectivity_check]
        normal_vecs = jsn.tools.mesh_tools.compute_face_normal_vectors_from_3d_simplices(positions_simplices, jsn.tools.mesh_tools._face_template(mesh[1].shape[1]))
        centroids   = jsn.tools.mesh_tools.create_face_centroids_from_simplices(positions_simplices, jsn.tools.mesh_tools._face_template(mesh[1].shape[1]))
        if plot:
            import pyvista as pv
            from jax_sbgeom.jax_utils import mesh_to_pyvista_mesh
            plotter = pv.Plotter()
            mesh_check = jsn.tools.mesh_tools.extract_elements_from_mesh(mesh, i_checks)
            plotter.add_mesh(mesh_to_pyvista_mesh(mesh), color="lightblue", show_edges=True, opacity=0.5)
            plotter.add_mesh(mesh_to_pyvista_mesh(mesh_check), color="orange", show_edges=True, opacity=1.0)


            plotter.add_arrows(onp.array(centroids.reshape(-1,3)), onp.array(normal_vecs.reshape(-1,3)), mag=0.1, color="red")
            plotter.show()
        return normal_vecs, centroids
    
    normals_line, centroids_line = get_normal_centroids(mesh_line, [0,5,8])
    normals_surf, centroids_surf = get_normal_centroids(mesh_surf, [101, 63, 1001])
    normals_tet, centroids_tet   = get_normal_centroids(mesh_tet, [100, 200, 1300])
    return normals_tet, centroids_tet, normals_surf, centroids_surf, normals_line, centroids_line

def test_normals():
    normals_tet, centroids_tet, normals_surf, centroids_surf, normals_line, centroids_line = get_normal_case()

    normals_tet_expected  = onp.array( [[[-0.48171403420675424, -0.012828024883804719, -0.8762345753426045], [0.9045644984521172, 0.024088515324778122, -0.42565574302424003], [-0.9049630122745709, 0.028706211035570237, 0.4245207884932329], [0.4466963052557006, -0.37449315058618826, 0.8125375628455194]], [[-0.4803485634402602, -0.038447712221161855, 0.8762345753426061], [-0.10629348564736354, 0.9943348002101373, 1.3050445248639804e-15], [-0.8359341549258195, -0.11986616528563952, -0.5355802377311656], [0.9020004121035098, 0.07219726446051544, 0.4256557430242389]], [[-0.43336485391955165, -0.21073175451611229, -0.8762345753426115], [0.8137742185112941, 0.3957129132550689, -0.4256557430242414], [-0.8359525433318875, -0.3478008703756095, 0.42452078849323405], [0.5615223217806451, -0.15644612845041392, 0.8125375628455278]]] )
    centroids_tet_expected= onp.array( [[[4.946275532821327, 0.08786433214808088, -0.09829563075681683], [4.963758795800336, 0.17656795688504529, -0.06702422610859404], [4.959682962340562, 0.17656795688504529, -0.0647835177025198], [4.962022785885564, 0.08870362473696441, -0.0647835177025198]], [[4.9275932998402805, 0.4383903042037992, 0.0982956307568168], [4.936260727126809, 0.5276817814679939, 0.06478351770251979], [4.943273565277761, 0.4400665103155369, 0.06478351770251979], [4.947326308326676, 0.44049974486089455, 0.06702422610859403]], [[4.467943883570577, 2.1238736404774654, -0.09829563075681683], [4.447211332119091, 2.211874542951139, -0.06702422610859404], [4.443499736442254, 2.2101903588983975, -0.0647835177025198], [4.481937073862664, 2.1311448863520237, -0.0647835177025198]]] )
    normals_tri_expected  = onp.array( [[[0.05644834300106321, -0.027449061114562686, 0.9980281226580558], [0.10755636188690847, 0.3547576141259899, -0.9287511314864103], [-0.4372123552178677, -0.8993567554011714, 0.0016682203459618575]], [[0.006671867236417652, -0.062412759682760485, 0.9980281226580551], [0.35779038619670933, 0.0969916249370204, -0.9287511314864096], [-0.9943222590045557, -0.10639766110659887, 0.0016682203459595529]], [[-0.046635109771647046, -0.8749926842907993, 0.4818848088227808], [0.4374954291272866, 0.7827232285528181, -0.44265347279399503], [-0.9993988443456472, 0.03279608904918527, 0.011241283897447543]]] )
    centroids_tri_expected= onp.array( [[[-5.386871712070242, 2.619466984852682, 0.12533323356430426], [-5.390462025658934, 2.621096135605322, 0.06266661678215213], [-5.460218323540786, 2.47764396385763, 0.06266661678215213]], [[-0.6366970396608814, 5.956056665842162, 0.12533323356430426], [-0.6372203298737942, 5.959964433899895, 0.06266661678215213], [-0.7958298734894184, 5.943009217966309, 0.06266661678215213]], [[0.288669436606143, 5.416169200534201, 0.9048270524660196], [0.290134197568045, 5.471173452435613, 0.8745774889840173], [0.14590183490964465, 5.478860707638952, 0.8745774889840173]]] )
    normals_lin_expected  = onp.array( [[[0.8600893382050481, 0.46986894694951375, 0.19866933079506116], [-0.8600893382050481, -0.46986894694951375, -0.19866933079506116]], [[0.8600893382050481, 0.46986894694951375, 0.19866933079506094], [-0.8600893382050481, -0.46986894694951375, -0.19866933079506094]], [[0.8600893382050481, 0.4698689469495137, 0.1986693307950613], [-0.8600893382050481, -0.4698689469495137, -0.1986693307950613]]] )
    centroids_lin_expected= onp.array( [[[4.483478291474647, 2.4493353537931832, 0.022074370088340133], [4.387912809451864, 2.397127693021015, 0.0]], [[4.961305701588562, 2.710373657654025, 0.1324462205300408], [4.865740219565779, 2.658165996881857, 0.11037185044170068]], [[5.248002147656911, 2.8669966399705302, 0.19866933079506122], [5.152436665634128, 2.814788979198362, 0.17659496070672107]]] )

    onp.testing.assert_allclose(normals_tet, normals_tet_expected, atol = 1e-12)
    onp.testing.assert_allclose(centroids_tet, centroids_tet_expected,atol = 1e-12)
    onp.testing.assert_allclose(normals_surf, normals_tri_expected, atol = 1e-12)
    onp.testing.assert_allclose(centroids_surf, centroids_tri_expected, atol = 1e-12)
    onp.testing.assert_allclose(normals_line, normals_lin_expected, atol = 1e-12)
    onp.testing.assert_allclose(centroids_line, centroids_lin_expected, atol = 1e-12)
