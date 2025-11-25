import jax 
import jax.numpy as jnp
from functools import partial
import numpy as onp

def _face_template(n_nodes_per_element: int) -> jnp.ndarray:
    '''
    Get the face template for a given element type. This defines which nodes belong to each face of the element.

    Parameters
    ----------
    n_nodes_per_element : int
        Number of nodes per element
    Returns
    -------
    face_template : jnp.ndarray [n_faces_per_element, n_nodes_per_face]
        The face template for the given element type    
    
    '''
    if n_nodes_per_element == 2:
        return jnp.array([[1],
                          [0]])
    elif n_nodes_per_element == 3:
        return jnp.array([[1,2],
                          [0,2],
                          [0,1]])
    elif n_nodes_per_element == 4:
        return jnp.array([[1,2,3],
                          [2,3,0],
                          [3,0,1],
                          [0,1,2]])    
    else:
        raise NotImplementedError(f"Face template for elements with {n_nodes_per_element} nodes not implemented")


@jax.jit
def _compute_adjacency(mesh_connectivity : jnp.ndarray, face_template : jnp.ndarray):
    '''
    Compute the adjacency list for a mesh.count

    See [1], pp. 428 for algorithm description. Generalised to arbitrary element types.

    Use 64-bit enabled jax to ensure you have sufficient range for face hashing. 
    For n_nodes_per_face, the maximum number of vertices is (2^63 -1)^(1/(n_nodes_per_face-1)) before face collisions will occur.

    [1] Hesthaven, J. S., & Warburton, T. (2008). Nodal discontinuous Galerkin methods: algorithms, analysis, and applications. New York, NY: Springer New York.

    Parameters
    ----------
    mesh_connectivity : jnp.ndarray [n_elements, n_nodes_per_element]
        The connectivity array of the mesh  
    
    Returns
    -------
    element_to_element : jnp.ndarray [n_elements, n_faces_per_element]
        The adjacency list of elements. element_to_element[i,f] gives the index of the element adjacent to element i at face f. -1 if boundary face.
    element_to_face : jnp.ndarray [n_elements, n_faces_per_element]
        The adjacency list of faces. element_to_face[i,f] gives the index of the face of the adjacent element to element i at face f. -1 if boundary face.     
    '''


    n_elements          = mesh_connectivity.shape[0]
    n_nodes_per_element = mesh_connectivity.shape[1]    
    n_nodes_per_face    = face_template.shape[1]

    n_vertices = jnp.max(mesh_connectivity) + 1 # is not static


    # assert n_vertices < (2**63 -1)**(1/(n_nodes_per_face-1)), f"Number of vertices {n_vertices} too large for face hashing with {n_nodes_per_face} nodes per face"

    n_faces_per_element = face_template.shape[0]
    n_total_faces       = n_elements * n_faces_per_element    
    
    faces               = mesh_connectivity[:, face_template]                                # [n_elements, n_faces_per_element, n_nodes_per_face]
    faces               = jnp.take_along_axis(faces , jnp.argsort(faces, axis=-1), axis=-1)  # [n_elements, n_faces_per_element, n_nodes_per_face]


    # Create a unique index for each face based on its vertices [hashing works until (2^63 -1)^(1/(n_nodes_per_face-1) > n_vertices]
    powers         = jnp.arange(n_nodes_per_face - 1, -1, -1, dtype=jnp.int64)   # [n_nodes_per_face]    
    bases          = n_vertices ** powers                                        # [n_nodes_per_face]
    face_keys      = jnp.sum(faces * bases, axis=-1).flatten()                   # [n_elements * n_faces_per_element]    
    sorted_idx     = jnp.argsort(face_keys)

    # Adjacency 
    elem_ids        = jnp.repeat(jnp.arange(n_elements), n_faces_per_element)   # [n_total_faces]
    ids_sorted      = face_keys[sorted_idx]                                     # [n_total_faces]    
    elem_ids_sorted = elem_ids[sorted_idx]                                      # [n_total_faces]    
    
    same = jnp.where(ids_sorted[:-1] == ids_sorted[1:], True, False)            # [n_total_faces -1] 
    same_idx = jnp.where(same, jnp.arange(same.shape[0]), -1)                   # [n_total_faces -1]   
    
    # Adjacency matrix:    
    left_element     = jnp.where(same, elem_ids_sorted[:-1][same_idx], -1) # -1 placeholder
    right_element    = jnp.where(same, elem_ids_sorted[1:][same_idx] , -1) # -1 placeholder

    face_indices = jnp.tile(jnp.arange(n_faces_per_element), n_elements)   # [n_total_faces]
 
    face_indices_sorted = face_indices[sorted_idx]

    face1 = jnp.where(same, face_indices_sorted[:-1][same_idx], -1)        # -1 placeholder 
    face2 = jnp.where(same, face_indices_sorted[1:][same_idx], -1)         # -1 placeholder    


    # Pad to account for -1 placeholders
    element_to_element = jnp.full((n_elements + 1, n_faces_per_element + 1), -1)  # initialize to -1
    
    element_to_element = element_to_element.at[left_element, face1].set(right_element)
    element_to_element = element_to_element.at[right_element, face2].set(left_element)

    element_to_face    = jnp.full((n_elements + 1, n_faces_per_element + 1), -1)  # initialize to -1

    element_to_face    = element_to_face.at[left_element, face1].set(face2)
    element_to_face    = element_to_face.at[right_element, face2].set(face1)    
    
    return element_to_element[:-1, :-1], element_to_face[:-1, :-1]


def _create_face_centroids_from_simplex_internal(simplex : jnp.ndarray, face_template : jnp.ndarray) -> jnp.ndarray:
    '''
    Create centroids of each edge in the simplex given by the face_template.

    Use the vectorized version create_face_centroids if you want to compute centroids for multiple simplices at once.

    Parameters
    ----------
    simplex : jnp.ndarray [n_nodes_per_element, n_dim]
        Array representing the vertices of the simplex.
    face_template : jnp.ndarray [n_faces_per_element, n_nodes_per_face]
        Array defining which nodes belong to each face of the simplex.
    Returns
    -------
    jnp.ndarray [n_faces_per_element, n_dim]
        Array of centroids for each face of the simplex.

    '''

    assert simplex.ndim == 2, "Simplex must be of shape [n_nodes_per_element, n_dim]"
    
    total_faces = simplex[face_template]  # [n_faces, n_nodes_per_face, n_dim]
    
    centroids = jnp.mean(total_faces, axis=-2)
    
    return centroids

_create_face_centroids_from_simplices_vec = jax.jit(jnp.vectorize(_create_face_centroids_from_simplex_internal, excluded =(1,), signature='(n_nodes_per_elem,3)->(n_faces_per_element,3)'))

@jax.jit
def create_face_centroids_from_simplices(simplices : jnp.ndarray, face_template : jnp.ndarray) -> jnp.ndarray:
    '''
    Create centroids of each face in the simplex given by the face_template.

    Fully vectorized.

    Parameters
    ----------
    simplices : jnp.ndarray [..., n_nodes_per_element, n_dim]
        Array representing the vertices of the simplices.
    face_template : jnp.ndarray [n_faces_per_element, n_nodes_per_face]
        Array defining which nodes belong to each face of the simplex.
    Returns
    -------
    jnp.ndarray [..., n_faces_per_element, n_dim]
        Array of centroids for each face of the simplex.

    '''
    return _create_face_centroids_from_simplices_vec(simplices, face_template)



def _create_normals_from_simplex(total_faces : jnp.ndarray, centroid : jnp.ndarray) -> jnp.ndarray:
    '''
    *internal*
    Create normal vectors from simplex faces.

    Checks whether faces are triangles, edges or points and computes normals accordingly.
    Asserts false if shape does not match expected dimensions.

    Parameters
    ----------
    total_faces : jnp.ndarray [n_faces, n_nodes_per_face, 3]
        Array representing the vertices of each face.

    Returns
    -------
    jnp.ndarray [n_faces, 3]
        Array of normal vectors for each face.
    '''
    assert total_faces.ndim == 3, "total_faces must be a 3D array"
    assert total_faces.shape[2] == 3, "total_faces must have 3 coordinates per vertex"

    if total_faces.shape[1] == 3:
        v1 = total_faces[:, 1, :] - total_faces[:, 0, :]
        v2 = total_faces[:, 2, :] - total_faces[:, 0, :]
        normals = jnp.cross(v1, v2) # we can just use the cross product of the two triangle edges
    elif total_faces.shape[1] == 2:
        normal_tri = jnp.cross(total_faces[0,1,:] - total_faces[0,0,:], centroid - total_faces[0,0,:])
        normals = jnp.cross(total_faces[:,1,:] - total_faces[:,0,:], normal_tri) # we use the cross product of the edge with the normal of the triangle itself.
    elif total_faces.shape[1] == 1:
        direction = total_faces[1,0 ,:] - total_faces[0,0,:]
        normals = jnp.array([-direction, direction])   # we just use the difference between the two faces
    return normals

def _create_face_normal_vectors_from_3d_simplex_internal(simplex : jnp.ndarray, face_template : jnp.ndarray):
    '''
    Create outward pointing normal vectors for each face of a 3D simplex.

    Use the vectorized version create_face_normal_vectors_3d if you want to compute normals for multiple simplices at once.

    Parameters:
    ----------
    simplex : jnp.ndarray [n_nodes_per_element, 3]
        Array representing the vertices of the simplex.
    face_template : jnp.ndarray [n_faces_per_element, n_nodes_per_face]
        Array defining which nodes belong to each face of the simplex.
    Returns:
    -------
    jnp.ndarray [n_faces_per_element, 3]
        Array of outward pointing normal vectors for each face of the simplex.

    '''
    

    centroid                  = jnp.mean(simplex, axis=0)
    total_faces               = simplex[face_template]  # [n_faces, n_nodes_per_face, 3 (cartesian)]

    assert total_faces.ndim == 3, "total_faces must be a 3D array"
    assert total_faces.shape[2] == 3, "total_faces must have 3 coordinates per vertex"

    normals                   = _create_normals_from_simplex(total_faces, centroid)
    normals_norm              = normals / jnp.linalg.norm(normals, axis=-1, keepdims=True)
    normals_pointing_outwards = jnp.sign( jnp.sum( (total_faces[:,0,:] - centroid)* normals_norm, axis=-1))
    return normals_norm * normals_pointing_outwards[:, jnp.newaxis]

_create_face_normal_vectors_from_3d_simplices_vec = jax.jit(jnp.vectorize(_create_face_normal_vectors_from_3d_simplex_internal, excluded =(1,), signature='(n_nodes_per_elem,3)->(n_faces_per_element,3)'))

@jax.jit
def create_face_normal_vectors_from_3d_simplices(simplices : jnp.ndarray, face_template : jnp.ndarray) -> jnp.ndarray:
    '''
    Create outward pointing normal vectors for each face of a 3D simplex.


    Parameters:
    ----------
    simplices : jnp.ndarray [..., n_nodes_per_element, 3]
        Array representing the vertices of the simplices
    face_template : jnp.ndarray [n_faces_per_element, n_nodes_per_face]
        Array defining which nodes belong to each face of the simplex.
    Returns:
    -------
    jnp.ndarray [..., n_faces_per_element, 3]
        Array of outward pointing normal vectors for each face of the simplex.

    '''
    return _create_face_normal_vectors_from_3d_simplices_vec(simplices, face_template)


#======================================================================================================================================================================
#                                                       Visualization and mesh manipulation tools               
#======================================================================================================================================================================
def extract_elements_from_mesh(mesh, element_indices):  
    '''
    Extract a subset of elements from a mesh, remapping vertex indices.

    Parameters
    ----------
    mesh : Tuple[jnp.ndarray, jnp.ndarray]
        The input mesh as (vertices, connectivity)
    element_indices : jnp.ndarray [n_selected_elements]
        The indices of the elements to extract
    '''  
    vertices, connectivity = onp.array(mesh[0]), onp.array(mesh[1])
    element_indices        = onp.array(element_indices)
    selected_connectivity = connectivity[element_indices]
    unique_vertex_indices = onp.unique(selected_connectivity)
    index_mapping         = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertex_indices)}
    remapped_connectivity = onp.array([[index_mapping[idx] for idx in element] for element in selected_connectivity])
    selected_vertices     = vertices[unique_vertex_indices]
    return selected_vertices, remapped_connectivity


def extract_boundaries_from_mesh(mesh, boundary_index, face_template):
    '''
    Extract boundary faces from a mesh.

    Parameters
    ----------
    mesh : Tuple[jnp.ndarray, jnp.ndarray]
        The input mesh as (vertices, connectivity)
    boundary_index : int or jnp.ndarray [n_elements]
        If int, the index of the boundary face to extract. If array, an array with one entry per element indicating which boundary face to extract for each element.
    face_template : jnp.ndarray [n_faces_per_element, n_nodes_per_face]
        The face template for the element type
    '''
    vertices, connectivity = onp.array(mesh[0]), onp.array(mesh[1])
    face_template = onp.array(face_template)    

    if onp.array(boundary_index).ndim == 0:        
        return vertices, connectivity[:, face_template[boundary_index]]
    elif onp.array(boundary_index).ndim == 1:
        assert boundary_index.shape[0] == mesh[1].shape[0], "If boundary_index is an array, it must have one entry per element"                
        return vertices, onp.take_along_axis(connectivity, face_template[boundary_index])
