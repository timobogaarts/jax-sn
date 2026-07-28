"""
NumPy implementations of mesh setup functions from mesh_tools.py.

Avoids JAX compilation overhead during mesh initialisation.  The main
bottleneck in the JAX version is the argsort / lexsort inside
`_compute_adjacency` and the subsequent `_topologicalsorted_angle` scan,
which each require XLA compilation the first time they are called.

The functions here are drop-in replacements that return plain NumPy arrays;
callers can wrap the results in `jnp.array(...)` as needed.
"""

import math
import numpy as np
from itertools import permutations as _itertools_permutations
from typing import Tuple

from jax_sn.domain.geometric_domain import _check_mesh_is_valid


def _pad_last_dimension_numpy(vertices: np.ndarray) -> np.ndarray:
    """Zero-pad the last axis to width 3."""
    pad = [(0, 0)] * (vertices.ndim - 1) + [(0, 3 - vertices.shape[-1])]
    return np.pad(vertices, pad, mode="constant")


def _compute_adjacency_numpy(
    mesh_connectivity: np.ndarray,
    face_template: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    NumPy version of `jax_sn.tools.mesh_tools._compute_adjacency`.

    Computes element-to-element and element-to-face adjacency via
    lexicographic sorting of sorted-vertex face tuples.  No JAX
    compilation is required.

    Parameters
    ----------
    mesh_connectivity : np.ndarray [n_elements, n_nodes_per_element]
    face_template     : np.ndarray [n_faces_per_element, n_vertices_per_face]

    Returns
    -------
    element_to_element : np.ndarray [n_elements, n_faces_per_element]
        Neighbour element index; -1 on boundary faces.
    element_to_face    : np.ndarray [n_elements, n_faces_per_element]
        Neighbour local-face index; -1 on boundary faces.
    """
    mesh_connectivity = np.asarray(mesh_connectivity)
    face_template = np.asarray(face_template)

    n_elements          = mesh_connectivity.shape[0]
    n_faces_per_element = face_template.shape[0]
    n_vertices_per_face = face_template.shape[1]

    # Sort vertex indices within each face so identical faces hash the same.
    faces      = mesh_connectivity[:, face_template]          # [E, F, V]
    faces      = np.sort(faces, axis=-1)
    faces_flat = faces.reshape(-1, n_vertices_per_face)       # [E*F, V]

    # Lexicographic sort: same as jnp.lexsort(faces_flat.T)
    # np.lexsort uses the *last* row as the primary key, matching JAX.
    sorted_idx = np.lexsort(faces_flat.T)

    elem_ids         = np.repeat(np.arange(n_elements), n_faces_per_element)
    face_indices     = np.tile(np.arange(n_faces_per_element), n_elements)

    faces_sorted     = faces_flat[sorted_idx]
    elem_sorted      = elem_ids[sorted_idx]
    face_idx_sorted  = face_indices[sorted_idx]

    # Adjacent entries in the sorted list that are identical are paired faces.
    same = np.all(faces_sorted[:-1] == faces_sorted[1:], axis=-1)

    left_elem  = elem_sorted[:-1][same]
    right_elem = elem_sorted[1:][same]
    face1      = face_idx_sorted[:-1][same]
    face2      = face_idx_sorted[1:][same]

    # Padded arrays so -1 indices are safe (they hit the padding row/col).
    etoe = np.full((n_elements + 1, n_faces_per_element + 1), -1, dtype=np.int64)
    etof = np.full((n_elements + 1, n_faces_per_element + 1), -1, dtype=np.int64)

    etoe[left_elem,  face1] = right_elem
    etoe[right_elem, face2] = left_elem
    etof[left_elem,  face1] = face2
    etof[right_elem, face2] = face1

    return etoe[:-1, :-1], etof[:-1, :-1]


def _compute_face_normals_numpy(
    simplices: np.ndarray,
    face_template: np.ndarray,
) -> np.ndarray:
    """
    NumPy version of `compute_face_normal_vectors_from_3d_simplices`.

    Parameters
    ----------
    simplices     : np.ndarray [n_elements, n_nodes_per_element, n_dim]
                    n_dim may be 1, 2, or 3; padded to 3 internally.
    face_template : np.ndarray [n_faces_per_element, n_vertices_per_face]

    Returns
    -------
    normals : np.ndarray [n_elements, n_faces_per_element, 3]
        Outward-pointing unit normals.
    """
    simplices     = np.asarray(simplices, dtype=float)
    face_template = np.asarray(face_template)

    if simplices.shape[-1] < 3:
        simplices = _pad_last_dimension_numpy(simplices)

    n_verts_per_face = face_template.shape[1]

    centroids    = np.mean(simplices, axis=1)                     # [E, 3]
    total_faces  = simplices[:, face_template, :]                 # [E, F, V, 3]

    if n_verts_per_face == 3:                                     # tetrahedral faces
        v1      = total_faces[:, :, 1, :] - total_faces[:, :, 0, :]
        v2      = total_faces[:, :, 2, :] - total_faces[:, :, 0, :]
        normals = np.cross(v1, v2)                                # [E, F, 3]

    elif n_verts_per_face == 2:                                   # triangle edges
        edge0          = total_faces[:, 0, 1, :] - total_faces[:, 0, 0, :]   # [E, 3]
        centroid_diff  = centroids - total_faces[:, 0, 0, :]                  # [E, 3]
        normal_tri     = np.cross(edge0, centroid_diff)                        # [E, 3]
        edges          = total_faces[:, :, 1, :] - total_faces[:, :, 0, :]   # [E, F, 3]
        normals        = np.cross(edges, normal_tri[:, np.newaxis, :])        # [E, F, 3]

    elif n_verts_per_face == 1:                                   # line element endpoints
        direction = total_faces[:, 1, 0, :] - total_faces[:, 0, 0, :]        # [E, 3]
        normals   = np.stack([-direction, direction], axis=1)                  # [E, 2, 3]

    else:
        raise NotImplementedError(
            f"Face normals not implemented for n_vertices_per_face={n_verts_per_face}"
        )

    norms         = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals_norm  = normals / norms

    # Flip normals that point inward.
    ref_verts   = total_faces[:, :, 0, :]                                     # [E, F, 3]
    outward_dot = np.sum(
        (ref_verts - centroids[:, np.newaxis, :]) * normals_norm, axis=-1
    )                                                                           # [E, F]
    return normals_norm * np.sign(outward_dot)[:, :, np.newaxis]


def _compute_face_permutations_numpy(
    connectivity: np.ndarray,
    etoe: np.ndarray,
    etof: np.ndarray,
    face_template: np.ndarray,
) -> np.ndarray:
    """
    NumPy version of `_compute_face_permutations`.

    For each (element, face) pair, finds which permutation of the
    self-face vertex ordering produces the neighbour-face vertex ordering.
    Returns the Lehmer (factorial-number-system) index of that permutation.

    Parameters
    ----------
    connectivity  : np.ndarray [n_elements, n_nodes_per_element]
    etoe          : np.ndarray [n_elements, n_faces_per_element]
    etof          : np.ndarray [n_elements, n_faces_per_element]
    face_template : np.ndarray [n_faces_per_element, n_vertices_per_face]

    Returns
    -------
    np.ndarray [n_elements, n_faces_per_element]
        Permutation number (Lehmer code); -1 for boundary faces.
    """
    connectivity  = np.asarray(connectivity)
    etoe          = np.asarray(etoe)
    etof          = np.asarray(etof)
    face_template = np.asarray(face_template)

    n_elements       = connectivity.shape[0]
    n_faces          = face_template.shape[0]
    n_verts_per_face = face_template.shape[1]

    # All permutations of vertex positions in lexicographic order.
    # argmax over this list gives the Lehmer code directly.
    all_perms = np.array(
        list(_itertools_permutations(range(n_verts_per_face))), dtype=np.int64
    )                                                                           # [n_perms, V]

    safe_etoe = np.maximum(etoe, 0)
    safe_etof = np.maximum(etof, 0)

    face_verts_self = connectivity[:, face_template]                           # [E, F, V]

    # Gather neighbour face vertex global IDs
    flat_etoe      = safe_etoe.ravel()                                         # [E*F]
    flat_face_tmpl = face_template[safe_etof.ravel()]                          # [E*F, V]
    neighbor_flat  = connectivity[flat_etoe[:, np.newaxis], flat_face_tmpl]   # [E*F, V]
    neighbor_verts = neighbor_flat.reshape(n_elements, n_faces, n_verts_per_face)

    # Check which permutation of self matches neighbour
    self_permuted     = face_verts_self[:, :, all_perms]                       # [E, F, n_perms, V]
    neighbor_expanded = neighbor_verts[:, :, np.newaxis, :]                    # [E, F, 1, V]
    matches           = np.all(self_permuted == neighbor_expanded, axis=-1)    # [E, F, n_perms]
    perm_numbers      = np.argmax(matches, axis=-1)                            # [E, F]

    return np.where(etoe >= 0, perm_numbers, -1).astype(np.int64)


def _compute_domain_setup_from_mesh_numpy(
    mesh: Tuple[np.ndarray, np.ndarray],
    face_template: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    NumPy drop-in for `_compute_domain_setup_from_mesh`.

    Computes face normals, element-to-element and element-to-face adjacency,
    and face permutation indices without incurring JAX compilation cost.

    Parameters
    ----------
    mesh          : (vertices [n_vertices, n_dim], connectivity [n_elements, n_nodes])
    face_template : np.ndarray [n_faces_per_element, n_vertices_per_face]

    Returns
    -------
    vertices, connectivity, face_normals, etoe, etof, face_perm_idx
        All as plain NumPy arrays.
    """
    assert np.asarray(face_template).ndim == 2, "face_template must be 2-D"
    assert len(mesh) == 2, "mesh must be a tuple of (vertices, connectivity)"
    _check_mesh_is_valid(mesh)

    vertices, connectivity = np.asarray(mesh[0]), np.asarray(mesh[1])
    face_template          = np.asarray(face_template)

    simplices     = vertices[connectivity]
    face_normals  = _compute_face_normals_numpy(simplices, face_template)
    etoe, etof    = _compute_adjacency_numpy(connectivity, face_template)
    face_perm_idx = _compute_face_permutations_numpy(connectivity, etoe, etof, face_template)

    return vertices, connectivity, face_normals, etoe, etof, face_perm_idx
