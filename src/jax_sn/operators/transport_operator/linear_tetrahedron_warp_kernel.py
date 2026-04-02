from __future__ import annotations

from tracemalloc import start
from typing import Tuple

import numpy as np
import warp as wp
from dataclasses import dataclass
import jax.numpy as jnp

NFACES = 4
NBASIS = 4
NDIM = 3

DTYPE        = wp.float64
NBASISDTYPE  = wp.vec4d
NMATRIXDTYPE = wp.mat44d
SPATVECDTYPE      = wp.vec3d
SPATMATDTYPE      = wp.mat33d

NFACESDTYPE = wp.vec4d
NFACESBOOLDTYPE = wp.vec4b

NFACESINTTYPE = wp.vec4l



@wp.struct
class SweepData:
    """All arrays needed to compute gij_inv and dij_face on-the-fly inside a kernel.

    Shapes (degree-1 3-D simplices: n_faces = n_basis = 4, n_dim = 3)
    -------------------------------------------------------------------
    face_normals        [n_elem, n_faces]              vec3d    outward unit face normals
    face_jacobians_det  [n_elem, n_faces]              float64  |J_face|
    vol_jacobians_det   [n_elem]                       float64  |J_vol|
    vol_jacobians_inv   [n_elem]                       mat33d   J_vol^{-1}
    sigma_t             [n_elem]                       float64  total cross-section σ_t

    etoe                [n_elem, n_faces]              int32    neighbour element, -1=boundary
    etof                [n_elem, n_faces]              int32    local face index on neighbour
    face_perm_idx       [n_elem, n_faces]              int32    permutation index

    mass_matrix         [1]                            mat44d   reference mass matrix
    derivative_matrix   [n_dim]                        mat44d   B_{a,ij}
    outflow_matrices    [n_faces]                      mat44d   C_{f,ij}
    inflow_matrices     [n_faces, n_perms, n_faces]    mat44d   inflow coupling matrices
    """

    # Per-element geometric
    face_normals        : wp.array2d(dtype=SPATVECDTYPE)    # [n_elem, n_faces]      vec3d
    face_jacobians_det  : wp.array2d(dtype=DTYPE)      # [n_elem, n_faces]      float64
    vol_jacobians_det   : wp.array(dtype=DTYPE)        # [n_elem]               float64
    vol_jacobians_inv   : wp.array(dtype=SPATMATDTYPE)      # [n_elem]               mat33d

    # Topology
    etoe                : wp.array(dtype= NFACESINTTYPE) # [n_elem, n_faces]
    etof                : wp.array(dtype= NFACESINTTYPE)      # [n_elem, n_faces]
    face_perm_idx       : wp.array(dtype= NFACESINTTYPE)   # [n_elem, n_faces]

    # Reference FEM matrices (same for every element)
    mass_matrix         : wp.array(dtype=NMATRIXDTYPE)    # [1]
    derivative_matrix   : wp.array(dtype=NMATRIXDTYPE)    # [n_dim]
    outflow_matrices    : wp.array(dtype=NMATRIXDTYPE)    # [n_faces]
    inflow_matrices     : wp.array3d(dtype=NMATRIXDTYPE)  # [n_faces, n_perms, n_faces]

    # Topological angle information 
    angles              : wp.array(dtype=SPATVECDTYPE)
    levels_argsort     : wp.array2d(dtype=wp.int64)


def sweepdata_from_transport_operator(transport_operator) -> SweepData:
    """Build a SweepData struct from a single-angle transport operator.

    Parameters
    ----------
    transport_operator
        A single-angle transport operator instance
        (e.g. ``mgop.transport_operator[angle_idx]``).
    
    
    """
    sd  = transport_operator.solution_domain_data
    dom = sd.domain
    pgd = sd.precomputed_geometric_data
    fem = sd.fem_matrices

    inst = SweepData()

    # face_normals: JAX [N, 4, 3] → warp 2D [N, 4] of vec3d
    inst.face_normals       = wp.from_jax(dom.face_normals,          dtype=SPATVECDTYPE)
    # face_jacobians_det: JAX [N, 4]
    inst.face_jacobians_det = wp.from_jax(pgd.face_jacobians_det)
    # vol_jacobians_det: JAX [N]
    inst.vol_jacobians_det  = wp.from_jax(pgd.vol_jacobians_det)
    # vol_jacobians_inv: JAX [N, 3, 3] → warp 1D [N] of mat33d
    inst.vol_jacobians_inv  = wp.from_jax(pgd.vol_jacobians_inv,      dtype=SPATMATDTYPE)


    # Topology
    inst.etoe          = wp.from_jax(dom.etoe,          dtype=NFACESINTTYPE) #[N, 4]
    inst.etof          = wp.from_jax(dom.etof,          dtype=NFACESINTTYPE) # [N, 4]
    inst.face_perm_idx = wp.from_jax(dom.face_perm_idx, dtype=NFACESINTTYPE) # [N, 4]

    # Reference FEM matrices
    # mass_matrix: JAX [4,4] → [1] × mat44d
    inst.mass_matrix       = wp.from_jax(jnp.array(fem.mass_matrix)[None, ...], dtype=NMATRIXDTYPE)
    # derivative_matrix: JAX [3,4,4] → [3] × mat44d
    inst.derivative_matrix = wp.from_jax(fem.derivative_matrix,   dtype=NMATRIXDTYPE)
    # outflow_matrices: JAX [4,4,4] → [4] × mat44d
    inst.outflow_matrices  = wp.from_jax(fem.outflow_matrices,     dtype=NMATRIXDTYPE)
    # inflow_matrices: JAX [4, n_perms, 4, 4, 4] → [4, n_perms, 4] × mat44d
    inst.inflow_matrices   = wp.from_jax(fem.inflow_matrices,      dtype=NMATRIXDTYPE)

    inst.angles           = wp.from_jax(transport_operator.topological_sorted_angle.angle, dtype=SPATVECDTYPE)
    inst.levels_argsort   = wp.from_jax(transport_operator.topological_sorted_angle.levels_argsort, dtype=wp.int64)

    return inst


# =============================================================================
# Device functions  (@wp.func — compiled to device code, called from kernels)
# =============================================================================

@wp.func
def compute_element_fused_multi_angle(elem_idx : wp.int32, angle_idx : wp.int32, sd : SweepData, xs : wp.array(dtype=DTYPE), result : wp.array2d(dtype = NBASISDTYPE), x_apply : wp.array2d(dtype = NBASISDTYPE)):
    
    sj = x_apply[angle_idx, elem_idx]
    vol_jacobian_det   = sd.vol_jacobians_det[elem_idx]   # scalar
    vol_jacobian_inv   = sd.vol_jacobians_inv[elem_idx]   # mat33d
    face_jacobians_det = sd.face_jacobians_det[elem_idx] # vec4d
    face_normals       = sd.face_normals[elem_idx]       # [4] vec3d
    sigma_t           = xs[elem_idx]
    angle             = sd.angles[angle_idx]              # vec3d
    face_perms        = sd.face_perm_idx[elem_idx]
    face_conns        = sd.etof[elem_idx]
    neighbours        = sd.etoe[elem_idx]
    Aij = vol_jacobian_det * sigma_t * sd.mass_matrix[0]
    
    angle_ref = vol_jacobian_inv * angle
    Bij = vol_jacobian_det * ( angle_ref[0] * sd.derivative_matrix[0] + angle_ref[1] * sd.derivative_matrix[1] + angle_ref[2] * sd.derivative_matrix[2])
         
    Cij = NMATRIXDTYPE()

    for i in range(NFACES):

        n_vec              = face_normals[i]

        omega_dot_n        = wp.dot(n_vec, angle)
        abs_omega_dot_n_Je = wp.abs(omega_dot_n) * face_jacobians_det[i]
        

        neighbour          = sd.etoe[elem_idx][i]

        if omega_dot_n < wp.float64(0.0):
            if neighbour >= 0: # inflow face with neighbour
                face_perm = face_perms[i]
                face_conn = face_conns[i]
                neighbour = neighbours[i]
                psi_conn  = result[angle_idx, neighbour]
                sj = sj + (sd.inflow_matrices[wp.int32(i), wp.int32(face_perm), wp.int32(face_conn)] @ psi_conn) * abs_omega_dot_n_Je
        else:
            Cij = Cij + abs_omega_dot_n_Je * sd.outflow_matrices[i]

    result[angle_idx, elem_idx] = wp.inverse((Aij + Bij + Cij)) @ sj

@wp.func
def compute_element_fused_single_angle(elem_idx : wp.int32, angle_idx : wp.int32, sd : SweepData, xs : wp.array(dtype=DTYPE), result : wp.array(dtype = NBASISDTYPE), x_apply : wp.array(dtype = NBASISDTYPE)):
    
    sj = x_apply[elem_idx]
    vol_jacobian_det   = sd.vol_jacobians_det[elem_idx]   # scalar
    vol_jacobian_inv   = sd.vol_jacobians_inv[elem_idx]   # mat33d
    face_jacobians_det = sd.face_jacobians_det[elem_idx] # vec4d
    face_normals       = sd.face_normals[elem_idx]       # [4] vec3d
    sigma_t           = xs[elem_idx]    
    face_perms        = sd.face_perm_idx[elem_idx]
    face_conns        = sd.etof[elem_idx]
    neighbours        = sd.etoe[elem_idx]
    Aij = vol_jacobian_det * sigma_t * sd.mass_matrix[0]
    angle             = sd.angles[angle_idx]
    
    angle_ref = vol_jacobian_inv * angle
    Bij = vol_jacobian_det * ( angle_ref[0] * sd.derivative_matrix[0] + angle_ref[1] * sd.derivative_matrix[1] + angle_ref[2] * sd.derivative_matrix[2])
         
    Cij = NMATRIXDTYPE()

    for i in range(NFACES):

        n_vec              = face_normals[i]

        omega_dot_n        = wp.dot(n_vec, angle)
        abs_omega_dot_n_Je = wp.abs(omega_dot_n) * face_jacobians_det[i]
        

        neighbour          = sd.etoe[elem_idx][i]

        if omega_dot_n < wp.float64(0.0):
            if neighbour >= 0: # inflow face with neighbour
                face_perm = face_perms[i]
                face_conn = face_conns[i]
                neighbour = neighbours[i]
                psi_conn  = result[neighbour]
                sj = sj + (sd.inflow_matrices[wp.int32(i), wp.int32(face_perm), wp.int32(face_conn)] @ psi_conn) * abs_omega_dot_n_Je
        else:
            Cij = Cij + abs_omega_dot_n_Je * sd.outflow_matrices[i]

    result[elem_idx] = wp.inverse((Aij + Bij + Cij)) @ sj


@wp.func
def compute_element_direct_multi_angle(elem_in_toposort_idx : wp.int32, angle_idx : wp.int32, sd : SweepData, xs : wp.array(dtype=DTYPE), result : wp.array2d(dtype = NBASISDTYPE), x_apply : wp.array2d(dtype = NBASISDTYPE)):
    elem_idx = wp.int32(sd.levels_argsort[angle_idx, elem_in_toposort_idx])
    compute_element_fused_multi_angle(elem_idx, angle_idx, sd, xs, result, x_apply)

@wp.func
def compute_element_direct_single_angle(elem_in_toposort_idx : wp.int32, angle_idx : wp.int32, sd : SweepData, xs : wp.array(dtype=DTYPE), result : wp.array(dtype = NBASISDTYPE), x_apply : wp.array(dtype = NBASISDTYPE)):
    elem_idx = wp.int32(sd.levels_argsort[angle_idx, elem_in_toposort_idx])    
    compute_element_fused_single_angle(elem_idx, angle_idx, sd, xs, result, x_apply)

@wp.kernel
def sequential_sweep_single_angle(angle_idx : wp.int32, sd : SweepData, xs : wp.array(dtype=DTYPE), result : wp.array1d(dtype = NBASISDTYPE), x_apply : wp.array1d(dtype = NBASISDTYPE)):    
    for elem_in_order in range(sd.face_normals.shape[0]):        
        compute_element_direct_single_angle(wp.int32(elem_in_order), angle_idx, sd, xs, result, x_apply)        

@wp.kernel
def sequential_sweep_multi_angle(sd : SweepData, xs : wp.array(dtype=DTYPE), result : wp.array2d(dtype = NBASISDTYPE), x_apply : wp.array2d(dtype = NBASISDTYPE)):    
    angle_idx = wp.int32(wp.tid())
    for elem_in_order in range(sd.face_normals.shape[0]):            
        compute_element_direct_multi_angle(wp.int32(elem_in_order), angle_idx, sd, xs, result, x_apply)


@wp.kernel
def _sweep_level_kernel(topo_offset : wp.int32, angle_idx : wp.int32,  sd : SweepData, xs : wp.array(dtype=DTYPE), result : wp.array2d(dtype = NBASISDTYPE), x_apply : wp.array2d(dtype = NBASISDTYPE)):
    in_level_id = wp.int32(wp.tid())
    compute_element_direct_multi_angle(in_level_id + topo_offset, angle_idx, sd, xs, result, x_apply)



@wp.kernel
def _sweep_level_kernel_ma(topo_offset_ids : wp.array2d(dtype = wp.int32), padded_level_size : wp.array2d(dtype = wp.int32), current_level : wp.int32,   sd : SweepData, xs : wp.array(dtype=DTYPE), result : wp.array2d(dtype = NBASISDTYPE), x_apply : wp.array2d(dtype = NBASISDTYPE)):
    angle_id, elem_in_level_id = wp.tid()

    offset_this_angle = topo_offset_ids[angle_id, current_level]
    if offset_this_angle < 0:
        return # no work for this angle
    else:
        if elem_in_level_id >= padded_level_size[angle_id, current_level]:
            return # out of bounds for this angle
        compute_element_direct_multi_angle(elem_in_level_id + offset_this_angle, angle_id, sd, xs, result, x_apply)



# =============================================================================
# Public Python wrapper
# =============================================================================

def _compute_level_bounds(levels_np: np.ndarray, levels_argsort_np: np.ndarray) -> list[tuple[int, int]]:
    """Return [(start, end), ...] index slices into levels_argsort, one per level."""
    sorted_levels      = levels_np[levels_argsort_np]
    _, first_occ       = np.unique(sorted_levels, return_index=True)
    ends               = np.append(first_occ[1:], len(sorted_levels))
    return list(zip(first_occ.tolist(), ends.tolist()))    

@dataclass
class WarpSweep:
    """Holds a SweepData struct and drives the on-device sweep."""

    _data  : SweepData
    device : str = "cuda"

    @classmethod
    def from_transport_operator(
        cls,
        transport_operator,        
        device: str = "cuda",
    ) -> "WarpSweep":
        """Construct from a single-angle transport operator.

        Parameters
        ----------
        transport_operator
            E.g. ``mgop.transport_operator[angle_idx]``.
        sigma_t : np.ndarray [n_elements]
            Total cross-sections.  Typical usage::

                sigma_t = np.asarray(
                    op.solution_domain_data.domain
                       .cross_sections_indexed
                       .map_to_full_array(op.energy_group).total
                )
        """
        return cls(sweepdata_from_transport_operator(transport_operator), device)

    
    def compute_single_angle(self, sigma_t, angle_i : int, x_apply : jnp.ndarray):
        
        x_apply_wp = wp.from_jax(x_apply, dtype=NBASISDTYPE)
        result = wp.zeros_like(x_apply_wp)        
        wp.launch(sequential_sweep_single_angle, dim=(1,), inputs=[wp.int32(angle_i), self._data, wp.from_jax(sigma_t), result, x_apply_wp])        
        wp.synchronize_device(self.device)        
        return wp.to_jax(result)

    def compute_all_angles(self, sigma_t, x_apply):
        
        n_angles   = self._data.angles.shape[0]    
        x_apply_wp = wp.from_jax(x_apply, dtype=NBASISDTYPE)
        result     = wp.zeros_like(x_apply_wp)        
        wp.launch(sequential_sweep_multi_angle, dim=(n_angles,), inputs=[ self._data, wp.from_jax(sigma_t), result, x_apply_wp])
        wp.synchronize_device(self.device)
        return wp.to_jax(result)

    def single_angle(self, angle : int, xs_wp : wp.array, x_wp: wp.array, level_bounds) -> None:
        """Run the sweep in-place on pre-allocated Warp arrays (no host copies)."""
        x_apply_bc = x_wp[None, ...]
        x_apply_wp = wp.from_jax(x_apply_bc, dtype=NBASISDTYPE)
        result = wp.zeros_like(x_apply_wp)
        for start, end in level_bounds:
            wp.launch(kernel = _sweep_level_kernel,
                dim=end - start,
                inputs=[
                    start,
                    wp.int32(angle),
                    self._data,
                    xs_wp,
                    result, 
                    x_apply_wp,
                    ],
                device=self.device
            )

        wp.synchronize()
        return result

    def multi_angle_levels(self, xs : jnp.ndarray, x_apply : jnp.ndarray, levelbased_setup : LevelBasedParallelismSetup):
        
        """Run the multi-angle level-sweep in-place on pre-allocated Warp arrays (no host copies)."""
        xs_wp      = wp.from_jax(xs, dtype=DTYPE)
        x_apply_wp = wp.from_jax(x_apply, dtype=NBASISDTYPE)
        result     =   wp.zeros_like(x_apply_wp)
        n_angles   = self._data.angles.shape[0]

        padded_occ_wp = wp.from_numpy(np.array(levelbased_setup.padded_first_occ, dtype=np.int32), dtype=wp.int32)

        padded_level_size = wp.from_numpy(np.array(levelbased_setup.padded_sizes, dtype=np.int32), dtype=wp.int32)


        for levels in range(levelbased_setup.n_levels):
                n_work = levelbased_setup.max_level_size(levels)                
                wp.launch(kernel = _sweep_level_kernel_ma,
                    dim=(n_angles, n_work),
                    inputs=[
                        padded_occ_wp,
                        padded_level_size,
                        wp.int32(levels), 
                        self._data,
                        xs_wp,
                        result,
                        x_apply_wp
                    ]
                )

        wp.synchronize()
        return result
    
    def multi_angle_levels_prealloc(self, xs : jnp.ndarray, x_apply : jnp.ndarray, levelbased_setup : LevelBasedParallelismSetup, result_array : jnp.ndarray):
        
        """Run the multi-angle level-sweep in-place on pre-allocated Warp arrays (no host copies)."""
        xs_wp      = wp.from_jax(xs, dtype=DTYPE)
        x_apply_wp = wp.from_jax(x_apply, dtype=NBASISDTYPE)
        result     = wp.from_jax(result_array, dtype=NBASISDTYPE)
        n_angles   = self._data.angles.shape[0]
        padded_occ_wp = wp.from_numpy(np.array(levelbased_setup.padded_first_occ, dtype=np.int32), dtype=wp.int32)
        
        padded_level_size = wp.from_numpy(np.array(levelbased_setup.padded_sizes, dtype=np.int32), dtype=wp.int32)

        for levels in range(levelbased_setup.n_levels):
                n_work = levelbased_setup.max_level_size(levels)                
                wp.launch(kernel = _sweep_level_kernel_ma,
                    dim=(n_angles, n_work),
                    inputs=[
                        padded_occ_wp,
                        padded_level_size,
                        wp.int32(levels), 
                        self._data,
                        xs_wp,
                        result,
                        x_apply_wp
                    ]
                )

        wp.synchronize()
        return result
    
def _multiple_angle_level_bounds(levels_np: np.ndarray, levels_argsort_np: np.ndarray) -> list[tuple[int, int]]:
    """Return [(start, end), ...] index slices into levels_argsort, one per level.

    This is for the multi-angle kernel where levels_argsort is 2D with shape [n_angles, n_elem].
    We assume that the levels are the same for each angle, so we just look at the first angle's levels.
    """
    #levels_np[levels_argsort_np]    
    sorted_levels = np.take_along_axis(levels_np, levels_argsort_np, axis=-1)    

    n_angles = sorted_levels.shape[0]
    
    first_occ = [np.unique(sorted_levels[i],return_index=True)[1] for i in range(sorted_levels.shape[0])]
    ends      = [ np.append(occ[1:], sorted_levels.shape[-1]) for occ in first_occ]    
    sizes     = [end - occ for end, occ in zip(ends, first_occ)]

    # We now need to pad such that we have a 2D array of shape [n_angles, n_levels], where each row is the (start, end) pairs for that angle, padded with -1 if that angle has fewer levels than the max.

    n_levels_per_angle = np.array([len(occ) for occ in first_occ])
    max_levels = max(n_levels_per_angle)
    
    padded_first_occ = np.full((n_angles, max_levels), -1, dtype=int) # [n_angles, max_levels]
    padded_ends      = np.full((n_angles, max_levels), -1, dtype=int)
    padded_sizes     = np.full((n_angles, max_levels), -1, dtype=int)
    for i in range(n_angles):
        n_levels_in_this_angle = n_levels_per_angle[i]
        
        padded_first_occ[i, :n_levels_in_this_angle] = first_occ[i]
        padded_ends[     i, :n_levels_in_this_angle] = ends[i]
        padded_sizes[    i, :n_levels_in_this_angle] = sizes[i]

    return padded_first_occ, padded_ends, padded_sizes

@dataclass 
class LevelBasedParallelismSetup:
    padded_first_occ: np.ndarray #[n_angles, n_max_levels]
    padded_ends: np.ndarray
    padded_sizes: np.ndarray

    @classmethod
    def from_transport_operator(cls, transport_operator):
        return cls(*_multiple_angle_level_bounds(np.array(transport_operator.topological_sorted_angle.levels), np.array(transport_operator.topological_sorted_angle.levels_argsort)))
    @property 
    def n_levels(self):
        return self.padded_first_occ.shape[1]
    
    def max_level_size(self, level):
        return self.padded_sizes[:, level].max()


