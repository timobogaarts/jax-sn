from __future__ import annotations

from abc import abstractmethod
from tracemalloc import start
from typing import Tuple
from unittest import result

import jax
import numpy as np
import warp as wp
from dataclasses import dataclass
import jax.numpy as jnp


NFACES = 4
NBASIS = 4
NDIM = 3
  
DTYPE             = wp.float64
NBASISDTYPE       = wp.vec4d
NMATRIXDTYPE      = wp.mat44d
SPATVECDTYPE      = wp.vec3d
SPATMATDTYPE      = wp.mat33d
NFACESDTYPE       = wp.vec4d
NFACESBOOLDTYPE   = wp.vec4b
NFACESINTTYPE     = wp.vec4l

JAXINDEXDTYPE    = wp.int64
WARPINDEXDTYPE   = wp.int32

INFLOW_EPSILON = DTYPE(-1e-12) 
'''
Floating point differences can lead to JAX and Warp disagreeing on whether a face is inflow or outflow.
Then, an element can access a non-computed face neighbour as inflow in Warp (trying to get non-computed elements).
This leads to NaNs in the solution. Although, in principle, one could set the whole result array to zero before the sweep to avoid NaNs, 
this would not work for level-based parallelism, where the levels could *in principle* have two of these neighbours on the same level,
which could lead to a race condition. Although extremely rare, we don't want to introduce the possibility. This epsilon guards 
agains this by treaing faces with angle.face_normal > -INFLOW_EPSILON as outflow faces. Since the contribution is multiplied 
by this dot-product, even if it were physical, the contribution would be on the order of the INFLOW_EPSILON, which should
be chosen to be on the order of floating point errors.
'''

@wp.struct 
class SweepGeometricData:
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
    face_normals        : wp.array2d[SPATVECDTYPE]    # [n_elem, n_faces]      vec3d
    face_jacobians_det  : wp.array2d[DTYPE]      # [n_elem, n_faces]      float64
    vol_jacobians_det   : wp.array[DTYPE]        # [n_elem]               float64
    vol_jacobians_inv   : wp.array[SPATMATDTYPE]      # [n_elem]               mat33d

    # Topology
    etoe                : wp.array[NFACESINTTYPE] # [n_elem, n_faces]
    etof                : wp.array[NFACESINTTYPE]      # [n_elem, n_faces]
    face_perm_idx       : wp.array[NFACESINTTYPE]   # [n_elem, n_faces]

    # Reference FEM matrices (same for every element)
    mass_matrix         : wp.array[NMATRIXDTYPE]    # [1]
    derivative_matrix   : wp.array[NMATRIXDTYPE]    # [n_dim]
    outflow_matrices    : wp.array[NMATRIXDTYPE]    # [n_faces]
    inflow_matrices     : wp.array3d[NMATRIXDTYPE]  # [n_faces, n_perms, n_faces]

@wp.struct
class SweepDataMultiAngle:

    sweep_geometric_data  : SweepGeometricData
    # Topological angle information 
    angles              : wp.array[SPATVECDTYPE]
    levels_argsort      : wp.array2d[JAXINDEXDTYPE]

@wp.struct
class SweepDataSingleAngle:
    sweep_geometric_data  : SweepGeometricData
    # Topological angle information 
    angles              : wp.array[SPATVECDTYPE] # not actually an array! but have to do because of the kernel launch.
    levels_argsort      : wp.array[JAXINDEXDTYPE]


# =============================================================================
# Device functions  (@wp.func — compiled to device code, called from kernels)
# =============================================================================
#-----------------------------------------------------------------------------
# Single Element functions
#-----------------------------------------------------------------------------
@wp.func
def compute_element_fused_multi_angle(elem_idx : WARPINDEXDTYPE, angle_idx : WARPINDEXDTYPE, sd : SweepDataMultiAngle, xs : wp.array[DTYPE], result : wp.array2d[NBASISDTYPE], x_apply : wp.array2d[NBASISDTYPE]):
    
    sj                 = x_apply[angle_idx, elem_idx]
    vol_jacobian_det   = sd.sweep_geometric_data.vol_jacobians_det[elem_idx]   # scalar
    vol_jacobian_inv   = sd.sweep_geometric_data.vol_jacobians_inv[elem_idx]   # mat33d
    face_jacobians_det = sd.sweep_geometric_data.face_jacobians_det[elem_idx] # vec4d
    face_normals       = sd.sweep_geometric_data.face_normals[elem_idx]       # [4] vec3d
    sigma_t            = xs[elem_idx]
    angle              = sd.angles[angle_idx]              # vec3d
    face_perms         = sd.sweep_geometric_data.face_perm_idx[elem_idx]
    face_conns         = sd.sweep_geometric_data.etof[elem_idx]
    neighbours         = sd.sweep_geometric_data.etoe[elem_idx]
    angle_ref          = vol_jacobian_inv * angle
    
    Aij = vol_jacobian_det * sigma_t * sd.sweep_geometric_data.mass_matrix[0]    
    Bij = vol_jacobian_det * ( angle_ref[0] * sd.sweep_geometric_data.derivative_matrix[0] + angle_ref[1] * sd.sweep_geometric_data.derivative_matrix[1]\
                               + angle_ref[2] * sd.sweep_geometric_data.derivative_matrix[2])         
    Cij = NMATRIXDTYPE()

    for i in range(NFACES):

        n_vec              = face_normals[i]

        omega_dot_n        = wp.dot(n_vec, angle)
        abs_omega_dot_n_Je = wp.abs(omega_dot_n) * face_jacobians_det[i]
        

        neighbour          = sd.sweep_geometric_data.etoe[elem_idx][i]

        if omega_dot_n < INFLOW_EPSILON:
            if neighbour >= 0: # inflow face with neighbour
                face_perm = face_perms[i]
                face_conn = face_conns[i]
                neighbour = neighbours[i]
                psi_conn  = result[angle_idx, neighbour]
                sj = sj + (sd.sweep_geometric_data.inflow_matrices[WARPINDEXDTYPE(i), WARPINDEXDTYPE(face_perm), WARPINDEXDTYPE(face_conn)] @ psi_conn) * abs_omega_dot_n_Je
        else:
            Cij = Cij + abs_omega_dot_n_Je * sd.sweep_geometric_data.outflow_matrices[i]

    result[angle_idx, elem_idx] = wp.inverse((Aij + Bij + Cij)) @ sj

@wp.func
def compute_element_fused_single_angle(elem_idx : WARPINDEXDTYPE, sd : SweepDataSingleAngle, xs : wp.array[DTYPE], result : wp.array[NBASISDTYPE], x_apply : wp.array[NBASISDTYPE]):
    
    sj                 = x_apply[elem_idx]
    vol_jacobian_det   = sd.sweep_geometric_data.vol_jacobians_det[elem_idx]   # scalar
    vol_jacobian_inv   = sd.sweep_geometric_data.vol_jacobians_inv[elem_idx]   # mat33d
    face_jacobians_det = sd.sweep_geometric_data.face_jacobians_det[elem_idx] # vec4d
    face_normals       = sd.sweep_geometric_data.face_normals[elem_idx]       # [4] vec3d
    sigma_t            = xs[elem_idx]    
    face_perms         = sd.sweep_geometric_data.face_perm_idx[elem_idx]
    face_conns         = sd.sweep_geometric_data.etof[elem_idx]
    neighbours         = sd.sweep_geometric_data.etoe[elem_idx]    
    angle_ref          = vol_jacobian_inv * sd.angles[0]
    
    
    Aij = vol_jacobian_det * sigma_t * sd.sweep_geometric_data.mass_matrix[0]    
    Bij = vol_jacobian_det * ( angle_ref[0] * sd.sweep_geometric_data.derivative_matrix[0] + angle_ref[1] * sd.sweep_geometric_data.derivative_matrix[1] + angle_ref[2] * sd.sweep_geometric_data.derivative_matrix[2])         
    Cij = NMATRIXDTYPE()

    for i in range(NFACES):

        n_vec              = face_normals[i]

        omega_dot_n        = wp.dot(n_vec, sd.angles[0])
        abs_omega_dot_n_Je = wp.abs(omega_dot_n) * face_jacobians_det[i]
        

        neighbour          = sd.sweep_geometric_data.etoe[elem_idx][i]

        if omega_dot_n < wp.float64(0.0):
            if neighbour >= 0: # inflow face with neighbour
                face_perm = face_perms[i]
                face_conn = face_conns[i]
                neighbour = neighbours[i]
                psi_conn  = result[neighbour]
                sj = sj + (sd.sweep_geometric_data.inflow_matrices[WARPINDEXDTYPE(i), WARPINDEXDTYPE(face_perm), WARPINDEXDTYPE(face_conn)] @ psi_conn) * abs_omega_dot_n_Je
        else:
            Cij = Cij + abs_omega_dot_n_Je * sd.sweep_geometric_data.outflow_matrices[i]

    result[elem_idx] = wp.inverse((Aij + Bij + Cij)) @ sj

@wp.func
def compute_element_direct_multi_angle(elem_in_toposort_idx : WARPINDEXDTYPE, angle_idx : WARPINDEXDTYPE, sd : SweepDataMultiAngle, xs : wp.array[DTYPE], result : wp.array2d[NBASISDTYPE], x_apply : wp.array2d[NBASISDTYPE]):
    '''
    Directly calls the multi-angle kernel using the topological sort data
    '''
    elem_idx = WARPINDEXDTYPE(sd.levels_argsort[angle_idx, elem_in_toposort_idx])
    compute_element_fused_multi_angle(elem_idx, angle_idx, sd, xs, result, x_apply)

@wp.func
def compute_element_direct_single_angle(elem_in_toposort_idx : WARPINDEXDTYPE, sd : SweepDataSingleAngle, xs : wp.array[DTYPE], result : wp.array[NBASISDTYPE], x_apply : wp.array[NBASISDTYPE]):
    '''
    Directly calls the single-angle kernel using the topological sort data
    '''
    elem_idx = WARPINDEXDTYPE(sd.levels_argsort[elem_in_toposort_idx])
    compute_element_fused_single_angle(elem_idx, sd, xs, result, x_apply)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Sequential Sweep Kernels
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@wp.kernel
def sequential_sweep_single_angle(sd : SweepDataSingleAngle, xs : wp.array[DTYPE], result : wp.array[NBASISDTYPE], x_apply : wp.array[NBASISDTYPE]):
    for elem_in_order in range(sd.sweep_geometric_data.face_normals.shape[0]):        
        compute_element_direct_single_angle(WARPINDEXDTYPE(elem_in_order), sd, xs, result, x_apply)

    
@wp.kernel 
def sequential_sweep_multi_angle(sd : SweepDataMultiAngle, xs : wp.array[DTYPE], result : wp.array2d[NBASISDTYPE], x_apply : wp.array2d[NBASISDTYPE]):
    angle_idx = wp.tid()
    for elem_in_order in range(sd.sweep_geometric_data.face_normals.shape[0]):        
        compute_element_direct_multi_angle(WARPINDEXDTYPE(elem_in_order), angle_idx, sd, xs, result, x_apply)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Level-sweep kernels (multi-angle and single-angle variants)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@wp.kernel
def _sweep_level_kernel(topo_offset : WARPINDEXDTYPE, angle_idx : WARPINDEXDTYPE,  sd : SweepDataMultiAngle, xs : wp.array[DTYPE], result : wp.array2d[NBASISDTYPE], x_apply : wp.array2d[NBASISDTYPE]):
    in_level_id = wp.tid()
    compute_element_direct_multi_angle(in_level_id + topo_offset, angle_idx, sd, xs, result, x_apply)

@wp.kernel
def _sweep_level_kernel_ma(topo_offset_ids : wp.array2d[WARPINDEXDTYPE], padded_level_size : wp.array2d[WARPINDEXDTYPE], current_level : WARPINDEXDTYPE,   sd : SweepDataMultiAngle, xs : wp.array[DTYPE], result : wp.array2d[NBASISDTYPE], x_apply : wp.array2d[NBASISDTYPE]):
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
import equinox as eqx
from .transport_operator_base import TransportOperatorMultipleAngles, TransportOperator, TransportOperatorMultipleAnglesVmap


    # def single_angle(self, angle : int, xs_wp : wp.array, x_wp: wp.array, level_bounds) -> None:
    #     """Run the sweep in-place on pre-allocated Warp arrays (no host copies)."""
    #     x_apply_bc = x_wp[None, ...]
    #     x_apply_wp = wp.from_jax(x_apply_bc, dtype=NBASISDTYPE)
    #     result = wp.zeros_like(x_apply_wp)
    #     for start, end in level_bounds:
    #         wp.launch(kernel = _sweep_level_kernel,
    #             dim=end - start,
    #             inputs=[
    #                 start,
    #                 wp.int32(angle),
    #                 self._data,
    #                 xs_wp,
    #                 result, 
    #                 x_apply_wp,
    #                 ],
    #             device=self.device
    #         )

    #     wp.synchronize()
    #     return result

    # def multi_angle_levels(self, xs : jnp.ndarray, x_apply : jnp.ndarray, levelbased_setup : LevelBasedParallelismSetup):
        
    #     """Run the multi-angle level-sweep in-place on pre-allocated Warp arrays (no host copies)."""
    #     xs_wp      = wp.from_jax(xs, dtype=DTYPE)
    #     x_apply_wp = wp.from_jax(x_apply, dtype=NBASISDTYPE)
    #     result     =   wp.zeros_like(x_apply_wp)
    #     n_angles   = self._data.angles.shape[0]

    #     padded_occ_wp = wp.from_numpy(np.array(levelbased_setup.padded_first_occ, dtype=np.int32), dtype=wp.int32)

    #     padded_level_size = wp.from_numpy(np.array(levelbased_setup.padded_sizes, dtype=np.int32), dtype=wp.int32)


    #     for levels in range(levelbased_setup.n_levels):
    #             n_work = levelbased_setup.max_level_size(levels)                
    #             wp.launch(kernel = _sweep_level_kernel_ma,
    #                 dim=(n_angles, n_work),
    #                 inputs=[
    #                     padded_occ_wp,
    #                     padded_level_size,
    #                     wp.int32(levels), 
    #                     self._data,
    #                     xs_wp,
    #                     result,
    #                     x_apply_wp
    #                 ]
    #             )

    #     wp.synchronize()
    #     return result
    
    # def multi_angle_levels_prealloc(self, xs : jnp.ndarray, x_apply : jnp.ndarray, levelbased_setup : LevelBasedParallelismSetup, result_array : jnp.ndarray):
        
    #     """Run the multi-angle level-sweep in-place on pre-allocated Warp arrays (no host copies)."""
    #     xs_wp      = wp.from_jax(xs, dtype=DTYPE)
    #     x_apply_wp = wp.from_jax(x_apply, dtype=NBASISDTYPE)
    #     result     = wp.from_jax(result_array, dtype=NBASISDTYPE)
    #     n_angles   = self._data.angles.shape[0]
    #     padded_occ_wp = wp.from_numpy(np.array(levelbased_setup.padded_first_occ, dtype=np.int32), dtype=wp.int32)
        
    #     padded_level_size = wp.from_numpy(np.array(levelbased_setup.padded_sizes, dtype=np.int32), dtype=wp.int32)

    #     for levels in range(levelbased_setup.n_levels):
    #             n_work = levelbased_setup.max_level_size(levels)                
    #             wp.launch(kernel = _sweep_level_kernel_ma,
    #                 dim=(n_angles, n_work),
    #                 inputs=[
    #                     padded_occ_wp,
    #                     padded_level_size,
    #                     wp.int32(levels), 
    #                     self._data,
    #                     xs_wp,
    #                     result,
    #                     x_apply_wp
    #                 ]
    #             )

    #     wp.synchronize()
    #     return result
    
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

# =============================================================================
# Utilities to map from transport operator to flattened array
# =============================================================================
from warp.jax_experimental import jax_callable, GraphMode
from typing import Union
from jax_sn.domain import GeometricDomain, CrossSectionDataSingleGroup, TopologicalSortedAngle
from jax_sn.solution_domain import PrecomputedGeometricData, FEMMatrices

def _flatten_base_args(geometric_domain           : GeometricDomain, 
                       fem_matrices               : FEMMatrices, 
                       precomputed_geometric_data : PrecomputedGeometricData,                                      
                       cross_sections             : CrossSectionDataSingleGroup, 
                       topological_sorted_angle   : TopologicalSortedAngle
                       ):
    """Build warp arrays from a transport operator in the same order as the sweep requires them

    All arrays are zero-copy views of the underlying JAX buffers.    
    """
    return (
        geometric_domain.face_normals,
        precomputed_geometric_data.face_jacobians_det,
        precomputed_geometric_data.vol_jacobians_det,
        precomputed_geometric_data.vol_jacobians_inv,
        geometric_domain.etoe,
        geometric_domain.etof,
        geometric_domain.face_perm_idx,
        jnp.broadcast_to(fem_matrices.mass_matrix[None, ...], (1, NBASIS, NBASIS)),
        fem_matrices.derivative_matrix,
        fem_matrices.outflow_matrices,
        fem_matrices.inflow_matrices,
        jnp.atleast_2d(topological_sorted_angle.angle) , 
        topological_sorted_angle.levels_argsort,
        cross_sections.total
    )

def _flatten_transport_operator(transport_operator : Union[TransportOperatorMultipleAngles, TransportOperator]):
    """Build warp arrays from a transport operator in the same order as the sweep requires them

    All arrays are zero-copy views of the underlying JAX buffers.    
    """
    return _flatten_base_args(*transport_operator._base_inv_args(None)[:-1])

# =============================================================================
# Sequential warp launches
# =============================================================================
def _sequential_single_angle_sweep(    
    face_normals:       wp.array2d[SPATVECDTYPE],
    face_jacobians_det: wp.array2d[DTYPE],
    vol_jacobians_det:  wp.array[DTYPE],
    vol_jacobians_inv:  wp.array[SPATMATDTYPE],
    etoe:               wp.array[NFACESINTTYPE],
    etof:               wp.array[NFACESINTTYPE],
    face_perm_idx:      wp.array[NFACESINTTYPE],
    mass_matrix:        wp.array[NMATRIXDTYPE],
    derivative_matrix:  wp.array[NMATRIXDTYPE],
    outflow_matrices:   wp.array[NMATRIXDTYPE],
    inflow_matrices:    wp.array3d[NMATRIXDTYPE],
    angle:              wp.array[SPATVECDTYPE],
    levels_argsort:     wp.array[JAXINDEXDTYPE],
    xs:                 wp.array[DTYPE],
    x_apply:            wp.array[NBASISDTYPE],
    result:             wp.array[NBASISDTYPE],
) -> None:
    sd = SweepGeometricData()
    sd.face_normals       = face_normals
    sd.face_jacobians_det = face_jacobians_det
    sd.vol_jacobians_det  = vol_jacobians_det
    sd.vol_jacobians_inv  = vol_jacobians_inv
    sd.etoe               = etoe
    sd.etof               = etof
    sd.face_perm_idx      = face_perm_idx
    sd.mass_matrix        = mass_matrix
    sd.derivative_matrix  = derivative_matrix
    sd.outflow_matrices   = outflow_matrices
    sd.inflow_matrices    = inflow_matrices
    sd_tot = SweepDataSingleAngle()
    sd_tot.sweep_geometric_data = sd
    sd_tot.angles               = angle    
    sd_tot.levels_argsort       = levels_argsort

    wp.launch(
        sequential_sweep_single_angle,
        dim=(1,),
        inputs=[sd_tot, xs, result, x_apply],
    )
sweep_sa_jax = jax_callable(_sequential_single_angle_sweep)

def _sequential_multi_angle_sweep(    
    face_normals:       wp.array2d[SPATVECDTYPE],
    face_jacobians_det: wp.array2d[DTYPE],
    vol_jacobians_det:  wp.array[DTYPE],
    vol_jacobians_inv:  wp.array[SPATMATDTYPE],
    etoe:               wp.array[NFACESINTTYPE],
    etof:               wp.array[NFACESINTTYPE],
    face_perm_idx:      wp.array[NFACESINTTYPE],
    mass_matrix:        wp.array[NMATRIXDTYPE],
    derivative_matrix:  wp.array[NMATRIXDTYPE],
    outflow_matrices:   wp.array[NMATRIXDTYPE],
    inflow_matrices:    wp.array3d[NMATRIXDTYPE],
    angles:             wp.array[SPATVECDTYPE],
    levels_argsort:     wp.array2d[JAXINDEXDTYPE],
    xs:                 wp.array[DTYPE],
    x_apply:            wp.array2d[NBASISDTYPE],
    result:             wp.array2d[NBASISDTYPE],
) -> None:
    sd = SweepGeometricData()
    sd.face_normals       = face_normals
    sd.face_jacobians_det = face_jacobians_det
    sd.vol_jacobians_det  = vol_jacobians_det
    sd.vol_jacobians_inv  = vol_jacobians_inv
    sd.etoe               = etoe
    sd.etof               = etof
    sd.face_perm_idx      = face_perm_idx
    sd.mass_matrix        = mass_matrix
    sd.derivative_matrix  = derivative_matrix
    sd.outflow_matrices   = outflow_matrices
    sd.inflow_matrices    = inflow_matrices

    sd_tot = SweepDataMultiAngle()
    sd_tot.sweep_geometric_data = sd
    sd_tot.angles               = angles    
    sd_tot.levels_argsort       = levels_argsort

    n_angles = angles.shape[0]
    
    wp.launch(
        sequential_sweep_multi_angle,
        dim=(n_angles,),
        inputs=[sd_tot, xs, result, x_apply],
    )

sweep_ma_jax = jax_callable(_sequential_multi_angle_sweep)

@eqx.filter_jit
def _sweep_sequential_ma_with_warp(top : TransportOperatorMultipleAngles, x_arr : jnp.ndarray):    
    assert x_arr.shape == top.out_structure().shape, f"Expected x_arr shape {top.out_structure().shape}, got {x_arr.shape}"
    
    args_flat_ta = _flatten_transport_operator(top)        
    
    return sweep_ma_jax(*args_flat_ta, x_arr, output_dims = {"result" :  (x_arr.shape)})[0]



@eqx.filter_jit
def single_angle_warp_sweep(                   geometric_domain                : GeometricDomain, 
                                               fem_matrices               : FEMMatrices, 
                                               precomputed_geometric_data : PrecomputedGeometricData, 
                                               cross_sections             : CrossSectionDataSingleGroup,                                                
                                               topological_sorted_angle   : TopologicalSortedAngle, 
                                               x                          : jnp.ndarray
                                               ) -> jnp.ndarray:    

    args_flat_ta = _flatten_base_args(geometric_domain, fem_matrices, precomputed_geometric_data, cross_sections, topological_sorted_angle)

    return sweep_sa_jax(*args_flat_ta, x, output_dims = {"result" :  x.shape})[0]


@eqx.filter_jit
def multi_angle_warp_sweep(                    geometric_domain           : GeometricDomain, 
                                               fem_matrices               : FEMMatrices, 
                                               precomputed_geometric_data : PrecomputedGeometricData, 
                                               cross_sections             : CrossSectionDataSingleGroup,                                                
                                               topological_sorted_angle   : TopologicalSortedAngle, 
                                               x                          : jnp.ndarray
                                               ) -> jnp.ndarray:    

    args_flat_ta = _flatten_base_args(geometric_domain, fem_matrices, precomputed_geometric_data, cross_sections, topological_sorted_angle)
    return sweep_ma_jax(*args_flat_ta, x, output_dims = {"result" :  x.shape})[0]
    


from .transport_operator_base import forward_transport_operator_vmap, vmap_multiple_angles_forward, TransportOperatorBatched

class TransportOperatorSequentialWarpSingleAngle(TransportOperatorBatched):
    '''
    Transport operator for a single angle using a sequential sweep in Warp.

    Forward mode is using batched forward as in :class:`TransportOperatorBatched`, while inverse mode is using the single-angle sequential sweep kernel in Warp.
    The inverse does not use any batching.
    '''

    def __check_init__(self):
        assert self.solution_domain_data.precomputed_geometric_data.vol_jacobians_inv.shape[-2:] == (3,3), f"For Warp kernel, expected vol_jacobians_inv shape (3,3), got {self.solution_domain_data.precomputed_geometric_data.vol_jacobians_inv.shape}"
        assert self.solution_domain_data.n_basis == NBASIS, f"For warp kernel, expected {NBASIS} basis functions, got {self.solution_domatin_data.n_basis}"
        assert self.solution_domain_data.domain.etoe.shape[-1] == NFACES, f"For warp kernel, expected {NFACES} faces per elements, got {self.solution_domain_data.domain.etoe.shape[-1]}"
    

    @property 
    def _inv_op_args(self):
        return single_angle_warp_sweep
    
    def _inv_args(self, b):
        return self._base_inv_args(b)
    
    @property
    def _vmap_for_inv_multiple_angles(self):
        return (None, None, None, None, 0,0)

class TransportOperatorSequentialWarpMultipleAngles(TransportOperatorMultipleAngles):
    '''
    Transport operator for multiple angles using vmap over angles.
    '''
    def __check_init__(self):
        assert isinstance(self.transport_operator, TransportOperatorSequentialWarpSingleAngle), f"Expected transport_operator to be a TransportOperatorSequentialWarpSingleAngle, got {type(self.transport_operator)}"

    def mv(self, x : jnp.ndarray) -> jnp.ndarray:
        return vmap_multiple_angles_forward(self.transport_operator, x)
    
    def _invert_internal(self, b : jnp.ndarray) -> jnp.ndarray:
        return multi_angle_warp_sweep(*self._base_inv_args(b))
    


        



#     def _assert_shape_base(self):
#         assert self.topological_sorted_angle.ndim==1, "topological_sorted_angle must be 1D for base TransportOperator"
    
#     def mv(self, x : jnp.ndarray) -> jnp.ndarray:
#         return self._mv_op_args(*self._forward_args(x))
    
#     def _invert_internal(self, b : jnp.ndarray) -> jnp.ndarray:
#         return self._inv_op_args(*self._inv_args(b))
        
#     @property
#     @abstractmethod
#     def _mv_op_args(self):
#         ...

#     @property 
#     @abstractmethod
#     def _inv_op_args(self):
#         ...
    

#     def transpose(self):
#         return eqx.tree_at(lambda t_operator : t_operator.topological_sorted_angle, self, self.topological_sorted_angle.reverse)
    
#     def inv(self, b : jnp.ndarray) -> jnp.ndarray:
#         r'''
#         Solves the system :math:`\mathbf{T} x = b` for `x` using the internal sweep inversion.

#         Parameters
#         ----------
#         b : jnp.ndarray
#             Right-hand side array.
#         Returns
#         -------
#         jnp.ndarray
#             Solution array `x`.
#         '''
#         solver = TransportOperatorSolver()
#         return lx.linear_solve(self, b, solver, throw=False).value
  
#     def in_structure(self):        
#         return self._shape_inout
    
#     def out_structure(self):
#         return self._shape_inout
    
#     @property 
#     def _shape_inout(self):
#         n_elements = self.solution_domain_data.domain.n_elements
#         n_basis = self.solution_domain_data.fem_matrices.n_basis        
#         return jax.ShapeDtypeStruct((n_elements, n_basis), self.solution_domain_data.fem_matrices.mass_matrix.dtype)
    
#     def _base_forward_args(self, x):
#         return (
#             *self.solution_domain_data,
#             self.solution_domain_data.domain.cross_sections_indexed.map_to_full_array(self.energy_group),
#             self.topological_sorted_angle.angle,
#             x
#         )
#     def _base_inv_args(self, x):
#         return (
#             *self.solution_domain_data,
#             self.solution_domain_data.domain.cross_sections_indexed.map_to_full_array(self.energy_group),
#             self.topological_sorted_angle,
#             x
#         )    
    
#     def _forward_args(self, x):
#         return self._base_forward_args(x)
    
#     def _inv_args(self, b):
#         return self._base_inv_args(b)
    
#     @property
#     def _vmap_for_forward_multiple_angles(self):
#         return (None, None, None, None, 0,0)
    
#     @property
#     def _vmap_for_inv_multiple_angles(self):
#         return (None, None, None, None, 0,0)
    
#     @property 
#     def n_angles(self):
#         '''
#         Number of angles in the transport operator. Here is just one.
#         '''
#         return 1
# class TransportOperatorSequentialWarp(TransportOperator):


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


