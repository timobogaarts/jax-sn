from __future__ import annotations

import numpy as onp
import warp as wp
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
thus be chosen to be on the order of floating point errors.
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

        if omega_dot_n < INFLOW_EPSILON:
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

@wp.kernel(enable_backward=False)
def sequential_sweep_single_angle(sd : SweepDataSingleAngle, xs : wp.array[DTYPE], result : wp.array[NBASISDTYPE], x_apply : wp.array[NBASISDTYPE]):
    for elem_in_order in range(sd.sweep_geometric_data.face_normals.shape[0]):        
        compute_element_direct_single_angle(WARPINDEXDTYPE(elem_in_order), sd, xs, result, x_apply)

    
@wp.kernel(enable_backward=False)
def sequential_sweep_multi_angle(sd : SweepDataMultiAngle, xs : wp.array[DTYPE], result : wp.array2d[NBASISDTYPE], x_apply : wp.array2d[NBASISDTYPE]):
    angle_idx = wp.tid()
    for elem_in_order in range(sd.sweep_geometric_data.face_normals.shape[0]):        
        compute_element_direct_multi_angle(WARPINDEXDTYPE(elem_in_order), angle_idx, sd, xs, result, x_apply)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Level-sweep kernels (multi-angle and single-angle variants)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@wp.kernel(enable_backward=False)
def _sweep_level_kernel(topo_offset : WARPINDEXDTYPE, sd : SweepDataSingleAngle, padded_level_size : wp.array[JAXINDEXDTYPE], current_level : WARPINDEXDTYPE, xs : wp.array[DTYPE], result : wp.array[NBASISDTYPE], x_apply : wp.array[NBASISDTYPE]):
    elem_in_level_id = wp.tid()

    if elem_in_level_id >= padded_level_size[current_level]:
        return # out of bounds for this level: this does not happen when the kernel is launched with the exact level size.
               # It will not even happend in the below jax function since it is skipped conditionally in python if level size is <=0.
    
    compute_element_direct_single_angle(elem_in_level_id + topo_offset, sd, xs, result, x_apply)

@wp.kernel(enable_backward=False)
def _sweep_level_kernel_ma(topo_offset_ids : wp.array2d[JAXINDEXDTYPE], padded_level_size : wp.array2d[JAXINDEXDTYPE], current_level : WARPINDEXDTYPE,   sd : SweepDataMultiAngle, xs : wp.array[DTYPE], result : wp.array2d[NBASISDTYPE], x_apply : wp.array2d[NBASISDTYPE]):
    angle_id, elem_in_level_id = wp.tid()
    offset_this_angle = WARPINDEXDTYPE(topo_offset_ids[angle_id, current_level])
    if offset_this_angle < 0:
        return # no work for this angle
    else:
        if elem_in_level_id >= WARPINDEXDTYPE(padded_level_size[angle_id, current_level]):
            return # out of bounds for this angle
        compute_element_direct_multi_angle(elem_in_level_id + offset_this_angle, angle_id, sd, xs, result, x_apply)

# =============================================================================
# Public Python wrapper
# =============================================================================
import equinox as eqx


# =============================================================================
# Utilities to map from transport operator to flattened array
# =============================================================================
from warp.jax_experimental import jax_callable, GraphMode
from jax_sn.domain import GeometricDomain, CrossSectionDataSingleGroup
from ..sweep_data_implementations import BasicSweepData
from jax_sn.solution_domain import PrecomputedGeometricData, FEMMatrices

def _flatten_base_args(geometric_domain           : GeometricDomain,
                       fem_matrices               : FEMMatrices,
                       precomputed_geometric_data : PrecomputedGeometricData,
                       cross_sections             : CrossSectionDataSingleGroup,
                       angle                      : jnp.ndarray,
                       sweep_data   : BasicSweepData
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
        jnp.atleast_2d(angle) ,
        sweep_data.levels_argsort,
        cross_sections.total
    )

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
def single_angle_warp_sweep(                   geometric_domain           : GeometricDomain,
                                               fem_matrices               : FEMMatrices,
                                               precomputed_geometric_data : PrecomputedGeometricData,
                                               cross_sections             : CrossSectionDataSingleGroup,
                                               angle                      : jnp.ndarray,
                                               sweep_data   : BasicSweepData,
                                               x                          : jnp.ndarray
                                               ) -> jnp.ndarray:

    args_flat_ta = _flatten_base_args(geometric_domain, fem_matrices, precomputed_geometric_data, cross_sections, angle, sweep_data)
    return sweep_sa_jax(*args_flat_ta, x, output_dims = {"result" :  x.shape})[0]


@eqx.filter_jit
def multi_angle_warp_sweep(                    geometric_domain           : GeometricDomain,
                                               fem_matrices               : FEMMatrices,
                                               precomputed_geometric_data : PrecomputedGeometricData,
                                               cross_sections             : CrossSectionDataSingleGroup,
                                               angle                      : jnp.ndarray,
                                               sweep_data   : BasicSweepData,
                                               x                          : jnp.ndarray
                                               ) -> jnp.ndarray:

    args_flat_ta = _flatten_base_args(geometric_domain, fem_matrices, precomputed_geometric_data, cross_sections, angle, sweep_data)
    return sweep_ma_jax(*args_flat_ta, x, output_dims = {"result" :  x.shape})[0]

#==============================================================================
# Levels based
#==============================================================================

def _level_single_angle_sweep(    
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
    levels_first_occ :  wp.array[JAXINDEXDTYPE],
    levels_ends      :  wp.array[JAXINDEXDTYPE],
    levels_sizes     :  wp.array[JAXINDEXDTYPE],
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

    wp.synchronize()
    
    level_start_np = levels_first_occ.numpy()    
    level_sizes_np    = levels_sizes.numpy()

    for level in range(level_start_np.shape[0]):
        level_start = level_start_np[level]        
        level_size   = level_sizes_np[level]
        if level_size <= 0:
            break # no work for this level -> no work for next levels either.

        wp.launch(kernel = _sweep_level_kernel,
            dim=(level_size,),
            inputs=[
                level_start,                
                sd_tot,
                levels_sizes,
                wp.int32(level),
                xs,
                result, 
                x_apply,
                ]    
        )    
    wp.synchronize()
    
level_sweep_sa_jax = jax_callable(_level_single_angle_sweep, graph_mode=GraphMode.NONE)


from ..sweep_data_implementations import LevelBasedParallelism

@eqx.filter_jit
def level_single_angle_warp_sweep(             geometric_domain           : GeometricDomain,
                                               fem_matrices               : FEMMatrices,
                                               precomputed_geometric_data : PrecomputedGeometricData,
                                               cross_sections             : CrossSectionDataSingleGroup,
                                               angle                      : jnp.ndarray,
                                               sweep_data   : BasicSweepData,
                                               x                          : jnp.ndarray,
                                               levelsetup                 : LevelBasedParallelism
                                               ) -> jnp.ndarray:

    args_flat_ta = _flatten_base_args(geometric_domain, fem_matrices, precomputed_geometric_data, cross_sections, angle, sweep_data)
    return level_sweep_sa_jax(*args_flat_ta, levelsetup.first_occurences, levelsetup.ends, levelsetup.sizes, x, output_dims = {"result" :  x.shape})[0]



def _level_multi_angles_sweep(    
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
    levels_first_occ :  wp.array2d[JAXINDEXDTYPE],
    levels_ends      :  wp.array2d[JAXINDEXDTYPE],
    levels_sizes     :  wp.array2d[JAXINDEXDTYPE],
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

    result.zero_()
    

    wp.synchronize()
    
    level_start_np = levels_first_occ.numpy()    
    level_sizes_np    = levels_sizes.numpy()
    
    n_work_per_level = onp.max(level_sizes_np, axis=0)
    
    n_angles = angles.shape[0]
    n_levels = level_start_np.shape[-1]    

    for level in range(n_levels):

        n_work = n_work_per_level[level] # Total work i.e. maximum over angles

        if n_work <= 0:
            break # no work for this level -> no work for next levels either.

        wp.launch(kernel = _sweep_level_kernel_ma,
            dim=(n_angles, n_work),
            inputs=[
                levels_first_occ,                
                levels_sizes,
                WARPINDEXDTYPE(level),
                sd_tot,                
                xs,
                result, 
                x_apply,
                ]    
        )    
    wp.synchronize()

levels_sweep_ma_jax = jax_callable(_level_multi_angles_sweep, graph_mode=GraphMode.NONE)

@eqx.filter_jit
def level_multi_angle_warp_sweep(              geometric_domain           : GeometricDomain,
                                               fem_matrices               : FEMMatrices,
                                               precomputed_geometric_data : PrecomputedGeometricData,
                                               cross_sections             : CrossSectionDataSingleGroup,
                                               angle                      : jnp.ndarray,
                                               sweep_data   : BasicSweepData,
                                               x                          : jnp.ndarray,
                                               levelsetup                 : LevelBasedParallelism
                                               ) -> jnp.ndarray:

    args_flat_ta = _flatten_base_args(geometric_domain, fem_matrices, precomputed_geometric_data, cross_sections, angle, sweep_data)
    return levels_sweep_ma_jax(*args_flat_ta, levelsetup.first_occurences, levelsetup.ends, levelsetup.sizes, x, output_dims = {"result" :  x.shape})[0]
