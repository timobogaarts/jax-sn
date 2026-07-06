import jax
import jax.numpy as jnp
from jax_sn.domain import GeometricDomain, CrossSectionDataSingleGroup
from ..sweep_data import BasicSweepData
from jax_sn.solution_domain import PrecomputedGeometricData, FEMMatrices
from functools import partial
from typing import Tuple
from .transport_operator_jax import _setup_elements_idx_in_precomp_with_omega_n_calculation, _assert_shape

from jax.experimental import pallas as pl

from jax.experimental.pallas import triton as pltriton

#======================================================================================================================================
#  Pallas versions of batched and complete vmap sweeps
#======================================================================================================================================
# This file contains Pallas versions of the batched and complete vmap sweeps. 
# The Pallas versions use a custom kernel to perform the sweep update, which in my testing is faster than the fori loops using plain JAX.
# However, this is not the case in all situations, especially for higher order elements, which have so much work per element that it 
# seems to benefit from the matrix multiplications in plain JAX.

# Furthermore, due to Triton limitations, the Pallas versions require padding of arrays to the next power-of-two sizes.
# This padding can add significant overhead, especially as the number of basis functions in simplices does not map nicely to power of two.
# Therefore, the Pallas versions are best suited for low-order elements with a number of basis functions close to a power of two.
# 
# One issue is that due to the padding, we have to allocate large temporary arrays for each sweep. This is not re-used in the output, as 
# we need to copy and allocate a new, not padded array, to return the result.
# If it is exactly a power of two, this is not an issue, but for other sizes this can lead to significant memory overhead.

# Some timings (RTX3080) [auto generated from examples/time_pallas.py]:
# Timings for a degree 0 tetrahedron domain with 60000 elements and batch size 1000
#  Setup elements:                                       3.12 ms
#  Setup elements padded to power of two:                3.31 ms
#  transport_inverse_sweep_complete_vmap_fori:           1780.72 ms
#  transport_inverse_sweep_batched:                      1812.21 ms
#  transport_inverse_sweep_complete_vmap_pallas:         46.06 ms
#  transport_inverse_sweep_batched_pallas:               48.45 ms

# Timings for a degree 1 tetrahedron domain with 60000 elements and batch size 1000
#  Setup elements:                                       1.60 ms
#  Setup elements padded to power of two:                1.60 ms
#  transport_inverse_sweep_complete_vmap_fori:           571.63 ms
#  transport_inverse_sweep_batched:                      570.89 ms
#  transport_inverse_sweep_complete_vmap_pallas:         84.33 ms
#  transport_inverse_sweep_batched_pallas:               76.13 ms

# Timings for a degree 2 tetrahedron domain with 60000 elements and batch size 1000
#  Setup elements:                                       6.00 ms
#  Setup elements padded to power of two:                7.07 ms
#  transport_inverse_sweep_complete_vmap_fori:           583.50 ms
#  transport_inverse_sweep_batched:                      587.54 ms
#  transport_inverse_sweep_complete_vmap_pallas:         149.16 ms
#  transport_inverse_sweep_batched_pallas:               156.60 ms

# Timings for a degree 3 tetrahedron domain with 60000 elements and batch size 1000
#  Setup elements:                                       30.39 ms
#  Setup elements padded to power of two:                32.71 ms
#  transport_inverse_sweep_complete_vmap_fori:           704.43 ms
#  transport_inverse_sweep_batched:                      706.89 ms
#  transport_inverse_sweep_complete_vmap_pallas:         490.57 ms
#  transport_inverse_sweep_batched_pallas:               501.36 ms
# Timings for a degree 1 tetrahedron domain with 60000 elements and batch size 100 and 72 angles
#  transport_inverse_sweep_multiple_angles_batched:           1478.25 ms
#  transport_inverse_sweep_multiple_angles_batched_pallas:    179.32 ms

# Timings for a degree 2 tetrahedron domain with 60000 elements and batch size 100 and 72 angles
#  transport_inverse_sweep_multiple_angles_batched:           1883.58 ms
#  transport_inverse_sweep_multiple_angles_batched_pallas:    755.03 ms

# Timings for a degree 3 tetrahedron domain with 60000 elements and batch size 100 and 72 angles
#  transport_inverse_sweep_multiple_angles_batched:           4209.55 ms
#  transport_inverse_sweep_multiple_angles_batched_pallas:    3119.95 ms

# Timings for a degree 0 triangle domain with 2000 elements and batch size 1000
#  Setup elements:                                       0.36 ms
#  Setup elements padded to power of two:                0.47 ms
#  transport_inverse_sweep_complete_vmap:           82.23 ms
#  transport_inverse_sweep_batched:                      63.25 ms
#  transport_inverse_sweep_complete_vmap_pallas:         1.80 ms
#  transport_inverse_sweep_batched_pallas:               1.76 ms

# Timings for a degree 1 triangle domain with 2000 elements and batch size 1000
#  Setup elements:                                       0.36 ms
#  Setup elements padded to power of two:                1.62 ms
#  transport_inverse_sweep_complete_vmap:           72.29 ms
#  transport_inverse_sweep_batched:                      62.01 ms
#  transport_inverse_sweep_complete_vmap_pallas:         3.00 ms
#  transport_inverse_sweep_batched_pallas:               2.88 ms

# Timings for a degree 2 triangle domain with 2000 elements and batch size 1000
#  Setup elements:                                       1.60 ms
#  Setup elements padded to power of two:                1.61 ms
#  transport_inverse_sweep_complete_vmap:           82.82 ms
#  transport_inverse_sweep_batched:                      65.73 ms
#  transport_inverse_sweep_complete_vmap_pallas:         4.88 ms
#  transport_inverse_sweep_batched_pallas:               4.05 ms

# Timings for a degree 3 triangle domain with 2000 elements and batch size 1000
#  Setup elements:                                       2.59 ms
#  Setup elements padded to power of two:                4.47 ms
#  transport_inverse_sweep_complete_vmap:           52.49 ms
#  transport_inverse_sweep_batched:                      22.57 ms
#  transport_inverse_sweep_complete_vmap_pallas:         5.96 ms
#  transport_inverse_sweep_batched_pallas:               6.12 ms

# Timings for a degree 1 triangle domain with 2000 elements and batch size 100 and 72 angles
#  transport_inverse_sweep_multiple_angles_batched:           59.29 ms
#  transport_inverse_sweep_multiple_angles_batched_pallas:    7.28 ms

# Timings for a degree 2 triangle domain with 2000 elements and batch size 100 and 72 angles
#  transport_inverse_sweep_multiple_angles_batched:           66.84 ms
#  transport_inverse_sweep_multiple_angles_batched_pallas:    10.30 ms

# Timings for a degree 3 triangle domain with 2000 elements and batch size 100 and 72 angles
#  transport_inverse_sweep_multiple_angles_batched:           79.16 ms
#  transport_inverse_sweep_multiple_angles_batched_pallas:    26.56 ms

# Given these results, it seems there is plenty of performance to be gained: the setup time is extremely low compared to the sweep time.
# TODO Write better kernels
#======================================================================================================================================



@partial(jax.jit, static_argnums=(7,))
def transport_inverse_sweep_batched_pallas(geometric_domain    : GeometricDomain,
                                    fem_matrices               : FEMMatrices,
                                    precomputed_geometric_data : PrecomputedGeometricData,
                                    cross_sections             : CrossSectionDataSingleGroup,
                                    angle                      : jnp.ndarray,
                                    sweep_data   : BasicSweepData,
                                    x                          : jnp.ndarray,
                                    batch_size                 : int):
    '''
    Perform a complete transport inverse sweep over all elements in the geometric domain using batched vmap and fori_loop in Pallas.

    See transport_inverse_sweep_batched for more details. This is essentially the same functions but with padding to power-of-two sizes and using a Pallas kernel for the sweep update.

    Parameters  
    ----------
    geometric_domain           : GeometricDomain
        The geometric domain containing the mesh and element connectivity.
    fem_matrices               : FEMMatrices
        The finite element method matrices for the element type used in the domain.
    precomputed_geometric_data : PrecomputedGeometricData
        Precomputed geometric data for the domain and element type.
    cross_sections             : CrossSectionDataSingleGroup
        The cross section data for the domain.
    sweep_data   : BasicSweepData
        The topologically sorted angle for the sweep.
    x                          : jnp.ndarray [n_elements, n_basis]
        The array to apply the sweep to.
    batch_size                 : int
        The batch size for the batched sweep.
    Returns
    -------
    jnp.ndarray 
        The updated sweep result after performing the transport inverse sweep. Shape (n_elements, n_basis).
    '''
    _assert_shape(geometric_domain, fem_matrices, precomputed_geometric_data, cross_sections, angle, sweep_data, x, False)

    n_batches                  = jnp.ceil(geometric_domain.n_elements / batch_size).astype(int)

    etoe_pad                   = _pad_etoe_pow2(geometric_domain.etoe)

    def single_batch(batch_i, sweep_result):
        batch_indices                               = jnp.clip(jnp.arange(batch_size) + batch_i * batch_size, 0, geometric_domain.n_elements - 1)
        batch_elements                              = sweep_data.levels_argsort[batch_indices]
        gij_inv_pad, dij_f_p_fm_pad, abs_omega_dot_n_Je, x_i_pad = _setup_elements_idx_in_precomp_with_omega_n_calculation_padded(
                                                                        batch_elements,
                                                                        geometric_domain,
                                                                        fem_matrices,
                                                                        precomputed_geometric_data,
                                                                        cross_sections,
                                                                        angle,
                                                                        x
                                                                        )
        return sweep_pl_update(batch_elements, gij_inv_pad, x_i_pad, dij_f_p_fm_pad, etoe_pad, abs_omega_dot_n_Je, sweep_result)    

    buffer_sweep = jnp.zeros((geometric_domain.n_elements, _next_power_of_two(fem_matrices.n_basis)), dtype=x.dtype)
    return jax.lax.fori_loop(0, n_batches, single_batch, buffer_sweep)[:, :fem_matrices.n_basis]



@jax.jit
def transport_inverse_sweep_complete_vmap_pallas(geometric_domain           : GeometricDomain,
                                                 fem_matrices               : FEMMatrices,
                                                 precomputed_geometric_data : PrecomputedGeometricData,
                                                 cross_sections             : CrossSectionDataSingleGroup,
                                                 angle                      : jnp.ndarray,
                                                 sweep_data   : BasicSweepData,
                                                 x                          : jnp.ndarray,
                                                 ):
    '''
    Perform a complete transport inverse sweep over all elements in the geometric domain using batched vmap and fori_loop in Pallas.

    See transport_inverse_sweep_complete_vmap for more details. This is essentially the same functions but with padding to power-of-two sizes and using a Pallas kernel for the sweep update.

    Parameters  
    ----------
    geometric_domain           : GeometricDomain
        The geometric domain containing the mesh and element connectivity.
    fem_matrices               : FEMMatrices
        The finite element method matrices for the element type used in the domain.
    precomputed_geometric_data : PrecomputedGeometricData
        Precomputed geometric data for the domain and element type.
    cross_sections             : CrossSectionDataSingleGroup
        The cross section data for the domain.
    sweep_data   : BasicSweepData
        The topologically sorted angle for the sweep.
    x                          : jnp.ndarray [n_elements, n_basis]
        The array to apply the sweep to.
    batch_size                 : int
        The batch size for the batched sweep.
    Returns
    -------
    jnp.ndarray 
        The updated sweep result after performing the transport inverse sweep. Shape (n_elements, n_basis).
    '''
       
    _assert_shape(geometric_domain, fem_matrices, precomputed_geometric_data, cross_sections, angle, sweep_data, x, False)

    etoe_pad = _pad_etoe_pow2(geometric_domain.etoe)
    gij_inv_pad, dij_f_p_fm_pad, abs_omega_dot_n_Je, x_i_pad = _setup_elements_idx_in_precomp_with_omega_n_calculation_padded(
        sweep_data.levels_argsort,
        geometric_domain,
        fem_matrices,
        precomputed_geometric_data,
        cross_sections,
        angle,
        x)
    buffer_sweep = jnp.zeros((geometric_domain.n_elements, _next_power_of_two(fem_matrices.n_basis)), dtype=x.dtype)  
    return sweep_pl_update(sweep_data.levels_argsort, gij_inv_pad, x_i_pad, dij_f_p_fm_pad,etoe_pad, abs_omega_dot_n_Je, buffer_sweep)[:, :fem_matrices.n_basis]    

@jax.jit
def _setup_elements_idx_in_precomp_with_omega_n_calculation_padded(elements_idx: jnp.ndarray,
                                                         geometric_domain           : GeometricDomain, 
                                                         fem_matrices               : FEMMatrices, 
                                                         precomputed_geometric_data : PrecomputedGeometricData,                                      
                                                         cross_sections             : CrossSectionDataSingleGroup, 
                                                         omega                      : jnp.ndarray,
                                                         x                          : jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:    
    '''
    Setup function to prepare all data needed for the transport sweep, padding arrays to power-of-two sizes for Pallas.
    This function wraps _setup_elements_idx_in_precomp_with_omega_n_calculation and pads the resulting arrays to power-of-two sizes.

    Parameters
    ----------
    elements_idx                : jnp.ndarray
        The indices of the elements to setup.
    geometric_domain           : GeometricDomain
        The geometric domain containing the mesh and element connectivity.
    fem_matrices               : FEMMatrices
        The finite element method matrices for the element type used in the domain.
    precomputed_geometric_data : PrecomputedGeometricData
        Precomputed geometric data for the domain and element type.
    cross_sections             : CrossSectionDataSingleGroup
        The cross section data for the domain.
    omega                      : jnp.ndarray
        The direction vector for the sweep.
    x                          : jnp.ndarray
        The x vector to apply the transport inverse sweep to. Will be padded to power-of-two size.
    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
        The padded gij_inv, qm_i, dij_f_p_fm, and abs_omega_dot_n_Je arrays needed for the transport sweep.
    '''
    power_of_two = _next_power_of_two(fem_matrices.n_basis)
    power_two_faces = _next_power_of_two(geometric_domain.n_faces_per_element)
    gij_inv_batch, dij_f_p_fm_batch, _, abs_omega_dot_n_Je = _setup_elements_idx_in_precomp_with_omega_n_calculation(elements_idx, geometric_domain, fem_matrices,
                                                        precomputed_geometric_data, cross_sections, omega)      

    gij_inv_pad    = jnp.pad(gij_inv_batch,    ((0,0),(0, power_of_two - fem_matrices.n_basis), (0, power_of_two - fem_matrices.n_basis)), mode='constant')
    
    dij_f_p_fm_pad = jnp.pad(dij_f_p_fm_batch, ((0,0),(0,power_two_faces - geometric_domain.n_faces_per_element), (0,power_of_two - fem_matrices.n_basis),(0, power_of_two - fem_matrices.n_basis)), mode='constant')  
    abs_omega_dot_n_Je_pad = jnp.pad(abs_omega_dot_n_Je, ((0,0),(0, power_two_faces - geometric_domain.n_faces_per_element)), mode='constant')

    x_i_pad       = jnp.pad(x[elements_idx],       ((0,0),(0, power_of_two - fem_matrices.n_basis)), mode='constant')
    return gij_inv_pad, dij_f_p_fm_pad, abs_omega_dot_n_Je_pad, x_i_pad

def _setup_elements_pad_args(geometric_domain          : GeometricDomain,
                            fem_matrices               : FEMMatrices,
                            precomputed_geometric_data : PrecomputedGeometricData,
                            cross_sections             : CrossSectionDataSingleGroup,
                            angle                      : jnp.ndarray,
                            sweep_data   : BasicSweepData,
                            x                          : jnp.ndarray):
    '''
    Convenience function to setup elements using the same arguments as the transport sweep functions, but returning padded arrays for Pallas.
    '''
    return _setup_elements_idx_in_precomp_with_omega_n_calculation_padded(
        sweep_data.levels_argsort,
        geometric_domain,
        fem_matrices,
        precomputed_geometric_data,
        cross_sections,
        angle,
        x)

def _pad_etoe_pow2(etoe : jnp.ndarray):
    '''
    Pad the element-to-element connectivity array to the next power-of-two number of faces for Pallas.
    '''
    power_two_faces = _next_power_of_two(etoe.shape[1])
    etoe_pad = jnp.pad(etoe, ((0,0),(0, power_two_faces - etoe.shape[1])), mode='constant', constant_values=-1)
    return etoe_pad


def _sweep_pl_update_kernel(elements_toposort, gij_inv_all_ref,  qm_i_all_ref, dij_f_p_fm_all_ref, etoe_ref, abs_omega_dot_n_Je_ref, _,  sweep_result_ref):  
  '''
    Pallas kernel to perform the sweep update over a batch of elements. 
    This kernel is called from sweep_pl_update.

    Uses the buffer sweep_result_ref as input and output, updating it in place.
    Can be part of a larger fori_loop over batches of elements.

    Parameters
    ----------
    elements_toposort        : jnp.ndarray
        The indices of the elements in topological order for the sweep.
    gij_inv_all_ref          : jnp.ndarray
        The padded inverse of the gij matrices for all elements in the batch.
    qm_i_all_ref             : jnp.ndarray
        The padded source terms for all elements in the batch.
    dij_f_p_fm_all_ref       : jnp.ndarray
        The padded dij_f_p_fm matrices for all elements in the batch.
    etoe_ref                 : jnp.ndarray
        The padded element-to-element connectivity array.
    abs_omega_dot_n_Je_ref   : jnp.ndarray
        The padded absolute values of the dot products of omega and the element face normals for all elements in the batch.
    sweep_result_ref         : jnp.ndarray
        The buffer array to store the sweep results. Updated in place.
  '''
  n_elem = sweep_result_ref.shape[0]
  n_basis = sweep_result_ref.shape[1]
  n_faces = etoe_ref.shape[1]
  def _body(i, sweep_ref):
      elem_idx             = elements_toposort[i]                      #[1]
      connected_elems      = etoe_ref[elem_idx]                        #[n_faces]      
      conn_elem_safe       = jnp.clip(connected_elems, 0, n_elem - 1)  # pallas is less forgiving with -1 it seems so we clip and select
      conn_values          = jnp.where((connected_elems >= 0)[:, None], sweep_ref[conn_elem_safe], jnp.zeros((n_faces, n_basis)))            #[n_faces, n_basis]
      dijfdotconn          = ((dij_f_p_fm_all_ref[i] * conn_values[:, None, :]).sum(axis=2) * abs_omega_dot_n_Je_ref[i][:, None]).sum(axis=0)#[n_basis]
      sweep_ref[elem_idx]  = (gij_inv_all_ref[i] *(qm_i_all_ref[i] + dijfdotconn)[None, :]).sum(axis=1)
      return sweep_ref
  jax.lax.fori_loop(0, elements_toposort.shape[0], _body, sweep_result_ref) 

def sweep_pl_update(elements_toposort, gij_inv_all_ref,  qm_i_all_ref, dij_f_p_fm_all_ref, etoe_ref, abs_omega_dot_n_Je_ref, sweep_result):    
  '''
    Wrapper function to call the Pallas kernel for the sweep update.

    Parameters
    ----------
    elements_toposort        : jnp.ndarray
        The indices of the elements in topological order for the sweep.
    gij_inv_all_ref          : jnp.ndarray
        The padded inverse of the gij matrices for all elements in the batch.
    qm_i_all_ref             : jnp.ndarray
        The padded source terms for all elements in the batch.
    dij_f_p_fm_all_ref       : jnp.ndarray
        The padded dij_f_p_fm matrices for all elements in the batch.
    etoe_ref                 : jnp.ndarray
        The padded element-to-element connectivity array.
    abs_omega_dot_n_Je_ref   : jnp.ndarray
        The padded absolute values of the dot products of omega and the element face normals for all elements in the batch.
    sweep_result             : jnp.ndarray
        The buffer array to store the sweep results. Updated in place.
    Returns
    -------
    jnp.ndarray
        The updated sweep result after performing the transport inverse sweep over the batch of elements.
  '''
  return pl.pallas_call(_sweep_pl_update_kernel,
                        out_shape=jax.ShapeDtypeStruct(sweep_result.shape, sweep_result.dtype),
                        grid=(1,), input_output_aliases={6: 0},
                        compiler_params = pltriton.CompilerParams())(elements_toposort, gij_inv_all_ref,  qm_i_all_ref, dij_f_p_fm_all_ref, etoe_ref, abs_omega_dot_n_Je_ref, sweep_result)

def _next_power_of_two(n: int) -> int:
    """Return next power-of-two >= n."""
    # If n already a pow2, return n; else next pow2
    return 1 if n == 0 else 2 ** ((n - 1).bit_length())

#=============================================================================================================================================================================
# Multiple angle functions
#=============================================================================================================================================================================

@partial(jax.jit, static_argnums = (7,))
def transport_inverse_sweep_multiple_angles_batched_pallas(geometric_domain    : GeometricDomain,
                                                    fem_matrices               : FEMMatrices,
                                                    precomputed_geometric_data : PrecomputedGeometricData,
                                                    cross_sections             : CrossSectionDataSingleGroup,
                                                    angle                      : jnp.ndarray,
                                                    sweep_data  : BasicSweepData,
                                                    x                          : jnp.ndarray,
                                                    batch_size                 : int):
    '''
    Perform a complete transport inverse sweep over all elements in the geometric domain for multiple angles using batched vmap and fori_loop.

    vmap is used to compute all Gij_inv, QM_i and Dij_f_p_fm matrices for all elements in a vectorized manner for a fixed batch size and multiple angles.
    Then, a fori_loop is used to iterate over the elements in topological order to compute the sweep result in batches. The sweep kernel uses Pallas for improved performance.
    
    Parameters
    ----------
    geometric_domain : GeometricDomain
        The geometric domain data.
    fem_matrices : FEMMatrices
        The finite element method matrices.
    precomputed_geometric_data : PrecomputedGeometricData
        The precomputed geometric data (Jacobian determinants and inverses).
    cross_sections : CrossSectionDataSingleGroup
        The cross section data.
    sweep_data : BasicSweepData
        The topologically sorted angle data for multiple angles. (batch dimension is the first dimension)
    x : jnp.ndarray [n_angles, n_elements, n_basis]
        The source data. Should also be batched over angles (batch dimension is the first dimension).
    batch_size : int
        The batch size for processing elements.
    Returns
    -------
    jnp.ndarray [n_angles, n_elements, n_basis]
        The sweep result after performing the transport inverse sweep for all angles.
    '''
        
    _assert_shape(geometric_domain, fem_matrices, precomputed_geometric_data, cross_sections, angle, sweep_data, x, True)
    f_batch = jax.vmap(transport_inverse_sweep_batched_pallas, in_axes=(None, None, None, None, 0, 0, 0, None ))
    return f_batch(geometric_domain, fem_matrices, precomputed_geometric_data, cross_sections, angle, sweep_data, x, batch_size)

# ---------------------------------------------------------------------------
# Pallas strategies + convenience constructors
# ---------------------------------------------------------------------------
from ..transport_operator_strategies import InversionStrategy
from .angle_liftings import VmapAngles


class VmapPallasSweep(InversionStrategy):
    '''Inverse sweep with all per-element matrices via vmap, using Pallas for the matrix ops.'''

    def invert(self, operator, b):
        return transport_inverse_sweep_complete_vmap_pallas(*operator._inversion_args(), b)

    @classmethod
    def vmappable(cls):
        return True

    @classmethod
    def default_angle_lifting(cls):
        return VmapAngles()


class BatchedPallasSweep(InversionStrategy):
    '''Inverse sweep over batches of elements, using Pallas for the matrix ops.'''

    batch_size: int = None

    def invert(self, operator, b):
        return transport_inverse_sweep_batched_pallas(*operator._inversion_args(), b, self.batch_size)

    @classmethod
    def vmappable(cls):
        return True

    @classmethod
    def default_angle_lifting(cls):
        return VmapAngles()
