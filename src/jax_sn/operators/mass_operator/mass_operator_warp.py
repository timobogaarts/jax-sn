r'''
NVIDIA Warp mass strategy for n_basis <= MAX_NBASIS (=4).
'''
# NOTE: deliberately no ``from __future__ import annotations``. Warp resolves a kernel's annotations
# against the module globals, so stringised annotations would not see the per-basis-count types that
# live in build_kernel's local scope.
import functools

import jax
import warp as wp
import jax.numpy as jnp
import equinox as eqx

from warp import jax_callable

from jax_sn.solution_domain import PrecomputedGeometricData, FEMMatrices
from .mass_operator_strategies import MassStrategy


MAX_NBASIS   = 4            # Warp's largest fixed-size vector/matrix type
SUPPORTED    = (2, 3, 4)    # the basis counts that fit
DTYPE        = wp.float64   # Always double precision.

_VECD = {2: wp.vec2d, 3: wp.vec3d, 4: wp.vec4d}     # a per-element basis vector
_MATD = {2: wp.mat22d, 3: wp.mat33d, 4: wp.mat44d}  # the reference mass matrix


# =============================================================================
# Kernel + jax_callable bridge, one per basis count
# (result is the last param, allocated by jax_callable per output_dims)
# =============================================================================
@functools.lru_cache(maxsize=None)
def build_kernel(n_basis: int):
    '''Compile the mass kernel for one basis count. Memoised, so distinct counts coexist as
    separate Warp modules and a repeated call recompiles nothing.'''
    if n_basis not in SUPPORTED:
        raise ValueError(
            f"WarpMass supports n_basis in {SUPPORTED}; got {n_basis}. Warp's fixed-size "
            f"vector/matrix types stop at {MAX_NBASIS}, so a higher-degree element cannot be "
            f"expressed as a vecNd.")

    NBASISDTYPE  = _VECD[n_basis]
    NMATRIXDTYPE = _MATD[n_basis]

    # One Warp module per basis count. Warp otherwise keys the module on the *Python* module, so each
    # new count would register into one shared module, change its content hash and mark it modified;
    # the next launch reloads it, and a module load inside this kernel's CUDA graph capture is illegal.
    module = f"{__name__}_nb{n_basis}"

    @wp.kernel(enable_backward=False, module=module)
    def _mass_apply_kernel(
        mass_matrix : wp.array[NMATRIXDTYPE],    # [1]            reference mass matrix (same for every element)
        vol_det     : wp.array[DTYPE],           # [n_elem]       volume Jacobian determinants
        x           : wp.array2d[NBASISDTYPE],   # [batch, n_elem]
        result      : wp.array2d[NBASISDTYPE],   # [batch, n_elem]
    ):
        b, e = wp.tid()
        result[b, e] = vol_det[e] * (mass_matrix[0] @ x[b, e])

    def _mass_apply(
        mass_matrix : wp.array[NMATRIXDTYPE],
        vol_det     : wp.array[DTYPE],
        x           : wp.array2d[NBASISDTYPE],
        result      : wp.array2d[NBASISDTYPE],
    ) -> None:
        wp.launch(
            _mass_apply_kernel,
            dim=(x.shape[0], x.shape[1]),
            inputs=[mass_matrix, vol_det, x, result],
        )

    return jax_callable(_mass_apply)


def _warp_mass_primal(mass_matrix: jnp.ndarray, vol_det: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    # Batched other dimensions automatically by flattening and reshaping.
    n_elem, n_basis = x.shape[-2], x.shape[-1]
    x_flat  = x.reshape((-1, n_elem, n_basis))                                   # [batch, n_elem, n_basis]
    mass_wp = jnp.broadcast_to(mass_matrix[None, ...], (1, n_basis, n_basis))
    out = build_kernel(n_basis)(mass_wp, vol_det, x_flat, output_dims={"result": x_flat.shape})[0]
    return out.reshape(x.shape)


def _reference_mass(mass_matrix: jnp.ndarray, vol_det: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    '''Pure-JAX statement of the *same* linear map, used only to supply exact tangents (see JVP below).'''
    return vol_det[:, None] * jnp.einsum("ij,...ej->...ei", mass_matrix, x)


# The Warp kernel is an opaque FFI call, we will define a custom JVP from the a pure-JAX reference implementation.
_mass_diff = jax.custom_jvp(_warp_mass_primal)


@_mass_diff.defjvp
def _mass_diff_jvp(primals, tangents):
    primal_out = _warp_mass_primal(*primals)                       # fast Warp primal
    _, tangent_out = jax.jvp(_reference_mass, primals, tangents)    # exact tangent, pure JAX
    return primal_out, tangent_out


@eqx.filter_jit
def warp_mass_apply(
    fem_matrices               : FEMMatrices,
    precomputed_geometric_data : PrecomputedGeometricData,
    x                          : jnp.ndarray,   # [..., n_elem, n_basis]
) -> jnp.ndarray:
    '''Apply the (symmetric) mass operator via the Warp kernel. Forward == transpose; differentiable.'''
    return _mass_diff(fem_matrices.mass_matrix, precomputed_geometric_data.vol_jacobians_det, x)


# =============================================================================
# Strategy
# =============================================================================
class WarpMass(MassStrategy):
    '''
    Fused Warp kernel, supporting n_basis <= MAX_NBASIS (=4). Forward == transpose; differentiable.
    '''

    def forward(self, operator, x):
        assert operator.fem_matrices.n_basis in SUPPORTED, \
            f"WarpMass supports n_basis in {SUPPORTED}, got {operator.fem_matrices.n_basis}"
        return warp_mass_apply(operator.fem_matrices, operator.precomputed_geometric_data, x)
