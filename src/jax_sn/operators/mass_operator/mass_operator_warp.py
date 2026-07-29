r'''
NVIDIA Warp mass strategy.
Only supports 4 basis functions.
'''
from __future__ import annotations

import jax
import warp as wp
import jax.numpy as jnp
import equinox as eqx

from warp.jax_experimental import jax_callable

from jax_sn.solution_domain import PrecomputedGeometricData, FEMMatrices
from .mass_operator_strategies import MassStrategy


NBASIS       = 4
DTYPE        = wp.float64   # matches JAX_ENABLE_X64 (see tests/conftest.py)
NBASISDTYPE  = wp.vec4d     # a per-element basis vector
NMATRIXDTYPE = wp.mat44d    # the reference mass matrix


# =============================================================================
# Kernel
# =============================================================================
@wp.kernel(enable_backward=False)
def _mass_apply_kernel(
    mass_matrix : wp.array[NMATRIXDTYPE],    # [1]            reference mass matrix (same for every element)
    vol_det     : wp.array[DTYPE],           # [n_elem]       volume Jacobian determinants
    x           : wp.array2d[NBASISDTYPE],   # [batch, n_elem]
    result      : wp.array2d[NBASISDTYPE],   # [batch, n_elem]
):
    b, e = wp.tid()
    result[b, e] = vol_det[e] * (mass_matrix[0] @ x[b, e])


# =============================================================================
# jax_callable bridge  (result is the last param, allocated by jax_callable per output_dims)
# =============================================================================
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

_mass_apply_jax = jax_callable(_mass_apply)


def _warp_mass_primal(mass_matrix: jnp.ndarray, vol_det: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    '''Warp primal: ``out[..., e, :] = detJ[e] * (M @ x[..., e, :])`` (degree-1 tet, n_basis == 4).'''
    n_elem, n_basis = x.shape[-2], x.shape[-1]
    x_flat  = x.reshape((-1, n_elem, n_basis))                                   # [batch, n_elem, n_basis]
    mass_wp = jnp.broadcast_to(mass_matrix[None, ...], (1, n_basis, n_basis))
    out = _mass_apply_jax(mass_wp, vol_det, x_flat, output_dims={"result": x_flat.shape})[0]
    return out.reshape(x.shape)


def _reference_mass(mass_matrix: jnp.ndarray, vol_det: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    '''Pure-JAX statement of the *same* linear map, used only to supply exact tangents (see JVP below).'''
    return vol_det[:, None] * jnp.einsum("ij,...ej->...ei", mass_matrix, x)


# The Warp kernel is an opaque FFI call (``enable_backward=False``) that autodiff cannot traverse. The
# mass operator, however, sits directly on the AD path (its ``mv`` is applied inside the multigroup
# operator that lineax linearises), so -- unlike the Warp *transport* sweep, which is only ever
# differentiated implicitly through ``lx.linear_solve`` -- it must carry its own derivative rule.
#
# Mass is linear (and symmetric), so the exact JVP/VJP is just the same linear map applied to the
# tangent/cotangent. We keep the fast Warp kernel for the primal and express the tangent with the
# pure-JAX ``_reference_mass`` (verified to agree with Warp to machine precision); JAX transposes that
# tangent rule automatically for reverse-mode ``grad``. This differentiates correctly w.r.t. the input
# *and* the operator parameters (``mass_matrix``, ``vol_jacobians_det``).
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
    Apply the mass operator with a fused NVIDIA Warp kernel (one thread per ``(batch, element)``). Creates
    no temporaries -- peak memory is just input + output. Fixed to degree-1 tetrahedra (``n_basis == 4``),
    matching the Warp transport kernels.
    '''

    def forward(self, operator, x):
        assert operator.fem_matrices.n_basis == NBASIS, \
            f"WarpMass expects {NBASIS} basis functions (degree-1 tetrahedra), got {operator.fem_matrices.n_basis}"
        return warp_mass_apply(operator.fem_matrices, operator.precomputed_geometric_data, x)
