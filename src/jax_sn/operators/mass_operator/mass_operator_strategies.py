r'''
Strategy objects for the mass operator.

The mass operator applies a block-diagonal, per-element mass matrix scaled by the volume Jacobian
determinant. The *implementation* of that contraction is selected by a strategy object rather than
by subclassing, mirroring the transport, moment and discrete operators:

- :class:`MassStrategy` -- how to apply the (batched) block-diagonal mass matrix.

The mass operator is symmetric, so each strategy implements a single ``forward`` (forward ==
transpose); there is no separate transpose method and no ``transpose_bool``. A new implementation is a
single small class, not a subclass of the operator. The kernels live here next to the strategies that
dispatch to them, so the operator module depends only on the (array-free) strategy classes.
'''

import jax
import jax.numpy as jnp
from abc import abstractmethod

import equinox as eqx

from jax_sn.solution_domain import PrecomputedGeometricData, FEMMatrices
from .mass_operator_base import MassData, MassStrategy

class VmapMass(MassStrategy):
    r'''
    Uses a vmap of all elements to compute the mass operator. Alone, this creates a temporary.
    However, in a combined operator (e.g. ``SingleGroupOperator``), this temporary can often be
    re-used, such that it is not necessarily any worse than :class:`LaxMass` (which does not create
    temporaries).
    '''

    def forward(self, data: MassData, x):
        return _total_mass_kernel_idx_in_precomp(
            data.fem_matrices,
            data.precomputed_geometric_data,
            x,
        )


class LaxMass(MassStrategy):
    '''
    Batched matrix multiplication implementation of the mass operator using ``jax.lax``. This
    implementation does not create temporaries and is not particularly less or more performant than
    :class:`VmapMass`.
    '''

    def forward(self, data: MassData, x):
        return total_mass_kernel_lax(
            data.fem_matrices,
            data.precomputed_geometric_data,
            x,
        )


# ---------------------------------------------------------------------------
# Mass kernels
# ---------------------------------------------------------------------------
def _mass_kernel_idx_in_precomp(
        elem_idx                    : int,
        fem_matrices                : FEMMatrices,
        precomputed_geometric_data  : PrecomputedGeometricData,
        values                      : jnp.ndarray
    ):
    '''
    Compute the mass matrix map
        v_i = ∑_{j} Jv ∫H_i H_j dV v_j

    We index directly in the source instead of passing a subsource. This can save significant memory when vmapping over all elements and all directions with the same source independent of direction.

    Parameters
    ----------
    elem_idx : int
        The element index to compute the values vector for
    fem_matrices : FEMMatrices
        The finite element method matrices.
    precomputed_geometric_data : PrecomputedGeometricData
        The precomputed geometric data (Jacobian determinants and inverses).
    values : jnp.ndarray
        The source values array of shape [..., n_elements, n_basis]
    Returns
    -------
    values : jnp.ndarray [...., n_basis]
        The computed value vector for the specified element.
    '''
    return precomputed_geometric_data.vol_jacobians_det[elem_idx] * jnp.einsum("ij, ...j->...i", fem_matrices.mass_matrix , values[..., elem_idx, :])


@jax.jit
def _total_mass_kernel_idx_in_precomp(
        fem_matrices                : FEMMatrices,
        precomputed_geometric_data  : PrecomputedGeometricData,
        values                      : jnp.ndarray
    ):
    '''
    Compute the mass matrix map
        v_i = ∑_{j} Jv ∫H_i H_j dV v_j

    We index directly in the source instead of passing a subsource. This can save significant memory when vmapping over all elements and all directions with the same source independent of direction.

    Parameters
    ----------
    fem_matrices : FEMMatrices
        The finite element method matrices.
    precomputed_geometric_data : PrecomputedGeometricData
        The precomputed geometric data (Jacobian determinants and inverses).
    values : jnp.ndarray
        The source values array of shape [..., n_elements, n_basis]
    Returns
    -------
    values : jnp.ndarray [...., n_elements, n_basis]
        The computed value vector for all elements.
    '''
    return jax.vmap(_mass_kernel_idx_in_precomp, in_axes=(0, None, None, None), out_axes=-2)(
                jnp.arange(precomputed_geometric_data.n_elements),
                fem_matrices,
                precomputed_geometric_data,
                values)


@jax.jit
def total_mass_kernel_lax(
    fem_matrices: FEMMatrices,
    precomputed_geometric_data: PrecomputedGeometricData,
    values: jnp.ndarray,                     # [..., n_elem, n_dof]
):
    M = fem_matrices.mass_matrix              # [n_dof, n_dof]
    detJ = precomputed_geometric_data.vol_jacobians_det  # [n_elem]

    # Move first axes to to batch axis
    # (..., n_elem, n_dof) -> (batch, n_elem, n_dof)
    batch_shape = values.shape[:-2]
    values_b = values.reshape((-1,) + values.shape[-2:])

    # Batched matmul: (n_dof, n_dof) @ (..., n_elem, n_dof)^T
    out = jax.lax.dot_general(
        values_b,                             # (B, n_elem, n_dof)
        M,                                    # (n_dof, n_dof)
        dimension_numbers=(
            ((2,), (1,)),                     # contract dof
            ((), ())                          # no batch contraction
        ),
    )                                         # (B, n_elem, n_dof)

    # Scale by Jacobian determinant
    out = out * detJ[None, :, None]

    return out.reshape(batch_shape + out.shape[-2:])
