"""
Tests for the adjoint multi-group solve.

The key identity tested is adjoint reciprocity:

    <d, phi_star> = <d_star, phi>

where
    phi      = A^{-1}   d,       d      = M T^{-1} H b      (forward preprocessed RHS)
    phi_star = (A^T)^{-1} d_star, d_star = D^T (T^T)^{-1} H q_star (adjoint preprocessed RHS)

This is equivalent to the physical phase-space reciprocity

    integral psi_star * q  dP  =  integral psi * q_star dP

since d = M T^{-1} H b and d_star = D^T (T^T)^{-1} H q_star encode the same information
as the angular fluxes psi and psi_star via the transport operator.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as onp
import pytest
import lineax as lx
import equinox as eqx

from finite_elements.operator_util import _setup_operators, setup_from_verts
from finite_elements.test_multigroup_derivative import _get_bb_verts_conn_xs_sources
from finite_elements.test_breeding_blanket import base_region_sizes, base_resolutions


def _setup_mgop(dim=2, degree=2, tn_order=3):
    verts, conn, cross_sections, sources = _get_bb_verts_conn_xs_sources(
        dim, base_region_sizes(), base_resolutions(), 5.0, 1
    )
    solution_domain, sources, quadrature_set = setup_from_verts(
        verts, conn, cross_sections, sources, degree, tn_order
    )
    return _setup_operators(solution_domain, sources, quadrature_set)


@eqx.filter_jit
def _preprocess_forward(mgop, b):
    r"""Compute :math:`d_g = M T_g^{-1} H b_g` for each energy group."""
    def body(_, g_i):
        return None, mgop.moment_operator.mv(
            mgop.transport_operator.to_energy_group(g_i).inv(
                mgop.mass_operator.mv(b[g_i])
            )
        )
    _, d = jax.lax.scan(body, None, jnp.arange(mgop.n_energy_groups))
    return d


@eqx.filter_jit
def _preprocess_adjoint(mgop, q_star):
    r"""Compute :math:`d^*_g = D^T (T^T_g)^{-1} H q^*_g` for each energy group."""
    transport_t = mgop.transport_operator.transpose()
    discrete_t  = mgop.discrete_operator.transpose()
    def body(_, g_i):
        return None, discrete_t.mv(
            transport_t.to_energy_group(g_i).inv(
                mgop.mass_operator.mv(q_star[g_i])
            )
        )
    _, d_star = jax.lax.scan(body, None, jnp.arange(mgop.n_energy_groups))
    return d_star


def test_adjoint_solve_reciprocity():
    r"""
    Adjoint reciprocity: <d, phi_star> == <d_star, phi>

    Verifies that solving the forward system  A phi = d  and the adjoint system
    A^T phi_star = d_star with independent random sources gives inner products
    that satisfy the algebraic adjoint identity, confirming that the adjoint
    preprocessing and solve are consistent.
    """
    mgop, _, _ = _setup_mgop()
    solver = lx.BiCGStab(rtol=1e-10, atol=1e-10, max_steps=300)

    angular_shape = (mgop.n_energy_groups, mgop.n_angles, mgop.n_elements, mgop.n_basis)

    rng = onp.random.default_rng(20260324)
    b      = jnp.array(rng.random(angular_shape))   # forward angular source
    q_star = jnp.array(rng.random(angular_shape))   # adjoint angular source

    # Preprocess both RHS into moment space
    d      = _preprocess_forward(mgop, b)
    d_star = _preprocess_adjoint(mgop, q_star)

    # Solve forward:  A phi      = d
    # Solve adjoint:  A^T phi_star = d_star
    phi      = lx.linear_solve(mgop,            d,      solver).value
    phi_star = lx.linear_solve(mgop.transpose(), d_star, solver).value

    # Reciprocity check: <d, phi_star> == <d_star, phi>
    # i.e. <A phi, phi_star> == <phi, A^T phi_star>
    lhs = float(jnp.sum(d      * phi_star))
    rhs = float(jnp.sum(d_star * phi))

    onp.testing.assert_allclose(lhs, rhs, rtol=1e-6, atol=1e-8)
