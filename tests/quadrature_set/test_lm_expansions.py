
import jax.numpy as jnp
import numpy as onp
import pytest

import jax_sn
import jax
jax.config.update("jax_enable_x64", True)


def _reference_yklm(tn_set, l_max):
    # Frozen reference real spherical harmonics, already in jax-sn convention
    # (see tests/reference_data/_generate.py)
    from reference_data import load, key_lm
    return load(key_lm(tn_set, l_max))['yklm']


@pytest.mark.parametrize("l_max", [0,1,2,3,4,5])
def test_lm_expansion(l_max):
    tn_set = 8

    qs_jsn   = jax_sn.quadrature_set.create_tn_quadrature_set(tn_set)
    yklm_jsn = jax_sn.quadrature_set.quadrature_set_base.compute_3d_yklm_real_from_quadrature_set(qs_jsn, l_max)

    onp.testing.assert_allclose(yklm_jsn, _reference_yklm(tn_set, l_max), atol=1e-12)

def mu_func(quadrature_set):
    return 1 - quadrature_set.angles[:,0]**2 + quadrature_set.angles[:,0]

def mu_zeta_func(quadrature_set):
    return 1 - quadrature_set.angles[:,0]**2 + quadrature_set.angles[:,0] + quadrature_set.angles[:,1]**2 - quadrature_set.angles[:,1]

def integrate_quad_set_lm(quadrature_set, func, lmax = 3):
    wkyklm = quadrature_set.wk_ylm_real(lmax)
    return jnp.einsum("kl, k, k-> l", wkyklm, func(quadrature_set), quadrature_set.weights)

def test_reduction_1d():
    tn_set = 8 
    qs_jsn   = jax_sn.quadrature_set.create_tn_quadrature_set(tn_set)
    reduced_qs = jax_sn.quadrature_set.quadrature_set_base.QuadratureSetReduced.from_quadrature_set(qs_jsn, 1)
    original_integral = integrate_quad_set_lm(qs_jsn, mu_func, lmax = 3)
    reduced_integral = integrate_quad_set_lm(reduced_qs, mu_func, lmax = 3)
    onp.testing.assert_allclose(original_integral, reduced_integral, atol=1e-12)

def test_reduction_2d():
    tn_set = 8 
    qs_jsn   = jax_sn.quadrature_set.create_tn_quadrature_set(tn_set)
    reduced_qs = jax_sn.quadrature_set.quadrature_set_base.QuadratureSetReduced.from_quadrature_set(qs_jsn, 2)
    original_integral = integrate_quad_set_lm(qs_jsn, mu_zeta_func, lmax = 3)
    reduced_integral = integrate_quad_set_lm(reduced_qs, mu_zeta_func, lmax = 3)
    onp.testing.assert_allclose(original_integral, reduced_integral, atol=1e-12)


    

    