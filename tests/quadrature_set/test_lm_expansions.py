
import jax.numpy as jnp
import numpy as onp 
import pytest

from jax_sn.quadrature_set.quadrature_set_base import _lm_real_expansion
import jax_sn
import jax
jax.config.update("jax_enable_x64", True)

def _lm_sb_to_norm(l : jnp.ndarray, m : jnp.ndarray) -> jnp.ndarray:
    return jnp.where(m == 0, l**2, jnp.where(m < 0, l**2 + l + abs(m), l**2 + abs(m)))

def check_ylm_real_to_sb(values_jaxsn, values_sb, l_max : int):
    # This converts from StellBlanket's ordering and normalization to the one used in jax-sn

    # Differences: 
    #  - 1/sqrt(4pi) in the normalization over the entire range 
    #  - sqrt(2) factor for m != 0
    #  - idiosyncratic ordering of lm coefficients (see _lm_sb_to_norm; this is the indices used in StellBlanket)
    lm_real = _lm_real_expansion(l_max)
    indices_convert = _lm_sb_to_norm(*lm_real)
    convert_sb_to_jaxsn = jnp.where(lm_real[1] != 0, values_sb[..., indices_convert] * jnp.sqrt(2), values_sb[..., indices_convert]) / jnp.sqrt(4 * jnp.pi)
    onp.testing.assert_allclose(convert_sb_to_jaxsn, values_jaxsn, atol=1e-12)


def _expected_sb_function(tn_set, l_max):
    import StellBlanket as SB
    qs_sb = SB.Generic_Elements.TN_Quadrature_Set(tn_set)
    lm_real_sb = qs_sb.Calculate_Yklm_real(l_max)
    return lm_real_sb

@pytest.mark.parametrize("l_max", [0,1,2,3,4,5])
def test_lm_expansion(l_max, _expected_function = _expected_sb_function):
    tn_set = 8 

    qs_jsn   = jax_sn.quadrature_set.create_tn_quadrature_set(tn_set)
    yklm_jsn = jax_sn.quadrature_set.quadrature_set_base.compute_3d_yklm_real_from_quadrature_set(qs_jsn, l_max)

    yklm_expected = _expected_function(tn_set, l_max)

    check_ylm_real_to_sb(yklm_jsn, yklm_expected, l_max)

