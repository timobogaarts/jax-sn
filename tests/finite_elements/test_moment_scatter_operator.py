import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as onp
import jax_sn


def first_collision_moments(MGOP, source_total):
    '''Spherical-harmonic moments of the first-collision angular flux:
    moments( T^{-1} M q ) for the group-0 isotropic BB source. This exercises the full moment
    projection at the BB anisotropy order (lmax=5). Shared with the reference generator so the
    stored bundle and the test compute the identical quantity.'''
    return MGOP.moment_operator.mv(MGOP.transport_operator.to_energy_group(0).inv(MGOP.mass_operator.mv(source_total)))


def _expected_moments_reference(dim):
    # Self-pinned regression reference (frozen from the current implementation; see
    # tests/reference_data/__init__.py). Guards the higher-l moment projection against silent
    # drift -- BB's own oracle (key_bb) only checks the scalar l=0 flux.
    from reference_data import load, key_moment
    return jnp.array(load(key_moment(dim))['expected'])


def test_moment_operator(bb_mgop):
    '''The full moment vector of the first-collision flux matches the frozen reference on the shared
    BB problem across dimensions.'''
    MGOP, source_total, _, bb_args = bb_mgop
    moments = first_collision_moments(MGOP, source_total)
    onp.testing.assert_allclose(onp.array(moments), onp.array(_expected_moments_reference(bb_args[0])), rtol=1e-12, atol=1e-12)

