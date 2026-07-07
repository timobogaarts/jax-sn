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


def test_moment_operator_transpose():
    quadrature_set = jax_sn.quadrature_set.create_tn_quadrature_set(3)
    element_basis = jax.ShapeDtypeStruct(shape=(10, 15), dtype=quadrature_set.angles.dtype)
    moment_operator = jax_sn.operators.moment_operator.MomentOperator.from_quadrature_set(quadrature_set, 5, element_basis)
    from .operator_util import _check_adjoint_of_operator

    _check_adjoint_of_operator(moment_operator)


def test_discrete_operator_transpose():
    quadrature_set = jax_sn.quadrature_set.create_tn_quadrature_set(3)
    element_basis = jax.ShapeDtypeStruct(shape=(10, 15), dtype=quadrature_set.angles.dtype)
    discrete_operator = jax_sn.operators.moment_operator.DiscreteOperator.from_quadrature_set(quadrature_set, 5, element_basis)
    from .operator_util import _check_adjoint_of_operator

    _check_adjoint_of_operator(discrete_operator)


def test_scatter_operator_transpose():
    onp.random.seed(5012026)
    n_elem = 10

    xs_data_total = jnp.array(onp.random.rand(n_elem,2))
    xs_data_scat = jnp.array(onp.random.rand(n_elem, 16, 2,2))

    cross_section_data= jax_sn.domain.cross_sections.CrossSectionData(xs_data_total, xs_data_scat)
    cross_sections_indexed = jax_sn.domain.geometric_domain.CrossSectionsIndexed(jnp.arange(n_elem), cross_section_data)

    scatter_operator = jax_sn.operators.scatter_operator.ScatterOperator(cross_sections_indexed, jax.ShapeDtypeStruct((4,n_elem, 3), dtype = xs_data_total.dtype), energy_group_out=1, energy_group_in=0)
    from .operator_util import _check_adjoint_of_operator
    _check_adjoint_of_operator(scatter_operator)
