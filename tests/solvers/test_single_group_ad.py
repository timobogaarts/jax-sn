import numpy as onp
import jax
import pytest
import equinox as eqx
import lineax as lx

jax.config.update("jax_enable_x64", True)

from jax_sn.domain.cross_sections import CrossSectionData
from jax_sn.parametrisation import SingleGroupOperatorSigmaAllParametrised


def _sgop_xs(sgop):
    return sgop.data.operator_data.spatial_data.domain.cross_sections_indexed.cross_section_data


def _patch_sgop_xs(sgop, xs_data_indices, dx):
    '''sgop with a single cross-section entry (total or scattering) perturbed by dx. Cross-section
    perturbation touches only leaf data, so tree_at suffices -- no geometry rebuild (unlike the
    vertex-perturbing test_multigroup_derivative).'''
    xs = _sgop_xs(sgop)
    total_xs, scattering_xs = xs.total, xs.scattering
    kind, idx = xs_data_indices
    if kind == 'total':
        total_xs = total_xs.at[idx].add(dx)
    elif kind == 'scattering':
        scattering_xs = scattering_xs.at[idx].add(dx)
    else:
        raise ValueError("Invalid xs_data_indices")
    return eqx.tree_at(_sgop_xs, sgop, CrossSectionData(total_xs, scattering_xs))


def total_solve(sgop, source_eg0, solver):
    from bb_setup import source_operator_from_sgop
    rhs = source_operator_from_sgop(sgop).mv(source_eg0)
    return lx.linear_solve(sgop, rhs, solver)


def _fd_derivative(sgop, source_eg0, xs_data_indices, dx, solver):
    f0 = total_solve(sgop, source_eg0, solver).value.sum()
    f1 = total_solve(_patch_sgop_xs(sgop, xs_data_indices, dx), source_eg0, solver).value.sum()
    return (f1 - f0) / dx


def all_testing_derivatives():
    return [('total', (1, 0)), ('scattering', (1, 0, 0, 0)), ('scattering', (1, 1, 0, 0)), ('scattering', (2, 4, 0, 0))]


def _index_all_derivatives(xs_data_indices, all_derivatives):
    kind, idx = xs_data_indices
    if kind == 'total':
        return all_derivatives.total[idx]
    elif kind == 'scattering':
        return all_derivatives.scattering[idx]
    raise ValueError("Invalid xs_data_indices")


def sum_f(sgop, source_eg0, theta, solver):
    sgop_new = SingleGroupOperatorSigmaAllParametrised.reconstruct_operator(sgop, theta)
    return total_solve(sgop_new, source_eg0, solver).value.sum()


@pytest.mark.slow
def test_single_group_ad(bb_sgop):
    '''Derivatives of a single-group solve wrt the cross-section parameters, two ways:
     - finite difference: perturb a cross-section entry and re-solve;
     - automatic differentiation: jax.grad through the parametrised operator.
    Compared at a few arbitrary cross-section indices on the shared BB single-group operator.'''
    sgop, source_eg0, _, _ = bb_sgop
    dx = 1e-5
    bicgstab = lx.BiCGStab(1e-10, 1e-10, max_steps=100)

    sum_grad_f = eqx.filter_jit(jax.grad(sum_f, argnums=(2,)))
    all_derivatives = sum_grad_f(sgop, source_eg0, SingleGroupOperatorSigmaAllParametrised.theta_base(sgop), bicgstab)[0]

    for xs_data_indices in all_testing_derivatives():
        ad = _index_all_derivatives(xs_data_indices, all_derivatives)
        fd = _fd_derivative(sgop, source_eg0, xs_data_indices, dx, bicgstab)
        onp.testing.assert_allclose(ad, fd, rtol=1e-4, atol=1e-4)
