'''Accuracy of the multigroup solve against the frozen reference -- the oracle test.

This is the only multigroup test tied to a frozen reference bundle (``key_bb``): dim 1,2 are
external-solver ground truth, dim 3 is self-pinned to the current implementation (see
tests/reference_data/__init__.py). Everything else about the operator is checked algebraically
in test_multigroup_operator.py.
'''
import numpy as onp
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)

from finite_elements.bb_setup import _interp_x


def _expected_reference(dim):
    from reference_data import load, key_bb
    return load(key_bb(dim))['phi']  # per-group interpolated scalar flux


def test_multigroup_solve_accuracy(bb_mgop):
    '''4-group isotropic-source slab: check both solve_lower_triangular and lx.linear_solve (global
    BiCGStab) against the frozen scalar-flux reference along 1000 points spanning the slab.

    Tolerances are uniform across dimensions: rtol=1e-10 for the lower-triangular solve, 6e-4 for
    the global BiCGStab (which only converges to its own tolerance due to thermal upscattering).
    '''
    import lineax as lx
    MGOP, source_eg0, solution_domain, bb_args = bb_mgop

    rhs_eg0 = MGOP.moment_operator.mv(MGOP.transport_operator.to_energy_group(0).inv(MGOP.mass_operator.mv(source_eg0)))
    rhs_total = jnp.stack([rhs_eg0] + [jnp.zeros_like(rhs_eg0) for _ in range(MGOP.n_energy_groups - 1)], axis=0)
    bicgstab_solver = lx.BiCGStab(1e-20, 1e-20, max_steps=300)

    from jax_sn.operators.multi_group_operator import solve_lower_triangular
    from jax_sn.operators.multi_group_operator.lower_triangular_solver import SolverDiagnostics

    values_lower_triangular = solve_lower_triangular(MGOP, bicgstab_solver, rhs_total, SolverDiagnostics(False, False))[0]
    values_total            = lx.linear_solve(MGOP, rhs_total, bicgstab_solver).value

    dim = bb_args[0]
    d_other = bb_args[-3]
    x = _interp_x()
    xyz = jnp.stack([x] + [jnp.full_like(x, d_other / 2)] * (dim - 1), axis=-1)

    phi_lower_triangular = [solution_domain.interpolate(xyz, values_lower_triangular[i, 0]) for i in range(values_lower_triangular.shape[0])]
    phi_total            = [solution_domain.interpolate(xyz, values_total[i, 0]) for i in range(values_total.shape[0])]

    expected_values = _expected_reference(dim)
    onp.testing.assert_allclose(onp.array(expected_values), onp.array(phi_lower_triangular), rtol=1e-10)
    onp.testing.assert_allclose(onp.array(expected_values), onp.array(phi_total), rtol=6e-4)
