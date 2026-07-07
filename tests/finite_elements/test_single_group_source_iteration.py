'''Source (fixed-point) iteration on the single-group operator.

This is the only test in the suite that exercises Richardson / source iteration
(``x <- x - A x + b``) rather than a Krylov solver -- i.e. it verifies that ``I - A`` is a
contraction and that the fixed point matches the direct linear solve. Accuracy of the
group-0 flux against the external reference is already covered by ``test_bb`` (group 0 is
the first block of the lower-triangular multigroup solve), so here we only check the
solver-equivalence property.

Uses the shared ``bb_sgop`` fixture (group-0 operator of the breeding-blanket problem), so
it adds no extra compilation beyond the ``bb_mgop`` build.
'''
import jax
import jax.numpy as jnp
import numpy as onp
import equinox as eqx
import lineax as lx

jax.config.update("jax_enable_x64", True)

_N_SOURCE_ITER = 300


def test_single_group_source_iteration(bb_sgop):
    sgop, source_total, solution_domain, bb_args = bb_sgop

    rhs = sgop.moment_operator.mv(sgop.transport_operator.inv(sgop.mass_operator.mv(source_total)))

    @eqx.filter_jit
    def source_iterate(op, rhs):
        # fori_loop keeps op.mv compiled once (no 300x unrolling)
        return jax.lax.fori_loop(0, _N_SOURCE_ITER, lambda i, xk: xk - op.mv(xk) + rhs, jnp.zeros_like(rhs))

    xk_fixed_point = source_iterate(sgop, rhs)
    xk_direct = lx.linear_solve(sgop, rhs, lx.BiCGStab(1e-14, 1e-14, max_steps=300)).value

    onp.testing.assert_allclose(onp.array(xk_fixed_point), onp.array(xk_direct), rtol=1e-7, atol=1e-9)
