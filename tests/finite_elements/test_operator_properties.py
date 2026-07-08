"""
Property + consistency battery run over *every discovered strategy* of each operator axis (mass / moment /
discrete / scatter / transport forward+inversion), at FEM degree 1 and 2 -- see
``operator_strategy_harness.py``. A strategy added anywhere in the package is picked up and checked here
automatically, with no edit to this file (the same contract the perf suite provides).

Scope is deliberately **property + consistency**, not absolute correctness:

  * Property checks are self-contained algebraic identities every variant must satisfy (adjoint, transpose
    idempotence, forward-backward round-trip).
  * Consistency checks assert all variants of an axis *agree with each other*. The pure-JAX default
    strategy is the pivot -- for readable failures ("WarpMass disagrees with VmapMass"), not a ground
    truth. Absolute correctness (a bug shared by every variant, and gradient accuracy) is the job of the
    frozen-reference pins (test_moment_operator, test_bb, test_transport_sweep_pinned) and the
    finite-difference checks in test_operator_regression.py.

Both degrees run for the degree-agnostic (pure-JAX / pallas) strategies; the warp kernels are n_basis==4
only and so carry a specialisation flag restricting them to degree 1 (see harness ``supported_degrees``).
The BB setup is shared + cached with test_operator_regression.py via the ``strategy_setup`` fixture.
"""
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as onp
import equinox as eqx
import pytest

from jax_sn.tools import assert_pytree_allclose

from .operator_strategy_harness import (
    LINEAR_AXES, SCATTER_AXIS, DEGREES, discover, discover_forward, discover_inversion_cases,
    supported_degrees,
)

_ANGLE_I = 0  # single-angle transport tests pick this angle from the multi-angle operator


def _rand(shape, seed):
    return jnp.array(onp.random.default_rng(seed).random(shape))


# ---------------------------------------------------------------------------
# Case lists (built at collection time; each variant paired with the degrees its flag allows)
# ---------------------------------------------------------------------------
_LINEAR_CASES = [(deg, axis, label, strat)
                 for axis in LINEAR_AXES
                 for (label, strat) in discover(axis)
                 for deg in supported_degrees(strat)]
_LINEAR_IDS = [f"deg{d}-{ax.name}-{lb}" for (d, ax, lb, _) in _LINEAR_CASES]

_SCATTER_CASES = [(deg, label, cls)
                  for (label, cls) in discover(SCATTER_AXIS, instantiate=False)
                  for deg in DEGREES]
_SCATTER_IDS = [f"deg{d}-{lb}" for (d, lb, _) in _SCATTER_CASES]

_FWD_CASES = [(deg, label, fwd)
              for (label, fwd) in discover_forward()
              for deg in supported_degrees(fwd)]
_FWD_IDS = [f"deg{d}-{lb}" for (d, lb, _) in _FWD_CASES]

_INV_CASES = [(deg, label, inv, swkw)
              for (label, inv, swkw) in discover_inversion_cases()
              for deg in supported_degrees(inv)]
_INV_IDS = [f"deg{d}-{lb}" for (d, lb, _, _) in _INV_CASES]


def _linear_ops(setup, deg, axis, label, strat):
    variant = setup.variant(deg, (axis.name, label), **{axis.kwarg: strat})
    return axis.extract(variant), axis.extract(setup.pivot(deg))


# ===========================================================================
# Linear-operator axes (mass / moment / discrete)
# ===========================================================================
@pytest.mark.parametrize("deg,axis,label,strat", _LINEAR_CASES, ids=_LINEAR_IDS)
class TestLinearOperatorStrategy:

    def test_adjoint_identity(self, strategy_setup, deg, axis, label, strat):
        op, _ = _linear_ops(strategy_setup, deg, axis, label, strat)
        x = _rand(op.in_structure().shape, 1)
        y = _rand(op.out_structure().shape, 2)
        onp.testing.assert_allclose(float(jnp.sum(op.mv(x) * y)),
                                    float(jnp.sum(x * op.transpose().mv(y))), rtol=1e-11, atol=1e-11)

    def test_transpose_idempotence(self, strategy_setup, deg, axis, label, strat):
        op, _ = _linear_ops(strategy_setup, deg, axis, label, strat)
        x = _rand(op.in_structure().shape, 3)
        onp.testing.assert_allclose(op.transpose().transpose().mv(x), op.mv(x), rtol=1e-11, atol=1e-11)

    def test_forward_consistency(self, strategy_setup, deg, axis, label, strat):
        op, ref = _linear_ops(strategy_setup, deg, axis, label, strat)
        x = _rand(op.in_structure().shape, 4)
        onp.testing.assert_allclose(op.mv(x), ref.mv(x), rtol=1e-11, atol=1e-11)

    def test_transpose_consistency(self, strategy_setup, deg, axis, label, strat):
        op, ref = _linear_ops(strategy_setup, deg, axis, label, strat)
        y = _rand(op.out_structure().shape, 5)
        onp.testing.assert_allclose(op.transpose().mv(y), ref.transpose().mv(y), rtol=1e-11, atol=1e-11)

    def test_grad_wrt_input_consistency(self, strategy_setup, deg, axis, label, strat):
        op, ref = _linear_ops(strategy_setup, deg, axis, label, strat)
        x = _rand(op.in_structure().shape, 6)
        loss = lambda o, v: jnp.sum(jnp.sin(o.mv(v)))
        onp.testing.assert_allclose(jax.grad(loss, argnums=1)(op, x),
                                    jax.grad(loss, argnums=1)(ref, x), rtol=1e-11, atol=1e-11)

    def test_jvp_wrt_input_consistency(self, strategy_setup, deg, axis, label, strat):
        op, ref = _linear_ops(strategy_setup, deg, axis, label, strat)
        x = _rand(op.in_structure().shape, 7)
        t = _rand(op.in_structure().shape, 8)
        _, jv = jax.jvp(lambda v: op.mv(v), (x,), (t,))
        _, jv_ref = jax.jvp(lambda v: ref.mv(v), (x,), (t,))
        onp.testing.assert_allclose(jv, jv_ref, rtol=1e-11, atol=1e-11)

    @pytest.mark.slow
    def test_end_to_end_multigroup_grad(self, strategy_setup, deg, axis, label, strat):
        """Real-pipeline guard: gradient of the full multigroup apply w.r.t. the solution-domain data is
        unchanged by swapping in this strategy -- the tangent flows through the sub-operator inside the
        multigroup scan, so a non-differentiable kernel breaks here."""
        variant = strategy_setup.variant(deg, (axis.name, label), **{axis.kwarg: strat})
        x = _rand(variant.in_structure().shape, 9)
        loss = lambda mgop: jnp.sum(jnp.sin(mgop.mv(x)))
        g = eqx.filter_grad(loss)(variant).setup.solution_domain_data
        g_ref = eqx.filter_grad(loss)(strategy_setup.pivot(deg)).setup.solution_domain_data
        assert_pytree_allclose(g, g_ref, rtol=1e-9, atol=1e-9)


# ===========================================================================
# Scatter axis (gather/scatter adjoint per energy group)
# ===========================================================================
@pytest.mark.parametrize("deg,label,cls", _SCATTER_CASES, ids=_SCATTER_IDS)
def test_scatter_gather_adjoint(strategy_setup, deg, label, cls):
    """Full-operator adjoint of the block group-to-group scatter:
    Σ_g <gather_forward(x, g), y_g> == Σ_g <x_g, gather_backward(y, g)>."""
    mgop = strategy_setup.variant(deg, ("scatter", label), gather_scatter_operator_cls=cls)
    gso = mgop.gather_scatter_operator
    n_groups = mgop.n_energy_groups
    moment_full = (n_groups,) + mgop.moment_operator.out_structure().shape
    rng = onp.random.default_rng(11)
    x = jnp.array(rng.random(moment_full))
    y = jnp.array(rng.random(moment_full))
    lhs = sum(float(jnp.sum(gso.gather_forward(x, g) * y[g])) for g in range(n_groups))
    rhs = sum(float(jnp.sum(x[g] * gso.gather_backward(y, g))) for g in range(n_groups))
    onp.testing.assert_allclose(lhs, rhs, rtol=1e-11, atol=1e-11)


# ===========================================================================
# Transport axis -- forward strategies (only the forward apply differs)
# ===========================================================================
@pytest.mark.parametrize("deg,label,fwd", _FWD_CASES, ids=_FWD_IDS)
class TestTransportForward:

    def _op_ref(self, setup, deg, label, fwd):
        op = setup.variant(deg, ("fwd", label), forward_strategy=fwd).transport_operator[_ANGLE_I]
        return op, setup.pivot(deg).transport_operator[_ANGLE_I]

    def test_mv_adjoint_identity(self, strategy_setup, deg, label, fwd):
        op, _ = self._op_ref(strategy_setup, deg, label, fwd)
        x = _rand(op.in_structure().shape, 1)
        y = _rand(op.out_structure().shape, 2)
        onp.testing.assert_allclose(float(jnp.sum(op.mv(x) * y)),
                                    float(jnp.sum(x * op.transpose().mv(y))), rtol=1e-10, atol=1e-10)

    def test_mv_consistency(self, strategy_setup, deg, label, fwd):
        op, ref = self._op_ref(strategy_setup, deg, label, fwd)
        x = _rand(op.in_structure().shape, 3)
        onp.testing.assert_allclose(op.mv(x), ref.mv(x), rtol=1e-10, atol=1e-10)


# ===========================================================================
# Transport axis -- inversion strategies, single-angle inverse sweep
# ===========================================================================
@pytest.mark.parametrize("deg,label,inv,swkw", _INV_CASES, ids=_INV_IDS)
class TestTransportInversionSingleAngle:

    def _op_ref(self, setup, deg, label, inv, swkw):
        op = setup.variant(deg, ("inv", label), inversion_strategy=inv,
                           sweep_data_kwargs=swkw).transport_operator[_ANGLE_I]
        return op, setup.pivot(deg).transport_operator[_ANGLE_I]

    def test_inv_adjoint_identity(self, strategy_setup, deg, label, inv, swkw):
        op, _ = self._op_ref(strategy_setup, deg, label, inv, swkw)
        x = _rand(op.in_structure().shape, 4)
        y = _rand(op.out_structure().shape, 5)
        lhs = float(jnp.sum(op.inv(x) * y))
        rhs = float(jnp.sum(x * op.transpose().inv(y)))
        onp.testing.assert_allclose(lhs, rhs, rtol=1e-9, atol=1e-9)

    def test_forward_backward_identity(self, strategy_setup, deg, label, inv, swkw):
        """T⁻¹(T x) == x and its transpose -- a self-contained round-trip property."""
        op, _ = self._op_ref(strategy_setup, deg, label, inv, swkw)
        x = _rand(op.in_structure().shape, 6)
        onp.testing.assert_allclose(op.inv(op.mv(x)), x, rtol=1e-9, atol=1e-9)
        onp.testing.assert_allclose(op.transpose().inv(op.transpose().mv(x)), x, rtol=1e-9, atol=1e-9)

    def test_inv_consistency(self, strategy_setup, deg, label, inv, swkw):
        op, ref = self._op_ref(strategy_setup, deg, label, inv, swkw)
        b = _rand(op.in_structure().shape, 7)
        onp.testing.assert_allclose(op.inv(b), ref.inv(b), rtol=1e-9, atol=1e-9)
        onp.testing.assert_allclose(op.transpose().inv(b), ref.transpose().inv(b), rtol=1e-9, atol=1e-9)

    @pytest.mark.slow
    def test_inv_grad_consistency(self, strategy_setup, deg, label, inv, swkw):
        """Gradient of the solve w.r.t. the solution-domain data matches the pivot (guards that an opaque
        sweep kernel still differentiates -- via lx.linear_solve implicit diff)."""
        op, ref = self._op_ref(strategy_setup, deg, label, inv, swkw)
        b = _rand(op.in_structure().shape, 8)
        loss = lambda o, v: jnp.sum(o.inv(v))
        g = eqx.filter_grad(loss)(op, b).solution_domain_data
        g_ref = eqx.filter_grad(loss)(ref, b).solution_domain_data
        assert_pytree_allclose(g, g_ref, rtol=1e-7, atol=1e-7)


# ===========================================================================
# Transport axis -- inversion strategies, multi-angle inverse (fused kernels vs stacked-single pivot)
# ===========================================================================
@pytest.mark.parametrize("deg,label,inv,swkw", _INV_CASES, ids=_INV_IDS)
class TestTransportInversionMultiAngle:

    def _op_ref(self, setup, deg, label, inv, swkw):
        op = setup.variant(deg, ("inv", label), inversion_strategy=inv,
                           sweep_data_kwargs=swkw).transport_operator
        return op, setup.pivot(deg).transport_operator

    def test_inv_adjoint_identity(self, strategy_setup, deg, label, inv, swkw):
        op, _ = self._op_ref(strategy_setup, deg, label, inv, swkw)
        x = _rand(op.in_structure().shape, 9)
        y = _rand(op.out_structure().shape, 10)
        lhs = float(jnp.sum(op.inv(x) * y))
        rhs = float(jnp.sum(x * op.transpose().inv(y)))
        onp.testing.assert_allclose(lhs, rhs, rtol=1e-9, atol=1e-9)

    def test_inv_consistency(self, strategy_setup, deg, label, inv, swkw):
        """Multi-angle inverse matches the vmap-of-single pivot -- for a fused multi-angle kernel
        (e.g. WarpLevelAngles) this is exactly the batching-correctness check."""
        op, ref = self._op_ref(strategy_setup, deg, label, inv, swkw)
        b = _rand(op.in_structure().shape, 11)
        onp.testing.assert_allclose(op.inv(b), ref.inv(b), rtol=1e-9, atol=1e-9)

    @pytest.mark.slow
    def test_inv_grad_consistency(self, strategy_setup, deg, label, inv, swkw):
        op, ref = self._op_ref(strategy_setup, deg, label, inv, swkw)
        b = _rand(op.in_structure().shape, 12)
        loss = lambda o, v: jnp.sum(o.inv(v))
        g = eqx.filter_grad(loss)(op, b).solution_domain_data
        g_ref = eqx.filter_grad(loss)(ref, b).solution_domain_data
        assert_pytree_allclose(g, g_ref, rtol=1e-7, atol=1e-7)
