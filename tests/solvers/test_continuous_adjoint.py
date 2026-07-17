r'''
Tests for the continuous adjoint of the multi-group operator (:mod:`jax_sn.variance_reduction`).

The continuous adjoint :math:`\mathbf{A}^\dagger` negates the streaming direction but keeps
:math:`\mathbf{M}, \mathbf{D}` at :math:`+\Omega` and gathers the transposed group scatter. It is *not*
the algebraic transpose :math:`\mathbf{A}^T`.

Two tests:

1. ``test_continuous_adjoint_structural`` -- the *primary*, exact gate. The data-transform used to build
   :math:`\mathbf{A}^\dagger` (negate angles + reverse sweep + transpose group scatter, run the forward
   strategy) must match an independent explicit assembly
   :math:`\mathbf{M}(\mathbf{T}^\dagger)^{-1}\mathbf{H}\mathbf{D}\,\mathbf{U}^\dagger` built from the
   *original* data via ``transport.transpose()`` + ``outgoing_gather``. Pure ``mv`` (no solver), random
   anisotropic input, every dimension -> tight tolerance.

2. ``test_continuous_adjoint_reciprocity`` -- a *secondary*, 1D physical smoke check that response
   reciprocity :math:`\langle\psi^\dagger, q\rangle \approx \langle\psi, q^\dagger\rangle` holds. Loose
   by construction (discrete adjoint != adjoint of the discrete operator; quadrature-point line integral;
   noisy random source).
'''

import jax.numpy as jnp
import numpy as onp
import pytest
import lineax as lx
import equinox as eqx

from jax_sn.operators.single_group_operator import SingleGroupData
from jax_sn.operators.scatter_operator import GatherScatterData
from jax_sn.solvers.right_hand_side import TotalSource, solve_multigroup_operator_rhs
from jax_sn.variance_reduction.continuous_adjoint import solve_continuous_adjoint, _continuous_adjoint_data


def test_continuous_adjoint_structural(bb_mgop):
    r'''
    Exact: :math:`\mathbf{A}^\dagger` built by the data transform equals the explicit adjoint assembly.

    The reference computes, per group,
    :math:`\psi_g - \mathbf{M}(\mathbf{T}_g^\dagger)^{-1}\mathbf{H}\mathbf{D}\sum_{g'}\mathbf{S}_{g'g}\psi_{g'}`
    from the *original* operator data, using ``transport.transpose()`` (which negates the angle + reverses
    the sweep) and ``outgoing_gather``. This is an independent code path from the data-transform trick, so
    agreement to float64 tolerance pins down that the transform is correct (in particular the scatter-axis
    ``swapaxes`` and the sweep reversal).
    '''
    mgop, _, _, _ = bb_mgop

    # data-transform path: the ordinary forward operator run on adjoint-transformed data.
    adjoint_op = eqx.tree_at(lambda o: o.data, mgop, _continuous_adjoint_data(mgop.data))

    onp.random.seed(11072026)
    x = jnp.asarray(onp.random.random(mgop.in_structure().shape))
    got = adjoint_op.mv(x)

    # independent reference on the ORIGINAL data.
    chain   = mgop.strategy.single_group_chain_strategy
    gather  = mgop.strategy.single_group_gather_scatter_strategy
    gs_data = GatherScatterData(mgop.data.domain.cross_sections_indexed)

    def reference_body(g):
        transport, moment, discrete, mass, _ = chain.all_operators(SingleGroupData(mgop.data, g))
        scatter_source = gather.outgoing_gather(gs_data, x, g)  # sum_{g'} S_{g'g} x_{g'}
        return x[g] - moment.mv(transport.transpose().inv(discrete.mv(mass.mv(scatter_source))))

    expected = jnp.stack([reference_body(g) for g in range(mgop.n_energy_groups)])

    onp.testing.assert_allclose(onp.asarray(got), onp.asarray(expected), rtol=1e-9, atol=1e-9)


def test_continuous_adjoint_reciprocity(bb_mgop):
    r'''
    End-to-end physical reciprocity in 1D: :math:`\langle\psi^\dagger, q\rangle == \langle\psi, q^\dagger\rangle`,
    computed as a 1D line integral (interpolated), not a full phase-space integral. Isotropic sources =>
    only the scalar flux (moment 0) contributes.

    Loose tolerance by construction (see module docstring). This is a smoke check that the pipeline is
    wired up and gives a non-trivial, reciprocal result -- the exact gate is the structural test.
    '''
    mgop, _, solution_domain, _ = bb_mgop

    if solution_domain.dimension != 1:
        pytest.skip("Continuous adjoint reciprocity test is 1D-only (line integral, not a full space integral).")

    onp.random.seed(25032026)
    source_shape_isotropic = (mgop.n_energy_groups, solution_domain.n_elements, solution_domain.n_basis)
    source_total        = onp.random.random(source_shape_isotropic)
    source_dagger_total = onp.random.random(source_shape_isotropic)

    rhs        = TotalSource.from_operator_and_isotropic_source(mgop, jnp.asarray(source_total        * onp.sqrt(onp.pi * 4)))
    rhs_dagger = TotalSource.from_operator_and_isotropic_source(mgop, jnp.asarray(source_dagger_total * onp.sqrt(onp.pi * 4)))

    solver = lx.BiCGStab(rtol=1e-12, atol=1e-15, max_steps=160)

    solution         = solve_multigroup_operator_rhs(mgop, rhs, solver, mgop.strategy.single_group_chain_strategy)
    solution_adjoint = solve_continuous_adjoint(mgop, rhs_dagger, solver)

    line = jnp.linspace(0, jnp.max(mgop.data.spatial_data.domain.vertices), 800)[:, None]
    result               = solution_domain.interpolate(line, solution.value[:, 0])
    result_adjoint       = solution_domain.interpolate(line, solution_adjoint.value[:, 0])
    source_interp        = solution_domain.interpolate(line, jnp.asarray(source_total))
    source_interp_dagger = solution_domain.interpolate(line, jnp.asarray(source_dagger_total))

    psidagger_q = jnp.sum(result_adjoint * source_interp)
    psi_qdagger = jnp.sum(result * source_interp_dagger)

    onp.testing.assert_allclose(onp.array(psidagger_q), onp.array(psi_qdagger), rtol=1e-2, atol=1e-2)
