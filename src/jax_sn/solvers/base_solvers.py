r'''
Direct block-substitution solver for lower-block-triangular (downscattering-only) multigroup systems.

Design
------
The system is solved directly by block forward substitution over energy groups, one single-group solve
per group. Two things are supplied *separately* and play distinct roles:

- ``operator`` (the :class:`MultiGroupOperator` passed to ``lx.linear_solve``) is the **differentiation
  surface** only. lineax differentiates ``linear_solve`` implicitly through ``operator.mv`` /
  ``operator.transpose()`` -- the solver's ``compute`` below is *not* a differentiation surface -- so the
  operator may be an arbitrary (even fused) differentiable kernel.
- the strategies carried by the solver (``diagonal_strategy``, ``chain_strategy``, ``gather``) plus the
  inner ``single_group_solver`` define the **value** of the direct solve. They are decoupled from
  ``operator.strategy`` on purpose: the diagonal inverse ``A_g^{-1}`` may be a non-differentiable / fused
  single-group solver kernel, expressed simply as the ``single_group_solver`` (a fused direct inverse
  *is* an ``lx.AbstractLinearSolver``).

The solver holds the three single-group operators it genuinely needs, each at its abstract type:

- ``diagonal_strategy``  -- the full block ``A_g = I - M T_g^{-1} H D S_gg`` (diagonal solve),
- ``chain_strategy``     -- the chain ``C_g = M T_g^{-1} H D`` alone (off-diagonal scatter source),
- ``gather``             -- the per-group scatter gather ``U_g`` / ``V_g``.

The full block and the chain overlap in the composed case (they share the same sub-strategies), but that
redundancy is resolved once, in :meth:`MultiGroupLowerTriangularBlockSolver.from_composed`, not by
reconstructing one from the other inside the solver.

Consistency is the caller's responsibility: the blocks the substitution builds must be the same linear
operator ``operator`` represents, else the (implicit) gradient describes a different system than the
computed value.
'''

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, ArrayLike
from typing_extensions import Self

from jax_sn.operators.multi_group_operator import MultiGroupOperator
from jax_sn.operators.multi_group_operator.composed_multi_group_strategies import ComposedMultiGroupStrategy
from jax_sn.operators.single_group_operator import (
    SingleGroupData,
    SingleGroupOperator,
    SingleGroupStrategy,
    SingleGroupChainStrategy,
    ComposedSingleGroupStrategy,
    ComposedSingleGroupChainStrategy,
)
from jax_sn.operators.scatter_operator import GatherScatterData, SingleGroupGatherScatterStrategy
import equinox as eqx

# lineax solver state carried between init/compute/transpose is just the (differentiable) operator.
_State = MultiGroupOperator
_Solution = Tuple[Array, lx.RESULTS, Dict[str, Array]]


class SolverDiagnostics(eqx.Module):
    print_sweep : bool 
    print_single_group : bool



class MultiGroupLowerTriangularBlockSolver(lx.AbstractLinearSolver[_State]):
    r'''
    Block forward-substitution solver for a lower-block-triangular (downscattering-only) multigroup
    operator. Both directions are implemented, as implicit differentiation requires the transpose solve:
    the primal is forward substitution over ascending groups; its transpose is an upper-triangular back
    substitution over descending groups, carrying the per-group intermediate ``zeta_g = C_g^T x_g``.
    '''

    single_group_solver : lx.AbstractLinearSolver
    diagonal_strategy   : SingleGroupStrategy
    chain_strategy      : SingleGroupChainStrategy
    gather              : SingleGroupGatherScatterStrategy
    diagnostics         : SolverDiagnostics = SolverDiagnostics(print_sweep=False, print_single_group=False)

    # ---- construction ----------------------------------------------------------------------------
    @classmethod
    def from_composed(cls, composed_strategy: ComposedMultiGroupStrategy,
                      single_group_solver: lx.AbstractLinearSolver, diagnostics : SolverDiagnostics = SolverDiagnostics(print_sweep=False, print_single_group=False)) -> Self:
        r'''
        Build the solver from a ComposedMultiGroupStrategy with a ComposedSingleGroupChainStrategy. The diagonal block is assembled from the chain's sub-strategies.
        '''
        chain = composed_strategy.single_group_chain_strategy
        assert isinstance(chain, ComposedSingleGroupChainStrategy), (
            "from_composed needs a ComposedSingleGroupChainStrategy to assemble the diagonal block; "
            f"got {type(chain).__name__}. Construct the solver directly with an explicit diagonal_strategy."
        )
        diagonal = ComposedSingleGroupStrategy(
            transport_strategy           = chain.transport_strategy,
            transport_inversion_strategy = chain.transport_inversion_strategy,
            moment_strategy              = chain.moment_strategy,
            discrete_strategy            = chain.discrete_strategy,
            mass_strategy                = chain.mass_strategy,
            scatter_strategy             = chain.scatter_strategy,
        )
        return cls(
            single_group_solver = single_group_solver,
            diagonal_strategy   = diagonal,
            chain_strategy      = chain,
            gather              = composed_strategy.single_group_gather_scatter_strategy,
            diagnostics         = diagnostics,
        )

    # ---- lineax plumbing -------------------------------------------------------------------------
    def init(self, operator: MultiGroupOperator, options: Dict[str, Any]) -> _State:
        assert isinstance(operator, MultiGroupOperator), \
            f"{type(self).__name__} only solves MultiGroupOperator systems, got {type(operator)}."
        # The downscattering-only (lower-block-triangular) precondition is value-dependent, so it cannot be
        # asserted here -- lineax jits `init`, making the scattering data a tracer. It is the caller's
        # responsibility (see module docstring); the block substitution simply ignores upper blocks.
        return operator                                    # solver state = the differentiable operator

    def transpose(self, state: _State, options: Dict[str, Any]) -> Tuple[_State, Dict[str, Any]]:
        return state.transpose(), options                  # compute then solves the transposed system

    def conj(self, state: _State, options: Dict[str, Any]) -> Tuple[_State, Dict[str, Any]]:
        return state.conj(), options                       # real operators: conj is identity

    def assume_full_rank(self) -> bool:
        return True                                         # nonsingular diagonal blocks -> full rank

    # ---- the diagonal solve (shared by both directions; value only, never differentiated) --------
    def _solve_diagonal(self, operator: MultiGroupOperator, g: ArrayLike, src: Array) -> Tuple[Array, Array]:
        r'''Invert the diagonal block ``A_g`` (or ``A_g^T``, following ``operator``) on ``src``, guarding
        the all-zero source: BiCGStab breaks on a zero rhs, which arises for adjoint sources with groups
        the objective does not depend on.'''
        index_dtype = operator.data.domain.cross_sections_indexed.indices.dtype
        block = SingleGroupOperator(SingleGroupData(operator.data, g), self.diagonal_strategy, operator.transpose_bool)

        def zero(s: Array) -> Tuple[Array, Array]:
            return jnp.zeros_like(s), jnp.array(0, index_dtype)

        def nonzero(s: Array) -> Tuple[Array, Array]:
            out = lx.linear_solve(block, s, self.single_group_solver, throw=False, options={"y0": s})
            return out.value, out.stats["num_steps"].astype(index_dtype)

        return jax.lax.cond(jnp.all(src == 0.0), zero, nonzero, src)

    def _finalize(self, sol: Array, nsteps: Array) -> _Solution:
        result = lx.RESULTS.where(jnp.all(nsteps < self.single_group_solver.max_steps),
                                  lx.RESULTS.successful, lx.RESULTS.max_steps_reached)
        return sol, result, {"num_steps_per_group": nsteps}

    # ---- the two directions ----------------------------------------------------------------------
    def _solve_forward(self, operator: MultiGroupOperator, rhs: Array) -> _Solution:
        r'''Forward substitution, groups ascending: ``A_g x_g = rhs_g + C_g[ U_g(x) ]``. The incoming
        gather over the partial solution only picks up already-solved lower groups (upper groups, and the
        diagonal itself, are still zero), so the visiting order alone enforces the lower-triangular
        structure.'''
        n = operator.n_energy_groups
        index_dtype = operator.data.domain.cross_sections_indexed.indices.dtype
        gs = GatherScatterData(operator.data.domain.cross_sections_indexed)

        if self.diagnostics.print_single_group:
            jax.debug.print("Using lower triangular block solver for multi-group operator with {n_groups} energy groups.", n_groups = operator.n_energy_groups)


        def body(g: ArrayLike, state: Tuple[Array, Array]) -> Tuple[Array, Array]:
            sol, nsteps = state
            source = self.gather.incoming_gather(gs, sol, g)
            src = rhs[g] + self.chain_strategy.forward(SingleGroupData(operator.data, g), source)
            val, ns = self._solve_diagonal(operator, g, src)

            if self.diagnostics.print_single_group:
                jax.debug.print("Solved energy group {eg} in {n_steps} steps", eg = g, n_steps = ns)

            return sol.at[g].set(val), nsteps.at[g].set(ns)

        sol = jnp.zeros(operator.out_structure().shape, operator.out_structure().dtype)
        sol, nsteps = jax.lax.fori_loop(0, n, body, (sol, jnp.zeros(n, index_dtype)))
        return self._finalize(sol, nsteps)

    def _solve_transpose(self, operator: MultiGroupOperator, rhs: Array) -> _Solution:
        r'''Back substitution, groups descending: ``A_g^T x_g = rhs_g + V_g(zeta)`` with
        ``zeta_g = C_g^T x_g``. Unlike the forward pass the gather runs over the per-group transposed-chain
        intermediate ``zeta`` rather than over ``x`` directly, so ``zeta`` is carried through the loop.'''
        n = operator.n_energy_groups
        index_dtype = operator.data.domain.cross_sections_indexed.indices.dtype
        gs = GatherScatterData(operator.data.domain.cross_sections_indexed)

        if self.diagnostics.print_single_group:
            jax.debug.print("Using upper triangular transpose solver for multi-group operator with {n_groups} energy groups.", n_groups = operator.n_energy_groups)

        def body(i: ArrayLike, state: Tuple[Array, Array, Array]) -> Tuple[Array, Array, Array]:
            sol, zeta, nsteps = state
            g = n - 1 - i
            src = rhs[g] + self.gather.outgoing_gather(gs, zeta, g)
            val, ns = self._solve_diagonal(operator, g, src)
            zeta_g = self.chain_strategy.transpose(SingleGroupData(operator.data, g), val)

            if self.diagnostics.print_single_group:
                jax.debug.print("Solved energy group {eg} in {n_steps} steps", eg = g, n_steps = ns)

            return sol.at[g].set(val), zeta.at[g].set(zeta_g), nsteps.at[g].set(ns)

        sol = jnp.zeros(operator.out_structure().shape, operator.out_structure().dtype)
        sol, _, nsteps = jax.lax.fori_loop(0, n, body, (sol, jnp.zeros_like(sol), jnp.zeros(n, index_dtype)))
        return self._finalize(sol, nsteps)

    def compute(self, state: _State, rhs: Array, options: Dict[str, Any]) -> _Solution:
        return self._solve_transpose(state, rhs) if state.transpose_bool else self._solve_forward(state, rhs)
