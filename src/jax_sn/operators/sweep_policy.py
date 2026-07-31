r'''
How an operator obtains its sweep plan.

The sweep order is a combinatorial function of the geometry: it must be rebuilt when the vertices
move, but no gradient may flow through it. Both of those are the policy's business, not the caller's,
which is why this is an object rather than a flag.
'''

import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Any, Tuple

from jax_sn.domain.geometric_domain import GeometricDomain
from .sweep_data import SweepData


__all__ = ["NoSweepData", "SweepPolicy", "Resort", "Reuse", "NoSweep"]


class NoSweepData(SweepData):
    '''
    Placeholder standing in for a sweep plan that was never built (see :class:`NoSweep`).

    Operators that do not sweep -- mass, moment, discrete, scatter -- never read the sweep data and
    never retain it, so building them does not need a topological sort. This makes that explicit and
    fails loudly if something does try to sweep.
    '''

    def _unavailable(self):
        raise RuntimeError(
            "This operator data was built with sweep=NoSweep(), so it carries no topological sort. "
            "Rebuild it with sweep=Resort() (or sweep=Reuse(sweep_data)) if you need to sweep."
        )

    # SweepData declares `levels`/`levels_argsort` abstract: satisfy them with properties that raise,
    # so the placeholder stays leaf-free and any actual read is a clear error rather than a wrong sort.
    @property
    def levels(self) -> jnp.ndarray:
        self._unavailable()

    @property
    def levels_argsort(self) -> jnp.ndarray:
        self._unavailable()

    def __getitem__(self, idx : int):
        self._unavailable()

    @property
    def reverse(self):
        self._unavailable()

    def reconstruct(self, geometric_domain : GeometricDomain, angle : jnp.ndarray, n_iterations_per_loop : int = 100):
        self._unavailable()


class SweepPolicy(eqx.Module):
    '''How to obtain the sweep plan when building an operator.'''

    def build(self, geometric_domain : GeometricDomain, sweep_data_type : type, angles : jnp.ndarray) -> SweepData:
        raise NotImplementedError


class Resort(SweepPolicy):
    '''
    Recompute the topological sort for the current geometry. The default, and the only correct choice
    when the vertices may have moved in a way that reorders elements (any dimension > 1).

    Constructor keywords are passed to the sweep data type's ``from_domain_angle`` (e.g.
    ``n_iterations_per_loop``, ``maximum_number_of_levels``).
    '''

    sweep_data_kwargs : Tuple[Tuple[str, Any], ...] = eqx.field(static=True, default=())

    def __init__(self, **sweep_data_kwargs):
        self.sweep_data_kwargs = tuple(sorted(sweep_data_kwargs.items()))

    def build(self, geometric_domain : GeometricDomain, sweep_data_type : type, angles : jnp.ndarray) -> SweepData:
        # The sort is discrete: cut the gradient here rather than relying on every caller to remember.
        return jax.lax.stop_gradient(
            sweep_data_type.from_domain_angle(geometric_domain, angles, **dict(self.sweep_data_kwargs))
        )


class Reuse(SweepPolicy):
    '''
    Reuse an existing sweep plan instead of recomputing it.

    Valid only when the sweep order cannot have changed -- a 1D geometry (a point face has no
    orientation, so it cannot rotate), or a sensitivity computed at fixed geometry. In any other case
    this silently produces the wrong sweep; see ``tests/solvers/test_parametrisation_toposort.py``.
    '''

    sweep_data : SweepData

    def build(self, geometric_domain : GeometricDomain, sweep_data_type : type, angles : jnp.ndarray) -> SweepData:
        return self.sweep_data


class NoSweep(SweepPolicy):
    '''
    Do not build a sweep plan at all. For operators that never sweep (mass, moment, discrete,
    scatter), so that building them does not pay for a topological sort.
    '''

    def build(self, geometric_domain : GeometricDomain, sweep_data_type : type, angles : jnp.ndarray) -> SweepData:
        return NoSweepData()
