r'''
Non-differentiable sweep-execution data: the abstract interface the shared operator-data bundle
'''

import jax
import jax.numpy as jnp
import equinox as eqx
from abc import abstractmethod

from jax_sn.domain.geometric_domain import GeometricDomain
from jax_sn.operators.base_linear_operator import BaseOperatorData


class SweepData(BaseOperatorData):
    '''
    Abstract base for the data required to invert (sweep) the transport operator.
    '''

    levels         : eqx.AbstractVar[jnp.ndarray] #[..., n_elements]
    '''Levels of the elements in the topological sort. Each level can be computed in parallel.'''

    levels_argsort : eqx.AbstractVar[jnp.ndarray] #[..., n_elements]
    '''The indices that sort the elements by their levels.'''

    @abstractmethod
    def __getitem__(self, idx : int) -> "SweepData":
        '''
        Indexing for batched instances: get a single (unbatched) sweep data from a batched one.

        Parameters
        ----------
        idx : int
            The index of the angle to get.
        Returns
        -------
        sweep_data : SweepData
            The instance for the given index, of the same concrete type as ``self``.
        '''
        ...

    @property
    @abstractmethod
    def reverse(self) -> "SweepData":
        '''
        Reverses the topological sort (the partner of negating the angle, done on the operator). This
        is also correct for batched data.

        Returns
        -------
        reversed_sweep_data : SweepData
            The reversed sweep data, of the same concrete type as ``self``.
        '''
        ...

    def _reversed_base(self) -> "SweepData":
        '''
        Final helper (not part of the polymorphic interface): returns a copy of ``self`` with only the
        base fields reversed, leaving any extra leaf fields untouched. Leaves compose this with their
        own field reversal. The levels are reversed by subtracting from the maximum level, which is
        non-optimal but produces a valid topological sort.
        '''
        return eqx.tree_at(
            lambda t : (t.levels, t.levels_argsort),
            self,
            (
                jnp.max(self.levels, axis = -1, keepdims=True) - self.levels, # technically not optimal, but produces valid levels [not entirely sure, but in general we don't use this anyway except for level-based parallelism....]
                self.levels_argsort[..., ::-1],
            ),
        )

    def _indexed_base(self, idx : int) -> "SweepData":
        '''
        Final helper (not part of the polymorphic interface): returns a copy of ``self`` with only the
        base fields indexed at ``idx``, leaving any extra leaf fields untouched. Leaves compose this
        with their own field slicing.
        '''
        assert self.levels.ndim == 2, "Indexing is only supported for batched SweepData instances."
        return eqx.tree_at(
            lambda t : (t.levels, t.levels_argsort),
            self,
            (self.levels[idx], self.levels_argsort[idx]),
        )

    @abstractmethod
    def reconstruct(self, geometric_domain : GeometricDomain, angle : jnp.ndarray, n_iterations_per_loop : int = 100) -> "SweepData":
        '''
        Recompute this sweep data for a new (deformed) geometric domain and the given ``angle`` -- so a
        parametrisation can rebuild it after moving the vertices. Type-preserving: each leaf returns its
        own concrete type and rebuilds its own extra data (e.g. a leveled sweep rebuilds its level setup
        with the same padding).

        Parameters
        ----------
        geometric_domain : GeometricDomain
            The deformed domain to recompute the sort for.
        angle : jnp.ndarray [n_angles, n_dim]
            The angles the sort is for.

        Returns
        -------
        sweep_data : SweepData
            The recomputed sweep data for ``geometric_domain``, of the same concrete type as ``self``.
        '''
        ...

    def _rebuilt_sort(self, geometric_domain : GeometricDomain, angle : jnp.ndarray, n_iterations_per_loop : int = 100):
        '''
        Final helper (not part of the polymorphic interface): recompute ``(levels, levels_argsort)`` for
        the given ``angle`` on a new domain. Leaves wrap this into their own concrete type.
        '''
        assert angle.ndim == 2, "reconstruct is only supported for batched (quadrature-set) angles."
        levels, levels_argsort = _levels_argsort_from_domain_angles(geometric_domain, angle, n_iterations_per_loop)
        return levels, levels_argsort


@eqx.filter_jit
def _levels_argsort_from_domain_angles(geometric_domain : GeometricDomain, angles : jnp.ndarray, n_iterations_per_loop : int = 100):
    '''
    Compute the (levels, levels_argsort) for a batch of angles on a domain by scanning the safe level
    computation over the angle axis. Shared by ``BasicSweepData.from_domain_quadrature_set`` (angles from
    the quadrature set) and ``SweepData.reconstruct`` (the instance's own stored angles).
    '''
    from jax_sn.domain.graph_generation import compute_levels_domain_safe_jax

    def body_fn(carry, angle):
        levels = compute_levels_domain_safe_jax(geometric_domain, angle, n_iterations_per_loop)
        return carry, levels

    levels = jax.lax.scan(body_fn, None, angles)[1] #[n_angles, n_elements]
    levels_argsort = jnp.argsort(levels, axis=-1) #[n_angles, n_elements]
    return levels, levels_argsort
