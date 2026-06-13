r'''
Sweep data: the precomputed structure required to *invert* (sweep) the transport operator.

The transport operator's forward action is element-local, but its inverse is a directional sweep that
must visit elements in upwind-dependency order. The data describing that traversal -- per angle, the
topological order of elements -- is the :class:`SweepData` family. It is a *sibling* family (abstract
base + final concrete leaves), not a subclass-with-optional-fields, so a half-initialised instance is
never constructible:

- :class:`SweepData`        -- abstract base: the angles + the per-angle topological order.
- :class:`BasicSweepData`   -- just the order. Cheap; serves every non-level sweep.
- :class:`LeveledSweepData` -- the order *plus* the level-based-parallelism setup (a longest-path
  layering of the same sweep DAG) for the fused warp kernels. Large -- O(n_elements x n_angles) -- so
  it is built only when a level sweep is actually selected.

Capabilities
------------
Richer structure that only *some* sweeps need is advertised as a *capability* -- a small interface a
``SweepData`` provides -- rather than as a concrete type. An :class:`InversionStrategy` declares the
capabilities it requires (``required_capabilities``) and accepts *any* ``SweepData`` that provides
them; this is what lets a future richer plan (e.g. levels + domain decomposition) satisfy a
level-only strategy without an enumerated isinstance per combination. See
:class:`ProvidesLevels`.
'''

import jax
import jax.numpy as jnp
import equinox as eqx
from abc import abstractmethod
from typing import Union

from jax_sn.domain.geometric_domain import GeometricDomain
from jax_sn.quadrature_set import QuadratureSetBase
from jax_sn.tools.graph_tools import multiple_angle_level_bounds_safe


class SweepData(eqx.Module):
    '''
    Abstract base for the data required to invert (sweep) the transport operator. Holds the shared
    state -- the element ``levels``, the ``angle``, and the ``levels_argsort`` -- and declares the
    polymorphic interface (``reverse``, ``__getitem__``, ``reconstruct``) as abstract. Concrete
    (final) leaves decide what additional precomputed data, if any, rides along with the sort (see
    :class:`BasicSweepData` and :class:`LeveledSweepData`).

    ``reverse`` / ``__getitem__`` / ``reconstruct`` are abstract: each leaf provides its own final
    implementation so that nothing is overridden (abstract-or-final, never "maybe"). The shared
    mechanical part -- the transform of the three base fields -- lives in the final, non-polymorphic
    helpers :meth:`_reversed_base`, :meth:`_indexed_base` and :meth:`_rebuilt_sort`, which the leaves
    compose with their own fields. All are written with :func:`equinox.tree_at`, so they preserve the
    concrete type.
    '''

    levels         : eqx.AbstractVar[jnp.ndarray] #[..., n_elements]
    '''Levels of the elements in the topological sort. Each level can be computed in parallel.'''

    angle          : eqx.AbstractVar[jnp.ndarray] #[..., n_dim]
    '''The angle for which the topological sort is computed.'''

    levels_argsort : eqx.AbstractVar[jnp.ndarray] #[..., n_elements]
    '''The indices that sort the elements by their levels.'''

    @abstractmethod
    def __getitem__(self, idx : int) -> "SweepData":
        '''
        Indexing for batched instances: get a single (unbatched) sweep data from a batched one.
        Type-preserving -- each leaf returns its own concrete type, slicing its own extra fields.

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
        Reverses the angle and topological sort. This is also correct for batched angles.
        Type-preserving -- each leaf returns its own concrete type, reversing its own extra fields.

        Returns
        -------
        reversed_sweep_data : SweepData
            The reversed sweep data, of the same concrete type as ``self``.
        '''
        ...

    def _reversed_base(self) -> "SweepData":
        '''
        Final helper (not part of the polymorphic interface): returns a copy of ``self`` with only the
        three base fields reversed, leaving any extra leaf fields untouched. Leaves compose this
        with their own field reversal. The levels are reversed by subtracting from the maximum level,
        which is non-optimal but produces a valid topological sort.
        '''
        return eqx.tree_at(
            lambda t : (t.levels, t.angle, t.levels_argsort),
            self,
            (
                jnp.max(self.levels, axis = -1, keepdims=True) - self.levels, # technically not optimal, but produces valid levels [not entirely sure, but in general we don't use this anyway except for level-based parallelism....]
                -1.0 * self.angle,
                self.levels_argsort[..., ::-1],
            ),
        )

    def _indexed_base(self, idx : int) -> "SweepData":
        '''
        Final helper (not part of the polymorphic interface): returns a copy of ``self`` with only the
        three base fields indexed at ``idx``, leaving any extra leaf fields untouched. Leaves
        compose this with their own field slicing.
        '''
        assert self.levels.ndim == 2, "Indexing is only supported for batched SweepData instances."
        return eqx.tree_at(
            lambda t : (t.levels, t.angle, t.levels_argsort),
            self,
            (self.levels[idx], self.angle[idx], self.levels_argsort[idx]),
        )

    @abstractmethod
    def reconstruct(self, geometric_domain : GeometricDomain, n_iterations_per_loop : int = 100) -> "SweepData":
        '''
        Recompute this sweep data for a new (deformed) geometric domain, reusing this instance's own
        stored ``angle`` -- so a parametrisation can rebuild it after moving the vertices without
        re-deriving which angles to use. Type-preserving: each leaf returns its own concrete type and
        rebuilds its own extra data (e.g. a leveled sweep rebuilds its level setup with the same
        padding). Only the (batched) sweep data produced from a quadrature set is supported.

        Returns
        -------
        sweep_data : SweepData
            The recomputed sweep data for ``geometric_domain``, of the same concrete type as ``self``.
        '''
        ...

    def _rebuilt_sort(self, geometric_domain : GeometricDomain, n_iterations_per_loop : int = 100):
        '''
        Final helper (not part of the polymorphic interface): recompute ``(levels, angle, levels_argsort)``
        for this instance's stored angles on a new domain. ``angle`` is unchanged (the directions don't
        move with the vertices). Leaves wrap this into their own concrete type.
        '''
        assert self.angle.ndim == 2, "reconstruct is only supported for batched (quadrature-set) sweep data."
        levels, levels_argsort = _levels_argsort_from_domain_angles(geometric_domain, self.angle, n_iterations_per_loop)
        return levels, self.angle, levels_argsort


class BasicSweepData(SweepData):
    '''
    The plain sweep data for a geometric domain: just the topological sort itself, with no
    precomputed level-parallelism data. Used by the vmap/batched/pallas sweeps. Also the home of the
    factories that build a plain sort from a domain/quadrature set.
    '''

    levels         : jnp.ndarray #[..., n_elements]
    angle          : jnp.ndarray #[..., n_dim]
    levels_argsort : jnp.ndarray #[..., n_elements]

    @classmethod
    def from_domain_angle(cls, geometric_domain : GeometricDomain, angle : jnp.ndarray, n_iterations_per_loop : int = 100):
        '''
        Computes a (plain) topological sort for the given angle and geometric domain.
        This function is not jittable because it uses Python control flow to safely compute levels, which can throw if there is a cycle in the graph.

        Parameters
        ----------
        geometric_domain : GeometricDomain
            The geometric domain for which to compute the topological sort.
        angle : jnp.ndarray [n_dim]
            The angle for which to compute the topological sort.
        n_iterations_per_loop : int, optional
            The number of iterations per loop for the level computation. The default is 100.

        Returns
        -------
        sweep_data : BasicSweepData
            The computed sweep data.
        '''
        from jax_sn.domain.graph_generation import compute_levels_domain_safe, compute_levels_domain_safe_jax
        levels = compute_levels_domain_safe_jax(geometric_domain, angle, n_iterations_per_loop)
        levels_argsort = jnp.argsort(levels)
        return cls(levels=levels, angle=angle, levels_argsort=levels_argsort)

    @classmethod
    def from_domain_quadrature_set(cls, geometric_domain : GeometricDomain, quadrature_set : QuadratureSetBase, n_iterations_per_loop : int = 100):
        '''
        Computes (plain) topological sorts for all angles in the given quadrature set and geometric domain.
        This function is not jittable because it uses Python control flow to safely compute levels, which can throw if there is a cycle in the graph.

        Parameters
        ----------
        geometric_domain : GeometricDomain
            The geometric domain for which to compute the topological sorts.
        quadrature_set : QuadratureSetBase
            The quadrature set containing the angles for which to compute the topological sorts.
        n_iterations_per_loop : int, optional
            The number of iterations per loop for the level computation. The default is 100.
        Returns
        -------
        sweep_data : BasicSweepData
            The computed sweep data for all angles in the quadrature set. Batched over angles.
        '''
        return cls(*_sweep_data_from_domain_quadrature_set_scan(geometric_domain, quadrature_set, n_iterations_per_loop))

    def __getitem__(self, idx : int) -> "BasicSweepData":
        return self._indexed_base(idx)

    @property
    def reverse(self) -> "BasicSweepData":
        return self._reversed_base()

    def reconstruct(self, geometric_domain : GeometricDomain, n_iterations_per_loop : int = 100) -> "BasicSweepData":
        return BasicSweepData(*self._rebuilt_sort(geometric_domain, n_iterations_per_loop))


@eqx.filter_jit
def _sweep_data_from_domain_quadrature_set_scan(geometric_domain : GeometricDomain, quadrature_set : QuadratureSetBase, n_iterations_per_loop : int = 100):
    '''
    Jittable version of BasicSweepData.from_domain_quadrature_set.
    Computes topological sorts for all angles in the given quadrature set and geometric domain.

    Parameters
    ----------
    geometric_domain : GeometricDomain
        The geometric domain for which to compute the topological sorts.
    quadrature_set : QuadratureSetBase
        The quadrature set containing the angles for which to compute the topological sorts.
    n_iterations_per_loop : int, optional
        The number of iterations per loop for the level computation. The default is 100.
    Returns
    -------
    (levels, angles, levels_argsort)
        The computed sweep data fields for all angles in the quadrature set. Batched over angles.
    '''
    levels, levels_argsort = _levels_argsort_from_domain_angles(geometric_domain, quadrature_set.angles, n_iterations_per_loop)
    return levels, quadrature_set.angles, levels_argsort


@eqx.filter_jit
def _levels_argsort_from_domain_angles(geometric_domain : GeometricDomain, angles : jnp.ndarray, n_iterations_per_loop : int = 100):
    '''
    Compute the (levels, levels_argsort) for a batch of angles on a domain by scanning the safe level
    computation over the angle axis. Shared by ``from_domain_quadrature_set`` (angles from the
    quadrature set) and ``SweepData.reconstruct`` (the instance's own stored angles).
    '''
    from jax_sn.domain.graph_generation import compute_levels_domain_safe_jax

    def body_fn(carry, angle):
        levels = compute_levels_domain_safe_jax(geometric_domain, angle, n_iterations_per_loop)
        return carry, levels

    levels = jax.lax.scan(body_fn, None, angles)[1] #[n_angles, n_elements]
    levels_argsort = jnp.argsort(levels, axis=-1) #[n_angles, n_elements]
    return levels, levels_argsort


class LevelBasedParallelism(eqx.Module):
    '''
    Class providing utilities for level-based parallelism in the context of transport sweeps. Stores the first occurrence, end, and size of each level for each angle, based on the sweep data provided by the transport operator.

    Since JAX does not allow dynamic shapes, the class method `from_sweep_data` optionally is given a maximum number of levels, which is used to pad the resulting arrays to a fixed shape. If this is not
    sufficiently high enough, an equinox error is given.

    '''
    first_occurences : jnp.ndarray
    '''
    First occurrence of each level, per angle. Shape [..., max_levels]
    '''
    ends : jnp.ndarray
    '''
    End of each level (i.e. start of next level), per angle. Shape [..., max_levels]
    '''
    sizes : jnp.ndarray
    '''
    Number of elements in each level, per angle. Shape [..., max_levels]
    '''

    @classmethod
    def from_sweep_data(cls, sweep_data : "SweepData", maximum_number_of_levels : Union[int, None] = None):
        '''
        Create a LevelBasedParallelism instance from a SweepData, which contains the levels and their argsort for each angle. Optionally, a maximum number of levels can be provided to pad the resulting arrays to a fixed shape. This has
        the advantage that array sizes are known at compile time. If not sufficiently high enough, it will raise an equinox error.

        Parameters
        ----------
        sweep_data : SweepData
            The sweep data containing the levels and their argsort for each angle.
        maximum_number_of_levels : int, optional
            The maximum number of levels to pad to. If None, it will be set to the maximum number of levels present in the sweep data. Default is None.

        Returns
        -------
        LevelBasedParallelism
            An instance of LevelBasedParallelism containing the first occurrence, end, and size of each level for each angle, based on the sweep data provided by the transport operator.
        '''
        first_occurences, ends, sizes = multiple_angle_level_bounds_safe(sweep_data.levels, sweep_data.levels_argsort, maximum_number_of_levels)
        return cls(first_occurences=first_occurences, ends=ends, sizes=sizes)

    @property
    def n_levels(self):
        '''
        Number of distinct levels
        '''
        return self.first_occurences.shape[-1]

    def __getitem__(self, idx : int) -> "LevelBasedParallelism":
        return eqx.tree_at(
            lambda lbp: (lbp.first_occurences, lbp.ends, lbp.sizes),
            self,
            (self.first_occurences[idx], self.ends[idx], self.sizes[idx])
        )


class ProvidesLevels(eqx.Module):
    '''
    Capability: a :class:`SweepData` that carries the level-based-parallelism setup for both sweep
    directions (``level_fwd`` for the angle, ``level_rev`` for the reversed angle) -- a longest-path
    layering of the sweep DAG, consumed by the fused (warp) level sweeps.

    A strategy that needs level parallelism declares ``required_capabilities = (ProvidesLevels,)`` and
    accepts *any* ``SweepData`` that also inherits this capability, rather than a single concrete type.
    This is an abstract equinox module (``AbstractVar`` fields): any concrete ``SweepData`` that
    provides levels inherits it and supplies ``level_fwd`` / ``level_rev`` as real fields.
    '''
    level_fwd : eqx.AbstractVar[LevelBasedParallelism]
    level_rev : eqx.AbstractVar[LevelBasedParallelism]


class LeveledSweepData(SweepData, ProvidesLevels):
    '''
    Sweep data that additionally carries the level-based-parallelism setup for both sweep directions
    (``level_fwd`` for this angle, ``level_rev`` for the reversed angle), i.e. it provides the
    :class:`ProvidesLevels` capability.

    Both directions are precomputed and stored because building a :class:`LevelBasedParallelism`
    is a non-jittable numpy step, so it cannot be recomputed inside a (jitted) ``reverse``. As a
    result ``reverse`` simply swaps the two, and ``__getitem__`` slices both -- composing the shared
    base-field helpers with its own field handling rather than reimplementing the base transform.
    '''

    levels         : jnp.ndarray #[..., n_elements]
    angle          : jnp.ndarray #[..., n_dim]
    levels_argsort : jnp.ndarray #[..., n_elements]

    level_fwd : LevelBasedParallelism
    '''Level-based-parallelism setup for the (forward) sweep along ``angle``.'''
    level_rev : LevelBasedParallelism
    '''Level-based-parallelism setup for the reversed sweep (along ``-angle``).'''

    @classmethod
    def from_basic_sweep_data(cls, sweep_data : "SweepData", maximum_number_of_levels : Union[int, None] = None) -> "LeveledSweepData":
        '''
        Build leveled sweep data from a plain one, precomputing the level setup for both the forward
        and the reversed sweep.
        '''
        return cls(
            levels         = sweep_data.levels,
            angle          = sweep_data.angle,
            levels_argsort = sweep_data.levels_argsort,
            level_fwd      = LevelBasedParallelism.from_sweep_data(sweep_data, maximum_number_of_levels),
            level_rev      = LevelBasedParallelism.from_sweep_data(sweep_data.reverse, maximum_number_of_levels),
        )

    @classmethod
    def from_domain_quadrature_set(cls, geometric_domain : GeometricDomain, quadrature_set : QuadratureSetBase, maximum_number_of_levels : Union[int, None] = None, n_iterations_per_loop : int = 100) -> "LeveledSweepData":
        '''
        Build leveled sweep data for all angles in a quadrature set directly from a domain (the level
        computation runs once), padding the level arrays to ``maximum_number_of_levels``. Mirrors
        :meth:`BasicSweepData.from_domain_quadrature_set` so creation helpers can build either sweep
        data type polymorphically from ``InversionStrategy.default_sweep_data_type``.
        '''
        plain = BasicSweepData.from_domain_quadrature_set(geometric_domain, quadrature_set, n_iterations_per_loop)
        return cls.from_basic_sweep_data(plain, maximum_number_of_levels)

    def __getitem__(self, idx : int) -> "LeveledSweepData":
        return eqx.tree_at(lambda t : (t.level_fwd, t.level_rev), self._indexed_base(idx), (self.level_fwd[idx], self.level_rev[idx]))

    @property
    def reverse(self) -> "LeveledSweepData":
        return eqx.tree_at(lambda t : (t.level_fwd, t.level_rev), self._reversed_base(), (self.level_rev, self.level_fwd))

    def reconstruct(self, geometric_domain : GeometricDomain, n_iterations_per_loop : int = 100) -> "LeveledSweepData":
        # Rebuild the plain sort for the new domain, then rebuild the level setup with the same padding
        # (read back from the current level setup) so the recomputed sweep data stays drop-in compatible.
        rebuilt_plain = BasicSweepData(*self._rebuilt_sort(geometric_domain, n_iterations_per_loop))
        return LeveledSweepData.from_basic_sweep_data(rebuilt_plain, self.level_fwd.n_levels)
