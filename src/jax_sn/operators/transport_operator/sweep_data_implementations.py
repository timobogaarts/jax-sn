r'''
Concrete sweep orderings: the precomputed topological-sort structure required to *invert* (sweep) the
transport operator. The abstract :class:`~jax_sn.operators.sweep_data.SweepData` base lives one level up
(referenced by the shared operator-data bundle); this module holds the concrete plain / level-parallel
sweep data.

Level-parallel data exists to allow for different capabilities: a normal sweep does not require data on
the levels. Since this can be particularly large (on the same order as the base data), it is not given
by default.
'''

import jax.numpy as jnp
import equinox as eqx
from typing import Union

from jax_sn.domain.geometric_domain import GeometricDomain
from jax_sn.quadrature_set import QuadratureSetBase
from jax_sn.tools.graph_tools import multiple_angle_level_bounds_safe
from jax_sn.operators.sweep_data import SweepData, _levels_argsort_from_domain_angles

__all__ = ["SweepData", "BasicSweepData", "LevelBasedParallelism", "ProvidesLevels", "LeveledSweepData"]


class BasicSweepData(SweepData):
    '''
    The plain sweep data for a geometric domain: just the topological sort itself, with no
    precomputed level-parallelism data. Used by the vmap/batched/pallas sweeps. Also the home of the
    factories that build a plain sort from a domain/quadrature set.
    '''

    levels         : jnp.ndarray #[..., n_elements]
    levels_argsort : jnp.ndarray #[..., n_elements]

    @classmethod
    def from_domain_angle(cls, geometric_domain : GeometricDomain, angle : jnp.ndarray, n_iterations_per_loop : int = 100):
        '''
        Computes a (plain) topological sort for the given angle and geometric domain. The ``angle`` is
        used to compute the order but is *not* stored (it lives in the angular setup).

        Can throw if there are cycles using equinox.         

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
        from jax_sn.domain.graph_generation import compute_levels_domain_safe_jax
        levels = compute_levels_domain_safe_jax(geometric_domain, angle, n_iterations_per_loop)
        levels_argsort = jnp.argsort(levels)
        return cls(levels=levels, levels_argsort=levels_argsort)
    
    @classmethod
    def from_domain_quadrature_set(cls, geometric_domain : GeometricDomain, quadrature_set : QuadratureSetBase, n_iterations_per_loop : int = 100):
        '''
        Computes (plain) topological sorts for all angles in the given quadrature set and geometric domain.

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
        levels, levels_argsort = _levels_argsort_from_domain_angles(geometric_domain, quadrature_set.angles, n_iterations_per_loop)
        return cls(levels=levels, levels_argsort=levels_argsort)

    def __getitem__(self, idx : int) -> "BasicSweepData":
        return self._indexed_base(idx)

    @property
    def reverse(self) -> "BasicSweepData":
        return self._reversed_base()

    def reconstruct(self, geometric_domain : GeometricDomain, angle : jnp.ndarray, n_iterations_per_loop : int = 100) -> "BasicSweepData":
        return BasicSweepData(*self._rebuilt_sort(geometric_domain, angle, n_iterations_per_loop))


class LevelBasedParallelism(eqx.Module):
    '''
    Class providing utilities for level-based parallelism in the context of transport sweeps. Stores the first occurrence, end, and size of each level for each angle, based on the sweep data provided by the transport operator.

    Since JAX does not allow dynamic shapes, the class method `from_sweep_data` optionally is given a maximum number of levels, which is used to pad the resulting arrays to a fixed shape. If this is not
    sufficiently high, an equinox error is given.

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
    Mixin for sweep data the provides levels forward and backward.
    '''
    level_fwd : eqx.AbstractVar[LevelBasedParallelism]
    level_rev : eqx.AbstractVar[LevelBasedParallelism]


class LeveledSweepData(SweepData, ProvidesLevels):
    '''
    Sweep data that also provides Levels for LevelBasedParallelism.
    '''

    levels         : jnp.ndarray #[..., n_elements]
    levels_argsort : jnp.ndarray #[..., n_elements]

    level_fwd : LevelBasedParallelism
    '''Level-based-parallelism setup for the (forward) sweep along the angle.'''
    level_rev : LevelBasedParallelism
    '''Level-based-parallelism setup for the reversed sweep (along the negated angle).'''

    @classmethod
    def from_basic_sweep_data(cls, sweep_data : "SweepData", maximum_number_of_levels : Union[int, None] = None) -> "LeveledSweepData":
        '''
        Build leveled sweep data from a plain one, precomputing the level setup for both the forward
        and the reversed sweep.
        '''
        return cls(
            levels         = sweep_data.levels,
            levels_argsort = sweep_data.levels_argsort,
            level_fwd      = LevelBasedParallelism.from_sweep_data(sweep_data, maximum_number_of_levels),
            level_rev      = LevelBasedParallelism.from_sweep_data(sweep_data.reverse, maximum_number_of_levels),
        )

    @classmethod
    def from_domain_quadrature_set(cls, geometric_domain : GeometricDomain, quadrature_set : QuadratureSetBase, maximum_number_of_levels : Union[int, None] = None, n_iterations_per_loop : int = 100) -> "LeveledSweepData":
        '''
        Build leveled sweep data for all angles in a quadrature set directly from a domain (the level
        computation runs once), padding the level arrays to ``maximum_number_of_levels``.        
        '''
        plain = BasicSweepData.from_domain_quadrature_set(geometric_domain, quadrature_set, n_iterations_per_loop)
        return cls.from_basic_sweep_data(plain, maximum_number_of_levels)

    def __getitem__(self, idx : int) -> "LeveledSweepData":
        return eqx.tree_at(lambda t : (t.level_fwd, t.level_rev), self._indexed_base(idx), (self.level_fwd[idx], self.level_rev[idx]))

    @property
    def reverse(self) -> "LeveledSweepData":
        return eqx.tree_at(lambda t : (t.level_fwd, t.level_rev), self._reversed_base(), (self.level_rev, self.level_fwd))

    def reconstruct(self, geometric_domain : GeometricDomain, angle : jnp.ndarray, n_iterations_per_loop : int = 100) -> "LeveledSweepData":
        # Rebuild the plain sort for the new domain, then rebuild the level setup with the same padding
        # (read back from the current level setup) so the recomputed sweep data stays drop-in compatible.
        rebuilt_plain = BasicSweepData(*self._rebuilt_sort(geometric_domain, angle, n_iterations_per_loop))
        return LeveledSweepData.from_basic_sweep_data(rebuilt_plain, self.level_fwd.n_levels)
