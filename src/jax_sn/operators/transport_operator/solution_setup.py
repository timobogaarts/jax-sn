import equinox as eqx

from jax_sn.solution_domain import SolutionDomainData
from .sweep_data import SweepData


class SolutionSetup(eqx.Module):
    r'''
    The canonical bundle of *precomputed data* an operator family is built on -- the ``setup`` in the
    ``operator = setup + spec + strategy`` decomposition. A composite operator owns exactly one of
    these and derives its sub-operators from it, so shared data lives in a single pytree location.

    The members are split along the **operator-defining / operator-agnostic** axis -- an intrinsic,
    mathematical distinction (independent of whether any particular user differentiates through a
    given array):

    - :attr:`solution_domain_data` -- **operator-defining**: geometry, FEM reference matrices,
      precomputed geometric data, and cross sections. Change any of it and the operator's action
      changes. (Which *subset* of these leaves a parametrisation differentiates is a separate concern,
      chosen at the ``ParametrisedModule.theta_base`` boundary -- not encoded here.)
    - :attr:`sweep_data` -- **operator-agnostic**: the precomputed sweep execution plan (the per-angle
      topological ordering of elements). Its *value* never changes the result -- any valid topological
      order yields the identical inverse -- it only makes the sequential sweep computable/efficient.

    .. note::
        :attr:`sweep_data` currently also carries its ``angle`` (the transport direction), which *is*
        operator-defining and is a cached copy of the quadrature angles. When the angular data
        (quadrature + harmonics) folds into this setup (Stage C), the sweep should read its angle from
        there and carry only the ordering, making ``sweep_data`` purely operator-agnostic.
    '''

    solution_domain_data : SolutionDomainData
    '''Operator-defining data: domain (geometry + cross sections), FEM matrices, geometric data.'''

    sweep_data           : SweepData
    '''Operator-agnostic data: the precomputed sweep execution plan. See :class:`~.sweep_data.SweepData`.'''

    @property
    def domain(self):
        '''Convenience accessor for the geometric domain.'''
        return self.solution_domain_data.domain
