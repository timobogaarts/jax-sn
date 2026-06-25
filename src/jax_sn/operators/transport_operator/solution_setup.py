import equinox as eqx
import jax.numpy as jnp

from jax_sn.solution_domain import SolutionDomainData
from jax_sn.quadrature_set import QuadratureSetBase
from .sweep_data import SweepData


class AngularSetup(eqx.Module):
    r'''
    The precomputed angular discretisation: the quadrature set together with the spherical-harmonic
    matrices that the moment / discrete operators contract against. This is the **single home for the
    angular data** -- the moment and discrete operators are derived views onto :attr:`wk_ylm` /
    :attr:`ylm` rather than each storing their own harmonics.

    All of it is **operator-defining** (it sets the angular action), and non-differentiable: the
    harmonics are fixed angular basis functions, recomputed only if the quadrature changes.
    '''

    quadrature_set : QuadratureSetBase
    '''The angular quadrature set (angles + weights).'''

    wk_ylm         : jnp.ndarray
    r'''Weighted harmonics :math:`w_k Y_{l,m}(\Omega_k)` -- the moment operator's matrix. [n_angles, n_moments]'''

    ylm            : jnp.ndarray   
    r'''Unweighted harmonics :math:`Y_{l,m}(\Omega_k)` -- the discrete operator's matrix. [n_angles, n_moments]'''

    @classmethod
    def from_quadrature_set(cls, quadrature_set : QuadratureSetBase, l_max : int) -> "AngularSetup":
        '''Precompute the harmonics up to moment order ``l_max`` for the given quadrature set.'''
        return cls(quadrature_set, quadrature_set.wk_ylm_real(l_max), quadrature_set.ylm_real(l_max))


class SolutionSetup(eqx.Module):
    r'''
    The canonical bundle of *precomputed data* an operator family is built on -- the ``setup`` in the
    ``operator = setup + spec + strategy`` decomposition. A composite operator owns exactly one of
    these and derives **every** sub-operator from it, so shared data lives in a single pytree location.

    The members are split along the **operator-defining / operator-agnostic** axis -- an intrinsic,
    mathematical distinction (independent of whether any particular user differentiates through a
    given array):

    - :attr:`solution_domain_data` -- **operator-defining** (spatial): geometry, FEM reference
      matrices, precomputed geometric data, and cross sections.
    - :attr:`angular_setup` -- **operator-defining** (angular): quadrature + precomputed harmonics.
    - :attr:`sweep_data` -- **operator-agnostic**: the precomputed sweep execution plan (the per-angle
      topological ordering of elements). Its *value* never changes the result -- any valid topological
      order yields the identical inverse -- it only makes the sequential sweep computable/efficient.

    Which *subset* of the operator-defining leaves a parametrisation differentiates is a separate
    concern, chosen at the ``ParametrisedModule.theta_base`` boundary -- not encoded here.

    The transport direction :math:`\Omega` lives in exactly one place: :attr:`angular_setup`'s
    quadrature. :attr:`sweep_data` carries only the (operator-agnostic) ordering; the derived transport
    operator is handed the angle from the angular setup at build time, so there is no second copy.
    '''

    solution_domain_data : SolutionDomainData
    '''Operator-defining (spatial) data: domain (geometry + cross sections), FEM matrices, geometric data.'''

    angular_setup        : AngularSetup
    '''Operator-defining (angular) data: quadrature + precomputed harmonics. See :class:`AngularSetup`.'''

    sweep_data           : SweepData
    '''Operator-agnostic data: the precomputed sweep execution plan. See :class:`~.sweep_data.SweepData`.'''

    @property
    def domain(self):
        '''Convenience accessor for the geometric domain.'''
        return self.solution_domain_data.domain

    @property
    def elements_basis_shape(self):
        '''The ``[n_elements, n_basis]`` structure the moment/discrete operators batch over.'''
        return self.solution_domain_data.elements_basis_shape
