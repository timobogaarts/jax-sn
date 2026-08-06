'''
Common data structures for all operators, containing differentiable and non-differentiable data.
'''

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp

from jax_sn.domain.geometric_domain import GeometricDomain
from jax_sn.solution_domain import SolutionDomain, SpatialData
from jax_sn.quadrature_set import QuadratureSetBase
from jax_sn.operators.sweep_data import SweepData


class AngularData(eqx.Module):
    r'''Operator-defining angular data: the quadrature set and its precomputed harmonics.'''

    quadrature_set : QuadratureSetBase
    '''Angular quadrature set (angles + weights).'''

    wk_ylm         : jnp.ndarray
    r'''Weighted harmonics :math:`w_k Y_{l,m}(\Omega_k)` -- the moment operator's matrix. [n_angles, n_moments]'''

    ylm            : jnp.ndarray
    r'''Unweighted harmonics :math:`Y_{l,m}(\Omega_k)` -- the discrete operator's matrix. [n_angles, n_moments]'''

    @classmethod
    def from_quadrature_set(cls, quadrature_set : QuadratureSetBase, l_max : int) -> "AngularData":
        '''Precompute the harmonics up to moment order ``l_max`` for ``quadrature_set``.'''
        return cls(quadrature_set, quadrature_set.wk_ylm_real(l_max), quadrature_set.ylm_real(l_max))


class DiscretisationData(eqx.Module):
    r'''
    Operator-defining differentiable data.
    '''

    spatial : SpatialData
    '''Spatial + material data: geometry, FEM matrices, geometric factors, cross sections.'''

    angular : AngularData
    '''Angular data: quadrature + harmonics.'''


class NonDifferentiableData(eqx.Module):
    r'''
    Non-differentiable data (setup data) that is not part of the operator's definition and as such does not have to take part in the VJP/JVP of the operator.
    '''

    sweep_data : SweepData
    '''The sweep execution plan (per-angle topological ordering).'''


class RequiresNonDifferentiableData(eqx.Module):
    '''
    Mixin for strategies that require non-differentiable data (transport / single-group/ multi-group).
    '''

    @abstractmethod
    def accepts(self, non_differentiable_data : NonDifferentiableData) -> bool:
        '''Whether it can accept this non-differentiable data.'''
        ...

    def check_compatible(self, non_differentiable_data : NonDifferentiableData):
        '''The asserting twin of :meth:`accepts`, for data assembled by hand.'''
        assert self.accepts(non_differentiable_data), \
            f"{type(self).__name__} cannot use the given {type(non_differentiable_data).__name__}. " \
            f"Build the tier it needs with " \
            f"{type(self).__name__}.build_default_non_differentiable_data(geometric_domain, angles)."

    @abstractmethod
    def build_default_non_differentiable_data(
            self, geometric_domain : GeometricDomain,
            angles : jnp.ndarray) -> NonDifferentiableData:
        '''
        Require each strategy to provide a way to construct this non-differentiable data from the
        geometric domain and angles.
        '''
        ...


class OperatorData(eqx.Module):
    r'''
    Data for neutron transport operators, split into differentiable and non-differentiable data.
    '''

    discretisation_data     : DiscretisationData
    '''Operator-defining, differentiable data.'''

    non_differentiable_data : NonDifferentiableData
    '''Discrete, non-differentiable execution data'''

    @property
    def spatial_data(self) -> SpatialData:
        return self.discretisation_data.spatial

    @property
    def angular_data(self) -> AngularData:
        return self.discretisation_data.angular

    @property
    def sweep_data(self) -> SweepData:
        return self.non_differentiable_data.sweep_data

    @property
    def domain(self):
        return self.discretisation_data.spatial.domain

    @property
    def solution_domain(self) -> SolutionDomain:        
        return SolutionDomain(self.discretisation_data.spatial)

    @property
    def elements_basis_shape(self):
        return self.discretisation_data.spatial.elements_basis_shape
    
    @property
    def n_moments(self):
        return self.angular_data.wk_ylm.shape[1]
    
    @property
    def n_angles(self):
        return self.angular_data.wk_ylm.shape[0]
    
    @property
    def n_basis(self):
        return self.elements_basis_shape.shape[-1]

    @property
    def n_energy_groups(self):
        return self.domain.cross_sections_indexed.n_energy_groups


