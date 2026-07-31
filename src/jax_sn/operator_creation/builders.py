

import equinox as eqx
import jax.numpy as jnp
from typing import Optional, Union
from jax_sn.domain.cross_sections import CrossSectionsIndexed
from jax_sn.domain import GeometricDomain, Domain
from jax_sn.solution_domain import SpatialData
from jax_sn.domain.geometric_domain import _check_cross_section_compatibility
from jax_sn.operators.multi_group_operator import MultiGroupOperator, MultiGroupStrategyBase
from jax_sn.operators.operator_data import AngularData, DiscretisationData, NonDifferentiableData, OperatorData
from jax_sn.operators.transport_operator import (
    MultiAngleForwardStrategy, MultiAngleInversionStrategy, SingleAngleForwardStrategy,
    SingleAngleInversionStrategy, TransportData, TransportOperatorMultiAngle, TransportOperatorSingleAngle,
)
from jax_sn.quadrature_set import QuadratureSetBase
from jax_sn.solution_domain.base_fem import Element, FEMMatrices, PrecomputedGeometricData
from jax_sn.solution_domain.solution_domain import SolutionDomain
from jax_sn.tools.derivation import DerivedModule

from jax_sn.operators.sweep_policy import Resort, SweepPolicy


def _resolve(sweep : Optional[SweepPolicy]) -> SweepPolicy:
    return Resort() if sweep is None else sweep

class TopologyBuild(eqx.Module):
    etoe          : jnp.ndarray
    etof          : jnp.ndarray    
    face_perm_idx : jnp.ndarray    

def face_template(element : Element) -> jnp.ndarray:        
    return jnp.asarray(element.face_template)

def fem_matrices(element : Element) -> FEMMatrices:
    return FEMMatrices.from_element(element)

def topology_build(connectivity : jnp.ndarray, face_template : jnp.ndarray) -> TopologyBuild:    
    from jax_sn.domain.geometric_domain import compute_topology        
    return TopologyBuild(*compute_topology(connectivity, face_template))
import jax
def face_normals(vertices : jnp.ndarray, connectivity : jnp.ndarray,
                 face_template : jnp.ndarray) -> jnp.ndarray:
    from jax_sn.domain.geometric_domain import compute_face_normals    
    return compute_face_normals(vertices, connectivity, face_template)


def geometric_domain(vertices : jnp.ndarray, connectivity : jnp.ndarray, face_normals : jnp.ndarray,
                     topology_build : TopologyBuild, face_template : jnp.ndarray) -> GeometricDomain:        
    return GeometricDomain(
        vertices      = vertices,
        connectivity  = connectivity,
        face_normals  = face_normals,
        etoe          = topology_build.etoe,
        etof          = topology_build.etof,
        face_perm_idx = topology_build.face_perm_idx,
        face_template = face_template,
    )


def domain(geometric_domain : GeometricDomain,
           cross_sections_indexed : CrossSectionsIndexed) -> Domain:    
    return Domain(
        vertices               = geometric_domain.vertices,
        connectivity           = geometric_domain.connectivity,
        face_normals           = geometric_domain.face_normals,
        etoe                   = geometric_domain.etoe,
        etof                   = geometric_domain.etof,
        face_perm_idx          = geometric_domain.face_perm_idx,
        face_template          = geometric_domain.face_template,
        cross_sections_indexed = cross_sections_indexed,
    )


def precomputed_geometric_data(element : Element,
                               geometric_domain : GeometricDomain) -> PrecomputedGeometricData:    
    return element.compute_geometric_data(geometric_domain)


def spatial_data(domain : Domain, fem_matrices : FEMMatrices,
                 precomputed_geometric_data : PrecomputedGeometricData,
                 element : Element) -> SpatialData:    
    return SpatialData(domain, fem_matrices, precomputed_geometric_data, element)


def solution_domain(spatial_data : SpatialData) -> SolutionDomain:
    return SolutionDomain(spatial_data)


def angular_data(quadrature_set : QuadratureSetBase,
                 cross_sections_indexed : CrossSectionsIndexed) -> AngularData:
    return AngularData.from_quadrature_set(quadrature_set, cross_sections_indexed.l_max)


def discretisation_data(spatial_data : SpatialData, angular_data : AngularData) -> DiscretisationData:    
    return DiscretisationData(spatial_data, angular_data)


# ==================================================================================================
#                                          The interface
# ==================================================================================================

class Builder(DerivedModule, abstract=True):
    '''
    Interface for builders.
    '''

    DERIVATIONS =  {
    "face_template"              : face_template,
    "fem_matrices"               : fem_matrices,
    "topology_build"             : topology_build,
    "face_normals"               : face_normals,
    "geometric_domain"           : geometric_domain,
    "domain"                     : domain,
    "precomputed_geometric_data" : precomputed_geometric_data,
    "spatial_data"               : spatial_data,
    "solution_domain"            : solution_domain,
    "angular_data"               : angular_data,
    "discretisation_data"        : discretisation_data}


    vertices               : jnp.ndarray    
    connectivity           : jnp.ndarray    
    element                : Element = eqx.field(static=True)
    cross_sections_indexed : CrossSectionsIndexed    
    quadrature_set         : Optional[QuadratureSetBase]    

    def __check_init__(self):
        assert self.vertices.ndim == 2, \
            f"vertices must be [n_vertices, n_dim] but got shape {self.vertices.shape}"
        _check_cross_section_compatibility(self.connectivity.shape[0], self.cross_sections_indexed)

    def _require_quadrature_set(self) -> QuadratureSetBase:
        if self.quadrature_set is None:
            raise ValueError(
                "This builder carries no quadrature_set, which this build requires. Attach one with "
                "builder.updated(quadrature_set=quadrature_set)."
            )
        return self.quadrature_set

    @classmethod
    def from_solution_domain(cls, solution_domain : SolutionDomain,
                             quadrature_set : Optional[QuadratureSetBase] = None) -> "Builder":
        '''
        Recover a builder from an already-built :class:`SolutionDomain`. Nothing is recomputed: a
        solution domain holds exactly a :class:`SpatialData`, which is what
        :class:`FixedGeometryBuilder` stores, so the whole spatial tier is a field read.
        '''
        spatial = solution_domain.spatial_data
        domain  = spatial.domain
        full    = FixedGeometryBuilder(
            vertices               = domain.vertices,
            connectivity           = domain.connectivity,
            element                = spatial.element,
            cross_sections_indexed = domain.cross_sections_indexed,
            quadrature_set         = quadrature_set,
            spatial_data           = spatial,
        )
        target = FixedGeometryBuilder if cls is Builder else cls
        return full if target is FixedGeometryBuilder else full.materialise(target)

    def operator_data(self, sweep_data_type : type, sweep : SweepPolicy = None) -> OperatorData:
        '''
        The derived discretisation plus a sweep plan.

        Parameters
        ----------
        sweep_data_type : type
            Concrete :class:`~jax_sn.operators.sweep_data.SweepData` subclass to build, normally a
            strategy's ``default_sweep_data()``.
        sweep : SweepPolicy, optional
            How to obtain the sweep plan. Defaults to
            :class:`~jax_sn.operators.sweep_policy.Resort`.
        '''
        discretisation = self.discretisation_data
        sweep_data     = _resolve(sweep).build(discretisation.spatial.domain, sweep_data_type,
                                               self._require_quadrature_set().angles)
        return OperatorData(discretisation, NonDifferentiableData(sweep_data))

    def multi_group_operator(self, strategy : MultiGroupStrategyBase,
                             sweep : SweepPolicy = None) -> MultiGroupOperator:
        '''The multigroup operator. ``strategy`` also selects the sweep data type.'''
        return MultiGroupOperator(self.operator_data(strategy.default_sweep_data(), sweep),
                                  strategy, transpose_bool=False)

    def transport_operator(self, angle : jnp.ndarray,
                           forward_strategy   : Union[SingleAngleForwardStrategy, MultiAngleForwardStrategy],
                           inversion_strategy : Union[SingleAngleInversionStrategy, MultiAngleInversionStrategy],
                           energy_group : int = 0, sweep : SweepPolicy = None
                           ) -> Union[TransportOperatorSingleAngle, TransportOperatorMultiAngle]:
        '''
        The transport operator, single- or multi-angle (dispatched on ``angle.ndim``).

        The sweep plan is built for ``angle`` rather than for the builder's quadrature set, so a
        single-angle operator sorts only its own angle -- and no quadrature set is needed at all.
        '''
        single_angle   = _check_transport_angle(angle, forward_strategy, inversion_strategy)
        spatial        = self.spatial_data
        sweep_data     = _resolve(sweep).build(spatial.domain, inversion_strategy.default_sweep_data_type(), angle)
        transport_data = TransportData(spatial, energy_group, angle, sweep_data)

        operator_cls = TransportOperatorSingleAngle if single_angle else TransportOperatorMultiAngle
        return operator_cls(transport_data, forward_strategy, inversion_strategy)


def _check_transport_angle(angle, forward_strategy, inversion_strategy) -> bool:
    '''Validate the angle/strategy combination; returns whether this is the single-angle case.'''
    if angle.ndim == 1:
        assert angle.shape[0] == 3, f"For single angle transport operator, angle must be a 1D array of shape (3,) but got {angle.shape}"
        assert isinstance(forward_strategy, SingleAngleForwardStrategy), f"For single angle transport operator, forward strategy must be a SingleAngleForwardStrategy but got {type(forward_strategy)}"
        assert isinstance(inversion_strategy, SingleAngleInversionStrategy), f"For single angle transport operator, inversion strategy must be a SingleAngleInversionStrategy but got {type(inversion_strategy)}"
        return True
    if angle.ndim == 2:
        assert angle.shape[1] == 3, f"For multi angle transport operator, angle must be a 2D array of shape (N, 3) but got {angle.shape}"
        assert isinstance(forward_strategy, MultiAngleForwardStrategy), f"For multi angle transport operator, forward strategy must be a MultiAngleForwardStrategy but got {type(forward_strategy)}"
        assert isinstance(inversion_strategy, MultiAngleInversionStrategy), f"For multi angle transport operator, inversion strategy must be a MultiAngleInversionStrategy but got {type(inversion_strategy)}"
        return False
    raise ValueError(f"Angle must be either a 1D array of shape (3,) for single angle or a 2D array of shape (N, 3) for multi angle but got {angle.shape}")


# ==================================================================================================
#                                            The ladder
# ==================================================================================================

class BaseBuilder(Builder):
    '''
    Stores nothing derived: every primitive is free to update, and every quantity is rebuilt from
    scratch on access.

    The reference implementation -- whatever a rung further up produces must agree with this. Also
    the right choice when the connectivity itself is what varies, since nothing here is frozen.
    '''

class FixedFemBuilder(Builder):
    
    fem_matrices : FEMMatrices
    

class FixedFemTopologyBuilder(Builder):

    fem_matrices   : FEMMatrices
    topology_build : TopologyBuild



class FixedGeometryBuilder(Builder):

    spatial_data : SpatialData
