

import equinox as eqx
import jax.numpy as jnp
from typing import Optional, Union
from jax_sn.domain.cross_sections import CrossSectionsIndexed
from jax_sn.domain import GeometricDomain, Domain
from jax_sn.solution_domain import SpatialData
from jax_sn.domain.geometric_domain import _check_cross_section_compatibility
from jax_sn.operators.mass_operator import MassOperator, MassStrategy
from jax_sn.operators.operator_data import AngularData
from jax_sn.operators.moment_operator import DiscreteOperator, FluxMomentStrategy, MomentOperator
from jax_sn.operators.multi_group_operator import MultiGroupOperator, MultiGroupStrategyBase
from jax_sn.operators.operator_data import DiscretisationData, NonDifferentiableData, OperatorData
from jax_sn.operators.scatter_operator import ScatterOperator, ScatterStrategy
from jax_sn.operators.single_group_operator import (
    SingleGroupChainOperator, SingleGroupChainStrategy, SingleGroupData, SingleGroupOperator,
    SingleGroupSourceOperator, SingleGroupSourceStrategy, SingleGroupStrategy, SingleGroupStrategyBase,
)
from jax_sn.operators.single_group_operator.composed_single_group_strategies import (
    _build_discrete_operator, _build_inscatter_operator, _build_mass_operator, _build_moment_operator,
)
from jax_sn.operators.transport_operator import (
    MultiAngleForwardStrategy, MultiAngleInversionStrategy, SingleAngleForwardStrategy,
    SingleAngleInversionStrategy, TransportData, TransportOperatorMultiAngle, TransportOperatorSingleAngle,
)
from jax_sn.quadrature_set import QuadratureSetBase
from jax_sn.solution_domain.base_fem import Element, FEMMatrices, PrecomputedGeometricData
from jax_sn.solution_domain.solution_domain import SolutionDomain
from jax_sn.tools.derivation import DerivedModule

from jax_sn.operators.sweep_policy import NoSweep, NoSweepData, Resort, SweepPolicy


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
    jax.debug.print("COMPUTING TOPOLOGY")
    return TopologyBuild(*compute_topology(connectivity, face_template))
import jax
def face_normals(vertices : jnp.ndarray, connectivity : jnp.ndarray,
                 face_template : jnp.ndarray) -> jnp.ndarray:
    from jax_sn.domain.geometric_domain import compute_face_normals
    jax.debug.print("COMPUTING NORMALS")
    return compute_face_normals(vertices, connectivity, face_template)


def geometric_domain(vertices : jnp.ndarray, connectivity : jnp.ndarray, face_normals : jnp.ndarray,
                     topology_build : TopologyBuild, face_template : jnp.ndarray) -> GeometricDomain:    
    jax.debug.print("COMPUTING GEOMETRIC DOMAIN")
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

def operator_data(spatial_data : SpatialData, angular_data : AngularData) -> OperatorData:
    return OperatorData(spatial_data, angular_data)

from typing import Mapping, Callable

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
    "operator_data"              : operator_data}


    vertices               : jnp.ndarray    
    connectivity           : jnp.ndarray    
    element                : Element = eqx.field(static=True)
    cross_sections_indexed : CrossSectionsIndexed    
    quadrature_set         : Optional[QuadratureSetBase]    

    def __check_init__(self):
        assert self.vertices.ndim == 2, \
            f"vertices must be [n_vertices, n_dim] but got shape {self.vertices.shape}"
        _check_cross_section_compatibility(self.connectivity.shape[0], self.cross_sections_indexed)

    
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

    face_template : jnp.ndarray
    fem_matrices : FEMMatrices
    

class FixedFemTopologyBuilder(Builder):

    face_template : jnp.ndarray
    fem_matrices   : FEMMatrices
    topology_build : TopologyBuild



class FixedGeometryBuilder(Builder):


    face_template              : jnp.ndarray
    fem_matrices               : FEMMatrices
    topology_build             : TopologyBuild
    face_normals               : jnp.ndarray
    precomputed_geometric_data : PrecomputedGeometricData
