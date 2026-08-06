'''
Builders: a derivation graph per operator kind.

Two axes. The *operator* axis is this file's class ladder -- each rung adds the primitives that
operator is built from (its strategy, its angle, its energy group) and the quantities that follow.
The *storage* axis is generated: ``.varying(...)`` / ``.freezing(...)`` on any rung returns a class
storing everything the named primitives do not determine (see :mod:`jax_sn.tools.parametrisation`).

Which tier an operator needs is visible in its registry rather than declared: only the rungs whose
operator actually sweeps carry ``sweep_data`` and mark it ``NON_DIFFERENTIABLE``. The others build
their narrow data type directly, so there is no sweep to skip and no placeholder to invent.
'''

import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Any, Tuple, Union

from jax_sn.domain import Domain, GeometricDomain
from jax_sn.domain.cross_sections import CrossSectionsIndexed
from jax_sn.domain.geometric_domain import _check_cross_section_compatibility
from jax_sn.operators.mass_operator import MassData, MassOperator, MassStrategy
from jax_sn.operators.moment_operator import (
    DiscreteData, DiscreteOperator, FluxMomentStrategy, MomentData, MomentOperator,
)
from jax_sn.operators.multi_group_operator import MultiGroupOperator, MultiGroupStrategy
from jax_sn.operators.operator_data import (
    AngularData, DiscretisationData, NonDifferentiableData, OperatorData,
)
from jax_sn.operators.scatter_operator import ScatterData, ScatterOperator, ScatterStrategy
from jax_sn.operators.single_group_operator import (
    SingleGroupChainOperator, SingleGroupChainStrategy, SingleGroupData, SingleGroupOperator,
    SingleGroupSourceOperator, SingleGroupSourceStrategy, SingleGroupStrategy,
)
from jax_sn.operators.sweep_data import SweepData
from jax_sn.operators.transport_operator import (
    MultiAngleForwardStrategy, MultiAngleInversionStrategy, SingleAngleForwardStrategy,
    SingleAngleInversionStrategy, TransportData, TransportOperatorMultiAngle,
    TransportOperatorSingleAngle,
)
from jax_sn.quadrature_set import QuadratureSetBase
from jax_sn.solution_domain import SpatialData
from jax_sn.solution_domain.base_fem import Element, FEMMatrices, PrecomputedGeometricData
from jax_sn.solution_domain.solution_domain import SolutionDomain
from jax_sn.tools.parametrisation import ParametrisedModule


class TopologyBuild(eqx.Module):
    etoe          : jnp.ndarray
    etof          : jnp.ndarray
    face_perm_idx : jnp.ndarray


# ==================================================================================================
#                                          Derivations
# ==================================================================================================
# A derivation's parameter names are its dependencies; anything without a derivation is a primitive.

# -- spatial ---------------------------------------------------------------------------------------

def face_template(element : Element) -> jnp.ndarray:
    return jnp.asarray(element.face_template)


def fem_matrices(element : Element) -> FEMMatrices:
    return FEMMatrices.from_element(element)


def topology_build(connectivity : jnp.ndarray, face_template : jnp.ndarray) -> TopologyBuild:
    from jax_sn.domain.geometric_domain import compute_topology
    return TopologyBuild(*compute_topology(connectivity, face_template))


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


# -- angular ---------------------------------------------------------------------------------------

def angular_data(quadrature_set : QuadratureSetBase,
                 cross_sections_indexed : CrossSectionsIndexed) -> AngularData:
    return AngularData.from_quadrature_set(quadrature_set, cross_sections_indexed.l_max)


def discretisation_data(spatial_data : SpatialData, angular_data : AngularData) -> DiscretisationData:
    return DiscretisationData(spatial_data, angular_data)


# -- the combinatorial tier -------------------------------------------------------------------------

def non_differentiable_data(geometric_domain : GeometricDomain, quadrature_set : QuadratureSetBase,
                            strategy : MultiGroupStrategy) -> NonDifferentiableData:
    # The strategy builds the tier it needs, so the two cannot disagree -- and built whole, since what
    # the tier holds is its business and need not stay one sweep ordering. Any knob that build takes
    # (a level-padding width, ...) is a static field on the sub-strategy that wants it, so it rides in
    # on `strategy` and this rung has nothing to pass through.
    return strategy.build_default_non_differentiable_data(geometric_domain, quadrature_set.angles)


def sweep_data(non_differentiable_data : NonDifferentiableData) -> SweepData:
    return non_differentiable_data.sweep_data


def operator_data(discretisation_data : DiscretisationData,
                  non_differentiable_data : NonDifferentiableData) -> OperatorData:
    return OperatorData(discretisation_data, non_differentiable_data)


# -- the operators themselves ------------------------------------------------------------------------

def multi_group_operator(operator_data : OperatorData,
                         strategy : MultiGroupStrategy) -> MultiGroupOperator:
    return MultiGroupOperator(operator_data, strategy, transpose_bool=False)


def single_group_data(operator_data : OperatorData, energy_group : int) -> SingleGroupData:
    return SingleGroupData(operator_data, energy_group)


def single_group_operator(single_group_data : SingleGroupData,
                          strategy : SingleGroupStrategy) -> SingleGroupOperator:
    return SingleGroupOperator(single_group_data, strategy, transpose_bool=False)


def single_group_chain_operator(single_group_data : SingleGroupData,
                                strategy : SingleGroupChainStrategy) -> SingleGroupChainOperator:
    return SingleGroupChainOperator(single_group_data, strategy, transpose_bool=False)


def single_group_source_operator(single_group_data : SingleGroupData,
                                 strategy : SingleGroupSourceStrategy) -> SingleGroupSourceOperator:
    return SingleGroupSourceOperator(single_group_data, strategy, transpose_bool=False)


def mass_data(fem_matrices : FEMMatrices,
              precomputed_geometric_data : PrecomputedGeometricData) -> MassData:
    return MassData(fem_matrices, precomputed_geometric_data)


def mass_operator(mass_data : MassData, additional_shape : Tuple[int, ...],
                  strategy : MassStrategy) -> MassOperator:
    return MassOperator(mass_data, additional_shape, strategy)


def moment_data(angular_data : AngularData) -> MomentData:
    return MomentData(angular_data.wk_ylm)


def moment_operator(moment_data : MomentData, spatial_data : SpatialData,
                    strategy : FluxMomentStrategy) -> MomentOperator:
    return MomentOperator(moment_data, spatial_data.elements_basis_shape, strategy,
                          transpose_bool=False)


def discrete_data(angular_data : AngularData) -> DiscreteData:
    return DiscreteData(angular_data.ylm)


def discrete_operator(discrete_data : DiscreteData, spatial_data : SpatialData,
                      strategy : FluxMomentStrategy) -> DiscreteOperator:
    return DiscreteOperator(discrete_data, spatial_data.elements_basis_shape, strategy,
                            transpose_bool=False)


def scatter_data(energy_group_out : int, energy_group_in : int,
                 cross_sections_indexed : CrossSectionsIndexed) -> ScatterData:
    return ScatterData(energy_group_out, energy_group_in, cross_sections_indexed)


def scatter_operator(scatter_data : ScatterData, strategy : ScatterStrategy,
                     spatial_data : SpatialData, angular_data : AngularData) -> ScatterOperator:
    elements_basis = spatial_data.elements_basis_shape
    return ScatterOperator(scatter_data, strategy, jax.ShapeDtypeStruct(
        (angular_data.wk_ylm.shape[1], *elements_basis.shape), elements_basis.dtype))


# -- transport: its own angle, so its own sort ---------------------------------------------------------

def transport_sweep_data(geometric_domain : GeometricDomain, angle : jnp.ndarray,
                         inversion_strategy : Union[SingleAngleInversionStrategy, MultiAngleInversionStrategy]) -> SweepData:
    return inversion_strategy.build_default_sweep_data(geometric_domain, angle)


def transport_data(spatial_data : SpatialData, energy_group : int, angle : jnp.ndarray,
                   sweep_data : SweepData) -> TransportData:
    return TransportData(spatial_data, energy_group, angle, sweep_data)


def single_angle_transport_operator(
        transport_data : TransportData, forward_strategy : SingleAngleForwardStrategy,
        inversion_strategy : SingleAngleInversionStrategy) -> TransportOperatorSingleAngle:
    return TransportOperatorSingleAngle(transport_data, forward_strategy, inversion_strategy)


def multi_angle_transport_operator(
        transport_data : TransportData, forward_strategy : MultiAngleForwardStrategy,
        inversion_strategy : MultiAngleInversionStrategy) -> TransportOperatorMultiAngle:
    return TransportOperatorMultiAngle(transport_data, forward_strategy, inversion_strategy)


# -- projections: reading a quantity back out of a container that already holds it ---------------------
# Only the derivations that repackage rather than compute have one, which is why this cannot be
# inferred from the graph.

def _geometric_domain_of(domain : Domain) -> GeometricDomain:
    return GeometricDomain(domain.vertices, domain.connectivity, domain.face_normals, domain.etoe,
                           domain.etof, domain.face_perm_idx, domain.face_template)


def _topology_build_of(geometric_domain : GeometricDomain) -> TopologyBuild:
    return TopologyBuild(geometric_domain.etoe, geometric_domain.etof,
                         geometric_domain.face_perm_idx)


# ==================================================================================================
#                                          The spine
# ==================================================================================================

class SpatialBuilder(ParametrisedModule, abstract=True):
    '''
    The problem's spatial description: a mesh, an element and its materials.
    '''

    DERIVATIONS = {
    "face_template"              : face_template,
    "fem_matrices"               : fem_matrices,
    "topology_build"             : topology_build,
    "face_normals"               : face_normals,
    "geometric_domain"           : geometric_domain,
    "domain"                     : domain,
    "precomputed_geometric_data" : precomputed_geometric_data,
    "spatial_data"               : spatial_data,
    "solution_domain"            : solution_domain}

    PROJECTIONS = {
    "spatial_data"               : {"solution_domain"    : lambda sd : sd.spatial_data},
    "domain"                     : {"spatial_data"       : lambda sd : sd.domain},
    "fem_matrices"               : {"spatial_data"       : lambda sd : sd.fem_matrices},
    "precomputed_geometric_data" : {"spatial_data"       : lambda sd : sd.precomputed_geometric_data},
    "geometric_domain"           : {"domain"             : _geometric_domain_of},
    "face_normals"               : {"geometric_domain"   : lambda gd : gd.face_normals},
    "face_template"              : {"geometric_domain"   : lambda gd : gd.face_template},
    "topology_build"             : {"geometric_domain"   : _topology_build_of}}

    vertices               : jnp.ndarray
    connectivity           : jnp.ndarray
    element                : Element = eqx.field(static=True)
    cross_sections_indexed : CrossSectionsIndexed

    def __check_init__(self):
        assert self.vertices.ndim == 2, \
            f"vertices must be [n_vertices, n_dim] but got shape {self.vertices.shape}"
        _check_cross_section_compatibility(self.connectivity.shape[0], self.cross_sections_indexed)

    @classmethod
    def from_solution_domain(cls, solution_domain : SolutionDomain,
                             **primitives : Any) -> "SpatialBuilder":
        '''
        Recover a builder from an already-built :class:`SolutionDomain`. Nothing is recomputed: the
        solution domain is stored whole and the spatial tier is projected back out. ``primitives``
        supplies whatever else the rung needs (a quadrature set, a strategy, an angle, ...).
        '''
        base    = cls._generation_base()
        spatial = solution_domain.spatial_data
        domain  = spatial.domain
        spine   = {"vertices", "connectivity", "element", "cross_sections_indexed"}
        seed    = base.storing("solution_domain")(
            vertices               = domain.vertices,
            connectivity           = domain.connectivity,
            element                = spatial.element,
            cross_sections_indexed = domain.cross_sections_indexed,
            solution_domain        = solution_domain,
            **primitives,
        )
        target = base.varying(*(base.graph().primitives - spine)) if cls._parametrisation_abstract else cls
        return seed if type(seed) is target else seed.materialise(target)


class DiscretisationBuilder(SpatialBuilder, abstract=True):
    '''
    The spatial description plus an angular discretisation.
    '''

    DERIVATIONS = {**SpatialBuilder.DERIVATIONS,
    "angular_data"                 : angular_data,
    "discretisation_data"          : discretisation_data}

    PROJECTIONS = {**SpatialBuilder.PROJECTIONS,
    "spatial_data"                 : {**SpatialBuilder.PROJECTIONS["spatial_data"],
                                      "discretisation_data": lambda dd : dd.spatial},
    "angular_data"                 : {"discretisation_data": lambda dd : dd.angular}}

    # The harmonics read the cross sections only for `l_max` (n_moments = (l_max+1)^2), so they can be
    # stored while the cross sections vary -- which is the whole point of varying them.
    RETAINED_UNDER = {"angular_data": {
        "cross_sections_indexed": lambda angular, xs : angular.wk_ylm.shape[1] == (xs.l_max + 1) ** 2}}

    quadrature_set : QuadratureSetBase


class SweptBuilder(DiscretisationBuilder, abstract=True):
    '''
    The rungs whose operator sweeps, and therefore carry the combinatorial tier.

    ``strategy`` builds the tier itself (:meth:`build_default_non_differentiable_data`), so a builder cannot
    disagree with the operator it builds, and a strategy that comes to need more of that tier than a
    sweep ordering -- or a knob to build it with -- changes nothing here. The tier is marked
    non-differentiable on the registry rather than inside the derivation, so it holds on the stored and
    projected paths too.
    '''

    DERIVATIONS = {**DiscretisationBuilder.DERIVATIONS,
    "non_differentiable_data" : non_differentiable_data,
    "sweep_data"              : sweep_data,
    "operator_data"           : operator_data}

    PROJECTIONS = {**DiscretisationBuilder.PROJECTIONS,
    "discretisation_data"     : {"operator_data"          : lambda od : od.discretisation_data},
    "non_differentiable_data" : {"operator_data"          : lambda od : od.non_differentiable_data},
    "sweep_data"              : {"non_differentiable_data": lambda nd : nd.sweep_data}}

    # The tier as a whole, so the mark holds however it grows; `sweep_data` reads out of an already
    # stopped tier, and is marked too because the transport rungs build it on its own.
    NON_DIFFERENTIABLE = frozenset({"non_differentiable_data", "sweep_data"})

    # The strategy reaches the tier only through what it asks of it, and `accepts` is the same
    # question the operator's `__check_init__` asks, so the two cannot drift.
    RETAINED_UNDER = {**DiscretisationBuilder.RETAINED_UNDER,
    "non_differentiable_data" : {"strategy": lambda tier, strategy : strategy.accepts(tier)}}


# ==================================================================================================
#                                          The operator rungs
# ==================================================================================================

class MultiGroupBuilder(SweptBuilder, abstract=True):
    '''The full multigroup operator :math:`I - M T^{-1} H D S`.'''

    DERIVATIONS = {**SweptBuilder.DERIVATIONS, "multi_group_operator": multi_group_operator}
    PROJECTIONS = {**SweptBuilder.PROJECTIONS,
                   "operator_data": {"multi_group_operator": lambda op : op.data}}

    strategy : MultiGroupStrategy


class SingleGroupBuilderBase(SweptBuilder, abstract=True):
    '''
    One energy group. ``energy_group`` is a primitive, so a per-group loop can hold the sort: the
    group index touches neither the geometry nor the angles it was built from.

    The three flavours below are three different operators, so they are three rungs rather than one
    that dispatches on the strategy.
    '''

    DERIVATIONS = {**SweptBuilder.DERIVATIONS, "single_group_data": single_group_data}
    PROJECTIONS = {**SweptBuilder.PROJECTIONS,
                   "operator_data": {"single_group_data": lambda sg : sg.operator_data}}

    energy_group : int


class SingleGroupBuilder(SingleGroupBuilderBase, abstract=True):
    r'''The full single-group operator :math:`I - M T^{-1} H D S_{gg}`.'''

    DERIVATIONS = {**SingleGroupBuilderBase.DERIVATIONS,
                   "single_group_operator": single_group_operator}
    PROJECTIONS = {**SingleGroupBuilderBase.PROJECTIONS,
                   "single_group_data": {"single_group_operator": lambda op : op.data}}

    strategy : SingleGroupStrategy


class SingleGroupChainBuilder(SingleGroupBuilderBase, abstract=True):
    r'''The chain operator :math:`M T^{-1} H D`.'''

    DERIVATIONS = {**SingleGroupBuilderBase.DERIVATIONS,
                   "single_group_chain_operator": single_group_chain_operator}
    PROJECTIONS = {**SingleGroupBuilderBase.PROJECTIONS,
                   "single_group_data": {"single_group_chain_operator": lambda op : op.data}}

    strategy : SingleGroupChainStrategy


class SingleGroupSourceBuilder(SingleGroupBuilderBase, abstract=True):
    r'''The source operator :math:`M T^{-1} H`.'''

    DERIVATIONS = {**SingleGroupBuilderBase.DERIVATIONS,
                   "single_group_source_operator": single_group_source_operator}
    PROJECTIONS = {**SingleGroupBuilderBase.PROJECTIONS,
                   "single_group_data": {"single_group_source_operator": lambda op : op.data}}

    strategy : SingleGroupSourceStrategy


class MassBuilder(SpatialBuilder, abstract=True):
    '''The mass operator. Needs no angular discretisation, so it hangs off the spatial spine.'''

    DERIVATIONS = {**SpatialBuilder.DERIVATIONS,
                   "mass_data"    : mass_data,
                   "mass_operator": mass_operator}
    PROJECTIONS = {**SpatialBuilder.PROJECTIONS,
                   "mass_data": {"mass_operator": lambda op : op.data}}

    strategy         : MassStrategy
    additional_shape : Tuple[int, ...] = eqx.field(static=True)


class MomentBuilder(DiscretisationBuilder, abstract=True):
    '''The discrete-to-moment operator :math:`M`.'''

    DERIVATIONS = {**DiscretisationBuilder.DERIVATIONS,
                   "moment_data"    : moment_data,
                   "moment_operator": moment_operator}
    PROJECTIONS = {**DiscretisationBuilder.PROJECTIONS,
                   "moment_data": {"moment_operator": lambda op : op.data}}

    strategy : FluxMomentStrategy


class DiscreteBuilder(DiscretisationBuilder, abstract=True):
    '''The moment-to-discrete operator :math:`D`.'''

    DERIVATIONS = {**DiscretisationBuilder.DERIVATIONS,
                   "discrete_data"    : discrete_data,
                   "discrete_operator": discrete_operator}
    PROJECTIONS = {**DiscretisationBuilder.PROJECTIONS,
                   "discrete_data": {"discrete_operator": lambda op : op.data}}

    strategy : FluxMomentStrategy


class ScatterBuilder(DiscretisationBuilder, abstract=True):
    '''The group-to-group scatter :math:`S_{\\tilde{g}g}`.'''

    DERIVATIONS = {**DiscretisationBuilder.DERIVATIONS,
                   "scatter_data"    : scatter_data,
                   "scatter_operator": scatter_operator}
    PROJECTIONS = {**DiscretisationBuilder.PROJECTIONS,
                   "scatter_data": {"scatter_operator": lambda op : op.data}}

    strategy         : ScatterStrategy
    energy_group_out : int
    energy_group_in  : int


class TransportBuilder(SpatialBuilder, abstract=True):
    '''
    Shared base for the transport rungs: no quadrature set, since the sort is for ``angle``.

    Single- and multi-angle are separate rungs rather than one that dispatches on ``angle.ndim``:
    they are different operators, and separating them turns the shape/strategy compatibility checks
    into ``__check_init__`` invariants that re-fire on every update.
    '''

    DERIVATIONS = {**SpatialBuilder.DERIVATIONS,
                   "sweep_data"    : transport_sweep_data,
                   "transport_data": transport_data}
    PROJECTIONS = {**SpatialBuilder.PROJECTIONS,
                   "spatial_data": {**SpatialBuilder.PROJECTIONS["spatial_data"],
                                    "transport_data": lambda td : td.spatial_data},
                   "sweep_data"  : {"transport_data": lambda td : td.sweep_data}}

    NON_DIFFERENTIABLE = frozenset({"sweep_data"})

    # As on `SweptBuilder`, one tier down: the sort is held bare here, and only the inversion strategy
    # asks anything of it (the forward one never touches it, so it is free whatever the policy).
    RETAINED_UNDER = {"sweep_data": {"inversion_strategy": lambda sweep_data, strategy: strategy.accepts(sweep_data)}}

    angle              : jnp.ndarray
    energy_group       : int


class SingleAngleTransportBuilder(TransportBuilder, abstract=True):
    '''Transport for one direction.'''

    DERIVATIONS = {**TransportBuilder.DERIVATIONS,
                   "transport_operator": single_angle_transport_operator}
    PROJECTIONS = {**TransportBuilder.PROJECTIONS,
                   "transport_data": {"transport_operator": lambda op : op.data}}

    forward_strategy   : SingleAngleForwardStrategy
    inversion_strategy : SingleAngleInversionStrategy

    def __check_init__(self):
        assert self.angle.ndim == 1 and self.angle.shape[0] == 3, \
            f"a single angle must have shape (3,), got {self.angle.shape}"


class MultiAngleTransportBuilder(TransportBuilder, abstract=True):
    '''Transport batched over the angle axis.'''

    DERIVATIONS = {**TransportBuilder.DERIVATIONS,
                   "transport_operator": multi_angle_transport_operator}
    PROJECTIONS = {**TransportBuilder.PROJECTIONS,
                   "transport_data": {"transport_operator": lambda op : op.data}}

    forward_strategy   : MultiAngleForwardStrategy
    inversion_strategy : MultiAngleInversionStrategy

    def __check_init__(self):
        assert self.angle.ndim == 2 and self.angle.shape[1] == 3, \
            f"batched angles must have shape (n_angles, 3), got {self.angle.shape}"
