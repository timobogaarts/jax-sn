from abc import abstractmethod
from typing import Self, Tuple

import jax
import jax.numpy as jnp

import equinox as eqx
import lineax as lx
from jax_sn.operators.base_linear_operator import (
    DataTransposeDataStrategyLinearOperator, DirectlyInvertibleOperator, Strategy
)
from typing import Type
from jax_sn.domain.geometric_domain import GeometricDomain
from .sweep_data_implementations import SweepData, BasicSweepData
from .transport_data import TransportData


class SingleAngleForwardStrategy(Strategy):
    '''    
    '''

    @abstractmethod
    def forward(self, data: TransportData, x):
        ...

    @classmethod
    def native_multi_angle(cls) -> Type["MultiAngleForwardStrategy"]:
        '''The fused multi-angle lift of this kernel, if it has one. ``None`` (the default) means the
        only lift is a combinator over the single-angle forward.'''
        return None


class MultiAngleForwardStrategy(Strategy):
    '''How to apply the forward operator over the angle axis. Consumes a batched :class:`TransportData`
    (angle axis leading). Mirrors :class:`MultiAngleInversionStrategy`.'''

    @abstractmethod
    def forward(self, data: TransportData, x):
        ...

    @abstractmethod
    def single_angle_strategy(self) -> SingleAngleForwardStrategy:
        '''The single-angle :class:`SingleAngleForwardStrategy` this angle lift corresponds to.'''
        ...


class AbstractInversionStrategy(eqx.Module):
    '''
    How to sweep-invert the operator, and the sweep-data precondition that inversion declares. Shared
    by the single- and multi-angle hierarchies, which differ only in whether ``data`` carries an angle
    axis -- so this is what a transport operator types its inversion strategy against.
    '''

    @abstractmethod
    def invert(self, data: TransportData, b):
        ...

    def required_capabilities(self) -> Tuple[type, ...]:
        '''Sweep-data capability interfaces this strategy requires (e.g. ``(ProvidesLevels,)``). Empty
        (the default) means any :class:`~jax_sn.operators.transport_operator.SweepData` is acceptable.'''
        return ()

    def build_default_sweep_data(self, geometric_domain : GeometricDomain,
                                 angle : jnp.ndarray) -> SweepData:
        '''
        The minimal sweep data this strategy needs, built for ``geometric_domain`` and ``angle``
        (single or batched -- dispatched on ``angle.ndim``).

        This *builds* rather than naming a type because a strategy may carry build-time knobs of its
        own (a level-padding width, say): those are properties of how this kernel sweeps, so they live
        as static fields on the strategy and are applied right here. Nothing upstream has to know they
        exist.
        '''
        return BasicSweepData.from_domain_angle(geometric_domain, angle)

    def accepts(self, sweep_data: SweepData) -> bool:
        '''Accepts any sweep data that provides every capability in :meth:`required_capabilities`.'''
        return all(isinstance(sweep_data, capability) for capability in self.required_capabilities())

    def check_compatible(self, data: TransportData):
        '''Asserts whether the sweep data is compatible with this inversion strategy (just typing information). '''
        assert self.accepts(data.sweep_data), \
            f"{type(self).__name__} requires sweep data providing " \
            f"{[c.__name__ for c in self.required_capabilities()]}, but got " \
            f"{type(data.sweep_data).__name__}. Build it via " \
            f"{type(self).__name__}.build_default_sweep_data(geometric_domain, angle)."


class SingleAngleInversionStrategy(AbstractInversionStrategy):
    '''
    How to sweep-invert the single-angle operator.
    '''

    @classmethod
    @abstractmethod
    def vmappable(cls) -> bool:
        '''Whether an outer ``vmap`` over this inversion vectorises.'''
        ...

    @classmethod
    def native_multi_angle(cls) -> Type["MultiAngleInversionStrategy"]:
        '''The fused multi-angle lift of this kernel, if it has one. ``None`` (the default) means the
        only lift is a combinator over the single-angle sweep.'''
        return None


class MultiAngleInversionStrategy(AbstractInversionStrategy):
    '''
    How to sweep-invert over the angle axis. Consumes a batched :class:`TransportData` (angle axis leading).
    '''

    @abstractmethod
    def invert(self, data: TransportData, b):
        '''Invert over the angle axis: ``data`` is a batched :class:`TransportData` (angle axis leading).'''
        ...

    @abstractmethod
    def single_angle_strategy(self) -> SingleAngleInversionStrategy:
        '''The single-angle :class:`SingleAngleInversionStrategy` this angle inversion corresponds to (used for composited strategy, error if does not exist)'''
        ...


class AbstractTransportOperator(DataTransposeDataStrategyLinearOperator, DirectlyInvertibleOperator):
    r'''
    Shared for single-angle and multi-angle transport operators (and even for future multi-group transport operators).
    '''

    data               : eqx.AbstractVar[TransportData]
    '''Transport Data'''

    strategy           : eqx.AbstractVar[Strategy]
    '''How to apply the forward operator.'''

    inversion_strategy : eqx.AbstractVar[AbstractInversionStrategy]
    '''How to invert the operator'''

    def __check_init__(self):
        '''Checks whether the sweep data is compatible with the inversion strategy (just typing information)'''
        self.inversion_strategy.check_compatible(self.data)

    def _invert_internal(self, b : jnp.ndarray) -> jnp.ndarray:
        return self.inversion_strategy.invert(self.data, b)

    def to_energy_group(self, energy_group : int) -> "AbstractTransportOperator":
        '''Derive a new transport operator for a different energy group, keeping the same strategies.'''
        return eqx.tree_at(lambda t : t.data.energy_group, self, energy_group)

    @abstractmethod
    def _leading_shape(self) -> Tuple[int, ...]:
        '''Shape prefixed to ``(n_elements, n_basis)``: the angle axis, if this operator has one.'''
        ...

    def _structure(self):
        spatial_data = self.data.spatial_data
        return jax.ShapeDtypeStruct(self._leading_shape() + (spatial_data.n_elements, spatial_data.n_basis),
                                    spatial_data.fem_matrices.mass_matrix.dtype)

    @property
    @abstractmethod
    def n_angles(self) -> int:
        '''Number of angles this operator covers.'''
        ...


class TransportOperatorSingleAngle(AbstractTransportOperator):
    r'''
    Base class for transport operators.
    Since the implementation of the transport operator can vary (e.g., vmap-based, batched, Pallas, etc.), this is an abstract base class that defines the interface for transport operators.

    This operator can be constructed on its own or through a :class:`MultiAngleTransportSpec` and a :class:`OperatorData`. The spec defines the strategies, the ``operator_data`` provides the data, and the energy group is passed to :meth:`MultiAngleTransportSpec.build`.

    Mathematical description
    ---------------------------

    The transport operator maps angular fluxes to transported angular fluxes for a specific energy group :math:`g` and angle :math:`\Omega_k`:

    .. math::
        \mathbf{T}_{gk} : \mathcal{F} \to \mathcal{F}

    where :math:`\mathcal{F} = \mathcal{E}\otimes \mathcal{B}` is the space of elements and basis functions.

    Its definition is:

    .. math::
        \mathbf{T}_{gk}[\psi_{gk, ej}] = \sum_j \psi_j \int \Omega_k \cdot \nabla (H_j) H_i  + \sigma^t_{ge} H_j H_i dV
    
    This is then implemented using discontinuous Galerkin elements, where the surface terms are handled through upwinding based on the angle direction:

    .. math::
        \mathbf{T}_{gk}[\psi_{gk, ej}] &= \sum_j \psi_{ej} \left[ \int_{e} -  \Omega_k \cdot \nabla (H_i) H_j  + \sigma^t_{ge} H_i H_j dV + \sum_{\gamma^+} |\Omega_k \cdot n_{\gamma}| \int_{\gamma} H_i H_j dA \right] \\ 
                                       & - \sum_{\gamma^-} |\Omega_k \cdot n_{\gamma}| \sum_{p} \psi_{gk, e(\gamma),p } \int_{\gamma} H_i  H_p^{(e(\gamma))}dA
    where :math:`\gamma` denotes a face of element :math:`e`, with :math:`n_{\gamma}` the outward normal of that face, and :math:`e(\gamma)` the neighboring element on the other side of face :math:`\gamma`. 
    :math:`\gamma^+` and :math:`\gamma^-` denote the faces where the dot product with the angle direction is positive (outflow) and negative (inflow) respectively.

    The required geometric factors and finite elements are provided through the :class:`SpatialData` object.

    Inversion
    ---------------------------

    The inverse can be computed through a so-called sweep. Since a topological ordering can be computed for the elements based on the angle direction, the transport equation can be solved element by element in that order. For each element, one then computes:

    .. math::
        G_{e, ij} &= \int_{e} -  \Omega_k \cdot \nabla (H_i) H_j  + \sigma^t_{ge} H_i H_j dV + \sum_{\gamma^+} |\Omega_k \cdot n_{\gamma}| \int_{\gamma} H_i H_j dA \\ 
        G^{-1}_{e, ij} &= \left( G_{e, ij} \right)^{-1} \\
        D_{e \gamma, ij} &= |\Omega_k \cdot n_{\gamma}|  \int_{\gamma} H_i  H_j^{(e(\gamma))}dA \\
        
    These can be computed in parallel since they do not depend on the angular fluxes, or even stored a priori (although this is expensive in memory).

    The 'sweep' then consists of iterating over the elements in topological order, and for each element collapsing (:math:`b_{ej}` is the right-hand side for element :math:`e`, :math:`a_{ej}` the unknown angular fluxes for element `e`):

    .. math::
        a_{ei} = \sum_j G_{e,ij}^{-1} \left( b_{ej} + \sum_{\gamma^-} \sum_{p} D_{e \gamma,jp} a_{e(\gamma),p}  \right)

    This is the only truly sequential part of the transport operator inversion.
        
    This inversion can be implemented in various ways, e.g., using vmap over angles, batched over angles, using Pallas, etc. These are implemented in subclasses.

    Transposition
    ---------------------------

    The transpose of the transport operator is obtained by simply reversing the angle (including the topological sort information). This is implemented in :meth:`transpose`.
    See :ref:`here <transport_operator_transposition>` for a detailed proof of this fact.

    The transpose of the inverse is obtained by the inverse of the transpose, i.e., by performing a sweep in reverse topological order with reversed angles.
        
    '''

    data                      : TransportData
    strategy                  : SingleAngleForwardStrategy
    inversion_strategy        : SingleAngleInversionStrategy    

    def _leading_shape(self) -> Tuple[int, ...]:
        return ()

    @property
    def n_angles(self) -> int:
        '''
        Number of angles in the transport operator. Here is just one.
        '''
        return 1


class TransportOperatorMultiAngle(AbstractTransportOperator):
    r'''
    Transport operator for multiple angles.

    Mathematical description
    --------------------------- 
    The transport operator for multiple angles maps angular fluxes to transported angular fluxes for a specific energy group :math:`g` and multiple angles :math:`\Omega_k`:

    .. math::
        &\mathbf{T}_{g} : \mathcal{D} \otimes \mathcal{F} \to \mathcal{D} \otimes \mathcal{F} \\ 
        &\mathbf{T}_g = \bigoplus_{k} \mathbf{T}_{gk}

    where :math:`\mathcal{F} = \mathcal{E}\otimes \mathcal{B}` is the space of elements and basis functions, and :math:`\mathcal{D}` is the space of discrete angles.
    The :math:`\mathbf{T}_{gk}` are the single-angle transport operators defined in :class:`TransportOperatorSingleAngle`.
    
    Therefore, we have for an angular flux :math:`\psi_g \in \mathcal{D} \otimes \mathcal{F}` with components :math:`\psi_{gk} \in \mathcal{F}` for each angle :math:`k`:

    .. math::
        (\mathbf{T}_{g}[\psi_{g}])_k = T_{gk}[\psi_{gk}]

    Inversion
    ---------------------------
    The inverse can be computed by inverting each single-angle transport operator independently due to the block-diagonal structure:

    .. math::
        \mathbf{T}_g^{-1} = \bigoplus_{k} \mathbf{T}_{gk}^{-1}
    
    Transpose
    ---------------------------
    The transpose of the multiple-angle transport operator is obtained by transposing each single-angle transport operator, as in :class:`TransportOperatorSingleAngle`.

    Structure
    ---------------------------
    Since this is by definition a block-diagonal operator with the single-angle transport operators on the diagonal, this operator does not couple angles and can be computed independently for each angle (and thus in parallel)
    '''

    data                      : TransportData    
    strategy                  : MultiAngleForwardStrategy    
    inversion_strategy        : MultiAngleInversionStrategy    
        
    def _leading_shape(self) -> Tuple[int, ...]:
        return (self.n_angles,)

    @property
    def n_angles(self) -> int:
        '''
        Number of angles in the transport operator.
        '''
        return self.data.angle.shape[0]

    def to_single_angle(self, idx) -> "TransportOperatorSingleAngle":
        '''Derive the genuine single-angle transport operator for angle ``idx``, slicing the angle and the
        sweep ordering in lockstep and using the natural single-angle inversion named by
        :attr:`inversion_strategy`.'''
        return TransportOperatorSingleAngle(
            self.data.single_angle(idx),
            self.strategy.single_angle_strategy(),
            self.inversion_strategy.single_angle_strategy(),
        )

    def __getitem__(self, idx) -> "TransportOperatorSingleAngle":
        '''Get the single-angle transport operator for angle ``idx`` (see :meth:`to_single_angle`).'''
        return self.to_single_angle(idx)

