import jax
import jax.numpy as jnp

from jax_sn.solution_domain import SpatialData
from .sweep_data_implementations import SweepData
from .transport_data import ForwardData, InversionData
import equinox as eqx
import lineax as lx
from jax_sn.operators.base_linear_operator import AbstractLinearOperator_sn
from .transport_operator_strategies import SingleAngleForwardStrategy, SingleAngleInversionStrategy, MultiAngleInversionStrategy

class TransportOperatorSolver(lx.AbstractLinearSolver):
    '''
    Class that implements a linear solver for TransportOperatorSingleAngle and TransportOperatorMultiAngle using the internal sweep inversion.
    Serves as an interface between Lineax and the internal inversion methods of the TransportOperatorSingleAngle classes.

    Has no members or state.

    '''
    def __init__(self):
        '''
        Initializes the TransportOperatorSolver, does nothing.
        '''
        pass

    def init(self, operator : "TransportOperatorSingleAngle", options):
        '''
        Initializes the solver state for a given TransportOperatorSingleAngle. Does nothing except check that the operator is of the correct type.
        The solver state is simply the operator.
        '''
        assert isinstance(operator, (TransportOperatorSingleAngle, TransportOperatorMultiAngle)), "TransportOperatorSolver can only be used with TransportOperatorSingleAngle or TransportOperatorMultiAngle"
        return operator
        
    def compute(self, solver_state, rhs, options):
        '''
        Computes the solution to the linear system for the given rhs using the internal inversion method of the TransportOperatorSingleAngle. Is always by definition succesfull.
        '''
        return solver_state._invert_internal(rhs), lx.RESULTS.successful, {}

    def assume_full_rank(self):
        '''
        Assumes that the operator is full rank, which is true for the transport operator.
        '''
        return True
    
    def transpose(self, solver_state, options):
        '''
        Returns the transposed solver state and options. (the solver_state is simply the operator here, so that is what is transposed).
        '''
        return solver_state.transpose(), options
    
    def conj(self, solver_state, options):
        '''
        Returns the conjugated solver state and options. (the solver_state is simply the operator here, so that is what is conjugated).
        '''
        return solver_state.conj(), options


class TransportOperatorSingleAngle(AbstractLinearOperator_sn):
    r'''
    Base class for transport operators.
    Since the implementation of the transport operator can vary (e.g., vmap-based, batched, Pallas, etc.), this is an abstract base class that defines the interface for transport operators.

    This operator can be constructed on its own or through a :class:`MultiAngleTransportSpec` and a :class:`OperatorData`. The spec defines the strategies and energy group, while the setup provides the data.

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

    spatial_data      : SpatialData
    '''Data for the solution domain, including geometric factors, finite element matrices, cross sections, etc.'''

    sweep_data                : SweepData    #[the forward operator does not *require* sweep data; but since the inverse does, we require it here.]
    '''The (operator-agnostic) data required to invert (sweep) the operator: the per-angle topological order of elements. See :class:`~jax_sn.operators.transport_operator.SweepData`. Does *not* carry the angle.'''

    angle                     : jnp.ndarray   #[ n_dim]
    r'''The transport direction(s) :math:`\Omega`. Operator-defining; supplied from the angular setup (not stored in :attr:`sweep_data`). Shape ``[n_dim]`` or, when lifted over angles, ``[n_angles, n_dim]``.'''

    energy_group              : int
    '''Energy group index for which this transport operator is defined.'''

    forward_strategy          : SingleAngleForwardStrategy
    '''How to apply the forward operator ``mv`` (e.g. vmap-over-elements, batched). See :mod:`.transport_operator_strategies`.'''

    inversion_strategy        : SingleAngleInversionStrategy
    '''How to sweep-invert the operator ``_invert_internal``. Composes freely with any forward strategy.'''

    def __check_init__(self):
        # The inversion strategy declares its precondition on the toposort/data (e.g. a level-based
        # sweep needs a level-parallel toposort -- which the caller builds explicitly). Validating here
        # means construction is "free" (just a check) and the check also fires on every eqx.tree_at
        # (transpose, indexing, reconstruction), with a clear message instead of an AttributeError deep
        # in a kernel.
        self.inversion_strategy.check_compatible(self)

    def mv(self, x : jnp.ndarray) -> jnp.ndarray:
        return self.forward_strategy.forward(self._forward_data(), x)

    def _invert_internal(self, b : jnp.ndarray) -> jnp.ndarray:
        return self.inversion_strategy.invert(self._inversion_data(), b)

    def transpose(self):
        # The transpose is the operator for the reversed direction: negate the angle and reverse the
        # sweep ordering together (the ordering's reverse is the partner of negating the angle).
        return eqx.tree_at(lambda t : (t.angle, t.sweep_data), self, (-self.angle, self.sweep_data.reverse))

    def inv(self, b : jnp.ndarray) -> jnp.ndarray:
        r'''
        Solves the system :math:`\mathbf{T} x = b` for `x` using the internal sweep inversion.

        Parameters
        ----------
        b : jnp.ndarray
            Right-hand side array.
        Returns
        -------
        jnp.ndarray
            Solution array `x`.
        '''
        solver = TransportOperatorSolver()
        return lx.linear_solve(self, b, solver, throw=False).value
  
    def in_structure(self):        
        return self._shape_inout
    
    def out_structure(self):
        return self._shape_inout
    
    @property
    def _shape_inout(self):
        n_elements = self.spatial_data.domain.n_elements
        n_basis = self.spatial_data.fem_matrices.n_basis        
        return jax.ShapeDtypeStruct((n_elements, n_basis), self.spatial_data.fem_matrices.mass_matrix.dtype)
    
    def _forward_data(self) -> ForwardData:
        '''Data a forward strategy consumes. The strategy appends ``x``.'''
        return ForwardData(
            *self.spatial_data,
            self.spatial_data.domain.cross_sections_indexed.map_to_full_array(self.energy_group),
            self.angle,
        )

    def _inversion_data(self) -> InversionData:
        '''Data an inversion strategy consumes. The strategy appends ``b``.'''
        return InversionData(
            *self.spatial_data,
            self.spatial_data.domain.cross_sections_indexed.map_to_full_array(self.energy_group),
            self.angle,
            self.sweep_data,
        )

    @property
    def n_angles(self):
        '''
        Number of angles in the transport operator. Here is just one.
        '''
        return 1


@lx.is_symmetric.register(TransportOperatorSingleAngle)
def _(operator : TransportOperatorSingleAngle):
    return False

@lx.linearise.register(TransportOperatorSingleAngle)
def _(operator : TransportOperatorSingleAngle):
    return operator

#===============================================================================================================================================
# Batching angles
#===============================================================================================================================================
def _map_over_angles(data: "InversionData", leading, fn):
    '''vmap over the sweep data and angle axis (the only ones that change for multiple angles)'''
    def one_angle(angle_i, sweep_i, v_i):
        return fn(data._replace(angle=angle_i, sweep_data=sweep_i), v_i)
    return eqx.filter_vmap(one_angle, in_axes=(0, 0, 0))(data.angle, data.sweep_data, leading)


class TransportOperatorMultiAngle(AbstractLinearOperator_sn):
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

    spatial_data              : SpatialData
    '''Data for the solution domain, including geometric factors, finite element matrices, cross sections, etc.'''

    sweep_data                : SweepData
    '''Topological sorted ordering.'''

    angle                     : jnp.ndarray   #[n_angles, n_dim]
    r'''The transport directions :math:`\Omega_k`, batched over the angle axis. Shape [n_angles, n_dim].'''

    energy_group              : int
    '''Energy group index for which this transport operator is defined.'''

    forward_strategy                         : SingleAngleForwardStrategy
    '''Forward strategy. Vmapped for multiple angles.'''

    multi_angle_inversion_strategy           : MultiAngleInversionStrategy
    '''Inversion strategy for multiple angles.'''

    def __check_init__(self):
        # Validate the single-angle inversion's data requirements against our (batched) sweep data,
        # mirroring TransportOperatorSingleAngle.__check_init__ (check_compatible reads only .sweep_data /
        # .spatial_data, both of which we expose). Fires at construction and on every eqx.tree_at.
        self.multi_angle_inversion_strategy.check_compatible(self)

    def _assert_shape_base(self):
        assert self.angle.ndim==2, "angle must be 2D [n_angles, n_dim] for TransportOperatorMultiAngle"

    def _single_angle(self, sweep_data, angle) -> "TransportOperatorSingleAngle":
        '''Single angle version from a multi-angle version which provides a natural single angle strategy.'''
        return TransportOperatorSingleAngle(
            self.spatial_data, sweep_data, angle, self.energy_group,
            self.forward_strategy, self.multi_angle_inversion_strategy.single_angle_strategy(),
        )

    def _forward_data(self) -> ForwardData:
        '''Batched (over angles) data a forward strategy consumes.'''
        return ForwardData(
            *self.spatial_data,
            self.spatial_data.domain.cross_sections_indexed.map_to_full_array(self.energy_group),
            self.angle,
        )

    def _inversion_data(self) -> InversionData:
        '''Batched (over angles) data an inversion strategy consumes.'''
        return InversionData(
            *self.spatial_data,
            self.spatial_data.domain.cross_sections_indexed.map_to_full_array(self.energy_group),
            self.angle,
            self.sweep_data,
        )

    def mv(self, x : jnp.ndarray) -> jnp.ndarray:
        '''Forward multiply: vmap the single-angle forward over the angle axis.'''
        data = self._forward_data()
        def one_angle(angle_i, x_i):
            return self.forward_strategy.forward(data._replace(angle=angle_i), x_i)
        return eqx.filter_vmap(one_angle, in_axes=(0, 0))(data.angle, x)

    def _invert_internal(self, b : jnp.ndarray) -> jnp.ndarray:
        return self.multi_angle_inversion_strategy.invert(self._inversion_data(), b)

    def inv(self, b : jnp.ndarray) -> jnp.ndarray:
        r'''
        Solves the system :math:`\mathbf{T}_g x = b` for `x` using the internal sweep inversion for multiple angles.
        '''
        solver = TransportOperatorSolver()
        return lx.linear_solve(self, b, solver, throw=False).value

    def transpose(self):
        # The transpose is the operator for the reversed direction: negate the angle and reverse the
        # sweep ordering together (as in the single-angle operator).
        return eqx.tree_at(lambda t : (t.angle, t.sweep_data), self, (-self.angle, self.sweep_data.reverse))


    def in_structure(self):
        return self._shape_inout

    def out_structure(self):
        return self._shape_inout

    def to_energy_group(self, energy_group : int) -> "TransportOperatorMultiAngle":
        '''
        Converts this operator to a different energy group. Since this is just an integer (no recomputation required), this is a cheap operation.
        '''
        return eqx.tree_at(lambda t : t.energy_group, self, energy_group)

    @property
    def _shape_inout(self):
        self._assert_shape_base()
        n_angles   = self.angle.shape[0]
        n_elements = self.spatial_data.domain.n_elements
        n_basis    = self.spatial_data.fem_matrices.n_basis
        return jax.ShapeDtypeStruct((n_angles, n_elements, n_basis), self.spatial_data.fem_matrices.mass_matrix.dtype)

    @property
    def n_angles(self):
        '''
        Number of angles in the transport operator.
        '''
        return self.angle.shape[0]

    def to_single_angle(self, idx) -> "TransportOperatorSingleAngle":
        '''Derive the genuine single-angle transport operator for angle ``idx``, slicing the angle and the
        sweep ordering in lockstep and using the natural single-angle inversion named by
        :attr:`multi_angle_inversion_strategy`.'''
        return self._single_angle(self.sweep_data[idx], self.angle[idx])

    def __getitem__(self, idx) -> "TransportOperatorSingleAngle":
        '''Get the single-angle transport operator for angle ``idx`` (see :meth:`to_single_angle`).'''
        return self.to_single_angle(idx)

@lx.is_symmetric.register(TransportOperatorMultiAngle)
def _(operator : TransportOperatorMultiAngle):
    return False

@lx.linearise.register(TransportOperatorMultiAngle)
def _(operator : TransportOperatorMultiAngle):
    return operator
