import jax
import jax.numpy as jnp

from jax_sn.solution_domain import SolutionDomainData
from .sweep_data import SweepData
import equinox as eqx
import lineax as lx
from jax_sn.operators.base_linear_operator import AbstractLinearOperator_sn
from .transport_operator_strategies import ForwardStrategy, InversionStrategy, AngleLifting

class TransportOperatorSolver(lx.AbstractLinearSolver):
    '''
    Class that implements a linear solver for TransportOperator and TransportOperatorMultipleAngles using the internal sweep inversion.
    Serves as an interface between Lineax and the internal inversion methods of the TransportOperator classes.

    Has no members or state.

    '''
    def __init__(self):
        '''
        Initializes the TransportOperatorSolver, does nothing.
        '''
        pass

    def init(self, operator : "TransportOperator", options):
        '''
        Initializes the solver state for a given TransportOperator. Does nothing except check that the operator is of the correct type.
        The solver state is simply the operator.
        '''
        assert isinstance(operator, (TransportOperator, TransportOperatorMultipleAngles)), "TransportOperatorSolver can only be used with TransportOperator or TransportOperatorMultipleAngles"
        return operator
        
    def compute(self, solver_state, rhs, options):
        '''
        Computes the solution to the linear system for the given rhs using the internal inversion method of the TransportOperator. Is always by definition succesfull.
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


class TransportOperator(AbstractLinearOperator_sn):
    r'''
    Base class for transport operators.
    Since the implementation of the transport operator can vary (e.g., vmap-based, batched, Pallas, etc.), this is an abstract base class that defines the interface for transport operators.

    This operator can be constructed on its own or through a :class:`TransportSpec` and a :class:`SolutionSetup`. The spec defines the strategies and energy group, while the setup provides the data.

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

    The required geometric factors and finite elements are provided through the :class:`SolutionDomainData` object.

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

    solution_domain_data      : SolutionDomainData
    '''Data for the solution domain, including geometric factors, finite element matrices, cross sections, etc.'''

    sweep_data                : SweepData    #[the forward operator does not *require* sweep data; but since the inverse does, we require it here.]
    '''The (operator-agnostic) data required to invert (sweep) the operator: the per-angle topological order of elements. See :class:`~jax_sn.operators.transport_operator.SweepData`. Does *not* carry the angle.'''

    angle                     : jnp.ndarray   #[..., n_dim]
    r'''The transport direction(s) :math:`\Omega`. Operator-defining; supplied from the angular setup (not stored in :attr:`sweep_data`). Shape ``[n_dim]`` or, when lifted over angles, ``[n_angles, n_dim]``.'''

    energy_group              : int
    '''Energy group index for which this transport operator is defined.'''

    forward_strategy          : ForwardStrategy
    '''How to apply the forward operator ``mv`` (e.g. vmap-over-elements, batched). See :mod:`.transport_operator_strategies`.'''

    inversion_strategy        : InversionStrategy
    '''How to sweep-invert the operator ``_invert_internal``. Composes freely with any forward strategy.'''

    def __check_init__(self):
        # The inversion strategy declares its precondition on the toposort/data (e.g. a level-based
        # sweep needs a level-parallel toposort -- which the caller builds explicitly). Validating here
        # means construction is "free" (just a check) and the check also fires on every eqx.tree_at
        # (transpose, indexing, reconstruction), with a clear message instead of an AttributeError deep
        # in a kernel.
        self.inversion_strategy.check_compatible(self)

    def mv(self, x : jnp.ndarray) -> jnp.ndarray:
        return self.forward_strategy.forward(self, x)

    def _invert_internal(self, b : jnp.ndarray) -> jnp.ndarray:
        return self.inversion_strategy.invert(self, b)

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
        n_elements = self.solution_domain_data.domain.n_elements
        n_basis = self.solution_domain_data.fem_matrices.n_basis        
        return jax.ShapeDtypeStruct((n_elements, n_basis), self.solution_domain_data.fem_matrices.mass_matrix.dtype)
    
    def _forward_data(self):
        '''Shared inputs for a forward strategy: ``(*solution_domain_data, cross_sections, angle)``. The strategy appends ``x``.'''
        return (
            *self.solution_domain_data,
            self.solution_domain_data.domain.cross_sections_indexed.map_to_full_array(self.energy_group),
            self.angle,
        )

    def _inversion_args(self):
        '''Shared inputs for an inversion strategy: ``(*solution_domain_data, cross_sections, angle, sweep_data)``. The strategy appends ``b``.'''
        return (
            *self.solution_domain_data,
            self.solution_domain_data.domain.cross_sections_indexed.map_to_full_array(self.energy_group),
            self.angle,
            self.sweep_data,
        )

    @property
    def n_angles(self):
        '''
        Number of angles in the transport operator. Here is just one.
        '''
        return 1


@lx.is_symmetric.register(TransportOperator)
def _(operator : TransportOperator):
    return False

@lx.linearise.register(TransportOperator)
def _(operator : TransportOperator):
    return operator

#===============================================================================================================================================
# Batching angles
#===============================================================================================================================================
class TransportOperatorMultipleAngles(AbstractLinearOperator_sn):
    r'''
    Transport operator for multiple angles. Wraps a :class:`TransportOperator` and extends it to multiple angles by batching or vmapping over the angles.

    Mathematical description
    --------------------------- 
    The transport operator for multiple angles maps angular fluxes to transported angular fluxes for a specific energy group :math:`g` and multiple angles :math:`\Omega_k`:

    .. math::
        &\mathbf{T}_{g} : \mathcal{D} \otimes \mathcal{F} \to \mathcal{D} \otimes \mathcal{F} \\ 
        &\mathbf{T}_g = \bigoplus_{k} \mathbf{T}_{gk}

    where :math:`\mathcal{F} = \mathcal{E}\otimes \mathcal{B}` is the space of elements and basis functions, and :math:`\mathcal{D}` is the space of discrete angles.
    The :math:`\mathbf{T}_{gk}` are the single-angle transport operators defined in :class:`TransportOperator`.
    
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
    The transpose of the multiple-angle transport operator is obtained by transposing each single-angle transport operator, as in :class:`TransportOperator`.

    Structure
    ---------------------------
    Since this is by definition a block-diagonal operator with the single-angle transport operators on the diagonal, this operator does not couple angles and can be computed independently for each angle (and thus in parallel)
    '''

    transport_operator : TransportOperator
    '''Single-angle transport operator that is extended to multiple angles.'''

    angle_lifting : AngleLifting
    '''How to lift the single-angle operator over the angle axis (e.g. vmap, or a fused warp kernel). See :mod:`.transport_operator_strategies`.'''

    def __check_init__(self):
        # The lift owns its precondition on the wrapped single-angle operator (e.g. a fused warp lift
        # requires a warp single-angle operator). Validating here catches a mispaired lift/operator at
        # construction instead of failing inside a kernel.
        self.angle_lifting.check_compatible(self.transport_operator)

    def _assert_shape_base(self):
        assert self.transport_operator.angle.ndim==2, "angle must be 2D [n_angles, n_dim] for base TransportOperatorMultipleAngles"

    def mv(self, x : jnp.ndarray) -> jnp.ndarray:
        return self.angle_lifting.forward(self.transport_operator, x)

    def _invert_internal(self, b : jnp.ndarray) -> jnp.ndarray:
        return self.angle_lifting.invert(self.transport_operator, b)

    def _inversion_args(self):
        '''Delegates to the wrapped single-angle operator (used by the warp flatten helper).'''
        return self.transport_operator._inversion_args()

    def inv(self, b : jnp.ndarray) -> jnp.ndarray:
        r'''
        Solves the system :math:`\mathbf{T}_g x = b` for `x` using the internal sweep inversion for multiple angles.
        '''
        solver = TransportOperatorSolver()
        return lx.linear_solve(self, b, solver, throw=False).value
    
    def transpose(self):
        return eqx.tree_at(lambda op: op.transport_operator, self, self.transport_operator.transpose())
    
    
    def in_structure(self):        
        return self._shape_inout
    
    def out_structure(self):
        return self._shape_inout
    
    def to_energy_group(self, energy_group : int) -> "TransportOperatorMultipleAngles":
        '''
        Converts this operator to a different energy group by changing the energy group of the underlying transport operator. Since this is just a integer (no recomputation required), this is a cheap operation.
        '''
        return eqx.tree_at(lambda t_operator : t_operator.transport_operator.energy_group, self, energy_group)
    
    @property
    def solution_domain_data(self):
        '''
        Solution domain data used by this transport operator.
        '''
        return self.transport_operator.solution_domain_data
    
    @property 
    def energy_group(self):
        '''
        Energy group index for which this transport operator is defined.
        '''
        return self.transport_operator.energy_group

    @property
    def sweep_data(self):
        '''
        Topological sort (ordering) information used by this transport operator.
        '''
        return self.transport_operator.sweep_data

    @property
    def angle(self):
        '''The transport directions used by this operator (batched over angles).'''
        return self.transport_operator.angle

    @property
    def _shape_inout(self):
        self._assert_shape_base()
        n_angles   = self.transport_operator.angle.shape[0]
        n_elements = self.transport_operator.solution_domain_data.domain.n_elements
        n_basis    = self.transport_operator.solution_domain_data.fem_matrices.n_basis
        return jax.ShapeDtypeStruct((n_angles, n_elements, n_basis), self.transport_operator.solution_domain_data.fem_matrices.mass_matrix.dtype)

    @property
    def n_angles(self):
        '''
        Number of angles in the transport operator.
        '''
        return self.transport_operator.angle.shape[0]

    def __getitem__(self, idx):
        '''
        Get the single-angle transport operator for a specific angle by indexing. Slices both the
        angle and the sweep ordering in lockstep.

        Parameters
        ----------
        idx : int
            Angle index.

        Returns
        -------
        TransportOperator
            Single-angle transport operator for the specified angle.
        '''
        return eqx.tree_at(
            lambda t : (t.angle, t.sweep_data),
            self.transport_operator,
            (self.transport_operator.angle[idx], self.transport_operator.sweep_data[idx]),
        )

@lx.is_symmetric.register(TransportOperatorMultipleAngles)
def _(operator : TransportOperator):
    return False

@lx.linearise.register(TransportOperatorMultipleAngles)
def _(operator : TransportOperator):
    return operator
