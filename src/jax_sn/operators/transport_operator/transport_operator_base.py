import jax 
import jax.numpy as jnp

from dataclasses import dataclass   
from abc import ABC, abstractmethod

from jax_sn.solution_domain import SolutionDomainData
from jax_sn.domain.geometric_domain import TopologicalSortedAngle
from jax_sn.source import Source
import equinox as eqx
import lineax as lx
from jax_sn.operators.base_linear_operator import AbstractLinearOperator_sn

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

    def allow_dependent_columns(self, operator):
        '''
        Dependent columns are not allowed (and not possible with a transport operator).
        '''
        return False
    
    def allow_dependent_rows(self, operator):
        '''
        Dependent rows are not allowed (and not possible with a transport operator).
        '''
        return False
    
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

    topological_sorted_angle  : TopologicalSortedAngle    #[this does not *require* a topological sorted angle for the forward operator. however, since the inverse does require it, we just require it here.]
    '''Angle directions along with topological sorting information for the elements.'''

    energy_group              : int
    '''Energy group index for which this transport operator is defined.'''

    def _assert_shape_base(self):
        assert self.topological_sorted_angle.ndim==1, "topological_sorted_angle must be 1D for base TransportOperator"
    
    def mv(self, x : jnp.ndarray) -> jnp.ndarray:
        return self._mv_op_args(*self._forward_args(x))
    
    def _invert_internal(self, b : jnp.ndarray) -> jnp.ndarray:
        return self._inv_op_args(*self._inv_args(b))
        
    @property
    @abstractmethod
    def _mv_op_args(self):
        ...

    @property 
    @abstractmethod
    def _inv_op_args(self):
        ...
    

    def transpose(self):
        return eqx.tree_at(lambda t_operator : t_operator.topological_sorted_angle, self, self.topological_sorted_angle.reverse)
    
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
    
    def _base_forward_args(self, x):
        return (
            *self.solution_domain_data,
            self.solution_domain_data.domain.cross_sections_indexed.map_to_full_array(self.energy_group),
            self.topological_sorted_angle.angle,
            x
        )
    def _base_inv_args(self, x):
        return (
            *self.solution_domain_data,
            self.solution_domain_data.domain.cross_sections_indexed.map_to_full_array(self.energy_group),
            self.topological_sorted_angle,
            x
        )    
    
    def _forward_args(self, x):
        return self._base_forward_args(x)
    
    def _inv_args(self, b):
        return self._base_inv_args(b)
    
    @property
    def _vmap_for_forward_multiple_angles(self):
        return (None, None, None, None, 0,0)
    
    @property
    def _vmap_for_inv_multiple_angles(self):
        return (None, None, None, None, 0,0)
    
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

from .transport_operator_imp import transport_inverse_sweep_batched, transport_inverse_sweep_complete_vmap, transport_inverse_sweep_multiple_angles_batched, transport_inverse_sweep_multiple_angles_complete_vmap
from .transport_operator_forward import forward_transport_operator_vmap, forward_transport_operator_multiple_angles_vmap, forward_transport_operator_batch, forward_transport_operator_multiple_angles_batch

class TransportOperatorVmap(TransportOperator):
    r'''
    Transport operator using vmap over all elements. This computes the :math:`G_{ij}^{-1}` and :math:`D_{e \gamma, ij}` for all elements using vmap, and then computes the sweep using a fori loop.

    Takes significant memory for large problems due to storing all :math:`G_{ij}^{-1}` and :math:`D_{e \gamma, ij}` matrices.

    Is not significantly faster than the batched version for batch sizes > 1000 (since parallelisation is limited).

    '''

    @property
    def _mv_op_args(self):
        return forward_transport_operator_vmap
    
    @property
    def _inv_op_args(self):
        return transport_inverse_sweep_complete_vmap

class TransportOperatorBatched(TransportOperator):
    r'''
    Transport operator using batched sweeps. This computes the :math:`G_{ij}^{-1}` and :math:`D_{e \gamma, ij}` for a batch of elements, performs the sweep over that batch, and then moves to the next batch.

    '''
    batch_size : int = None
    '''
    Batch size for the batched sweeps. If None, this reproduces the non-batched version.
    '''
    
    @property
    def _mv_op_args(self):
        return forward_transport_operator_batch

    @property 
    def _inv_op_args(self):
        return transport_inverse_sweep_batched

    def _forward_args(self, x):
        return (*self._base_forward_args(x), self.batch_size)
    
    def _inv_args(self, b):
        return (*self._base_inv_args(b), self.batch_size)
    
    @property
    def _vmap_for_forward_multiple_angles(self):
        return (None, None, None, None, 0,0, None)
    
    @property
    def _vmap_for_inv_multiple_angles(self):
        return (None, None, None, None, 0,0, None)

from .transport_operator_pallas import transport_inverse_sweep_batched_pallas, transport_inverse_sweep_complete_vmap_pallas, transport_inverse_sweep_multiple_angles_batched_pallas, transport_inverse_sweep_multiple_angles_complete_vmap_pallas, _next_power_of_two

class TransportOperatorVmapPallas(TransportOperatorVmap):
    '''
    Transport operator using vmap sweeps with Pallas. Similar to :class:`TransportOperatorVmap`, but uses Pallas for the matrix operations. This can be significantly faster on GPUs that support it.

    '''  
    @property 
    def _inv_op_args(self):
        return transport_inverse_sweep_complete_vmap_pallas
    

class TransportOperatorBatchedPallas(TransportOperatorBatched):
    '''
    Transport operator using batched sweeps with Pallas. Similar to :class:`TransportOperatorBatched`, but uses Pallas for the matrix operations. This can be significantly faster on GPUs that support it.

    '''    
    
    @property 
    def _inv_op_args(self):
        return transport_inverse_sweep_batched_pallas


#===============================================================================================================================================
# Batching angles
#===============================================================================================================================================
@eqx.filter_jit
def vmap_multiple_angles_forward(transport_operator : TransportOperator, x : jnp.ndarray) -> jnp.ndarray:
    r'''
    Computes the forward operation of the transport operator for multiple angles using vmap.
    In other words, computes

    .. math::
        \mathbf{T}_{gk}[\psi_{gk, ej}] 
    for multiple angles :math:`k` using vmap.
    '''
    args = transport_operator._forward_args(x)
    f_vmap = jax.vmap(transport_operator._mv_op_args, in_axes=transport_operator._vmap_for_forward_multiple_angles)
    return f_vmap(*args)

@eqx.filter_jit
def vmap_multiple_angles_inv(transport_operator : TransportOperator, b : jnp.ndarray) -> jnp.ndarray:
    r'''
    Computes the forward operation of the transport operator for multiple angles using vmap.
    In other words, computes

    .. math::
        \mathbf{T}^{-1}_{gk}[\psi_{gk, ej}] 
    for multiple angles :math:`k` using vmap.
    '''
    args = transport_operator._inv_args(b)
    f_vmap = jax.vmap(transport_operator._inv_op_args, in_axes=transport_operator._vmap_for_inv_multiple_angles)
    return f_vmap(*args)

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

    def _assert_shape_base(self):
        assert self.transport_operator.topological_sorted_angle.angle.ndim==2, "topological_sorted_angle must be 2D for base TransportOperatorMultipleAngles"

    @abstractmethod
    def mv(self, x : jnp.ndarray) -> jnp.ndarray:
        ...

    @abstractmethod
    def _invert_internal(self, b : jnp.ndarray) -> jnp.ndarray:
        ...

    def inv(self, b : jnp.ndarray) -> jnp.ndarray:
        r'''
        Solves the system :math:`\mathbf{T}_g x = b` for `x` using the internal sweep inversion for multiple angles.
        '''
        solver = TransportOperatorSolver()
        return lx.linear_solve(self, b, solver, throw=False).value
    
    def transpose(self):
        return eqx.tree_at(lambda t_operator : t_operator.transport_operator.topological_sorted_angle, self, self.transport_operator.topological_sorted_angle.reverse)    
    
    
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
    def topological_sorted_angle(self):
        '''
        Topological sorted angle information used by this transport operator.
        '''
        return self.transport_operator.topological_sorted_angle
        
    
    @property 
    def _shape_inout(self):
        self._assert_shape_base()
        n_angles   = self.transport_operator.topological_sorted_angle.angle.shape[0]
        n_elements = self.transport_operator.solution_domain_data.domain.n_elements
        n_basis    = self.transport_operator.solution_domain_data.fem_matrices.n_basis        
        return jax.ShapeDtypeStruct((n_angles, n_elements, n_basis), self.transport_operator.solution_domain_data.fem_matrices.mass_matrix.dtype)
    
    @property
    def n_angles(self):
        '''
        Number of angles in the transport operator.
        '''
        return self.transport_operator.topological_sorted_angle.angle.shape[0]
    
    def __getitem__(self, idx):
        '''
        Get a transport operator for a specific angle by indexing.

        Parameters
        ----------
        idx : int
            Energy group index.

        Returns
        -------
        TransportOperator
            Transport operator for the specified energy group.
        '''        
        return eqx.tree_at(lambda t_operator : t_operator.topological_sorted_angle, self.transport_operator, self.transport_operator.topological_sorted_angle[idx])

@lx.is_symmetric.register(TransportOperatorMultipleAngles)
def _(operator : TransportOperator):
    return False

@lx.linearise.register(TransportOperatorMultipleAngles)
def _(operator : TransportOperator):
    return operator

class TransportOperatorMultipleAnglesVmap(TransportOperatorMultipleAngles):
    '''
    Transport operator for multiple angles using vmap over angles.
    '''

    def mv(self, x : jnp.ndarray) -> jnp.ndarray:
        return vmap_multiple_angles_forward(self.transport_operator, x)
    
    def _invert_internal(self, b : jnp.ndarray) -> jnp.ndarray:
        return vmap_multiple_angles_inv(self.transport_operator, b)
    


        