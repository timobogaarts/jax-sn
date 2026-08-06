

from abc import abstractmethod
import jax
import lineax as lx
import equinox as eqx
from jax_sn.operators.base_linear_operator import TransposeBoolDataStrategyLinearOperator
from jax_sn.operators.operator_data import OperatorData, RequiresNonDifferentiableData
from ..base_linear_operator import TransposableStrategy

class MultiGroupStrategyBase(TransposableStrategy, RequiresNonDifferentiableData):
    r'''
    Apply a multigroup operator, base class for both full operator and chain.

    States its requirement on the whole non-differentiable tier (see
    :class:`~jax_sn.operators.operator_data.RequiresNonDifferentiableData`), not on the sweep data
    alone: it is handed an :class:`~jax_sn.operators.operator_data.OperatorData`, so what it needs of
    that tier is its own business and may grow past the sweep ordering.
    '''

    @abstractmethod
    def forward(self, operator_data: OperatorData, x):
        ...

    @abstractmethod
    def transpose(self, operator_data: OperatorData, x):
        ...

class MultiGroupStrategy(MultiGroupStrategyBase):
    r'''
    Multigroup strategy for $\mathbf{A}= \mathbf{I} - \mathbf{M}\mathbf{T}^{-1}\mathbf{H}\mathbf{D}\mathbf{S}$, with $\mathbf{S}$ the full scattering operator.
    '''


class MultiGroupChainStrategy(MultiGroupStrategyBase):
    r'''
    Multigroup strategy for chain $\mathbf{M}\mathbf{T}^{-1}\mathbf{H}\mathbf{D}$
    '''

class MultiGroupOperatorBase(TransposeBoolDataStrategyLinearOperator):
    '''
    Base operator acting on the full energy-group axis: shared operand structure and data accessors.
    '''

    data           : OperatorData
    strategy       : MultiGroupStrategyBase   # concrete real field (narrowed per concrete operator); declared before transpose_bool so field order stays data, strategy, transpose_bool
    transpose_bool : bool = False

    @property
    def n_energy_groups(self) -> int:
        '''Number of energy groups, derived from ``operator_data``'s cross sections.'''
        return self.data.n_energy_groups

    @property
    def solution_domain(self):
        '''Reconstruct the :class:`SolutionDomain <jax_sn.solution_domain.SolutionDomain>` this operator was built from (e.g. for interpolation).'''
        return self.data.solution_domain

    def _group_structure(self) -> jax.ShapeDtypeStruct:
        '''Full operand structure: the group axis over a single-group moment array.'''
        return jax.ShapeDtypeStruct((self.n_energy_groups, self.data.n_moments, ) + self.data.elements_basis_shape.shape, self.data.angular_data.wk_ylm.dtype)

    def _in_structure(self) -> jax.ShapeDtypeStruct:
        return self._group_structure()

    def _out_structure(self) -> jax.ShapeDtypeStruct:
        return self._group_structure()


class MultiGroupOperator(MultiGroupOperatorBase):
    r'''
      Multi energy group linear operator implementation base class.


    Mathematical Description
    ---------------------------

    This operator maps as:

    .. math::
        \mathbf{A} : \mathcal{G} \otimes \mathcal{M} \otimes \mathcal{F}  \rightarrow \mathcal{G}\otimes \mathcal{M} \otimes \mathcal{F}


    The structure of the operator is 

    .. math::
        \mathbf{A} = \begin{bmatrix} \mathbf{A}_0 & B_{01} & B_{02} & \dots & B_{0G} \\ B_{10} & \mathbf{A}_1 & B_{12} & \dots & B_{1G} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ B_{G0} & B_{G1} & B_{G2} & \dots & \mathbf{A}_G \end{bmatrix}

    where :math:`\mathbf{A}_g` is the single group operator for energy group g, and :math:`B_{ij}` is the scattering operator from group j to group i.

    These operators are defined as:

    .. math::
        \mathbf{A}_g = \mathbf{I} - \mathbf{M}\mathbf{T}_g^{-1}\mathbf{H} \mathbf{D} \mathbf{S}_{gg} 

    .. math::
        \mathbf{B}_{ij} = - \mathbf{M}\mathbf{T}_i^{-1}\mathbf{H} \mathbf{D}\mathbf{S}_{ij} 
    
    where :math:`\mathbf{M}` is the batched moment operator (see :class:`MomentOperator <jax_sn.operators.moment_operator.MomentOperator>`), :math:`\mathbf{T}_g` is the transport operator for group g (see :class:`TransportOperatorMultiAngle <jax_sn.operators.transport_operator.TransportOperatorMultiAngle>`),
    :math:`\mathbf{H}` is the mass operator (see :class:`MassOperator <jax_sn.operators.mass_operator.MassOperator>`), 
    :math:`\mathbf{D}` is the moment-to-discrete operator (see :class:`DiscreteOperator <jax_sn.operators.moment_operator.DiscreteOperator>`), and :math:`\mathbf{S}_{ij}` is the scattering operator from group j to group i (see :class:`ScatterOperator <jax_sn.operators.scatter_operator.ScatterOperator>`). These operators are all defined for a single energy group, as in their original definitions.

    These are combined to form the multi-group operator as shown above. However, the naive implementation would require :math:`G^2` transport sweeps to apply the operator (one for each block in the matrix). This is clearly very inefficient. 
    We can exploit the forward and transpose structure instead to limit this to only :math:`G` sweeps per application of the operator.

    Forward Structure
    ---------------------------

    Note that for a specific row (energy group), the operator acts as a single group operator plus a scattering source from all other energy groups. Writing out explicitly the action on a vector :math:`\psi` for energy group g, we have:

    .. math::
        \mathbf{A}[\psi]_g = \left[\mathbf{I} - \mathbf{M}\mathbf{T}_g^{-1}\mathbf{H}\mathbf{D}\mathbf{S}_{gg} \right] \psi_{g} - \sum_{g' \neq g} \mathbf{M}\mathbf{T}_g^{-1}\mathbf{H}\mathbf{D}\mathbf{S}_{gg'} \psi_{g'}

    which is equivalent to:

    .. math::
        \mathbf{A}[\psi]_g = \left[\mathbf{I} - \mathbf{M} \mathbf{T}_g^{-1} \mathbf{H}\mathbf{D}\left(\mathbf{S}_{gg}[\psi_g] - \sum_{g'\neq g} \mathbf{S}_{gg'}[\psi_{g'}]\right)\right] \equiv \left[\mathbf{I} - \mathbf{M} \mathbf{T}_g^{-1} \mathbf{H} \mathbf{D}\mathbf{U}_{g}[\psi]\right]

    This latter operator:

    .. math::
        \mathbf{U}_g[\psi] =  \sum_{\tilde{g}} \mathbf{S}_{g\tilde{g}}[\psi_{\tilde{g}}]

    is implemented efficiently in the :class:`GatherScatterOperator` class.

    This procedure minimizes the number of sweeps to just one per group in the forward direction.


    Transposed Structure
    ---------------------------
    The transposition structure is not as straightforward as the forward structure. We have 

    .. math::
        \mathbf{A}^T = \begin{bmatrix} \mathbf{A}_0^T & B_{10}^T & B_{20}^T & \dots & B_{G0}^T \\ B_{01}^T & \mathbf{A}_1^T & B_{21}^T & \dots & B_{G1}^T \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ B_{0G}^T & B_{1G}^T & B_{2G}^T & \dots & \mathbf{A}_G^T \end{bmatrix}

    For a single row, we have 

    .. math::
        \mathbf{A}^T[\psi]_g = \left[\mathbf{I} - \mathbf{S}_{gg}^T \mathbf{D}^T \mathbf{H}^T (\mathbf{T}_g^{-1})^T \mathbf{M}^T \right] \psi_{g} - \sum_{g' \neq g} \mathbf{S}_{g'g}^T \mathbf{D}^T\mathbf{H}^T (\mathbf{T}_{g'}^{-1})^T \mathbf{M}^T \psi_{g'}

    Now, we note that we cannot take together the :math:`\mathbf{T}_{g'}^{-1}` terms since this is a different energy group for each term in the sum.

    So, instead we precompute, for all energy groups :math:`g`:

    .. math::
        \zeta_g = \mathbf{D}^T \mathbf{H}^T (\mathbf{T}_{g}^{-1})^T  \mathbf{M}^T \psi_g

    and then combine these with the transposed indexed scattering operators to form the final result (note that the scatter operator transposition is itself, since it is element-wise multiplication):
 
    .. math::
        \mathbf{A}^T[\psi]_g = \psi_g - \sum_{g'} \mathbf{S}_{g'g}^T[\zeta_{g'}] \equiv \psi_g - \sum_{g'}\mathbf{S}_{g'g}[\zeta_{g'}] \equiv \psi_g - \mathbf{V}_g[g]

    with 

    .. math::
        \mathbf{V}_g[\zeta] = \sum_{\tilde{g}} \mathbf{S}_{\tilde{g} g}[\zeta_{\tilde{g}}]

    again implemented efficiently in the :class:`GatherScatterOperator` class.

    This again minimizes the number of sweeps to just one per group in the transposed direction, although some extra storage is required for the intermediate :math:`\zeta_g` vector


    Solution
    ---------------------------
    Since this is a iteratively defined linear operator, the standard way to solve systems involving this operator is to use an iterative solver (e.g., GMRES, CG, etc.) with the mv() method defined here. This can 
    be accomplished using e.g. lineax solvers. This allows for the propagation of the derivatives through the solver as well.
    However, in some cases, e.g. downscattering only problems (such as often found in fusion applications), the multi-group operator is lower triangular. In these cases, it is much more efficient to solve the system directly using a block forward substitution approach, 
    solving each energy group sequentially using the single-group operator for that group and the already computed higher energy group solutions to form the scattering source. This is implemented in the function :func:`solve_lower_triangular`.    

    '''
    strategy : MultiGroupStrategy


class MultiGroupChainOperator(MultiGroupOperatorBase):
    r'''
    The chain :math:`\mathbf{C} = \mathbf{M}\mathbf{T}^{-1}\mathbf{H}\mathbf{D}` applied independently to
    every group (no gather-scatter, so groups don't couple). Same operand structure as
    :class:`MultiGroupOperator`; used to precondition a per-group right-hand side.

    Wrapping the strategy as an operator is what buys ``transpose()`` and lineax composition for free.
    It is a *view*: it shares ``data`` with the multigroup operator it accompanies, so build it at the
    call site -- a stored copy would not follow an ``eqx.tree_at`` on the original's data.
    '''
    strategy : MultiGroupChainStrategy

