import jax
import jax.numpy as jnp
from typing import Type
from jax_sn.domain.cross_sections import CrossSectionData
from jax_sn.operators.base_linear_operator import AbstractLinearOperator_sn
from jax_sn.operators.single_group_operator import SingleGroupOperator
from jax_sn.operators.moment_operator import MomentOperator, DiscreteOperator, MomentStrategy, DiscreteStrategy, BatchedMoment, BatchedDiscrete
from jax_sn.operators.transport_operator import TransportOperatorMultipleAngles, TransportSpec, SolutionSetup, AngularSetup
from jax_sn.operators.mass_operator import MassOperator, MassStrategy, VmapMass
from jax_sn.operators.scatter_operator import GatherScatterOperator, GatherScatterOperatorBroadcasted
from jax_sn.quadrature_set import QuadratureSetBase
import lineax as lx
import equinox as eqx
from abc import abstractmethod

class MultiGroupOperator(AbstractLinearOperator_sn):
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
    
    where :math:`\mathbf{M}` is the batched moment operator (see :class:`MomentOperator <jax_sn.operators.moment_operator.MomentOperator>`), :math:`\mathbf{T}_g` is the transport operator for group g (see :class:`TransportOperatorMultipleAngles <jax_sn.operators.transport_operator.TransportOperatorMultipleAngles>`),
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

    Data ownership
    ---------------------------
    This operator follows the ``operator = setup + spec + strategy`` decomposition. It owns a single
    canonical :attr:`setup` (a :class:`~jax_sn.operators.transport_operator.SolutionSetup`) plus a
    data-free :attr:`transport_spec` and strategy/config fields, and exposes :attr:`transport_operator`,
    :attr:`mass_operator` and :attr:`gather_scatter_operator` as derived ``@property`` views. The data
    those sub-operators previously duplicated -- :class:`CrossSectionsIndexed`,
    :class:`PrecomputedGeometricData`, :class:`FEMMatrices` -- now lives in exactly **one** pytree
    location (:attr:`setup`), so a single :func:`equinox.tree_at` updates every consumer at once and
    there is no hand-synced invariant to maintain. (:attr:`moment_operator` and
    :attr:`discrete_operator` remain stored fields, carrying the angular harmonics.)

    The canonical locations are exposed for pytree surgery via :attr:`cross_sections_indexed_loc`,
    :attr:`precomputed_geometric_data_loc` and :attr:`fem_matrices_loc` (each now a single location).
    For example, replacing the cross sections::

        multi_group_operator_new = eqx.tree_at(multi_group_operator.cross_sections_indexed_loc,
                                               multi_group_operator,
                                               replace_fn = lambda _ : new_cross_sections_indexed)

    Possible improvements
    ---------------------------

    The operator form could be further exploited to reduce memory usage and increase efficiency. Some possible improvements include:

    - Merging the scattering and moment-to-discrete operators to avoid storing intermediate results. This reduces one vector :math:`\phi_{lm}\in \mathcal{M}\otimes \mathcal{F}` in total (as you don't store the intermediate result after the scattering operation.). However, XLA  might already do this anyway.

    - Noting that since the discrete operator and the mass operator are acting on different spaces (:math:`\mathcal{M}` and :math:`\mathcal{F}` respectively), these two operations commute and could be interchanged. Since the number of moments is often much smaller than the number of basis functions, it might be more efficient to first apply the mass operator and then the discrete operator.

    - One could note that in fact, the entire inversion operator can be materialised (in the sense that all :math:`G_{gkije}^{-1}` and :math:`D_{gk e\gamma ij} can be stored and computed)`, since it doesn't depend on the right-hand-side. This would however require significant memory (:math:`~5 G\cdot N_{angles} \cdot N_{elements} \cdot N_{basis}^2` floating point numbers), so is likely not feasible for large problems. Furthermore,
      it would not actually provide that much improvement, since the main cost is still the transport sweeps, which would still need to be done sequentially.

    '''
    setup                     : SolutionSetup
    '''Canonical shared data: operator-defining spatial + angular data + operator-agnostic sweep plan. The single home every sub-operator draws on.'''

    transport_spec            : TransportSpec
    '''Data-free transport configuration (strategies + energy group); the transport operator is derived from :attr:`setup` + this via :meth:`TransportSpec.build`.'''

    n_energy_groups           : int
    '''Number of energy groups.'''

    moment_strategy           : MomentStrategy = eqx.field(default_factory=BatchedMoment)
    '''Contraction strategy for the derived :attr:`moment_operator`.'''

    discrete_strategy         : DiscreteStrategy = eqx.field(default_factory=BatchedDiscrete)
    '''Contraction strategy for the derived :attr:`discrete_operator`.'''

    mass_strategy             : MassStrategy = eqx.field(default_factory=VmapMass)
    '''Contraction strategy for the derived :attr:`mass_operator`.'''

    gather_scatter_operator_cls : Type[GatherScatterOperator] = eqx.field(static=True, default=GatherScatterOperatorBroadcasted)
    '''Implementation class for the derived :attr:`gather_scatter_operator` (built on :attr:`setup`'s cross sections).'''

    transpose_bool : bool = False
    r'''Whether to use the transposed operator.'''

    @property
    def transport_operator(self) -> TransportOperatorMultipleAngles:
        '''The multi-angle transport operator, derived from :attr:`setup` + :attr:`transport_spec`.'''
        return self.transport_spec.build(self.setup)

    @property
    def moment_operator(self) -> MomentOperator:
        '''The moment operator, derived from :attr:`setup`'s angular harmonics.'''
        return MomentOperator(self.setup.angular_setup.wk_ylm, self.setup.elements_basis_shape, self.moment_strategy)

    @property
    def discrete_operator(self) -> DiscreteOperator:
        '''The discrete (moment-to-discrete) operator, derived from :attr:`setup`'s angular harmonics.'''
        return DiscreteOperator(self.setup.angular_setup.ylm, self.setup.elements_basis_shape, self.discrete_strategy)

    @property
    def mass_operator(self) -> MassOperator:
        '''The mass operator, derived from :attr:`setup` (shares its geometric data + FEM matrices).'''
        return MassOperator.from_transport_operator(self.transport_operator, self.mass_strategy)

    @property
    def gather_scatter_operator(self) -> GatherScatterOperator:
        '''The gather-scatter operator, derived from :attr:`setup`'s cross sections. This is not a linear operator, but a class that implements the required functionality.'''
        return self.gather_scatter_operator_cls(self.setup.solution_domain_data.domain.cross_sections_indexed)

    @property
    def quadrature_set(self) -> QuadratureSetBase:
        '''The angular quadrature set (the single angular home lives in :attr:`setup`).'''
        return self.setup.angular_setup.quadrature_set

    def __check_init__(self):
        assert isinstance(self.setup, SolutionSetup), f"setup must be SolutionSetup, but got {type(self.setup)}."
        assert isinstance(self.transport_spec, TransportSpec), f"transport_spec must be TransportSpec, but got {type(self.transport_spec)}."
        assert isinstance(self.moment_strategy, MomentStrategy), f"moment_strategy must be MomentStrategy, but got {type(self.moment_strategy)}."
        assert isinstance(self.discrete_strategy, DiscreteStrategy), f"discrete_strategy must be DiscreteStrategy, but got {type(self.discrete_strategy)}."
        assert isinstance(self.mass_strategy, MassStrategy), f"mass_strategy must be MassStrategy, but got {type(self.mass_strategy)}."
        assert self.n_energy_groups > 0, f"n_energy_groups must be positive, but got {self.n_energy_groups}."
        self.setup.solution_domain_data.domain.cross_sections_indexed.cross_section_data.assert_n_energy_groups(self.n_energy_groups)
        assert self.moment_operator.out_structure() == self.discrete_operator.in_structure(), f"Moment operator out structure {self.moment_operator.out_structure()} must match discrete operator in structure {self.discrete_operator.in_structure()}."
    
    @eqx.filter_jit
    def mv(self, x):
        if self.transpose_bool:
            return self._mv_transpose(x)
        else:   
            return self._mv_forward(x)
        
    @abstractmethod
    def _mv_forward(self,x):
        ...

    @abstractmethod
    def _mv_transpose(self,x):
        ...
    
    def transpose(self):
        return eqx.tree_at(lambda op: op.transpose_bool, self, not self.transpose_bool)

    def in_structure(self):
        return jax.ShapeDtypeStruct(
            shape=(self.n_energy_groups,) + self.moment_operator.out_structure().shape,
            dtype=self.moment_operator.out_structure().dtype # in out struture same as transpose
        )
    
    def out_structure(self):
        return jax.ShapeDtypeStruct(
            shape=(self.n_energy_groups,) + self.moment_operator.out_structure().shape,
            dtype  =self.moment_operator.out_structure().dtype            
        )
    
    @eqx.filter_jit
    def to_single_group_operator(self, g_i : int) -> SingleGroupOperator:
        r'''
        Convert to single group operator for energy group g_i.

        Parameters
        ----------
        g_i : int
            Energy group index.
        Returns
        -------
        SingleGroupOperator
            The single group operator for energy group g_i.
        ''' 
        # Shares the *same* canonical setup (spatial + angular + sweep); only the energy group differs.
        return SingleGroupOperator(self.setup,
                                   self.transport_spec.to_energy_group(g_i),
                                   self.moment_strategy,
                                   self.discrete_strategy,
                                   self.mass_strategy,
                                   self.transpose_bool)
        
    @property
    def lower_block_triangular(self):
        '''
        Whether the structure of the multi-group operator is lower block triangular (downscattering only)
        '''
        return self.setup.solution_domain_data.domain.cross_sections_indexed.cross_section_data.downscattering_only

    def convert_to_lower_block_triangular(self):
        r'''
        Convert to lower block triangular multi-group operator by discarding all upscattering cross sections.

        Returns
        -------
        MultiGroupOperator
            The lower block triangular multi-group operator.
        '''
        cs_data_lower = self.setup.solution_domain_data.domain.cross_sections_indexed.cross_section_data.convert_to_downscattering_only()
        return eqx.tree_at(self.cross_sections_indexed_loc, self, replace_fn = lambda xs_idx : eqx.tree_at(lambda xs : xs.cross_section_data, xs_idx, cs_data_lower))

    @property
    def n_elements(self):
        return self.setup.solution_domain_data.domain.n_elements

    @property
    def n_angles(self):
        return self.setup.angular_setup.quadrature_set.angles.shape[0]

    @property
    def n_moments(self):
        return self.moment_operator.n_moments

    @property
    def n_basis(self):
        return self.setup.solution_domain_data.n_basis

    @property
    def domain(self):
        return self.setup.solution_domain_data.domain

    @property
    def solution_domain_data(self):
        return self.setup.solution_domain_data

    @property
    def cross_sections_indexed_loc(self):
        '''
        Returns a ``where`` function locating the cross sections in the tree. Now a single canonical
        location (``setup``), shared by the derived transport and gather-scatter operators.
        '''
        return lambda m : m.setup.solution_domain_data.domain.cross_sections_indexed

    @property
    def precomputed_geometric_data_loc(self):
        '''
        Returns a ``where`` function locating the precomputed geometric data in the tree. Single
        canonical location (``setup``), shared by the derived transport and mass operators.
        '''
        return lambda m : m.setup.solution_domain_data.precomputed_geometric_data

    @property
    def fem_matrices_loc(self):
        '''
        Returns a ``where`` function locating the FEM matrices in the tree. Single canonical location
        (``setup``), shared by the derived transport and mass operators.
        '''
        return lambda m : m.setup.solution_domain_data.fem_matrices

    @classmethod
    def from_multigroup_operator(cls, mgop : "MultiGroupOperator"):
        return cls(mgop.setup, mgop.transport_spec, mgop.n_energy_groups, mgop.moment_strategy, mgop.discrete_strategy, mgop.mass_strategy, mgop.gather_scatter_operator_cls, mgop.transpose_bool)

    @classmethod
    def from_operators(cls, moment_operator : MomentOperator, discrete_operator : DiscreteOperator,
                       transport_operator : TransportOperatorMultipleAngles, mass_operator : MassOperator,
                       gather_scatter_operator : GatherScatterOperator, quadrature_set : QuadratureSetBase,
                       n_energy_groups : int, transpose_bool : bool = False) -> "MultiGroupOperator":
        r'''
        Build a multi-group operator from already-constructed sub-operators. The canonical
        :attr:`setup` bundles the transport operator's spatial data + sweep plan with the angular data
        (``quadrature_set`` + the harmonics read off the moment/discrete operators); the strategies are
        extracted from the supplied operators. All the data the sub-operators previously duplicated is
        therefore deduplicated into the single :attr:`setup`.
        '''
        setup = SolutionSetup(
            transport_operator.solution_domain_data,
            AngularSetup(quadrature_set, moment_operator.wk_yklm, discrete_operator.yklm),
            transport_operator.sweep_data,
        )
        return cls(
            setup,
            TransportSpec.from_transport_operator(transport_operator),
            n_energy_groups,
            moment_operator.strategy,
            discrete_operator.strategy,
            mass_operator.strategy,
            type(gather_scatter_operator),
            transpose_bool,
        )
    

@lx.is_symmetric.register(MultiGroupOperator)
def _(operator : MultiGroupOperator):
    return False

@lx.linearise.register(MultiGroupOperator)
def _(operator : MultiGroupOperator):
    return operator



@eqx.filter_jit
def _multi_group_operator_scan(n_energy_groups : int, moment_operator    : MomentOperator, transport_operator : TransportOperatorMultipleAngles,  mass_operator      : MassOperator,  discrete_operator : DiscreteOperator, gather_scatter_operator : GatherScatterOperator, x : jnp.ndarray, checkpoint = False) -> jnp.ndarray:
    r'''
    This computes the multi-group operator mv using a scan over energy groups.
    
    See :class:`MultiGroupOperator` for details on the mathematical formulation.

    The memory requirement does not scale with the number of energy groups, since we only need to store temporaries for a single energy group at a time.

    In fact, the only temporary memory required is for the transport operator application, which requires two temporaries of size :math:`N_{angles} \times N_{elements} \times N_{basis}` (one for input, one for output). All other operations can re-use these temporaries.

    Parameters
    ----------
    n_energy_groups : int
        Number of energy groups.
    moment_operator : MomentOperator
        The moment operator :math:`\mathbf{M}`.
    transport_operator : TransportOperatorMultipleAngles
        The transport operator :math:`\mathbf{T}^{-1}`.
    mass_operator : MassOperator
        The mass operator :math:`\mathbf{H}`.
    gather_scatter_operator : GatherScatterOperator
        The total gather scattering operator containing :math:`\mathbf{U}, \mathbf{V}`.
    x : jnp.ndarray
        The input vector.   

    Returns
    -------
    jnp.ndarray
        The result of applying the multi-group operator to x.   

    
    '''

    # We have exactly the same memory analysis as for the single group operator, (see _mv_single_group_operator).
    # Two discrete single-group sized temporaries [n_angles, n_elements, n_basis] are required as in/out for transport operator. 
    in_structure_shape = (n_energy_groups,) + moment_operator.out_structure().shape

    assert x.shape == in_structure_shape, f"Input x shape {x.shape} does not match expected in_structure shape {in_structure_shape}."
        
    def _single_energy_group(_, g_i):        
        y_g = (x[g_i] - moment_operator.mv(transport_operator.to_energy_group(g_i).inv(mass_operator.mv(discrete_operator.mv(gather_scatter_operator.gather_forward(x, g_i))))))
        return None, y_g
    
    f_seg = jax.checkpoint(_single_energy_group) if checkpoint else _single_energy_group
    
    _, result = jax.lax.scan(
        f_seg,
        init=None,
        xs=jnp.arange(n_energy_groups),
    )
    return result


@eqx.filter_jit
def _multi_group_operator_transpose_scan(n_energy_groups : int, moment_operator    : MomentOperator, transport_operator : TransportOperatorMultipleAngles,  mass_operator      : MassOperator,  discrete_operator : DiscreteOperator, gather_scatter_operator : GatherScatterOperator, x : jnp.ndarray, checkpoint = False) -> jnp.ndarray:    
    r'''
    This computes the multi-group operator transpose mv using a scan over energy groups.

    See :class:`MultiGroupOperator` for details on the mathematical formulation.

    Parameters
    ----------
    n_energy_groups : int
        Number of energy groups.
    moment_operator : MomentOperator
        The moment operator :math:`\mathbf{M}`.
    transport_operator : TransportOperatorMultipleAngles
        The transport operator :math:`\mathbf{T}^{-1}`.
    mass_operator : MassOperator
        The mass operator :math:`\mathbf{H}`.
    gather_scatter_operator : GatherScatterOperator
        The total gather scattering operator containing :math:`\mathbf{U}, \mathbf{V}`.
    x : jnp.ndarray
        The input vector.
    '''
    
    moment_operator_t    = moment_operator.transpose()
    transport_operator_t = transport_operator.transpose()
    mass_operator_t      = mass_operator.transpose()
    discrete_operator_t  = discrete_operator.transpose()

    in_structure_shape   = (n_energy_groups,) + moment_operator_t.in_structure().shape
    assert x.shape == in_structure_shape, f"Input x shape {x.shape} does not match expected in_structure shape {in_structure_shape}."
    
    def _single_energy_group_zeta(_, g_i):        
        return None, discrete_operator_t.mv(mass_operator_t.mv(transport_operator_t.to_energy_group(g_i).inv(moment_operator_t.mv(x[g_i]))))

    f_seg = jax.checkpoint(_single_energy_group_zeta) if checkpoint else _single_energy_group_zeta
    
    zeta_g_stored = jax.lax.scan(f_seg, None, jnp.arange(n_energy_groups))[1]

    def _scan_body(_, g_i):                
        return None, x[g_i] - gather_scatter_operator.gather_backward(zeta_g_stored, g_i)
    
    f_collapse = jax.checkpoint(_scan_body) if checkpoint else _scan_body

    _, result = jax.lax.scan(f_collapse, None, jnp.arange(n_energy_groups))
    return result

class MultiGroupOperatorScan(MultiGroupOperator):
    r'''
    Multi energy group linear operator implementation using scan over energy groups.
    Parallelisation is then over angles and elements only.
    
    '''    
    def _mv_forward(self, x : jnp.ndarray) -> jnp.ndarray:
        return _multi_group_operator_scan(self.n_energy_groups, self.moment_operator, self.transport_operator, self.mass_operator, self.discrete_operator, self.gather_scatter_operator, x)
    
    def _mv_transpose(self, x):
        return _multi_group_operator_transpose_scan(self.n_energy_groups, self.moment_operator, self.transport_operator, self.mass_operator, self.discrete_operator, self.gather_scatter_operator, x)
    


class MultiGroupOperatorScanCheckpoint(MultiGroupOperator):
    r'''
    Multi energy group linear operator implementation using scan over energy groups.
    Parallelisation is then over angles and elements only.

    Uses checkpointing of the scan loop to lower memory requirements with cost of computation time (value_and_grad ~ 3 times wrt base, while without checkpointing ~2 times wrt base).

    However, when using iterative solve, this does not matter nearly as much: the solve applies this operator O(10^2, 10^3) times. Differentiating through the solve means 
    computing the derivative of 1 times the operator application. Therefore, this means the total cost increase due to checkpointing is only 1/(O(10^2, 10^3)) compared to not using checkpointing, which is negligible
    in most cases.
    
    '''    
    def _mv_forward(self, x : jnp.ndarray) -> jnp.ndarray:
        return _multi_group_operator_scan(self.n_energy_groups, self.moment_operator, self.transport_operator, self.mass_operator, self.discrete_operator, self.gather_scatter_operator, x, True)
    
    def _mv_transpose(self, x):
        return _multi_group_operator_transpose_scan(self.n_energy_groups, self.moment_operator, self.transport_operator, self.mass_operator, self.discrete_operator, self.gather_scatter_operator, x, True)
    

@eqx.filter_jit
def _solve_multigroup_operator_rhs_scan(operator : MultiGroupOperator, rhs : jnp.ndarray, solver : lx.AbstractLinearSolver) :
    r'''
    Solve the multi-group operator system using an iterative solver.

    Parameters
    ----------
    operator : MultiGroupOperator
        The multi-group operator.
    rhs : jnp.ndarray
        The right-hand side.
    solver : lx.AbstractLinearSolver
        The lineax solver to use.

    Returns
    -------
    lx.LinearSolveOutput
        The solution output from lineax.
    '''
    expected_shape = tuple([operator.out_structure().shape[0]] + [operator.n_angles] + [operator.n_elements] + [operator.n_basis])
    assert rhs.shape == expected_shape, f"RHS shape {rhs.shape} does not match expected operator structure [n_energy_groups, n_angles, n_elements, n_basis]: \
                                                        {expected_shape}."

    def body_fn(_, g_i):        
        return None, operator.moment_operator.mv(operator.transport_operator.to_energy_group(g_i).inv(operator.mass_operator.mv(rhs[g_i])))

    rhs_mtinvh = jax.lax.scan(body_fn, None, jnp.arange(operator.n_energy_groups))[1]
    return lx.linear_solve(operator, rhs_mtinvh, solver)

