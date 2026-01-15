import jax
import jax.numpy as jnp
from jax_sn.domain.cross_sections import CrossSectionData
from jax_sn.operators.base_linear_operator import AbstractLinearOperator_sn
from jax_sn.operators.single_group_operator import SingleGroupOperator, SingleGroupOperatorTranspose
from jax_sn.operators.moment_operator import MomentOperator, MomentOperatorTransposed, DiscreteOperator, DiscreteOperatorTransposed
from jax_sn.operators.transport_operator import TransportOperatorMultipleAngles
from jax_sn.operators.mass_operator import MassOperator
from jax_sn.operators.scatter_operator import GatherScatterOperator
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

    Shared data
    ---------------------------
    Some operators have shared underlying data. The reason why these are not merged is that each operator has its own meaning and can be applied separately. In principle, the alternative would be to discard the operator mechanism
    and simply have all the data here.

    The places are 

    - The :class:`CrossSectionsIndexed <jax_sn.domain.cross_sections.CrossSectionsIndexed>` is shared between the transport operator and the gather-scatter operator. 
      This is because both operators need access to the cross section data to perform their operations. Specifically, the transport operator uses the total data, while the gather-scatter operator uses the scattering data.
      
      - :attr:`transport_operator.solution_domain_data.domain.cross_sections_indexed`,
      - :attr:`gather_scatter_operator.cross_sections_indexed`.

    - The :class:`PrecomputedGeometricData <jax_sn.solution_domain.PrecomputedGeometricData>` is shared between the transport operator and the mass operator.
      Both operators need access to geometric data to perform their operations. 
      
      - :attr:`transport_operator.solution_domain_data.precomputed_geometric_data`,
      - :attr:`mass_operator.precomputed_geometric_data <jax_sn.operators.mass_operator.MassOperator.precomputed_geometric_data>`.

    - The :class:`FEMMatrices <jax_sn.solution_domain.FEMMatrices>` is shared between the transport operator and the mass operator. Both require the application of the computed FEM matrices.

      - :attr:`transport_operator.solution_domain_data.fem_matrices`,
      - :attr:`mass_operator.fem_matrices <jax_sn.operators.mass_operator.MassOperator.fem_matrices>`.

    The location of the shared data is crucial for correct pytree surgery (i.e. not replacing just one, but replacing both).

    These are provided as property, see :attr:`cross_sections_indexed_loc`, :attr:`precomputed_geometric_data_loc`, and :attr:`fem_matrices_loc`.

    Pytree surgery can then be performed as (e.g., for replacing cross section data)::
        
        multi_group_operator_new = eqx.tree_at(multi_group_operator.cross_sections_indexed_loc, 
                                               multi_group_operator, 
                                               replace_fn = lambda _ : new_cross_sections_indexed)

    (the normal replace will not work directly as it expects that the replace is a tuple of the same length as the number of locations, which we do not want to assume a priori).

    If the sub-data needs to be replaced, the easiest is to stack two replaces. An example is :meth:`convert_to_lower_block_triangular`, which internally replaces the required cross section data in both locations::
    
        xs_indexed_new = eqx.tree_at(lambda xs_idx : xs_idx.cross_section_data, 
                                     self.gather_scatter_operator.cross_sections_indexed, 
                                     cs_data_lower)
        return eqx.tree_at(self.cross_sections_indexed_loc, 
                           self, 
                           replace_fn = lambda _ : xs_indexed_new)

    Possible improvements
    ---------------------------

    The operator form could be further exploited to reduce memory usage and increase efficiency. Some possible improvements include:

    - Merging the scattering and moment-to-discrete operators to avoid storing intermediate results. This reduces one vector :math:`\phi_{lm}\in \mathcal{M}\otimes \mathcal{F}` in total (as you don't store the intermediate result after the scattering operation.). However, XLA  might already do this anyway.

    - Noting that since the discrete operator and the mass operator are acting on different spaces (:math:`\mathcal{M}` and :math:`\mathcal{F}` respectively), these two operations commute and could be interchanged. Since the number of moments is often much smaller than the number of basis functions, it might be more efficient to first apply the mass operator and then the discrete operator.

    - One could note that in fact, the entire inversion operator can be materialised (in the sense that all :math:`G_{gkije}^{-1}` and :math:`D_{gk e\gamma ij} can be stored and computed)`, since it doesn't depend on the right-hand-side. This would however require significant memory (:math:`~5 G\cdot N_{angles} \cdot N_{elements} \cdot N_{basis}^2` floating point numbers), so is likely not feasible for large problems. Furthermore,
      it would not actually provide that much improvement, since the main cost is still the transport sweeps, which would still need to be done sequentially.

    '''
    moment_operator           : MomentOperator
    '''The single group moment operator used in the multi-group operator.'''
    
    discrete_operator         : DiscreteOperator
    '''The single group discrete operator used in the multi-group operator.'''

    transport_operator        : TransportOperatorMultipleAngles
    '''The single group transport operator used in the multi-group operator.'''

    mass_operator             : MassOperator
    '''The single group mass operator used in the multi-group operator.'''

    gather_scatter_operator   : GatherScatterOperator
    '''The gather-scatter operator used to efficiently compute the scattering sources over all energy groups. This is not a linear operator, but a class that implements the required functionality.'''

    n_energy_groups           : int    
    '''Number of energy groups.'''

    transpose_bool : bool = False
    r'''Whether to use the transposed operator.'''

    def __check_init__(self):
        assert isinstance(self.moment_operator, MomentOperator), f"moment_operator must be MomentOperator, but got {type(self.moment_operator)}."
        assert isinstance(self.discrete_operator, DiscreteOperator), f"discrete_operator must be DiscreteOperator, but got {type(self.discrete_operator)}."
        assert isinstance(self.transport_operator, TransportOperatorMultipleAngles), f"transport_operator must be TransportOperatorMultipleAngles, but got {type(self.transport_operator)}."
        assert isinstance(self.mass_operator, MassOperator), f"mass_operator must be MassOperator, but got {type(self.mass_operator)}."
        assert isinstance(self.gather_scatter_operator, GatherScatterOperator), f"scatter_operator must be GatherScatterOperator, but got {type(self.gather_scatter_operator)}."
        assert self.n_energy_groups > 0, f"n_energy_groups must be positive, but got {self.n_energy_groups}."        
        self.transport_operator.solution_domain_data.domain.cross_sections_indexed.cross_section_data.assert_n_energy_groups(self.n_energy_groups)
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
        return type(self)(self.moment_operator,
                          self.discrete_operator,
                                  self.transport_operator,
                                  self.mass_operator,
                                  self.gather_scatter_operator,
                                  self.n_energy_groups, 
                                    not self.transpose_bool)        

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
        if self.transpose_bool:
            return SingleGroupOperatorTranspose(self.moment_operator,
                                                self.transport_operator.to_energy_group(g_i),
                                                self.mass_operator,
                                                self.discrete_operator,
                                                self.gather_scatter_operator.to_scatter_operator(g_i, g_i, self.moment_operator.out_structure()))
        else:
            return SingleGroupOperator(self.moment_operator, self.transport_operator.to_energy_group(g_i), self.mass_operator, self.discrete_operator, self.gather_scatter_operator.to_scatter_operator(g_i, g_i, self.moment_operator.out_structure()))
        
    @property
    def lower_block_triangular(self):
        '''
        Whether the structure of the multi-group operator is lower block triangular (downscattering only)
        '''
        return self.gather_scatter_operator.cross_sections_indexed.cross_section_data.downscattering_only
    
    def convert_to_lower_block_triangular(self):
        r'''
        Convert to lower block triangular multi-group operator by discarding all upscattering cross sections.

        Returns
        -------
        MultiGroupOperator
            The lower block triangular multi-group operator.
        ''' 
        cs_data_lower = self.gather_scatter_operator.cross_sections_indexed.cross_section_data.convert_to_downscattering_only()
        xs_indexed_new = eqx.tree_at(lambda xs_idx : xs_idx.cross_section_data, self.gather_scatter_operator.cross_sections_indexed, cs_data_lower)

        return eqx.tree_at(self.cross_sections_indexed_loc, self, replace_fn = lambda _ : xs_indexed_new)
        
    @property 
    def n_elements(self):
        return self.transport_operator.solution_domain_data.domain.n_elements
    
    @property 
    def n_angles(self):
        return self.transport_operator.n_angles
    
    @property 
    def n_moments(self):
        return self.moment_operator.n_moments
    
    @property 
    def n_basis(self):
        return self.transport_operator.solution_domain_data.n_basis
        
    @property
    def cross_sections_indexed_loc(self):
        '''
        Function that returns a function to get the location of the cross section data in the tree.
        Since multiple places hold cross section data, we return a tuple of locations.
        '''
        return lambda m :  (m.transport_operator.solution_domain_data.domain.cross_sections_indexed, m.gather_scatter_operator.cross_sections_indexed)
    
    @property
    def precomputed_geometric_data_loc(self):
        '''
        Function that returns a function to get the location of the precomputed geometric data in the tree.
        Since multiple places hold precomputed geometric data, we return a tuple of locations.
        '''
        return lambda m : (m.transport_operator.solution_domain_data.precomputed_geometric_data, m.mass_operator.precomputed_geometric_data)
    
    @property
    def fem_matrices_loc(self):
        '''
        Function that returns a function to get the location of the FEM matrices in the tree.
        Since multiple places hold FEM matrices, we return a tuple of locations.
        '''
        return lambda m : (m.transport_operator.solution_domain_data.fem_matrices, m.mass_operator.fem_matrices)
    

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
    