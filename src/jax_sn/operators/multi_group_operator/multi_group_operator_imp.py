import jax
import jax.numpy as jnp
from jax_sn.operators.base_linear_operator import AbstractLinearOperator_sn
from jax_sn.operators.single_group_operator import SingleGroupOperator, SingleGroupOperatorTranspose
from jax_sn.operators.moment_operator import MomentOperator, MomentOperatorTransposed
from jax_sn.operators.scatter_operator.total_scatter_operator import TotalGroupScatterOperator
from jax_sn.operators.transport_operator import TransportOperatorMultipleAngles
from jax_sn.operators.mass_operator import MassOperator
from jax_sn.operators.scatter_operator.total_scatter_operator import TotalGroupScatterOperator
from jax_sn.operators.scatter_operator.scatter_operator_util import _total_scattering_kernel_xs_indexed_multigroup_backwards
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
    
    where :math:`\mathbf{M}` is the batched moment operator (see :class:`MomentOperator`), :math:`\mathbf{T}_g` is the transport operator for group g (see :class:`TransportOperatorMultipleAngles`),  :math:`\mathbf{H}` is the mass operator (see :class:`MassOperator`), 
    :math:`\mathbf{D}` is the moment-to-discrete operator (see :class:`DiscreteOperator`), and :math:`\mathbf{S}_{ij}` is the scattering operator from group j to group i.

    This is defined as:

    .. math::
        \mathbf{S}_{ij} = \bar{\mathbf{S}}_{ij} \otimes \mathbf{I}_B : \mathcal{M} \otimes \mathcal{F} \to  \mathcal{M} \otimes \mathcal{F}

    with :math:`\bar{\mathbf{S}}_{ij}` the single-element scattering operator defined as:
    
    .. math::
        \bar{\mathbf{S}}_{ij}[\psi] = s_{\ell e} = \sigma_{\ell i j e} \psi_{\ell e}

    These are simply combined to form the multi-group operator.
    

    Forward Structure
    ---------------------------

    Note that for a specific row (energy group), the operator acts as a single group operator plus a scattering source from all other energy groups. Writing out explicitly the action on a vector :math:`\psi` for energy group g, we have:

    .. math::
        \mathbf{A}[\psi]_g = \left[\mathbf{I} - \mathbf{M}\mathbf{T}_g^{-1}\mathbf{H}\mathbf{S}_{gg} \right] \psi_{g} - \sum_{g' \neq g} \mathbf{M}\mathbf{T}_g^{-1}\mathbf{H}\mathbf{S}_{gg'} \psi_{g'}

    which is equivalent to:

    .. math::
        \mathbf{A}[\psi]_g = \left[\mathbf{I} - \mathbf{M} \mathbf{T}_g^{-1} \mathbf{H}\left(\mathbf{S}_{gg}[\psi_g] - \sum_{g'\neq g} \mathbf{S}_{gg'}[\psi_{g'}]\right)\right] \equiv \left[\mathbf{I} - \mathbf{M} \mathbf{T}_g^{-1} \mathbf{H} \mathbf{S}_g^{total}[\psi]\right]

    Since the inversion of the transport operator is by far the most expensive operation, in this way, we minimize the number of sweeps per energy group to just one.

    This latter operator:

    .. math::
        \mathbf{S}_g^{total}[\psi] = \mathbf{S}_{gg}[\psi_g] + \sum_{\tilde{g}} \mathbf{S}_{g\tilde{g}}[\psi_{\tilde{g}}]

    is implemented efficiently in the :class:`TotalGroupScatterOperator` class.

    Transposed Structure
    ---------------------------
    The transposition structure is not as straightforward as the forward structure, since the transpose of the multi-group operator. We have 

    .. math::
        \mathbf{A}^T = \begin{bmatrix} \mathbf{A}_0^T & B_{10}^T & B_{20}^T & \dots & B_{G0}^T \\ B_{01}^T & \mathbf{A}_1^T & B_{21}^T & \dots & B_{G1}^T \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ B_{0G}^T & B_{1G}^T & B_{2G}^T & \dots & \mathbf{A}_G^T \end{bmatrix}

    For a single row, we have 

    .. math::
        \mathbf{A}^T[\psi]_g = \left[\mathbf{I} - \mathbf{S}_{gg}^T \mathbf{H}^T (\mathbf{T}_g^{-1})^T \mathbf{M}^T \right] \psi_{g} - \sum_{g' \neq g} \mathbf{S}_{g'g}^T \mathbf{H}^T (\mathbf{T}_{g'}^{-1})^T \mathbf{M}^T \psi_{g'}

    Now, we note that this is not possible take together in one sweep: the transport operator group index is different for all terms in the sum. Naively, we therefore would need to perform G sweeps for each energy group in the transpose case.

    However, a more careful examination shows that although the transport operator is different for each term, the combination of 

    .. math::
        \mathbf{H}^T (\mathbf{T}_{g'}^{-1})^T \mathbf{M}^T
    
    stays the same for all resulting groups :math:`g`. Therefore, we can first compute with :math:`G` sweeps the intermediate quantity:

    .. math::
        \xi_g = \mathbf{H}^T (\mathbf{T}_g^{-1})^T  \mathbf{M}^T \psi_g

    and then combine these with the transposed scattering operators to form the final result:

    .. math::
        \mathbf{A}^T[\psi]_g = \psi_g  - \sum_{g' } \mathbf{S}_{g'g}^T[\xi_{g'}]


    However, this is suboptimal as well: :math:`\xi \in \mathcal{G}\otimes\mathcal{D}\otimes \mathcal{F}` needs to be stored. This can be quite large as the discrete angles are often much larger than the number of moments.

    So we deconstruct the scattering operator as follows (see :class:`InScatterOperator` as well):

    .. math::
        \mathbf{S}_{g\tilde{g}}[\phi]& = \sum_{\ell}Y_{k\ell} \sigma_{\ell g \tilde{g} e} \phi_{\ell e}
                                     &=\mathbf{D} \mathcal{S}_{g\tilde{g}}[\phi]]\\
        \mathcal{S}_{g\tilde{g}}[\phi] & = \sigma_{\ell g \tilde{g} e} \phi_{\ell e}\\
        \mathbf{D}[\chi] & = \sum_{\ell} Y_{k\ell} \chi_{\ell e}        
                                     
    where :math:`\mathbf{D}` is the operator that maps from moments to discrete angles, and :math:`\mathcal{S}_{g\tilde{g}}` is the operator that applies the indexed cross sections and spherical harmonics.

    Therefore, we can write :math:`\mathbf{S}^T_{g\tilde{g}}` as:

    .. math::
        \mathbf{S}_{g\tilde{g}}^T[\xi] & = \mathcal{S}_{g\tilde{g}}^T \mathbf{D}^T[\xi]\\

    with 

    .. math::
        \mathbf{D}^T[\xi] & = \sum_{k} Y_{k\ell} \xi_{ke}\\
        \mathcal{S}_{g\tilde{g}}^T[\chi] & = \sigma_{\ell g \tilde{g} e} \chi_{\ell e} equiv \mathcal{S}_{g\tilde{g}}[\chi]

    So, instead we compute

    .. math::
        \zeta_g = \mathbf{D}^T \mathbf{H}^T (\mathbf{T}_{g}^{-1})^T  \mathbf{M}^T \psi_g

    and then combine these with the transposed indexed scattering operators to form the final result:

    .. math::
        \mathbf{A}^T[\psi]_g = \psi_g - \sum_{g'} \mathcal{S}_{g'g}^T[\zeta_{g'}]

    Note that the transpose of this operator is *not* the same as what would be obtained when one discretizes the adjoint equation.

    Solution
    ---------------------------
    Since this is a iteratively defined linear operator, the standard way to solve systems involving this operator is to use an iterative solver (e.g., GMRES, CG, etc.) with the mv() method defined here. This can 
    be accomplished using e.g. lineax solvers. This allows for the propagation of the derivatives through the solver as well.
    However, in some cases, e.g. downscattering only problems (such as often found in fusion applications), the multi-group operator is lower triangular. In these cases, it is much more efficient to solve the system directly using a block forward substitution approach, solving each energy group sequentially using the single-group operator for that group and the already computed higher energy group solutions to form the scattering source. This is implemented in the function :func:`solve_lower_triangular`.    

    '''
    moment_operator    : MomentOperator
    transport_operator : TransportOperatorMultipleAngles
    mass_operator      : MassOperator
    scatter_operator   : TotalGroupScatterOperator    
    n_energy_groups    : int    
    transpose_bool     : bool = False

    def __check_init__(self):
        self.transport_operator.solution_domain_data.domain.cross_sections_indexed.cross_section_data.assert_n_energy_groups(self.n_energy_groups)
    
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
        return MultiGroupOperator(self.moment_operator,
                                  self.transport_operator,
                                  self.mass_operator,
                                  self.scatter_operator,
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
        Singl   eGroupOperator
            The single group operator for energy group g_i.
        ''' 
        if self.transpose_bool:
            return SingleGroupOperatorTranspose(self.moment_operator,
                                                self.transport_operator.to_energy_group(g_i),
                                                self.mass_operator,
                                                self.scatter_operator.to_inscatter(g_i))
        else:
            return SingleGroupOperator(self.moment_operator, self.transport_operator.to_energy_group(g_i), self.mass_operator, self.scatter_operator.to_inscatter(g_i))
    
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

@lx.is_symmetric.register(MultiGroupOperator)
def _(operator : MultiGroupOperator):
    return False

@lx.linearise.register(MultiGroupOperator)
def _(operator : MultiGroupOperator):
    return operator



@eqx.filter_jit
def _multi_group_operator_scan(n_energy_groups : int, moment_operator    : MomentOperator, transport_operator : TransportOperatorMultipleAngles,  mass_operator      : MassOperator,  scatter_operator   : TotalGroupScatterOperator, x : jnp.ndarray) -> jnp.ndarray:
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
    scatter_operator : TotalGroupScatterOperator    
        The total group scattering operator :math:`\mathbf{S}^{total}`.
    x : jnp.ndarray
        The input vector.   

    Returns
    -------
    jnp.ndarray
        The result of applying the multi-group operator to x.   

    
    '''

    in_structure_shape = (n_energy_groups,) + moment_operator.out_structure().shape

    assert x.shape == in_structure_shape, f"Input x shape {x.shape} does not match expected in_structure shape {in_structure_shape}."
    result = jnp.zeros(in_structure_shape, dtype=moment_operator.in_structure().dtype)
        
    def _single_energy_group(_, g_i):
        scatter_source = scatter_operator.DS_g_mv(g_i, x)
        y_g = (x[g_i] - moment_operator.mv(transport_operator.to_energy_group(g_i).inv(mass_operator.mv(scatter_source))))
        return None, y_g
    
    _, result = jax.lax.scan(
        _single_energy_group,
        init=None,
        xs=jnp.arange(n_energy_groups),
    )

    return result

@eqx.filter_jit
def _compute_zeta_g_scan(n_energy_groups : int, moment_operator_t : MomentOperatorTransposed, transport_operator_t : TransportOperatorMultipleAngles, mass_operator_t : MassOperator, scatter_operator : TotalGroupScatterOperator, x : jnp.ndarray) -> jnp.ndarray:
    
    in_structure_shape = (n_energy_groups,) + moment_operator_t.in_structure().shape
    
    assert x.shape == in_structure_shape, f"Input x shape {x.shape} does not match expected in_structure shape {in_structure_shape}."
    result             = jnp.zeros(in_structure_shape, dtype=moment_operator_t.in_structure().dtype)
            
    def _single_energy_group(_, g_i):        
        return None, scatter_operator.D_transpose_mv(mass_operator_t.mv(transport_operator_t.to_energy_group(g_i).inv(moment_operator_t.mv(x[g_i]))))
            
    result = jax.lax.scan(_single_energy_group, None, jnp.arange(n_energy_groups))[1]
    return result


@eqx.filter_jit
def _multi_group_operator_transpose_scan(n_energy_groups : int, moment_operator    : MomentOperator, transport_operator : TransportOperatorMultipleAngles,  mass_operator      : MassOperator,  scatter_operator   : TotalGroupScatterOperator, x : jnp.ndarray) -> jnp.ndarray:    
    
    moment_operator_t    = moment_operator.transpose()
    transport_operator_t = transport_operator.transpose()
    mass_operator_t      = mass_operator.transpose()

    in_structure_shape   = (n_energy_groups,) + moment_operator_t.in_structure().shape
    assert x.shape == in_structure_shape, f"Input x shape {x.shape} does not match expected in_structure shape {in_structure_shape}."

    zeta_g_stored = _compute_zeta_g_scan(n_energy_groups, moment_operator_t, transport_operator_t, mass_operator_t, scatter_operator, x)

    def _scan_body(_, g_i):                
        return None, x[g_i] - _total_scattering_kernel_xs_indexed_multigroup_backwards(g_i, scatter_operator.cross_sections_indexed, zeta_g_stored)

    _, result = jax.lax.scan(_scan_body, None, jnp.arange(n_energy_groups))
    return result





class MultiGroupOperatorScan(MultiGroupOperator):
    r'''
    Multi energy group linear operator implementation using fori_loop over energy groups.
    Parallelisation is then over angles and elements only.
    
    '''    
    def _mv_forward(self, x : jnp.ndarray) -> jnp.ndarray:
        return _multi_group_operator_scan(self.n_energy_groups, self.moment_operator, self.transport_operator, self.mass_operator, self.scatter_operator, x)
    
    def _mv_transpose(self, x):
        return _multi_group_operator_transpose_scan(self.n_energy_groups, self.moment_operator, self.transport_operator, self.mass_operator, self.scatter_operator, x)
    

@eqx.filter_jit
def solve_lower_triangular(mgop : MultiGroupOperator, solver, rhs : jnp.ndarray) -> jnp.ndarray:
    '''
    Solve the linear system using a given solver, assuming the multi-group operator is lower triangular.
    This is useful for e.g. downscattering only problems. 

    Parameters
    ----------
    mgop : MultiGroupOperator
        The multi group operator.
    solver : lx.LinearSolver
        The linear solver to use.
    rhs : jnp.ndarray
        The right-hand side.
    Returns
    -------
    jnp.ndarray
        The solution.   
    '''
    n_groups = mgop.n_energy_groups    
    solution = jnp.zeros_like(rhs)
    def body_fun(i, solution):        
        top_i = mgop.transport_operator.to_energy_group(i)
        sgop = SingleGroupOperator(mgop.moment_operator, mgop.transport_operator.to_energy_group(i), mgop.mass_operator, mgop.scatter_operator.to_inscatter(i))
        rhs_i = rhs[i] + mgop.moment_operator.mv(top_i.inv(mgop.mass_operator.mv(mgop.scatter_operator.DS_g_mv(i,solution))))        
        return solution.at[i].set(lx.linear_solve(sgop, rhs_i, solver, throw = False).value)
    return jax.lax.fori_loop(0, n_groups, body_fun, solution) 

@eqx.filter_jit
def solve_total(mgop : MultiGroupOperator, solver, rhs : jnp.ndarray) -> jnp.ndarray:
    r'''
    Solve the linear system using a given solver.

    Parameters
    ----------
    mgop : MultiGroupOperator
        The multi group operator.
    solver : lx.LinearSolver
        The linear solver to use.
    rhs : jnp.ndarray
        The right-hand side.

    Returns
    -------
    jnp.ndarray
        The solution.
    '''    
    return lx.linear_solve(mgop, rhs, solver).value