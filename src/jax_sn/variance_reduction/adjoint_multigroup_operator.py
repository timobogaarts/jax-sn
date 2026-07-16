import jax
import jax.numpy as jnp
from jax_sn.domain.cross_sections import CrossSectionData
from jax_sn.operators.base_linear_operator import AbstractLinearOperator_sn
from jax_sn.operators.multi_group_operator.lower_triangular_solver import MultiGroupLowerTriangularBlockSolver
from jax_sn.operators.single_group_operator import SingleGroupOperator
from jax_sn.operators.moment_operator import MomentOperator, DiscreteOperator
from jax_sn.operators.transport_operator import TransportOperatorMultiAngle
from jax_sn.operators.mass_operator import MassOperator
# REFACTOR-MARKER(gather-scatter): GatherScatterOperator is now a lineax Data+Strategy operator; the
# .gather_backward(x, g) calls below become gs_strategy.transpose_group(GatherScatterData(cross_sections), x, g).
from jax_sn.operators.scatter_operator import GatherScatterOperator, GatherScatterStrategy
import lineax as lx 
import equinox as eqx
from abc import abstractmethod
from jax_sn.operators.multi_group_operator import MultiGroupOperatorScan, MultiGroupOperator


class AdjointMultiGroupOperator(MultiGroupOperator):
    
    @abstractmethod
    def _mv_forward(self,x):
        ...
    
    def _mv_transpose(self,x):
        raise NotImplementedError("Transpose not implemented for the Adjoint operator. Be aware that this is not the same as the transpose of a MultiGroupOperator!.")
    
    def transpose(self):
        raise NotImplementedError("Transpose not implemented for the Adjoint operator. Be aware that this is not the same as the transpose of a MultiGroupOperator!.")
        # return type(self)(self.moment_operator,
        #                   self.discrete_operator,
        #                           self.transport_operator,
        #                           self.mass_operator,
        #                           self.gather_scatter_operator,
        #                           self.n_energy_groups, 
        #                             not self.transpose_bool)        

class AdjointMultiGroupOperatorScan(AdjointMultiGroupOperator):
    '''
    Adjoint multi-group operator implemented using a scan over energy groups. This is memory efficient, since we only need to store temporaries for a single energy group at a time. 

    See :class:`MultiGroupOperator` for details on the mathematical formulation.
    '''
    def _mv_forward(self, x):
        return _adjoint_multi_group_operator_scan(self.n_energy_groups, self.moment_operator, self.transport_operator, self.mass_operator, self.discrete_operator, self.gather_scatter_operator, x, checkpoint=False)

@eqx.filter_jit
def _adjoint_multi_group_operator_scan(n_energy_groups : int, moment_operator    : MomentOperator, transport_operator : TransportOperatorMultiAngle,  mass_operator      : MassOperator,  discrete_operator : DiscreteOperator, gather_scatter_operator : GatherScatterOperator, x : jnp.ndarray, checkpoint = False) -> jnp.ndarray:
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
    transport_operator : TransportOperatorMultiAngle
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
    transport_operator_t = transport_operator.transpose()

    # We have exactly the same memory analysis as for the single group operator, (see ComposedSingleGroupStrategy.mv_forward).
    # Two discrete single-group sized temporaries [n_angles, n_elements, n_basis] are required as in/out for transport operator. 
    in_structure_shape = (n_energy_groups,) + moment_operator.out_structure().shape

    assert x.shape == in_structure_shape, f"Input x shape {x.shape} does not match expected in_structure shape {in_structure_shape}."
        
    def _single_energy_group(_, g_i):
        # Mass before discrete (moment space): mass and discrete commute (independent tensor factors),
        # mirrors MultiGroupOperator's forward scan.
        y_g = (x[g_i] - moment_operator.mv(transport_operator_t.to_energy_group(g_i).inv(discrete_operator.mv(mass_operator.mv(gather_scatter_operator.gather_backward(x, g_i))))))
        return None, y_g
    
    f_seg = jax.checkpoint(_single_energy_group) if checkpoint else _single_energy_group
    
    _, result = jax.lax.scan(
        f_seg,
        init=None,
        xs=jnp.arange(n_energy_groups),
    )
    return result

from .right_hand_side import RightHandSide

@eqx.filter_jit
def solve_multigroup_operator_rhs_adjoint(operator : AdjointMultiGroupOperator, rhs : RightHandSide, solver : lx.AbstractLinearSolver) :
    r'''
    Solve the multi-group operator system using an iterative solver. The right-hand-side is provided as a RightHandSide object, which can override the default behaviour for specific applications.
    It also conditions the right hand side using 

    .. math::
        \mathbf{M} \mathbf{T}^{-1} \mathbf{H} \mathbf{D}[b]

    Parameters
    ----------
    operator : MultiGroupOperator
        The multi-group operator.
    rhs : RightHandSide
        The right-hand side.
    solver : lx.AbstractLinearSolver
        The lineax solver to use.

    Returns
    -------
    lx.LinearSolveOutput
        The solution output from lineax.
    '''    
    
    assert rhs.shape == operator.out_structure().shape, f"RHS shape {rhs.shape} does not match expected operator structure [n_energy_groups, n_angles, n_elements, n_basis]: \
                                                        {operator.out_structure().shape}."
    
    assert isinstance(operator, AdjointMultiGroupOperator), f"Operator must be an instance of AdjointMultiGroupOperator but got {type(operator)}."
    assert not isinstance(solver, MultiGroupLowerTriangularBlockSolver), "MultiGroupLowerTriangularBlockSolver is not compatible with the adjoint operator. Please use a different solver, e.g. GMRES or BiCGSTAB from lineax."
    
    transport_operator_t = operator.transport_operator.transpose()
    def body_fn(_, g_i):
        # Mass before discrete (moment space): see _adjoint_multi_group_operator_scan.
        return None, operator.moment_operator.mv(transport_operator_t.to_energy_group(g_i).inv(operator.discrete_operator.mv(operator.mass_operator.mv(rhs[g_i]))))

    rhs_mtinvh = jax.lax.scan(body_fn, None, jnp.arange(operator.n_energy_groups))[1]
    return lx.linear_solve(operator, rhs_mtinvh, solver)
