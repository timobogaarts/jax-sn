import jax
import jax.numpy as jnp
from .multi_group_operator_imp import MultiGroupOperator
import lineax as lx 
import equinox as eqx
from abc import abstractmethod


class RightHandSide(eqx.Module):
    '''
    Class that can override the default right-hand-side behaviour for specific applications.

    Example is a fixed source problem with only neutrons on the first energy group.
    '''

    structure : jax.ShapeDtypeStruct
    '''ShapeDtypeStruct for static shape/dtype checking. [n_groups, n_moments, n_elements, n_basis]'''

    @property
    def shape(self) -> tuple:
        return self.structure.shape

    @abstractmethod
    def __getitem__(self, group_idx : int) -> jnp.ndarray:
        r'''
        Get the right-hand-side for the given energy group index. 

        Parameters
        ----------
        group_idx : int
            The energy group index for which to get the right-hand-side.

        Returns
        -------
        jnp.ndarray
            The right-hand-side for the given energy group index.
        '''
        pass


class IsotropicSourceSingleGroup(RightHandSide):
    
    energy_group : int
    isotropic_value : jnp.ndarray
    
    @classmethod
    def from_shape(cls, shape : jax.ShapeDtypeStruct, isotropic_value : jnp.ndarray, energy_group : int = 0):
        r'''
        Create an isotropic source right-hand-side for a single energy group. The first energy group will have the isotropic value, while the other groups will be zero.

        Parameters
        ----------
        shape : jax.ShapeDtypeStruct
            The shape of the right-hand-side, which should be [n_groups, n_moments, n_elements, n_basis].
        isotropic_value : jnp.ndarray
            The isotropic value to use for the first energy group. Should have shape [n_elements, n_basis].

        Returns
        -------
        IsotropicSourceSingleGroup
            The created isotropic source right-hand-side for a single energy group.
        '''
        return cls(shape, energy_group, isotropic_value)  # shape is already a ShapeDtypeStruct

    @classmethod
    def from_operator(cls, operator : MultiGroupOperator, isotropic_value : jnp.ndarray, energy_group : int = 0):
        r'''
        Create an isotropic source right-hand-side for a single energy group from a multi-group operator. The first energy group will have the isotropic value, while the other groups will be zero.

        Parameters
        ----------
        operator : MultiGroupOperator
            The multi-group operator for which to create the right-hand-side. The shape of the right-hand-side will be determined from the operator's output structure.
        isotropic_value : jnp.ndarray
            The isotropic value to use for the first energy group. Should have shape [n_elements, n_basis].

        Returns
        -------
        IsotropicSourceSingleGroup
            The created isotropic source right-hand-side for a single energy group.
        '''
        return cls(operator.out_structure(), energy_group, isotropic_value)

    def __check_init__(self):
        assert len(self.shape) == 4, f"Shape must be of the form [n_groups, n_moments, n_elements, n_basis], but got {self.shape}."
        assert self.isotropic_value.shape == self.shape[2:], f"Isotropic value must have shape [n_elements, n_basis] ={self.shape[2:]}, but got {self.isotropic_value.shape}."        

    def __getitem__(self, group_idx: int) -> jnp.ndarray:        
        return jnp.where(group_idx == self.energy_group, jnp.stack([self.isotropic_value] + [jnp.zeros_like(self.isotropic_value)] * (self.shape[1] - 1), axis=0), jnp.zeros(self.shape[1:]))

class TotalSource(RightHandSide):

    total_source : jnp.ndarray

    @classmethod
    def from_operator_and_isotropic_source(cls, operator : MultiGroupOperator, total_isotropic_source : jnp.ndarray):
        r'''
        Create an isotropic total source right-hand-side from a multi-group operator. The shape of the right-hand-side will be determined from the operator's output structure.

        Parameters
        ----------
        operator : MultiGroupOperator
            The multi-group operator for which to create the right-hand-side. The shape of the right-hand-side will be determined from the operator's output structure.
        total_source : jnp.ndarray
            The total source to use for all energy groups. Should have shape [n_groups, n_moments, n_elements, n_basis].

        Returns
        -------
        IsotropicTotalSource
            The created isotropic total source right-hand-side.
        '''
        total_source = jnp.stack([total_isotropic_source] + [jnp.zeros_like(total_isotropic_source)] * (operator.n_moments -1), axis=1)
        return cls(operator.out_structure(), total_source)

    def __check_init__(self):
        assert self.structure == jax.ShapeDtypeStruct(self.total_source.shape, self.total_source.dtype), f"Structure of total source {jax.ShapeDtypeStruct(self.total_source.shape, self.total_source.dtype)} must match expected structure {self.structure}."

    def __getitem__(self, group_idx: int) -> jnp.ndarray:
        return self.total_source[group_idx]


def _to_struct(x) -> jax.ShapeDtypeStruct:
    """Normalise a RightHandSide or a plain jax.Array to a ShapeDtypeStruct."""
    if isinstance(x, RightHandSide):
        return x.structure
    return jax.ShapeDtypeStruct(x.shape, x.dtype)


@eqx.filter_jit
def solve_multigroup_operator_rhs(operator : MultiGroupOperator, rhs : RightHandSide, solver : lx.AbstractLinearSolver) :
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
    assert _to_struct(rhs) == operator.out_structure(), f"RHS structure {_to_struct(rhs)} does not match expected operator structure [n_energy_groups, n_angles, n_elements, n_basis]: \
                                                        {operator.out_structure()}."
    
    def body_fn(_, g_i):
        # Mass before discrete (moment space): mass and discrete commute (independent tensor factors),
        # mirrors MultiGroupOperator's forward scan.
        return None, operator.moment_operator.mv(operator.transport_operator.to_energy_group(g_i).inv(operator.discrete_operator.mv(operator.mass_operator.mv(rhs[g_i]))))

    rhs_mtinvh = jax.lax.scan(jax.checkpoint(body_fn), None, jnp.arange(operator.n_energy_groups))[1]
    return lx.linear_solve(operator, rhs_mtinvh, solver)


