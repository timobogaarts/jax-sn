import jax
import jax.numpy as jnp
import lineax as lx
import equinox as eqx
from abc import abstractmethod

from jax_sn.operators.multi_group_operator import MultiGroupOperator, MultiGroupChainStrategyBase


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
        r'''Get the right-hand-side for energy group ``group_idx`` (shape ``[n_moments, n_elements, n_basis]``).'''
        ...


class IsotropicSourceSingleGroup(RightHandSide):

    energy_group : int
    isotropic_value : jnp.ndarray

    @classmethod
    def from_shape(cls, shape : jax.ShapeDtypeStruct, isotropic_value : jnp.ndarray, energy_group : int = 0):
        r'''Isotropic source living in a single energy group; all other groups are zero.'''
        return cls(shape, energy_group, isotropic_value)

    @classmethod
    def from_operator(cls, operator : MultiGroupOperator, isotropic_value : jnp.ndarray, energy_group : int = 0):
        r'''As :meth:`from_shape`, taking the operand structure from ``operator.out_structure()``.'''
        return cls(operator.out_structure(), energy_group, isotropic_value)

    def __check_init__(self):
        assert len(self.shape) == 4, f"Shape must be [n_groups, n_moments, n_elements, n_basis], but got {self.shape}."
        assert self.isotropic_value.shape == self.shape[2:], f"Isotropic value must have shape [n_elements, n_basis] = {self.shape[2:]}, but got {self.isotropic_value.shape}."

    def __getitem__(self, group_idx: int) -> jnp.ndarray:
        return jnp.where(group_idx == self.energy_group,
                         jnp.stack([self.isotropic_value] + [jnp.zeros_like(self.isotropic_value)] * (self.shape[1] - 1), axis=0),
                         jnp.zeros(self.shape[1:]))


class TotalSource(RightHandSide):

    total_source : jnp.ndarray

    @classmethod
    def from_operator_and_isotropic_source(cls, operator : MultiGroupOperator, total_isotropic_source : jnp.ndarray):
        r'''Isotropic total source (all groups), zero in the higher moments.'''
        total_source = jnp.stack([total_isotropic_source] + [jnp.zeros_like(total_isotropic_source)] * (operator.data.n_moments - 1), axis=1)
        return cls(operator.out_structure(), total_source)

    def __check_init__(self):
        assert self.structure == jax.ShapeDtypeStruct(self.total_source.shape, self.total_source.dtype), \
            f"Structure of total source {jax.ShapeDtypeStruct(self.total_source.shape, self.total_source.dtype)} must match expected structure {self.structure}."

    def __getitem__(self, group_idx: int) -> jnp.ndarray:
        return self.total_source[group_idx]


def _to_struct(x) -> jax.ShapeDtypeStruct:
    """Normalise a RightHandSide or a plain jax.Array to a ShapeDtypeStruct."""
    if isinstance(x, RightHandSide):
        return x.structure
    return jax.ShapeDtypeStruct(x.shape, x.dtype)


@eqx.filter_jit
def solve_multigroup_operator_rhs(operator : MultiGroupOperator, rhs : RightHandSide,
                                  solver : lx.AbstractLinearSolver,
                                  chain_strategy : MultiGroupChainStrategyBase):
    r'''
    Solve :math:`\mathbf{A}\psi = \mathbf{C}[b]` for the multi-group operator ``operator``.

    Parameters
    ----------
    operator : MultiGroupOperator
        The multi-group operator; its ``mv`` defines the differentiation surface.
    rhs : RightHandSide
        The physical right-hand side (or a bare array of matching structure).
    solver : lx.AbstractLinearSolver
        The lineax solver applied to the preconditioned system.
    chain_strategy : MultiGroupChainStrategyBase
        Per-group chain, lifted over groups, used to precondition the source. Build via
        ``ComposedMultiGroupChainStrategy.from_composed(operator.strategy)`` to match the operator's own lift.

    Returns
    -------
    lx.LinearSolveOutput
    '''
    assert _to_struct(rhs) == operator.out_structure(), \
        f"RHS structure {_to_struct(rhs)} does not match operator structure {operator.out_structure()}."

    rhs_conditioned = chain_strategy.forward(operator.data, rhs)
    return lx.linear_solve(operator, rhs_conditioned, solver)
