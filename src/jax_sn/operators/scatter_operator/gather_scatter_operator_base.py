import jax

from jax_sn.operators.base_linear_operator import (
    BaseOperatorData, TransposableStrategy, TransposeBoolDataStrategyLinearOperator,
)
from jax_sn.domain.cross_sections import CrossSectionsIndexed
import equinox as eqx
from abc import abstractmethod
import jax.numpy as jnp

class GatherScatterData(BaseOperatorData):
    '''Data for the gather-scatter operator: the indexed cross sections it contracts against.'''
    cross_sections_indexed : CrossSectionsIndexed

class SingleGroupGatherScatterStrategy(eqx.Module):
    r'''
    Gathers the scatter source in a single energy group, either from incoming groups (normal scattering) or
    from outgoing groups (transpose/adjoint scattering). 

    .. math::
        \sum_{g'} \sigma_{\ell g g',e}\, x_{g'\ell e b} \quad\text{or}\quad \sum_{g'} \sigma_{\ell g' g,e}\, x_{g'\ell e b }

    Should be batched over trailing axes (e.g. elements, basis functions). Note this is not a Transpose: it doesn't satisfy the adjoint identity in this single-group form! 
    This is why this is a separate object, not a strategy.

    This can be used either to define the total GatherScatter operation (by lifting over the group axis) or to define a single-group operator (by fixing the group index).
    This just applies the single-group version.
    '''
    
    @abstractmethod
    def gather_group(self, data : GatherScatterData, x : jnp.ndarray, g : int, sum_over_outgoing : bool) -> jnp.ndarray:
        ...

    def incoming_gather(self, data : GatherScatterData, x : jnp.ndarray, g : int) -> jnp.ndarray:
        return self.gather_group(data, x, g, sum_over_outgoing=False)
    
    def outgoing_gather(self, data : GatherScatterData, x : jnp.ndarray, g : int) -> jnp.ndarray:
        return self.gather_group(data, x, g, sum_over_outgoing=True)
    

class GatherScatterStrategy(TransposableStrategy):
    r'''
    The total gather scatter operator:

    .. math::
        \mathbf{U} : [G,M,E,B] \to [G,M,E,B], \quad (\mathbf{U} x)_{g\ell e b} = \sum_{g'} \sigma_{\ell g g',e}\, x_{g'\ell e b}

    This is transposable:

    .. math::
        \mathbf{U}^T : [G,M,E,B] \to [G,M,E,B], \quad (\mathbf{U}^T x)_{g\ell e b} = \sum_{g'} \sigma_{\ell g' g,e}\, x_{g' \ell e b}

    and is therefore a :class:`TransposableStrategy`. 
    '''


class GatherScatterOperator(TransposeBoolDataStrategyLinearOperator):
    r'''
    The all-groups scatter operator :math:`\mathbf{U}` (or :math:`\mathbf{U}^T`) as a lineax operator on
    ``[n_groups, n_moments, n_elements, n_basis]``.
    '''
    data                               : GatherScatterData
    strategy                           : GatherScatterStrategy
    group_moments_elements_basis_shape : jax.ShapeDtypeStruct
    r'''Full operand structure ``[n_groups, n_moments, n_elements, n_basis]`` (cross sections fix only groups + elements).'''
    transpose_bool                     : bool = False

    def _in_structure(self):
        return self.group_moments_elements_basis_shape

    def _out_structure(self):
        return self.group_moments_elements_basis_shape


class ComposedGatherScatterStrategy(GatherScatterStrategy):
    '''The full scatter operator built from a per-group row + something that maps it to the full operator'''
    single_group_strategy     : SingleGroupGatherScatterStrategy
    
    @abstractmethod
    def forward(self, data : GatherScatterData, x : jnp.ndarray) -> jnp.ndarray:
        ...

    @abstractmethod
    def transpose(self, data : GatherScatterData, x : jnp.ndarray) -> jnp.ndarray:
        ...

