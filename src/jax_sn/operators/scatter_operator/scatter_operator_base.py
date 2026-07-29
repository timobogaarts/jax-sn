import jax

from jax_sn.operators.base_linear_operator import (
    BaseOperatorData, SymmetricDataStrategyLinearOperator, SymmetricStrategy,
)
from jax_sn.domain.cross_sections import CrossSectionsIndexed
import equinox as eqx


class ScatterData(BaseOperatorData):
    energy_group_out       : int
    energy_group_in        : int
    cross_sections_indexed : CrossSectionsIndexed

class ScatterStrategy(SymmetricStrategy):
    '''How to apply the (symmetric-shaped) scatter contraction. Reads ``data``.'''

class ScatterOperator(SymmetricDataStrategyLinearOperator):
    r'''
    Maps the flux moments to scattering for a specific energy group in and a specific energy group out, using indexed cross sections.

    Mathematical description
    ---------------------------
    We have:

    .. math::
        \mathbf{S}_{\tilde{g} g} = \overline{\mathbf{S}}_{\tilde{g} g}  \otimes \mathbf{I}_{\mathcal{B}} : \mathcal{M} \otimes \mathcal{E}\otimes \mathcal{B} \to \mathcal{M} \otimes \mathcal{E}\otimes \mathcal{B}

    with :math:`\tilde{g}` the outgoing energy group, :math:`g` the incoming energy group, :math:`\mathcal{F} = \mathcal{E}\otimes \mathcal{B}` the space of elements and basis functions, and :math:`\mathbf{S}_{\tilde{g} g}` the single-element scattering operator defined as:

    .. math:: 
        \overline{\mathbf{S}}_{\tilde{g} g} [\psi] 
         =\sigma_{\ell \tilde{g} g e} \psi_{g\ell e}    

    for an arbitrary :math:`\psi\in \mathcal{M} \otimes \mathcal{E}` (we use :math:`\ell` to denote :math:`(l,m)` moment index for brevity; for :math:`\sigma_{\ell}` this just equals :math:`\sigma_{l}` for all :math:`m`).

    Transposition
    ---------------------------
    Since this is a element wise multiplication, the transpose is itself.

    Technical details
    ---------------------------
    We use a :class:`jax.ShapeDtypeStruct` to represent the shape of the moments, elements and basis functions over which this operator is defined. This is used to define a consistent in and output structure.    
    Since we want to have the flexibility to have a different number of moments than is present in the cross sections, we explicitly provide it here.

    '''
    data                   : ScatterData
    r'''ScatterData containing the indexed cross sections for the scatter operator. [differentiable except for groups]'''
    
    strategy               : ScatterStrategy
    r'''How to apply the scatter contraction.'''

    moments_elements_basis_shape : jax.ShapeDtypeStruct
    r'''The shape of the moments, elements and basis functions over which this operator is batched. Identified with :math:`\mathcal{M}\otimes \mathcal{E}\otimes \mathcal{B}` in the documentation above.'''

    def __check_init__(self):
        assert self.moments_elements_basis_shape.ndim == 3, f"elements_basis_shape must be 2D [moments_n_elements, n_basis], but has shape {self.moments_elements_basis_shape.shape}."
        assert self.moments_elements_basis_shape.shape[1] == self.data.cross_sections_indexed.indices.shape[0], f"Number of elements in elements_basis_shape {self.moments_elements_basis_shape.shape[0]} must match number of elements in cross_sections_indexed {self.data.cross_sections_indexed.indices.shape[0]}."

    def _structure(self):
        return self.moments_elements_basis_shape

    def to_energy_group(self, energy_group_out : int, energy_group_in : int) -> "ScatterOperator":
        return eqx.tree_at(lambda op: (op.data.energy_group_out, op.data.energy_group_in), self,
                           (energy_group_out, energy_group_in))
