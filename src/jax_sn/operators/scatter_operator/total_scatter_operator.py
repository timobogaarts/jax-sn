import jax
import jax.numpy as jnp
from jax_sn.operators.moment_operator.moment_operator_imp import MomentOperator
from jax_sn.quadrature_set import QuadratureSet
from jax_sn.quadrature_set.quadrature_set_base import _lm_real_expansion
from jax_sn.domain.geometric_domain import CrossSectionIndexed
from jax_sn.operators.base_linear_operator import AbstractLinearOperator_sn
import lineax as lx
import equinox as eqx

from .scatter_operator_util import total_scattering_kernel_xs_indexed_multigroup_discrete#, _total_scattering_kernel_xs_indexed_multigroup_discrete_transpose

class TotalGroupScatterOperator(AbstractLinearOperator_sn):
    r'''
    TotalScatterOperator represents the total scattering operator that maps angular flux moments of all groups to discrete angles in one group using indexed cross sections.

    Implements:

    .. math::
        \mathbf{S} = \mathbf{S}_G \otimes \mathbf{I}_F : \mathcal{G}\otimes\mathcal{M} \otimes F \to  \mathcal{D} \otimes F

    where :math:`\mathbf{S}_G` is the single-element in-scattering operator defined as:
    .. math::
        s_{ki} = \sum_{g_{in}}\sum_{lm} Y_{klm} \sigma_{l, g_{out}\from g_{in}}^{(e)} \psi_{lm, i}^{ g_{in})} =\sum_{lm} S_{klm}^{g_{out}\from g_{in}} \psi_{lm, i}^{ g_{in})}        
    
        
    Transpose does not make sense since this operator is merely a convenience operator to gather scattering from all groups into one group efficiently.
    '''
    yklm                         : jnp.ndarray           # [n_angles, n_moments]
    cross_section_indexed        : CrossSectionIndexed
    energy_group                 : int
    groups_elements_basis_shape  : jax.ShapeDtypeStruct  #[n_groups, n_elements, n_basis]
    
    @classmethod 
    def from_quadrature_set_and_cross_sections(cls, quadrature_set : QuadratureSet, cross_sections : CrossSectionIndexed, energy_group : int, l_max : int, groups_elements_basis_shape : jax.ShapeDtypeStruct) -> "TotalGroupScatterOperator":
        '''
        Constructs a TotalScatterOperator from a quadrature set and cross sections.
        Note this is merely a convenience function: the yklm matrix is explicitly calculated here.
        The default constructor can be used if the yklm matrix is already available.

        Parameters
        ----------
        quadrature_set : QuadratureSet
            The quadrature set to use.
        cross_sections : CrossSectionIndexed
            The cross sections to use.
        energy_group : int
            The energy group to use.
        l_max : int [0, ....]
            The maximum moment to use.  

        Returns
        -------
        TotalScatterOperator
            The constructed TotalScatterOperator.
        
        '''
        yklm = quadrature_set.ylm_real(l_max)  # [n_angles, n_moments]
        return cls(yklm, cross_sections, energy_group, groups_elements_basis_shape)
    
    def __check_init__(self):
        
        # assert self.energy_group >=0 and self.energy_group < self.cross_section_indexed.n_energy_groups, f"Energy group {self.energy_group} out of bounds for cross sections with {self.cross_section_indexed.n_energy_groups} energy groups."
        # assert self.energy_group < self.groups_elements_basis_shape.shape[0], f"Energy group {self.energy_group} out of bounds for groups_elements_basis_shape with {self.groups_elements_basis_shape.shape[0]} energy groups."
        assert self.yklm.ndim    == 2, f"yklm must be 2D [n_angles, n_moments], but has shape {self.yklm.shape}."
        assert self.groups_elements_basis_shape.ndim == 3, f"groups_elements_basis_shape must be 3D [n_elements, n_basis], but has shape {self.groups_elements_basis_shape.shape}."
        assert self.groups_elements_basis_shape.shape[1] == self.cross_section_indexed.indices.shape[0], f"Number of elements in elements_basis_shape {self.groups_elements_basis_shape.shape[1]} must match number of elements in cross_section_indexed {self.cross_section_indexed.indices.shape[0]}."
        self.cross_section_indexed.cross_section_data.assert_n_energy_groups(self.groups_elements_basis_shape.shape[0]), f"Number of energy groups in cross_section_indexed {self.cross_section_indexed.n_energy_groups} must match number of groups in groups_elements_basis_shape {self.groups_elements_basis_shape.shape[0]}."
    
    def mv(self, x : jnp.ndarray) -> jnp.ndarray:
        assert x.shape == self.in_structure().shape, f"Input x shape {x.shape} does not match expected in_structure shape {self.in_structure().shape}."
        return total_scattering_kernel_xs_indexed_multigroup_discrete(self.energy_group, self.cross_section_indexed, x, self.yklm)
    def transpose(self):
        raise ValueError("Transpose not implemented for TotalGroupScatterOperator because it is ill-defined")        
    
    def to_energy_group(self, energy_group : int) -> "TotalGroupScatterOperator":
        return TotalGroupScatterOperator(self.yklm, self.cross_section_indexed, energy_group, self.groups_elements_basis_shape)
    
    def in_structure(self):
        return jax.ShapeDtypeStruct(
            shape=(self.groups_elements_basis_shape.shape[0], ) + (self.yklm.shape[1],) + self.groups_elements_basis_shape.shape[1:],
            dtype=self.yklm.dtype)
    def out_structure(self):
        return jax.ShapeDtypeStruct(
            shape=(self.yklm.shape[0],) + self.groups_elements_basis_shape.shape[1:],
            dtype=self.yklm.dtype)
    
    def to_inscatter(self, energy_group : int):
        from jax_sn.operators.scatter_operator.inscatter_operator import InScatterOperator
        return InScatterOperator(self.yklm, self.cross_section_indexed, energy_group, jax.ShapeDtypeStruct(self.groups_elements_basis_shape.shape[1:], dtype=self.groups_elements_basis_shape.dtype))

@lx.is_symmetric.register(TotalGroupScatterOperator)
def _(operator):
    return False
    
@lx.linearise.register(TotalGroupScatterOperator)
def _(operator):
    return operator


