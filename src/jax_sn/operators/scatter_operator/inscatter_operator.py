import jax
import jax.numpy as jnp

from .scatter_operator_util import _scatter_kernel_xs_indexed_discrete
from jax_sn.operators.base_linear_operator import AbstractLinearOperator_sn
from jax_sn.domain.geometric_domain import CrossSectionIndexed

from jax_sn.quadrature_set import QuadratureSet

import lineax as lx

class InScatterOperator(AbstractLinearOperator_sn):
    yklm                  : jnp.ndarray           # [n_angles, n_moments]
    cross_section_indexed : CrossSectionIndexed
    energy_group          : int
    elements_basis_shape  : jax.ShapeDtypeStruct  #[n_elements, n_basis]
    
    @classmethod 
    def from_quadrature_set_and_cross_sections(cls, quadrature_set : QuadratureSet, cross_sections : CrossSectionIndexed, energy_group : int, l_max : int, elements_basis_shape : jax.ShapeDtypeStruct) -> "InScatterOperator":
        '''
        Constructs an InScatterOperator from a quadrature set and cross sections.
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
        InScatterOperator
            The constructed InScatterOperator.
        
        '''
        yklm = quadrature_set.ylm_real(l_max)  # [n_angles, n_moments]
        return cls(yklm, cross_sections, energy_group, elements_basis_shape)
    
    def __check_init__(self):
        assert self.energy_group >=0 and self.energy_group < self.cross_section_indexed.n_energy_groups, f"Energy group {self.energy_group} out of bounds for cross sections with {self.cross_section_indexed.n_energy_groups} energy groups."
        assert self.yklm.ndim    == 2, f"yklm must be 2D [n_angles, n_moments], but has shape {self.yklm.shape}."
        assert self.elements_basis_shape.ndim == 2, f"elements_basis_shape must be 2D [n_elements, n_basis], but has shape {self.elements_basis_shape.shape}."
        assert self.elements_basis_shape.shape[0] == self.cross_section_indexed.indices.shape[0], f"Number of elements in elements_basis_shape {self.elements_basis_shape.shape[0]} must match number of elements in cross_section_indexed {self.cross_section_indexed.indices.shape[0]}."

    
    def mv(self, x : jnp.ndarray) -> jnp.ndarray:
        assert x.shape == self.in_structure().shape, f"Input x shape {x.shape} does not match expected in_structure shape {self.in_structure().shape}."
        return total_scattering_kernel_xs_indexed_discrete(
                                                    self.energy_group,  # g_out = self energy group
                                                    self.energy_group,  # g_in  = self energy group
                                                    self.cross_section_indexed,
                                                    x,
                                                    self.yklm)
    def transpose(self) -> "InScatterOperator":
        raise NotImplementedError("Transpose of InScatterOperator is not implemented. [but will be required!]")        
    
    def in_structure(self):
        return jax.ShapeDtypeStruct(
            shape=(self.yklm.shape[1],) + self.elements_basis_shape.shape,
            dtype=self.yklm.dtype)
    def out_structure(self):
        return jax.ShapeDtypeStruct(
            shape=(self.yklm.shape[0],) + self.elements_basis_shape.shape,
            dtype=self.yklm.dtype)

@lx.is_symmetric.register(InScatterOperator)
def _(operator):
    return False
    
@lx.linearise.register(InScatterOperator)
def _(operator):
    return operator


@jax.jit
def total_scattering_kernel_xs_indexed_discrete(        
        g_out                       : int,
        g_in                        : int,
        cross_section_indexed       : CrossSectionIndexed,
        moments_array               : jnp.ndarray,
        yklm                        : jnp.ndarray
    ) -> jnp.ndarray:
    r'''
    Computes the total scattering source in discrete angles for multiple elements.

    Computes:

    .. math::
        s_{kei} = \sum_{lm} Y_{klm} \sigma_{l, g_{out}\from g_{in}}^{(e)} \psi_{lm, ei}^{ g_{in})}

    Ensures the angles are the leading dimension. 

    Parameters
    ----------
    elem_indices : jnp.ndarray [n_elements]
        The element indices to compute the scattering source for.
    g_out : int
        The outgoing energy group index.
    g_in : int
        The incoming energy group index.
    cross_section_indexed : CrossSectionIndexed
        The indexed cross section data structure.
    moments_array : jnp.ndarray [n_moments, n_elements, n_basis]
        The moments array containing angular flux moments.
    yklm : jnp.ndarray [n_angles, n_moments]
        The yklm matrix for the quadrature set.

    Returns
    -------
    jnp.ndarray [n_angles, n_elements, n_basis]
        The total scattering source in discrete angles for the specified elements and energy groups.
    
    '''    
    return jax.vmap(
        _scatter_kernel_xs_indexed_discrete,
        in_axes=(0, None, None, None, None, None),
        out_axes=1
    )(jnp.arange(cross_section_indexed.indices.shape[0]), g_out, g_in, cross_section_indexed, moments_array, yklm)