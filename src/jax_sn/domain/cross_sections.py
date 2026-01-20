r'''
Dataclasses for cross section data representation.
It contains matrices for total and scattering cross sections, both for single and multiple energy groups.

.. math::
    \sigma_t 

.. math::
        \sigma_{s,l}^{g' \to g}

'''
import jax 
from dataclasses import dataclass
import jax.numpy as jnp
from typing import List
import equinox as eqx

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CrossSectionDataSingleGroup:
    '''
    Dataclass to hold cross section data for a single energy group.
    Batched over leading dimensions
    '''

    total           : jnp.ndarray    
    '''total cross section, batched arbitrarily over leading dimensions. Shape is [...].'''
    
    scattering     : jnp.ndarray    #[..., n_L]
    '''scattering cross section moments, batched arbitrarily over leading dimensions. Shape is [..., n_L]'''


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CrossSectionData:
    '''
    Dataclass to hold cross section data for multiple energy groups.
    Batched over leading dimensions.
    '''

    total           : jnp.ndarray    #[...,       n_energy_groups]    
    '''total cross section, batched arbitrarily over leading dimensions. Shape is [..., n_energy_groups]'''
    scattering      : jnp.ndarray    #[...., n_L, n_energy_groups, n_energy_groups]
    r'''scattering cross section moments, batched arbitrarily over leading dimensions. Shape is [..., n_L, n_energy_groups, n_energy_groups]. Here, :math:`\sigma^{l}_{g' \leftarrow g}` is stored at index [..., l, g', g]
      In other words, the first energy group index corresponds to the *outgoing* energy group, and the second to the *incoming* energy group.
      '''

    @classmethod
    def zero(cls, n_eg : int, n_L : int):
        '''
        Generates a CrossSectionData object with zero total and scattering cross sections.
        
        :param n_eg: Number of energy groups.
        :param n_L: Number of scattering moments (L).        
        '''
        return cls(
            total=jnp.zeros((n_eg,)),
            scattering=jnp.zeros((n_L + 1, n_eg, n_eg))
        )
    @classmethod    
    def constant(cls, total : float, n_eg : int, n_L : int):
        '''
        Generates a CrossSectionData object with constant total cross section and zero scattering cross sections.
        
        :param total: Value for the total cross section.
        :param n_eg: Number of energy groups.
        :param n_L: Number of scattering moments (L).'''
        return cls(
            total=jnp.full((n_eg,), total),
            scattering=jnp.zeros((n_L + 1, n_eg, n_eg))            
        )    
    @classmethod
    def from_list(cls, cross_section_data : List["CrossSectionData"]):
        '''
        Constructs a batched CrossSectionData object from a list of CrossSectionData objects.
        
        :param cross_section_data: List of CrossSectionData objects to batch together.

        '''
        return jax.tree.map(lambda *xs : jnp.stack(xs), *cross_section_data)

    def __getitem__(self, index : int) -> CrossSectionDataSingleGroup:
        '''
        Returns the single-group cross section data for the given energy group index.
        '''
        return CrossSectionDataSingleGroup(
            total=self.total[..., index],
            scattering=self.scattering[..., index, index]
        )
    
    def assert_valid(self):
        '''
        Asserts that the cross section data is valid, checking dimensions and consistency.
        '''
        assert self.total.ndim >= 1, f"Total cross section must be at least 1D [ ..., n_energy_groups ]. Provided shape: {self.total.shape}"
        assert self.scattering.ndim >= 3, f"Scattering cross section must be at least 3D [ ..., n_L, n_energy_groups, n_energy_groups ]. Provided shape: {self.scattering.shape}"
        assert self.total.shape[-1] == self.scattering.shape[-2] == self.scattering.shape[-1],\
               f"Number of energy groups in total and scattering cross sections must match. Total shape: {self.total.shape}, Scattering shape: {self.scattering.shape}"        
    def assert_valid_batched(self):
        '''
        Asserts that the batched cross section data is valid, checking dimensions and consistency of the leading batch dimensions additionally.
        '''
        self.assert_valid()
        assert self.total.ndim > 1, f"Total cross section must be at least 1D [ ..., n_energy_groups ]. Provided shape: {self.total.shape}"
        assert self.total.ndim == self.scattering.ndim - 2,\
              f"Total and scattering cross sections must be batched over the same leading dimensions.\
                  total n_dim:{self.total.ndim}, sattering n_dim:{self.scattering.ndim}.\
                  Total shape: {self.total.shape}, Scattering shape: {self.scattering.shape}"
        assert self.total.shape[:-1] == self.scattering.shape[:-3], f"Total and scattering cross sections must be batched over the same leading dimensions. Total shape: {self.total.shape}, Scattering shape: {self.scattering.shape}"

    def assert_n_energy_groups(self, n_energy_groups : int) -> bool:
        '''
        Asserts that the cross section data has at least n_energy_groups energy groups.
        :param n_energy_groups: Number of energy groups to check against.
        '''
        assert self.total.shape[-1] >= n_energy_groups,\
                f"Total cross section number of energy groups must be at least {n_energy_groups}. Provided shape: {self.total.shape}"
        assert self.scattering.shape[-2] >= n_energy_groups and self.scattering.shape[-1] >= n_energy_groups,\
                f"Scattering cross section number of energy groups must be at least {n_energy_groups}. Provided shape: {self.scattering.shape}"
        
    @property
    def downscattering_only(self) -> bool:
        return _check_if_lower_triangular(self)
    
    def convert_to_downscattering_only(self) -> "CrossSectionData":
        r'''
        Converts the cross section data to a downscattering only cross section data by zeroing all upscattering cross sections.
        '''
        scattering_down = jnp.tril(self.scattering)
        return CrossSectionData(
            total=self.total,
            scattering=scattering_down
        )


class CrossSectionsIndexed(eqx.Module):
    '''
    Dataclass to hold indexed cross section data for multiple energy groups.
    Each element has an index into the cross section data. This avoids repeating cross section data for each element.
    '''
    
    indices            : jnp.ndarray      #[n_elements]
    '''Indices into the cross section data for each element. Shape is [n_elements]'''

    cross_section_data : CrossSectionData #[batched over n_cross_sections (where it should hold that n_cross_sections >= max(indices)+1])]
    '''The cross section data for the indexed cross sections. Batched over n_cross_sections (where it should hold that n_cross_sections >= max(indices)+1])'''

    def map_to_full_array(self, energy_group : int) -> CrossSectionDataSingleGroup:
        '''
        Maps the indexed cross sections to a full array for the given energy group.
        :param energy_group: Energy group index to map to.
        '''
        return _map_cross_sections_to_full_array(self, energy_group)
    
    def __check_init__(self):
        self.cross_section_data.assert_valid_batched()
    
    @property 
    def n_energy_groups(self):
        '''
        Returns the number of energy groups in the cross section data.
        '''
        return self.cross_section_data.total.shape[-1]
    
    @property
    def l_max(self):
        '''
        Returns the maximum scattering moment L in the cross section data.
        '''
        return self.cross_section_data.scattering.shape[-3] - 1

@jax.jit
def _map_cross_sections_to_full_array(cross_sections : CrossSectionsIndexed, energy_group : int) -> CrossSectionDataSingleGroup:
    return CrossSectionDataSingleGroup(cross_sections.cross_section_data.total[cross_sections.indices,energy_group], cross_sections.cross_section_data.scattering[cross_sections.indices, : , energy_group, energy_group])
    
def _check_if_lower_triangular(cross_section_data : CrossSectionData) -> bool:
    r'''
    Check if the multi-group operator is lower triangular.

    This is the case if all upscattering cross sections are zero.

    Parameters
    ----------
    mgop : MultiGroupOperator
        The multi group operator.
    Returns
    -------
    bool
        True if the operator is lower triangular, False otherwise.
    '''


    tril = jnp.tril(cross_section_data.scattering)    
    return jnp.all(cross_section_data.scattering == tril)