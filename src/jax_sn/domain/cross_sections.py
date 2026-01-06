import jax 
from dataclasses import dataclass
import jax.numpy as jnp
from typing import List

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CrossSectionDataSingleGroup:
    total           : jnp.ndarray   #[..., ]
    scattering     : jnp.ndarray    #[..., n_L]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CrossSectionData:
    total           : jnp.ndarray    #[...,       n_energy_groups]    
    scattering      : jnp.ndarray    #[...., n_L, n_energy_groups, n_energy_groups]

    @classmethod
    def zero(cls, n_eg : int, n_L : int):
        return cls(
            total=jnp.zeros((n_eg,)),
            scattering=jnp.zeros((n_L, n_eg, n_eg))
        )
    @classmethod    
    def constant(cls, total : float, n_eg : int, n_L : int):
        return cls(
            total=jnp.full((n_eg,), total),
            scattering=jnp.zeros((n_L, n_eg, n_eg))            
        )    
    @classmethod
    def from_list(cls, cross_section_data : List["CrossSectionData"]):
        return jax.tree.map(lambda *xs : jnp.stack(xs), *cross_section_data)

    def __getitem__(self, index : int) -> CrossSectionDataSingleGroup:
        return CrossSectionDataSingleGroup(
            total=self.total[..., index],
            scattering=self.scattering[..., index, index]
        )
    
    def assert_valid(self):
        assert self.total.ndim >= 1, f"Total cross section must be at least 1D [ ..., n_energy_groups ]. Provided shape: {self.total.shape}"
        assert self.scattering.ndim >= 3, f"Scattering cross section must be at least 3D [ ..., n_L, n_energy_groups, n_energy_groups ]. Provided shape: {self.scattering.shape}"
        assert self.total.shape[-1] == self.scattering.shape[-2] == self.scattering.shape[-1],\
               f"Number of energy groups in total and scattering cross sections must match. Total shape: {self.total.shape}, Scattering shape: {self.scattering.shape}"        
    def assert_valid_batched(self):
        self.assert_valid()
        assert self.total.ndim > 1, f"Total cross section must be at least 1D [ ..., n_energy_groups ]. Provided shape: {self.total.shape}"
        assert self.total.ndim == self.scattering.ndim - 2,\
              f"Total and scattering cross sections must be batched over the same leading dimensions.\
                  total n_dim:{self.total.ndim}, sattering n_dim:{self.scattering.ndim}.\
                  Total shape: {self.total.shape}, Scattering shape: {self.scattering.shape}"
        assert self.total.shape[:-1] == self.scattering.shape[:-3], f"Total and scattering cross sections must be batched over the same leading dimensions. Total shape: {self.total.shape}, Scattering shape: {self.scattering.shape}"

    def assert_n_energy_groups(self, n_energy_groups : int) -> bool:
        assert self.total.shape[-1] >= n_energy_groups,\
                f"Total cross section number of energy groups must be at least {n_energy_groups}. Provided shape: {self.total.shape}"
        assert self.scattering.shape[-2] >= n_energy_groups and self.scattering.shape[-1] >= n_energy_groups,\
                f"Scattering cross section number of energy groups must be at least {n_energy_groups}. Provided shape: {self.scattering.shape}"
        
               