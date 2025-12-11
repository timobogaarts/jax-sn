import jax
import jax.numpy as jnp
from dataclasses import dataclass

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Source:
    source_values : jnp.ndarray


    def __getitem__(self, idx):
        assert self.source_values.ndim >= 3, "Indexing is only supported for batched Source (..., n_elements, n_basis) instances but has shape {self.source_values.shape}."
        return Source(self.source_values[idx])

