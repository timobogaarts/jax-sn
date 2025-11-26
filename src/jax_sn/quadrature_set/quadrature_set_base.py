import jax 
from dataclasses import dataclass
import jax.numpy as jnp

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class QuadratureSet:
    angles : jnp.ndarray
    weights : jnp.ndarray

    @property 
    def n_angles(self) -> int:
        return self.angles.shape[0]
    
    def __getitem__(self, idx):
        return self.angles[idx], self.weights[idx]