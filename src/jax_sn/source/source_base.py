import jax
import jax.numpy as jnp
from dataclasses import dataclass

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Source:
    source_values : jnp.ndarray




