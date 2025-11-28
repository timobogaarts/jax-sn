import jax 
import jax.numpy as jnp
from dataclasses import dataclass

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LagrangianSolutionDomain:
    n_order : int
    jacobians          : jnp.ndarray
    fij_alpha          : jnp.ndarray
    boundary_jacobians : jnp.ndarray

    connection_offsets : jnp.ndarray
    index_map          : jnp.ndarray
    dijev_set          : jnp.ndarray

    aij_base_matrix    : jnp.ndarray
    bij_base_matrix    : jnp.ndarray
    cij_base_matrix    : jnp.ndarray

    vim_map            : jnp.ndarray
    face_template      : jnp.ndarray

