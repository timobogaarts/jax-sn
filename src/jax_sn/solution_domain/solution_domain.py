import jax 
import jax.numpy as jnp 
from dataclasses import dataclass
from .base_fem import Element, compute_volume_matrices_linear_jacobian, compute_face_matrices_linear_jacobian
from typing import List, Tuple
from jax_sn.domain.geometric_domain import GeometricDomain, TopologicalSortedAngle, PrecomputedGeometricData, CrossSectionData

from jax_sn.source import Source
import equinox as eqx


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FEMMatrices:
    mass_matrix       : jnp.ndarray 
    integration_coeff : jnp.ndarray
    derivative_matrix : jnp.ndarray
    outflow_matrices  : jnp.ndarray
    inflow_matrices   : jnp.ndarray

    @classmethod 
    def from_element(cls, element : Element):
        return cls(*compute_volume_matrices_linear_jacobian(element), *compute_face_matrices_linear_jacobian(element))

def remove_zeros(array):
    import numpy as onp
    a = onp.array(array)
    a[onp.abs(a) < 1e-14] = 0.0
    return a

#class SolutionDomain(eqx.Module):