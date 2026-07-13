'''Data bundles a transport operator hands to its forward / inversion strategies.'''

import jax.numpy as jnp
from typing import NamedTuple

from jax_sn.solution_domain import FEMMatrices, PrecomputedGeometricData
from jax_sn.domain.geometric_domain import Domain
from .sweep_data_implementations import SweepData


class ForwardData(NamedTuple):
    '''Positional inputs to a single-angle forward kernel (unpacked with ``*data``).'''
    domain                     : Domain
    fem_matrices               : FEMMatrices
    precomputed_geometric_data : PrecomputedGeometricData
    cross_sections             : jnp.ndarray
    angle                      : jnp.ndarray


class InversionData(NamedTuple):
    '''Positional inputs to a single-angle inverse-sweep kernel (unpacked with ``*data``).'''
    domain                     : Domain
    fem_matrices               : FEMMatrices
    precomputed_geometric_data : PrecomputedGeometricData
    cross_sections             : jnp.ndarray
    angle                      : jnp.ndarray
    sweep_data                 : SweepData
