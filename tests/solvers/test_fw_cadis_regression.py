r'''
Round-trip regression pin for FW-CADIS (:func:`jax_sn.variance_reduction.fw_cadis_scalar_flux`).

Runs the full FW-CADIS pipeline (forward solve -> 1/phi -> continuous-adjoint solve) on the breeding-
blanket problem, interpolates the resulting scalar-flux importance map along the domain diagonal, and
compares against a frozen ``.npz`` bundle. This is a *self-pinned* regression guard (frozen from jax-sn's
own output), not an external-solver oracle -- it just verifies the round trip stays put.

Freeze / refresh the bundles with::

    JAXSN_REFERENCE=freeze pytest tests/solvers/test_fw_cadis_regression.py
    # or: pytest --reference-mode=freeze tests/solvers/test_fw_cadis_regression.py

The set of (dim, degree) cases is deliberately reduced (one per dimension) -- the point is to exercise the
round trip, not to sweep every FEM order. Edit ``FW_CADIS_DIM_DEGREES`` to widen/narrow it.
'''

import os

import jax.numpy as jnp
import numpy as onp
import pytest
import lineax as lx

from bb_setup import _get_bb_args, _setup_bb_args
from reference_data import save, load, key_fw_cadis
from jax_sn.solvers.right_hand_side import TotalSource
from jax_sn.variance_reduction import fw_cadis_scalar_flux

# Reduced round-trip set: one FEM order per dimension. Widen for a fuller sweep.
FW_CADIS_DIM_DEGREES = [(1, 2), (2, 1), (3, 1)]

# Deterministic sample points: 40 points along the domain bounding-box diagonal, inset from the
# boundary to avoid element-boundary interpolation ambiguity.
_N_SAMPLE = 40
_INSET = (0.02, 0.98)


def _reference_mode() -> str:
    return os.environ.get("JAXSN_REFERENCE", "frozen")


def _diagonal_points(solution_domain) -> jnp.ndarray:
    verts = solution_domain.domain.vertices          # [n_vertices, dim]
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    t = jnp.linspace(_INSET[0], _INSET[1], _N_SAMPLE)
    return lo[None, :] + t[:, None] * (hi - lo)[None, :]   # [_N_SAMPLE, dim]


@pytest.mark.parametrize("dim,degree", FW_CADIS_DIM_DEGREES)
def test_fw_cadis_roundtrip(dim, degree):
    mgop, _, solution_domain = _setup_bb_args(*_get_bb_args(dim, degree))

    n_groups   = mgop.n_energy_groups
    n_elements = solution_domain.n_elements
    n_basis    = solution_domain.n_basis

    # Deterministic isotropic unit response over all groups.
    rhs = TotalSource.from_operator_and_isotropic_source(mgop, jnp.ones((n_groups, n_elements, n_basis)))
    solver = lx.BiCGStab(rtol=1e-10, atol=1e-12, max_steps=200)

    importance_map = fw_cadis_scalar_flux(mgop, rhs, solver)   # [n_groups, n_moments, n_elements, n_basis]
    assert bool(jnp.all(jnp.isfinite(importance_map))), "FW-CADIS produced non-finite values."

    points = _diagonal_points(solution_domain)
    interpolated = onp.asarray(solution_domain.interpolate(points, importance_map[:, 0]))   # scalar flux moment

    name = key_fw_cadis(dim, degree)
    mode = _reference_mode()

    if mode == "freeze":
        save(name, interpolated=interpolated)
        pytest.skip(f"froze FW-CADIS reference bundle '{name}'")

    reference = load(name)["interpolated"]
    onp.testing.assert_allclose(interpolated, reference, rtol=1e-5, atol=1e-8)
