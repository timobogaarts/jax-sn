'''
This file tests that the parametrisation actually performs a valid topological sort when geometry changes and 
that skipping it would have been wrong. 


The trick is a **dimension-uniform** mesh: two ``d``-simplices glued on their shared ``(d-1)``-face,
plus one apex each. The shared face sits in the hyperplane ``x_{d-1} = 0`` with the two apexes at
``+/- e_{d-1}``, so the base shared-face normal is ``e_{d-1}``. With an angle chosen *near-grazing*
to that face (mostly tangent, a sliver of ``+e_{d-1}``), lifting **one shared-face vertex** out of
the hyperplane rotates the shared face until ``Omega . n`` flips sign -- i.e. the two elements swap
upwind/downwind and the topological order changes. Because the angle grazes the face, a small,
non-degenerate lift suffices.

  d=2: two triangles sharing an edge   (tilt the edge)     -> order flips
  d=3: two tets sharing a triangle     (tilt the triangle) -> order flips
  d=1: two intervals sharing a *point* (a point has no orientation)   -> order CANNOT change

Naturally, in d=1, it does not matter.

'''
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as onp
import equinox as eqx
import pytest

from jax_sn.domain import Domain
from jax_sn.domain.geometric_domain import GeometricDomain
from jax_sn.domain.cross_sections import CrossSectionData, CrossSectionsIndexed
from jax_sn.solution_domain.basix_fem import BasixLagrangianSimplex
from jax_sn.solution_domain.solution_domain import SolutionDomain
from jax_sn.quadrature_set.quadrature_set_base import QuadratureSet
from jax_sn.operators.transport_operator import BasicSweepData
from jax_sn.operator_creation.multi_group_operator_creation import create_multi_group_operator
from jax_sn.operator_creation.default_strategies import DefaultMultiGroupStrategy
from jax_sn.parametrisation.multigroup_parametrisation import (
    MultigroupOperatorParametrisedVerticesXS, MultigroupOperatorParametrisedVerticesNoToposort,
    VerticesXSTheta,
)

from bb_setup import transport_operator_from_mgop


# --------------------------------------------------------------------------------------------------
# The glued-simplex mesh, parametrised by dimension. `lift` displaces one shared-face vertex out of
# the shared hyperplane (x_{d-1}=0); lift=0 is the base, lift!=0 the deformed mesh.
# Per dim we also fix a near-grazing angle and a lift that demonstrably flips the order (d>=2). The
# tests self-check the flip, so a too-small lift fails loudly rather than passing vacuously.
# --------------------------------------------------------------------------------------------------
_ANGLE = {                       # 3D Cartesian (the operator uses angle[:dim]); near-grazing to e_{d-1}
    1: (1.0,  0.0,  0.0),        # flow +x along the line
    2: (1.0,  0.18, 0.0),        # mostly +x, sliver of +y (grazes the x-axis edge)
    3: (1.0,  0.35, 0.18),       # mostly in-plane, sliver of +z (grazes the z=0 triangle)
}
_LIFT = {1: 0.4, 2: 0.9, 3: 0.9}


def _glued_simplices(dim: int, lift: float = 0.0):
    '''Two d-simplices sharing their (d-1)-face. Returns (vertices [n_verts, dim], connectivity).'''
    if dim == 1:
        # intervals B=[-1, s] and A=[s, 1] sharing the point s (=lift, slid along the only axis).
        verts = onp.array([[-1.0], [lift], [1.0]])
        conn  = onp.array([[0, 1], [1, 2]])
    elif dim == 2:
        # shared edge (s0,s1) on the x-axis; one endpoint (s1) lifted in +y. Apexes at +/- y.
        verts = onp.array([[-1.0, 0.0], [1.0, lift], [0.0, 1.0], [0.0, -1.0]])
        conn  = onp.array([[0, 1, 2], [0, 1, 3]])   # share edge (0,1)
    elif dim == 3:
        # shared triangle (t0,t1,t2) in z=0; one vertex (t1) lifted in +z. Apexes at +/- z.
        verts = onp.array([[-1.0, -1.0, 0.0], [1.0, -1.0, lift], [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
        conn  = onp.array([[0, 1, 2, 3], [0, 1, 2, 4]])   # share face (0,1,2)
    else:
        raise ValueError(dim)
    return jnp.asarray(verts), _orient_positive(verts, conn)


def _orient_positive(verts, conn):
    '''Reorder each simplex so its signed volume is positive (swap the last two local vertices if not).'''
    verts, conn = onp.asarray(verts), onp.asarray(conn).copy()
    for e in range(conn.shape[0]):
        p = verts[conn[e]]                  # [d+1, d]
        if onp.linalg.det(p[1:] - p[0]) < 0:
            conn[e, [-1, -2]] = conn[e, [-2, -1]]
    return jnp.asarray(conn)

# --------------------------------------------------------------------------------------------------
# Minimal operator stack: one material (isotropic, one energy group) -- enough to build/sweep.
# --------------------------------------------------------------------------------------------------
def _cross_sections(conn, n_eg=1, n_L=0, total=1.0):
    cs = CrossSectionData.constant(total, n_eg, n_L)            # unbatched single material
    cs = jax.tree.map(lambda x: x[None], cs)                   # -> batched over 1 cross-section
    indices = jnp.zeros(onp.asarray(conn).shape[0], dtype=jnp.int32)
    return CrossSectionsIndexed(indices=indices, cross_section_data=cs)

def _quad(angle3):
    a = jnp.asarray(angle3, dtype=jnp.float64)
    a = a / jnp.linalg.norm(a)
    return QuadratureSet(angles=a[None, :], weights=jnp.array([4.0 * jnp.pi]))

def _solution_domain(verts, conn, element):
    domain = Domain.from_mesh_and_cross_sections((verts, conn), element.face_template, _cross_sections(conn))
    return SolutionDomain.from_element_and_domain(element, domain)


def _build_mgop(solution_domain, quad, batch_size=4):
    # Single energy group (n_eg=1): the multigroup operator is trivially lower-block-triangular, so no
    # convert step is needed -- only the transport sweep order matters for this test.
    strategy = DefaultMultiGroupStrategy.default_multi_group_strategy(batch_size=batch_size, checkpoint=False)
    return create_multi_group_operator(solution_domain, quad, strategy)


def _order(mgop):
    return onp.asarray(mgop.data.sweep_data.levels_argsort)


# --------------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_basic_sweep_data_reconstruct(dim):
    '''Lowest level: BasicSweepData.reconstruct recomputes the sort for the deformed domain.'''
    element = BasixLagrangianSimplex(1, dim)
    verts0, conn = _glued_simplices(dim, lift=0.0)
    verts1, _    = _glued_simplices(dim, lift=_LIFT[dim])
    quad  = _quad(_ANGLE[dim])

    geom0 = GeometricDomain.from_mesh((verts0, conn), element.face_template)
    geom1 = GeometricDomain.from_mesh((verts1, conn), element.face_template)

    sd_base  = BasicSweepData.from_domain_quadrature_set(geom0, quad, 100)
    sd_fresh = BasicSweepData.from_domain_quadrature_set(geom1, quad, 100)
    sd_recon = sd_base.reconstruct(geom1, quad.angles)

    # reconstruct reproduces the from-scratch sort on the deformed domain
    assert (onp.asarray(sd_recon.levels_argsort) == onp.asarray(sd_fresh.levels_argsort)).all()

    order_changed = not (onp.asarray(sd_base.levels_argsort) == onp.asarray(sd_fresh.levels_argsort)).all()
    if dim == 1:
        assert not order_changed, "1D point-face cannot rotate: the sweep order must be invariant"
    else:
        assert order_changed, "deformation did not change the sweep order -- test would be vacuous; increase the lift"


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_parametrisation_resort(dim):
    '''The parametrisation re-sorts (VerticesXS) and NoToposort keeps the stale order -- which is
    wrong exactly when the order changed (d>=2) and harmless when it did not (d=1).'''
    element = BasixLagrangianSimplex(1, dim)
    verts0, conn = _glued_simplices(dim, lift=0.0)
    verts1, _    = _glued_simplices(dim, lift=_LIFT[dim])
    quad = _quad(_ANGLE[dim])

    mgop_base  = _build_mgop(_solution_domain(verts0, conn, element), quad)
    mgop_fresh = _build_mgop(_solution_domain(verts1, conn, element), quad)

    base_xs   = MultigroupOperatorParametrisedVerticesXS.theta_base(mgop_base).cross_section_data
    mgop_with = MultigroupOperatorParametrisedVerticesXS.reconstruct_operator(mgop_base, VerticesXSTheta(verts1, base_xs))
    mgop_no   = MultigroupOperatorParametrisedVerticesNoToposort.reconstruct_operator(mgop_base, verts1)

    # --- the sort itself ---
    assert (_order(mgop_with) == _order(mgop_fresh)).all(), "VerticesXS must recompute the sort for the deformed mesh"
    assert (_order(mgop_no) == _order(mgop_base)).all(),    "NoToposort must keep the base (stale) sort"
    if dim == 1:
        assert (_order(mgop_base) == _order(mgop_fresh)).all()
    else:
        assert not (_order(mgop_base) == _order(mgop_fresh)).all(), "order did not change -- increase the lift"

    # --- and that the stale sort actually matters: the transport sweep diverges iff the order changed ---
    top_base, top_with = transport_operator_from_mgop(mgop_base), transport_operator_from_mgop(mgop_with)
    top_no, top_fresh  = transport_operator_from_mgop(mgop_no), transport_operator_from_mgop(mgop_fresh)
    b = jnp.ones(top_base.in_structure().shape)
    inv_with  = top_with.inv(b)
    inv_no    = top_no.inv(b)
    inv_fresh = top_fresh.inv(b)

    onp.testing.assert_allclose(inv_with, inv_fresh, atol=1e-10, rtol=1e-8)
    if dim == 1:
        onp.testing.assert_allclose(inv_no, inv_fresh, atol=1e-10, rtol=1e-8)
    else:
        assert not onp.allclose(inv_no, inv_fresh, atol=1e-8, rtol=1e-6), \
            "stale sort gave the same sweep result -- the test is not exercising the re-sort"


if __name__ == "__main__":
    for d in (1, 2, 3):
        test_basic_sweep_data_reconstruct(d)
        test_parametrisation_resort(d)
        print(f"dim {d}: OK")
