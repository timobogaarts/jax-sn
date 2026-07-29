'''
Tests for the BVH cache backing `GeometricDomain.containing_cell`.

The cache is keyed on the identity of the mesh *arrays* (`create_bvh_cached`), not on the identity of
the domain object, so that a domain rebuilt around the same mesh (an `eqx.tree_at` on the cross
sections, `Domain.from_geometric_domain_and_cross_sections`) still hits it. These tests pin that
behaviour, the eviction of entries once the arrays die, and the rejection of traced arrays.
'''
import gc

import jax
import jax.numpy as jnp
import numpy as onp
import pytest
import equinox as eqx

jax.config.update("jax_enable_x64", True)

pytest.importorskip("jax_sbgeom", reason="BVH construction requires the 'geometry' extra")

import jax_sn.tools.mesh_tools as mt
from jax_sn.domain.cross_sections import CrossSectionData, CrossSectionsIndexed
from jax_sn.domain.geometric_domain import Domain, GeometricDomain


@pytest.fixture(autouse=True)
def clear_bvh_cache():
    '''The cache is module-global; clear it around each test so sizes can be asserted absolutely.

    Finalizers left over from the cleared entries are harmless: they pop with a default.
    '''
    mt._bvh_cache.clear()
    yield
    mt._bvh_cache.clear()


def _unit_square_domain():
    '''Unit square split into two triangles. Small enough to be exact, and needs no mesher.'''
    vertices = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    connectivity = jnp.array([[0, 1, 2], [1, 3, 2]])
    return GeometricDomain.from_mesh((vertices, connectivity), mt._face_template(3))


def _with_cross_sections(geometric_domain):
    cross_sections = CrossSectionsIndexed(
        indices=jnp.zeros(geometric_domain.n_elements, dtype=jnp.int32),
        cross_section_data=jax.tree.map(lambda x: x[None], CrossSectionData.constant(1.0, 1, 0)))
    return Domain.from_geometric_domain_and_cross_sections(geometric_domain, cross_sections)


# ---------------------------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------------------------
def test_bvh_is_built_once_per_mesh():
    domain = _unit_square_domain()

    first = mt.create_bvh_cached(domain.vertices, domain.connectivity)
    second = mt.create_bvh_cached(domain.vertices, domain.connectivity)

    assert first is second
    assert len(mt._bvh_cache) == 1


def test_containing_cell_populates_the_cache():
    domain = _unit_square_domain()
    points = jnp.array([[0.25, 0.25], [0.75, 0.75], [5.0, 5.0]])

    onp.testing.assert_array_equal(onp.asarray(domain.containing_cell(points)), [0, 1, -1])
    assert len(mt._bvh_cache) == 1
    onp.testing.assert_array_equal(onp.asarray(domain.containing_cell(points)), [0, 1, -1])
    assert len(mt._bvh_cache) == 1


def test_cache_hits_when_the_domain_is_rebuilt_around_the_same_mesh():
    '''The motivating case: the mesh arrays are shared, only the wrapper object differs.'''
    geometric_domain = _unit_square_domain()
    bvh = mt.create_bvh_cached(geometric_domain.vertices, geometric_domain.connectivity)

    domain = _with_cross_sections(geometric_domain)
    assert mt.create_bvh_cached(domain.vertices, domain.connectivity) is bvh

    # cross-section parametrisation rewrites the domain but leaves the geometry untouched
    replaced = eqx.tree_at(lambda d: d.cross_sections_indexed.cross_section_data.total, domain,
                           replace=jnp.full_like(domain.cross_sections_indexed.cross_section_data.total, 2.0))
    assert mt.create_bvh_cached(replaced.vertices, replaced.connectivity) is bvh

    assert len(mt._bvh_cache) == 1


def test_deformed_vertices_miss_the_cache():
    '''A deformed mesh must not reuse the BVH of the undeformed one.'''
    domain = _unit_square_domain()
    points = jnp.array([[0.25, 0.25], [1.25, 0.25]])
    onp.testing.assert_array_equal(onp.asarray(domain.containing_cell(points)), [0, -1])

    # stretch by 2 in x: the second point now lies inside the mesh
    deformed = eqx.tree_at(lambda d: d.vertices, domain, replace=domain.vertices * jnp.array([2.0, 1.0]))
    bvh_deformed = mt.create_bvh_cached(deformed.vertices, deformed.connectivity)

    assert bvh_deformed is not mt.create_bvh_cached(domain.vertices, domain.connectivity)
    assert len(mt._bvh_cache) == 2
    onp.testing.assert_array_equal(onp.asarray(deformed.containing_cell(points)), [0, 0])


# ---------------------------------------------------------------------------------------------
# Lifetime and jit safety
# ---------------------------------------------------------------------------------------------
def test_cache_does_not_keep_the_mesh_alive():
    domain = _unit_square_domain()
    mt.create_bvh_cached(domain.vertices, domain.connectivity)
    assert len(mt._bvh_cache) == 1

    del domain
    gc.collect()

    assert len(mt._bvh_cache) == 0


def test_traced_arrays_are_rejected():
    domain = _unit_square_domain()

    with pytest.raises(RuntimeError, match="traced arrays"):
        jax.jit(mt.create_bvh_cached)(domain.vertices, domain.connectivity)

    with pytest.raises(RuntimeError):
        jax.jit(lambda d, p: d.containing_cell(p))(domain, jnp.array([[0.25, 0.25]]))

    assert len(mt._bvh_cache) == 0
