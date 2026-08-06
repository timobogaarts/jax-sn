'''
Builder-vs-manual equivalence: a re-derived operator is the operator you would have built.
'''

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as onp
import pytest

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bb_setup import bb_mesh, base_region_sizes, base_resolutions                     
from jax_sn.domain import Domain                                                      
from jax_sn.operator_creation.builders import MultiGroupBuilder                          
from jax_sn.operator_creation.default_strategies import DefaultMultiGroupStrategy        
from jax_sn.operators.multi_group_operator import MultiGroupOperator                     
from jax_sn.operators.operator_data import (                                             
    AngularData, DiscretisationData, OperatorData,
)
from jax_sn.quadrature_set import create_tn_quadrature_set                        
from jax_sn.solution_domain import BasixLagrangianSimplex         
from jax_sn.solution_domain.base_fem import FEMMatrices                           
from jax_sn.solution_domain.solution_domain import SpatialData                    


# --------------------------------------------------------------------------------------------------
# Problem: the 2D breeding blanket. The semantics under test are dimension-independent, so one
# dimension is enough -- 2D is the cheapest one with real faces to sort over.
# --------------------------------------------------------------------------------------------------
_DIM      = 2
_DEGREE   = 1
_TN_ORDER = 3


def _strategy():
    return DefaultMultiGroupStrategy.default_multi_group_strategy(checkpoint=False)

@pytest.fixture(scope="module")
def bb():
    '''``(connectivity, base primitives)`` for the 2D BB mesh. The connectivity is not perturbed by any
    case, so it sits outside the primitive dict.'''
    vertices, connectivity, cross_sections, _sources = bb_mesh(
        _DIM, base_region_sizes(), base_resolutions(), 5.0, 1)
    primitives = {
        "vertices":               vertices,
        "cross_sections_indexed": cross_sections,
        "element":                BasixLagrangianSimplex(_DEGREE, _DIM),
        "quadrature_set":         create_tn_quadrature_set(_TN_ORDER),
    }
    return connectivity, primitives


def _sheared(vertices):
    '''A shear along the transport direction. Being affine with unit determinant it cannot invert an
    element however large it is, and it is large enough to tilt faces past some of the T_N angles --
    which is what makes the sweep order change (asserted below; a shear too mild to flip any face
    leaves the ordering untouched and would make the moved-vertices case only test the geometry).'''
    return vertices.at[:, 0].add(0.3 * vertices[:, 1])


def _rescaled(cross_sections_indexed):
    '''Scale the cross-section values, leaving the element indices and ``l_max`` alone -- so the
    ``angular_data`` retention holds and a policy that stores the harmonics keeps them.'''
    return jax.tree.map(
        lambda a: 1.3 * a if jnp.issubdtype(a.dtype, jnp.inexact) else a,
        cross_sections_indexed,
    )


# --------------------------------------------------------------------------------------------------
# The two routes
# --------------------------------------------------------------------------------------------------
def _manual_operator(connectivity, primitives, strategy):
    '''The operator assembled by hand: each tier from the eager constructor that owns it, then the
    containers filled in directly. Touches nothing in ``builders``.

    The geometry comes from ``Domain.from_mesh_and_cross_sections``, not from ``compute_topology`` /
    ``compute_face_normals`` -- the builder's derivations call that pair, so assembling from them here
    would transcribe the code under test rather than check it.'''
    element     = primitives["element"]
    xs          = primitives["cross_sections_indexed"]
    quadrature  = primitives["quadrature_set"]

    domain      = Domain.from_mesh_and_cross_sections(
        (primitives["vertices"], connectivity), element.face_template, xs)
    spatial     = SpatialData(
        domain                     = domain,
        fem_matrices               = FEMMatrices.from_element(element),
        precomputed_geometric_data = element.compute_geometric_data(domain),
        element                    = element,
    )
    angular     = AngularData.from_quadrature_set(quadrature, xs.l_max)

    # The tier is the strategy's to build (`RequiresNonDifferentiableData`), and `check_compatible`
    # exists precisely for a tier assembled by hand like this one.
    non_differentiable = strategy.build_default_non_differentiable_data(domain, quadrature.angles)
    strategy.check_compatible(non_differentiable)

    return MultiGroupOperator(
        OperatorData(DiscretisationData(spatial, angular), non_differentiable),
        strategy, transpose_bool=False,
    )


def _builder(connectivity, primitives, strategy, policy):
    '''A builder straight from its primitives -- the ladder's own entry, not the ``from_solution_domain``
    adapter. ``freezing()`` is the all-free class, the one whose fields are exactly the primitives;
    ``materialise`` then converts it to the policy under test, deriving whatever that policy holds.'''
    free = MultiGroupBuilder.freezing()(connectivity = connectivity, strategy = strategy, **primitives)
    return free if type(free) is policy else free.materialise(policy)


# --------------------------------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------------------------------
def _assert_same_operator(rederived, manual, *, rtol=1e-12, atol=1e-14):
    '''Whole-pytree equality of two ``MultiGroupOperator``s: identical structure (which, for an equinox
    module, already covers every static strategy/config field), and array leaves equal to round-off.'''
    leaves_r, treedef_r = jax.tree_util.tree_flatten_with_path(rederived)
    leaves_m, treedef_m = jax.tree_util.tree_flatten_with_path(manual)
    assert treedef_r == treedef_m, (
        f"operator pytree structure differs:\n  re-derived: {treedef_r}\n  manual:     {treedef_m}"
    )

    for (path, a), (_, b) in zip(leaves_r, leaves_m):
        where = jax.tree_util.keystr(path)
        a, b  = onp.asarray(a), onp.asarray(b)
        assert a.shape == b.shape, f"{where}: shape {a.shape} != {b.shape}"
        assert a.dtype == b.dtype, f"{where}: dtype {a.dtype} != {b.dtype}"
        if onp.issubdtype(a.dtype, onp.inexact):
            onp.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=f"at {where}")
        else:
            assert (a == b).all(), f"{where}: integer/boolean leaf differs"


# --------------------------------------------------------------------------------------------------
# The hand-picked policy/update combinations
#
# Each entry is (id, policy, primitives -> the primitives to replace). The policy column is what the
# derivation graph is allowed to keep; the comment records which tier that puts on the *held* side,
# since that is what the case actually tests.
# --------------------------------------------------------------------------------------------------
_CASES = [
    # Nothing stored, no update: every quantity comes off its derivation. The baseline -- does the
    # graph reproduce a hand assembly at all, before any policy or update is in play?
    ("nothing_stored_no_update", MultiGroupBuilder.freezing(), lambda p: {}),

    # The most-stored end, no update: this is what `from_solution_domain` defaults to, and it holds the
    # solution domain and *projects* the whole spatial tier back out of it. Projection vs. derivation.
    ("solution_domain_stored_no_update", MultiGroupBuilder.varying("quadrature_set", "strategy"),
     lambda p: {}),

    # Mesh deformation (the shape-optimisation loop). Holds the FEM matrices, the topology and the
    # harmonics; recomputes the geometry and the whole sweep tier.
    ("moved_vertices", MultiGroupBuilder.varying("vertices"),
     lambda p: {"vertices": _sheared(p["vertices"])}),

    # The canonical theta policy. Holds the sweep tier *and* the harmonics (the latter only because the
    # `l_max` retention says the held value still serves) -- so this is the case that checks a retention
    # predicate numerically rather than by fiat.
    ("rescaled_cross_sections", MultiGroupBuilder.varying("cross_sections_indexed"),
     lambda p: {"cross_sections_indexed": _rescaled(p["cross_sections_indexed"])}),

    # Both at once: two branches of the graph re-derived in one update, with the cross sections riding
    # in on their retention (this policy freezes them outright, conditionally).
    ("moved_vertices_and_rescaled_cross_sections",
     MultiGroupBuilder.varying("vertices", "cross_sections_indexed"),
     lambda p: {"vertices": _sheared(p["vertices"]),
                "cross_sections_indexed": _rescaled(p["cross_sections_indexed"])}),

    # Element degree 1 -> 2: every spatial shape downstream changes, so nothing stale can survive
    # unnoticed. Only the harmonics are held (they do not depend on the element).
    ("higher_degree_element", MultiGroupBuilder.varying("element"),
     lambda p: {"element": BasixLagrangianSimplex(_DEGREE + 1, _DIM)}),

    # A coarser angular set: the angular tier and the sweep tier rebuild, the spatial tier is held.
    ("coarser_quadrature_set", MultiGroupBuilder.varying("quadrature_set"),
     lambda p: {"quadrature_set": create_tn_quadrature_set(_TN_ORDER - 1)}),
]


@pytest.mark.parametrize("policy,updates", [c[1:] for c in _CASES], ids=[c[0] for c in _CASES])
def test_rederived_operator_matches_manual_build(bb, policy, updates):
    '''``builder.updated(...)`` yields the same MultiGroupOperator as constructing one from scratch at
    the updated primitives.'''
    connectivity, base = bb
    strategy = _strategy()
    delta    = updates(base)

    rederived = _builder(connectivity, base, strategy, policy).updated(**delta).multi_group_operator
    manual    = _manual_operator(connectivity, {**base, **delta}, strategy)

    _assert_same_operator(rederived, manual)


def test_the_perturbations_actually_perturb(bb):
    '''Guard against a vacuous suite: every hand-crafted update must move the operator away from the
    unperturbed one, else the equality above would hold for the wrong reason.'''
    connectivity, base = bb
    strategy = _strategy()
    unperturbed = _manual_operator(connectivity, base, strategy)

    for name, _policy, updates in _CASES:
        delta = updates(base)
        if not delta:
            continue
        perturbed = _manual_operator(connectivity, {**base, **delta}, strategy)
        with pytest.raises(AssertionError):
            _assert_same_operator(perturbed, unperturbed)


def test_the_shear_reaches_the_sweep_tier(bb):
    '''The moved-vertices cases are meant to exercise the combinatorial tier too, not just the
    geometry -- so the shear must be big enough to reorder the sweep. It is the one perturbation whose
    effect is not visible in the leaf it edits.'''
    connectivity, base = bb
    strategy = _strategy()
    order = lambda p: onp.asarray(_manual_operator(connectivity, p, strategy).data.sweep_data.levels_argsort)

    assert not (order({**base, "vertices": _sheared(base["vertices"])}) == order(base)).all(), \
        "the shear left the sweep ordering untouched -- increase it, or the case only tests geometry"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
