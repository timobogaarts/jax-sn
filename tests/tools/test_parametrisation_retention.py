'''
Retentions: when a stored quantity survives a change to a primitive it depends on.

A dependency edge records only *that* a quantity is a function of a primitive, so a policy holding
that quantity freezes the primitive outright. Some edges are far coarser than that -- the sweep tier
reads the strategy only for what that strategy asks of it -- and ``RETAINED_UNDER`` declares the
coarsening, so ``updated`` can keep the held value whenever it still serves.

The toy graph below is the same shape as the real one (a tier built from a primitive it uses coarsely,
plus a view read out of that tier), so the mechanism is tested without building an operator; the last
tests then pin the real builder behaviour.
'''
import jax
jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import numpy as onp
import pytest

from jax_sn.tools.parametrisation import ParametrisedModule


# --------------------------------------------------------------------------------------------------
# A toy registry: `tier` is expensive and depends on `tool` only through what the tool asks of it.
# --------------------------------------------------------------------------------------------------
class _Tier(eqx.Module):
    payload : jnp.ndarray
    grade   : int = eqx.field(static=True)     # what the tool got, ie how rich this tier is


class _Tool(eqx.Module):
    '''Needs a tier of at least ``grade``; richer is fine (the permissive direction).'''
    grade : int = eqx.field(static=True, default=0)

    def accepts(self, tier : _Tier) -> bool:
        return tier.grade >= self.grade


def tier(seed : jnp.ndarray, tool : _Tool) -> _Tier:
    return _Tier(seed * 2, tool.grade)


def view(tier : _Tier) -> jnp.ndarray:
    return tier.payload


def result(view : jnp.ndarray, scale : float) -> jnp.ndarray:
    return view * scale


class Toy(ParametrisedModule, abstract=True):
    DERIVATIONS = {"tier": tier, "view": view, "result": result}
    PROJECTIONS = {"view": {"tier": lambda t : t.payload}}
    RETAINED_UNDER = {"tier": {"tool": lambda tier, tool : tool.accepts(tier)}}

    seed  : jnp.ndarray
    tool  : _Tool
    scale : float


def _held(seed=1.0, grade=0, scale=1.0):
    '''An instance under the policy that stores everything the seed and the tool determine.'''
    return Toy(jnp.asarray(seed), _Tool(grade), scale).frozen("seed", "tool")


# --------------------------------------------------------------------------------------------------
# The graph a retention implies
# --------------------------------------------------------------------------------------------------
def test_view_of_a_dependency_is_never_stored():
    '''`view` is a projection of `tier`, which it also depends on, so it is a view: storing it would
    duplicate a field its own source already provides, next to the source itself.'''
    graph = Toy.freezing("seed", "tool").graph()
    assert graph.stored == frozenset({"tier"})
    assert graph.projected == frozenset({"view"})


def test_conditionally_free_is_the_relaxable_half_of_frozen():
    graph = Toy.freezing("seed", "tool").graph()
    assert graph.frozen == frozenset({"seed", "tool"})       # the policy genuinely holds both
    assert graph.conditionally_free == frozenset({"tool"})   # but only one has a retention
    assert not graph.free & graph.conditionally_free         # they are disjoint by construction


def test_a_policy_storing_nothing_has_nothing_to_retain():
    graph = Toy.freezing().graph()
    assert graph.free == frozenset({"seed", "tool", "scale"})
    assert graph.conditionally_free == frozenset()


def test_a_retention_also_unblocks_storing():
    '''The storage side of the same statement: asking to vary `tool` no longer forces `tier` to be
    recomputed, since the retention says when the held one still serves.'''
    graph = Toy.varying("tool").graph()
    assert graph.stored == frozenset({"tier"})           # would be recomputed without the retention
    assert graph.free == frozenset({"scale"})
    assert graph.conditionally_free == frozenset({"tool"})


def test_only_a_quantitys_own_retention_unblocks_it():
    '''`result` depends on the tool only through `tier`, but nothing would hold a `tier` to check the
    retention against, so it stays recomputed.'''
    assert "result" not in Toy.varying("tool").graph().stored


# --------------------------------------------------------------------------------------------------
# What `updated` does with it
# --------------------------------------------------------------------------------------------------
def test_retained_update_keeps_the_very_same_stored_value():
    held = _held(grade=2)
    updated = held.updated(tool=_Tool(1))                 # a tier of grade 2 serves a grade-1 tool
    assert updated.tier is held.tier                      # not rebuilt: the identical object
    assert updated.tool.grade == 1                        # ... and the new primitive did land


def test_retention_is_permissive_not_an_equality():
    '''What is asked is "does the held value still serve", not "is it what I would have built".'''
    assert _held(grade=2).updated(tool=_Tool(2)).tier.grade == 2      # exactly right
    assert _held(grade=2).updated(tool=_Tool(0)).tier.grade == 2      # richer than needed


def test_a_refused_retention_raises_rather_than_rebuilding():
    held = _held(grade=0)
    with pytest.raises(ValueError, match="retention refused"):
        held.updated(tool=_Tool(1))                       # grade-0 tier cannot serve a grade-1 tool


def test_a_primitive_without_a_retention_is_still_hard_frozen():
    held = _held()
    with pytest.raises(ValueError, match="cannot update"):
        held.updated(seed=jnp.asarray(9.0))


def test_a_free_primitive_needs_no_retention():
    held = _held()
    assert held.updated(scale=3.0).scale == 3.0


def test_downstream_of_a_retained_swap_still_follows():
    '''Only the *stored* value is kept; everything derived from the new primitive is recomputed.'''
    held = _held(seed=1.0, grade=2)
    assert held.result == pytest.approx(2.0)
    assert held.updated(scale=5.0).result == pytest.approx(10.0)


def test_no_policy_no_check():
    '''With nothing stored the tier is rebuilt anyway, so an "incompatible" tool is simply fine.'''
    free = Toy(jnp.asarray(1.0), _Tool(0), 1.0)
    assert free.updated(tool=_Tool(3)).tier.grade == 3


# --------------------------------------------------------------------------------------------------
# Declaring one wrongly
# --------------------------------------------------------------------------------------------------
def test_retention_must_name_a_quantity():
    class Bad(Toy, abstract=True):
        RETAINED_UNDER = {"seed": {"tool": lambda a, b : True}}
    with pytest.raises(ValueError, match="not in DERIVATIONS"):
        Bad.freezing("seed", "tool").graph()


def test_retention_must_name_a_real_dependency():
    class Bad(Toy, abstract=True):
        RETAINED_UNDER = {"tier": {"scale": lambda a, b : True}}
    with pytest.raises(ValueError, match="does not depend on"):
        Bad.freezing("seed", "tool").graph()


# --------------------------------------------------------------------------------------------------
# The real thing: a builder holding the sweep tier, swapping strategies
# --------------------------------------------------------------------------------------------------
def _one_group_builder(strategy, n_L=0):
    '''Two intervals, one material, one angle: the smallest thing with a real sweep ordering.'''
    from jax_sn.domain import Domain
    from jax_sn.domain.cross_sections import CrossSectionData, CrossSectionsIndexed
    from jax_sn.operator_creation.builders import MultiGroupBuilder
    from jax_sn.quadrature_set.quadrature_set_base import QuadratureSet
    from jax_sn.solution_domain.basix_fem import BasixLagrangianSimplex
    from jax_sn.solution_domain.solution_domain import SolutionDomain

    verts   = jnp.asarray(onp.array([[-1.0], [0.0], [1.0]]))
    conn    = jnp.asarray(onp.array([[0, 1], [1, 2]]))
    element = BasixLagrangianSimplex(1, 1)
    cs      = jax.tree.map(lambda x : x[None], CrossSectionData.constant(1.0, 1, n_L))
    domain  = Domain.from_mesh_and_cross_sections(
        (verts, conn), element.face_template,
        CrossSectionsIndexed(indices=jnp.zeros(2, dtype=jnp.int32), cross_section_data=cs))
    quad    = QuadratureSet(angles=jnp.asarray([[1.0, 0.0, 0.0]]), weights=jnp.asarray([4.0 * jnp.pi]))

    builder = MultiGroupBuilder.from_solution_domain(
        SolutionDomain.from_element_and_domain(element, domain), quadrature_set=quad, strategy=strategy)
    # the policy that holds the whole sweep tier: only the cross sections stay free
    return builder.varied("cross_sections_indexed")


def _strategies():
    from jax_sn.operator_creation.default_strategies import DefaultMultiGroupStrategy
    return (DefaultMultiGroupStrategy.default_multi_group_strategy(checkpoint=False),
            DefaultMultiGroupStrategy.default_multi_group_strategy(batch_size=2, checkpoint=True))


def test_builder_retains_the_tier_across_a_compatible_strategy_swap():
    plain, batched = _strategies()
    builder = _one_group_builder(plain)
    graph   = type(builder).graph()
    assert "non_differentiable_data" in graph.stored
    assert graph.conditionally_free == frozenset({"strategy", "cross_sections_indexed"})

    swapped = builder.updated(strategy=batched)
    assert swapped.non_differentiable_data is builder.non_differentiable_data   # sort not recomputed
    assert type(swapped.multi_group_operator.strategy) is type(batched)


def test_builder_refuses_a_strategy_the_held_tier_cannot_serve():
    '''The warp level sweep needs ``ProvidesLevels``; a plain sort does not provide it, so the swap
    genuinely needs a rebuild and is refused rather than silently rebuilt.'''
    from jax_sn.operator_creation.default_strategies import DefaultMultiGroupStrategy
    try:
        warp = DefaultMultiGroupStrategy.default_multi_group_warp_strategy()
    except Exception:                                    # pragma: no cover - warp extra not installed
        pytest.skip("warp backend not available")

    plain, _ = _strategies()
    builder  = _one_group_builder(plain)
    with pytest.raises(ValueError, match="retention refused"):
        builder.updated(strategy=warp)

    # the other direction is fine: a leveled tier serves the plain sweep too
    rich = _one_group_builder(warp)
    assert rich.updated(strategy=plain).non_differentiable_data is rich.non_differentiable_data


def test_the_builder_carries_no_build_time_knobs():
    '''Everything a build needs beyond the primitives is a static field on the strategy that wants it,
    so the rung has no knob bag of its own to pass through.'''
    plain, _ = _strategies()
    assert type(_one_group_builder(plain)).graph().primitives == frozenset(
        {"vertices", "connectivity", "element", "cross_sections_indexed", "quadrature_set", "strategy"})


def test_a_pinned_level_width_is_what_makes_a_traced_rebuild_concrete():
    '''A leveled sweep pads its level arrays to a fixed width. Left unpinned it reads that width off
    the levels -- a tracer once the vertices it derives from are one -- so the rebuild a
    differentiating loop performs cannot be traced. The fix is a field on the sweep strategy; the
    builder is untouched either way.'''
    from jax_sn.operator_creation.default_strategies import DefaultMultiGroupStrategy
    try:
        warp = DefaultMultiGroupStrategy.default_multi_group_warp_strategy()
    except Exception:                                    # pragma: no cover - warp extra not installed
        pytest.skip("warp backend not available")

    builder  = _one_group_builder(warp).varied("vertices", "cross_sections_indexed")
    vertices = builder.vertices
    n_levels = builder.sweep_data.level_fwd.n_levels     # read off the concrete build

    with pytest.raises(jax.errors.ConcretizationTypeError):
        eqx.filter_jit(lambda v : builder.updated(vertices=v).sweep_data.level_fwd.sizes)(vertices)

    pinned = _one_group_builder(
        DefaultMultiGroupStrategy.default_multi_group_warp_strategy(maximum_number_of_levels=n_levels)
    ).varied("vertices", "cross_sections_indexed")
    sizes = eqx.filter_jit(lambda v : pinned.updated(vertices=v).sweep_data.level_fwd.sizes)(vertices)
    assert sizes.shape[-1] == n_levels


def test_builder_stores_the_harmonics_while_the_cross_sections_vary():
    '''The harmonics read the cross sections only for `l_max`, so the policy whose whole point is to
    vary them can still hold the harmonics -- and refuses only a change that moves `l_max`.'''
    plain, _ = _strategies()
    builder  = _one_group_builder(plain)
    assert "angular_data" in type(builder).graph().stored

    xs      = builder.cross_sections_indexed
    rescaled = jax.tree.map(lambda x : 2.0 * x if x.dtype == jnp.float64 else x, xs)
    assert builder.updated(cross_sections_indexed=rescaled).angular_data is builder.angular_data

    richer = _one_group_builder(plain, n_L=1).cross_sections_indexed       # l_max 0 -> 1
    assert richer.l_max != xs.l_max
    with pytest.raises(ValueError, match="retention refused"):
        builder.updated(cross_sections_indexed=richer)


def test_the_retention_check_survives_tracing():
    '''`updated` now runs a predicate on the way in, so it must read only static structure -- this is
    the path a differentiating rebuild takes, where the new primitive is a tracer.'''
    import equinox as eqx

    plain, _ = _strategies()
    builder  = _one_group_builder(plain)
    xs       = builder.cross_sections_indexed
    x        = jnp.ones(builder.multi_group_operator.in_structure().shape)

    @eqx.filter_jit
    def apply(scale):
        scaled = jax.tree.map(lambda a : a * scale if a.dtype == jnp.float64 else a, xs)
        return jnp.sum(builder.updated(cross_sections_indexed=scaled).multi_group_operator.mv(x))

    assert jnp.isfinite(apply(1.0))
    assert jnp.isfinite(jax.grad(apply)(1.0))
