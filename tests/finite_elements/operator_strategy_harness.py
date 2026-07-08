"""
Discovery-driven test harness for the per-operator *strategy* axes (mass / moment / discrete / scatter /
transport forward+inversion).

Every operator axis in ``jax_sn`` selects its kernel by a strategy object, not a subclass (see CLAUDE.md).
Rather than hand-list the strategy matrix -- which silently fails to cover a newly added strategy -- this
harness *discovers* every concrete, importable, default-constructible strategy subclass of each axis base
(the same principle as ``perf_tests/_discovery.py``). Add a strategy anywhere in the package and it is
tested automatically, with no edit here. Two test modules share this harness (and its cached setup):
``test_operator_properties.py`` (algebraic properties + cross-strategy consistency) and
``test_operator_regression.py`` (finite-difference gradient correctness).

Problem: the breeding-blanket slab (``bb_setup``) at **dim 3, degree 1 and degree 2**. Degree 1 is the
4-node linear tetrahedron the warp kernels require; degree 2 exercises higher-order FEM (10-node tet) that
the pure-JAX / pallas kernels must also handle. (Degree 3 is overkill; the parametrised ``bb_mgop`` fixture
uses degree 2 across dims 1-3 for the frozen regression pins -- this harness uses a degree-1/2, dim-3 build
so the warp variants are exercised.) A strategy that only supports a particular discretisation carries a
specialisation *flag* (``supported_degrees``): the warp kernels are hardcoded to n_basis == 4, so they run
only at degree 1.

The mass/moment/discrete axes are ordinary ``AbstractLinearOperator_sn`` (uniform ``mv`` / ``transpose`` /
``in_structure`` / ``out_structure``) and share one battery. Scatter is energy-group-indexed
(``gather_forward`` / ``gather_backward``, no single ``mv``) and gets its own discovered adjoint check.
"""
import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Type

from jax_sn.operators.mass_operator import MassStrategy, VmapMass
from jax_sn.operators.moment_operator import MomentStrategy, DiscreteStrategy, BatchedMoment, BatchedDiscrete
from jax_sn.operators.scatter_operator import GatherScatterOperator
from jax_sn.operators.transport_operator import ForwardStrategy, InversionStrategy, BatchedForward, BatchedSweep
from jax_sn.operators.multi_group_operator.multi_group_operator_creation import create_multi_group_operator


# ---------------------------------------------------------------------------
# Optional backends: import so their strategy subclasses register for discovery. Guarded exactly like
# perf_tests -- a machine without the warp extra simply never sees those variants (they drop out).
# ---------------------------------------------------------------------------
_OPTIONAL_STRATEGY_MODULES = (
    "jax_sn.operators.mass_operator.mass_operator_warp",
    "jax_sn.operators.transport_operator.implementations.transport_operator_pallas",
    "jax_sn.operators.transport_operator.implementations.transport_operator_warp",
)


def ensure_optional_backends() -> None:
    try:
        import warp
        warp.config.quiet = True
    except Exception:
        pass
    for mod in _OPTIONAL_STRATEGY_MODULES:
        try:
            __import__(mod)
        except Exception:
            pass


def concrete_subclasses(base: Type) -> List[Type]:
    """Every importable concrete (non-abstract) subclass of ``base``, recursively."""
    ensure_optional_backends()
    seen, out = set(), []

    def walk(cls):
        for sub in cls.__subclasses__():
            if sub in seen:
                continue
            seen.add(sub)
            if not getattr(sub, "__abstractmethods__", False):
                out.append(sub)
            walk(sub)

    walk(base)
    return out


# ---------------------------------------------------------------------------
# Axis descriptors
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Axis:
    """One strategy axis: how to build the sub-operator with a chosen strategy and what to compare to."""
    name: str
    base: Type                                   # strategy base class walked for variants
    kwarg: str                                   # create_multi_group_operator / setup_slab_mgop kwarg
    extract: Callable                            # mgop -> the sub-operator under test
    reference: object                            # the pure-JAX reference strategy (create_* default)
    overrides: Dict[Type, list] = field(default_factory=dict)  # non-default ctors (class -> [instances])


LINEAR_AXES = [
    Axis("mass",     MassStrategy,     "mass_strategy",     lambda m: m.mass_operator,     VmapMass()),
    Axis("moment",   MomentStrategy,   "moment_strategy",   lambda m: m.moment_operator,   BatchedMoment()),
    Axis("discrete", DiscreteStrategy, "discrete_strategy", lambda m: m.discrete_operator, BatchedDiscrete()),
]

SCATTER_AXIS = Axis(
    "scatter", GatherScatterOperator, "gather_scatter_operator_cls",
    lambda m: m.gather_scatter_operator, None,
)


def _label(cls: Type, inst) -> str:
    """``ClassName`` plus any set scalar init fields, e.g. ``BatchedSweep(64)`` (matches perf's labels)."""
    extras = []
    if dataclasses.is_dataclass(inst):
        for f in dataclasses.fields(inst):
            v = getattr(inst, f.name, None)
            if isinstance(v, (int, float, str, bool)) and v is not None:
                extras.append(str(v))
    return f"{cls.__name__}({','.join(extras)})" if extras else cls.__name__


def discover_instances(base: Type, overrides: Dict[Type, list] = None) -> List[tuple]:
    """``[(label, instance), ...]`` -- every concrete strategy of ``base``, default-constructed, with
    ``overrides`` replacing the default ctor for classes that need a representative arg (batch size)."""
    overrides = overrides or {}
    out = []
    for cls in concrete_subclasses(base):
        if cls in overrides:
            out.extend((_label(cls, inst), inst) for inst in overrides[cls])
        else:
            out.append((cls.__name__, cls()))
    return out


def discover(axis: Axis, instantiate: bool = True) -> List[tuple]:
    """``[(label, value), ...]`` for an axis. ``value`` is a strategy *instance* (linear axes) or the
    concrete *class* itself (the scatter axis, whose ``create_*`` kwarg takes a class)."""
    if not instantiate:
        return [(cls.__name__, cls) for cls in concrete_subclasses(axis.base)]
    return discover_instances(axis.base, axis.overrides)


# ---------------------------------------------------------------------------
# Transport axes: forward / inversion strategies. The inversion axis drives both the single-angle and the
# multi-angle transport operators (multi-angle uses each inversion's own default angle lifting -- the
# fused warp kernels for warp inversions, VmapAngles otherwise -- so no strategy *product* is needed).
# Batch sizes are injected the same way perf does (default-construct gives batch_size=None == unbatched).
# ---------------------------------------------------------------------------
_TRANSPORT_BATCH = 64
FWD_OVERRIDES: Dict[Type, list] = {BatchedForward: [BatchedForward(_TRANSPORT_BATCH)]}
INV_OVERRIDES: Dict[Type, list] = {BatchedSweep: [BatchedSweep(_TRANSPORT_BATCH)]}
try:
    from jax_sn.operators.transport_operator import BatchedPallasSweep
    INV_OVERRIDES[BatchedPallasSweep] = [BatchedPallasSweep(_TRANSPORT_BATCH)]
except ImportError:
    pass


def discover_forward() -> List[tuple]:
    ensure_optional_backends()
    return discover_instances(ForwardStrategy, FWD_OVERRIDES)


def discover_inversion() -> List[tuple]:
    ensure_optional_backends()
    return discover_instances(InversionStrategy, INV_OVERRIDES)


_PAD_LEVELS = 256   # a fixed level cap comfortably above the small BB slab's actual level count


def discover_inversion_cases() -> List[tuple]:
    """``[(label, inversion_strategy, sweep_data_kwargs), ...]``. Beyond the discovered strategies this
    adds a *level-padded* WarpLevelSweep case -- ``maximum_number_of_levels`` is a ``LeveledSweepData``
    build parameter (orthogonal to the strategy axis) whose fixed-size padding exercises the kernel's
    out-of-bounds guard; it is the one bit of coverage that is not itself a strategy variant."""
    cases = [(label, inv, {}) for label, inv in discover_inversion()]
    try:
        from jax_sn.operators.transport_operator.implementations.transport_operator_warp import WarpLevelSweep
        cases.append((f"{WarpLevelSweep.__name__}(padded)", WarpLevelSweep(),
                      {"maximum_number_of_levels": _PAD_LEVELS}))
    except Exception:
        pass
    return cases


# ---------------------------------------------------------------------------
# Per-strategy discretisation specialisation flag
# ---------------------------------------------------------------------------
DEGREES = (1, 2)   # FEM degrees the harness exercises: linear tet (n_basis 4) + higher-order (n_basis 10)

# The warp kernels are hardcoded to the 4-node linear tetrahedron (vec4d/mat44d, n_basis == 4), so they
# fire only at degree 1. Everything else (pure-JAX, pallas -- which sizes to _next_power_of_two(n_basis))
# is degree-agnostic. Guarded so the flag is empty (and nothing is restricted) without the warp extra.
_DEGREE1_ONLY: set = set()
try:
    from jax_sn.operators.mass_operator import WarpMass
    _DEGREE1_ONLY.add(WarpMass)
except Exception:
    pass
try:
    from jax_sn.operators.transport_operator.implementations.transport_operator_warp import (
        WarpSequentialSweep, WarpLevelSweep,
    )
    _DEGREE1_ONLY.update({WarpSequentialSweep, WarpLevelSweep})
except Exception:
    pass


def supported_degrees(value) -> tuple:
    """The FEM degrees a strategy (instance or class) is tested at: degree-1-only for the warp kernels
    (see ``_DEGREE1_ONLY``), every degree otherwise. This is the per-strategy specialisation flag."""
    cls = value if isinstance(value, type) else type(value)
    return (1,) if cls in _DEGREE1_ONLY else DEGREES


# ---------------------------------------------------------------------------
# Shared, cached setup: the BB problem per FEM degree, its pivot mgop, and per-variant mgops. Session-
# scoped (via the ``strategy_setup`` conftest fixture) so the properties and regression modules build once.
# ---------------------------------------------------------------------------
def _build_bb_sd_qs(degree: int):
    """(SolutionDomain, QuadratureSet) for the breeding-blanket slab at dim 3 and the given FEM degree."""
    from finite_elements.bb_setup import setup_bb, _get_bb_args
    dim, _deg, *rest = _get_bb_args(3, degree)     # (_get_bb_args fixes tn_order etc.; we override degree)
    solution_domain, _sources, quadrature_set = setup_bb(dim, degree, *rest)
    return solution_domain, quadrature_set


class StrategySetup:
    """Builds + caches the BB problem, pivot mgop, and per-variant mgops per FEM degree. One instance is
    shared session-wide by both test modules, so a degree's expensive build/toposort happens once."""

    def __init__(self):
        self._sdqs: Dict[int, tuple] = {}
        self._pivot: Dict[int, object] = {}
        self._variant: Dict[tuple, object] = {}

    def sd_qs(self, degree: int):
        if degree not in self._sdqs:
            self._sdqs[degree] = _build_bb_sd_qs(degree)
        return self._sdqs[degree]

    def pivot(self, degree: int):
        """The all-defaults (pure-JAX) mgop at this degree -- the consistency pivot."""
        if degree not in self._pivot:
            sd, qs = self.sd_qs(degree)
            self._pivot[degree] = create_multi_group_operator(sd, qs)
        return self._pivot[degree]

    def variant(self, degree: int, key, **create_kwargs):
        """An mgop with the given strategy kwargs swapped in, cached by ``(degree, key)``."""
        cache_key = (degree, key)
        if cache_key not in self._variant:
            sd, qs = self.sd_qs(degree)
            self._variant[cache_key] = create_multi_group_operator(sd, qs, **create_kwargs)
        return self._variant[cache_key]
