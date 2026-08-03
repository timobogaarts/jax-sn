"""
Bench-runner helpers. A ``bench_*.py`` module declares one or more benches (in a module list named
``BENCHES``); ``run.py`` discovers those modules and runs every one. Two runner shapes:

- :class:`Bench` -- one standalone operator axis (mass, moment, discrete, scatter, single-angle
  transport): it discovers every concrete strategy for that axis, composes each into the package's
  canonical default multi-group strategy tree (swapping only the one field under test, see
  :func:`compose_chain_strategy`), builds via ``create_multi_group_operator``, pulls the sub-operator out
  (``extract``), and measures. Auto-discovery is appropriate here because these are isolated-kernel
  costs -- what the rest of the pipeline defaults to doesn't matter.
- :class:`ComposedBench` -- a whole-pipeline axis (the aggregate MGOp/SGOp matvecs, the solvers): swept
  over the hand-picked backend recipes in :func:`default_multi_group_strategies` instead. Auto-discovery
  would combine an arbitrary sub-operator with defaults elsewhere, which can produce combinations nobody
  actually runs (e.g. a Warp transport lift paired with a non-Warp mass operator); hand-picking the
  package's own ``DefaultMultiGroupStrategy`` recipes keeps every row an internally-consistent, real
  configuration.
"""
import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Type

import equinox as eqx
import jax.numpy as jnp

from _common import Measurement, measure, LiveTable
from _discovery import discover_variants, Variant
from jax_sn.operator_creation.builders import MultiGroupBuilder
from jax_sn.operator_creation.default_strategies import DefaultMultiGroupStrategy
from jax_sn.operators.single_group_operator import (
    SingleGroupData, SingleGroupOperator, ComposedSingleGroupStrategy,
)
from jax_sn.operators.transport_operator import VmapAnglesForward, VmapAnglesInversion
from jax_sn.operators.scatter_operator import IndexedScatter


def create_multi_group_operator(sd, qs, strategy):
    """The whole-pipeline build every bench measures: solution domain + quadrature set + strategy tree
    -> ``MultiGroupOperator``, via the builder rung."""
    return MultiGroupBuilder.from_solution_domain(
        sd, quadrature_set=qs, strategy=strategy).multi_group_operator


def ones_in(op) -> jnp.ndarray:
    return jnp.ones(op.in_structure().shape)


def mv_call(op) -> Tuple[Callable, tuple]:
    """The standard single-application benchmark: ``op.mv(ones)``."""
    return (lambda o, x: o.mv(x), (op, ones_in(op)))


def inv_call(op) -> Tuple[Callable, tuple]:
    """The inverse-sweep benchmark: ``op.inv(ones)``."""
    return (lambda o, b: o.inv(b), (op, ones_in(op)))


def default_multi_group_strategy():
    """A fresh instance of the package's canonical default composition (non-batched jax kernels,
    checkpointed scan, see ``operator_creation/default_strategies.py``). Every ``Bench`` axis swaps
    exactly one field of this tree rather than hand-listing a default per axis."""
    return DefaultMultiGroupStrategy.default_multi_group_strategy()


def compose_chain_strategy(chain_field: str, instance, lift: Callable = lambda x: x, top_level: bool = False):
    """The default multi-group strategy tree with one field swapped for ``instance`` (after ``lift``).

    ``chain_field`` names an attribute of the single-group chain strategy (``mass_strategy``,
    ``moment_strategy``, ``discrete_strategy``, ``transport_strategy``, ``transport_inversion_strategy``,
    ``scatter_strategy``) by default, or of the multi-group strategy itself when ``top_level=True`` (the
    gather-scatter axis, whose strategy lives one level up)."""
    base = default_multi_group_strategy()
    value = lift(instance)
    where = (lambda s: getattr(s, chain_field)) if top_level \
        else (lambda s: getattr(s.single_group_chain_strategy, chain_field))
    return eqx.tree_at(where, base, value)


def lift_forward(inner):
    """A discovered ``SingleAngleForwardStrategy`` lifted to the multi-angle form the chain expects: its
    fused native lift if it has one, else the ``VmapAnglesForward`` combinator."""
    native = type(inner).native_multi_angle()
    return native() if native is not None else VmapAnglesForward(inner)


def lift_inversion(inner):
    """A discovered ``SingleAngleInversionStrategy`` lifted the way the old operator-creation factory
    used to auto-resolve before the strategy tree became explicit: ``VmapAnglesInversion`` if the kernel
    vectorises, else its fused native multi-angle lift (e.g. a Warp level sweep)."""
    if inner.vmappable():
        return VmapAnglesInversion(inner)
    native = type(inner).native_multi_angle()
    if native is not None:
        return native()
    raise ValueError(f"{type(inner).__name__} is neither vmappable nor has a native multi-angle lift")


def single_group_operators(mgop, g: int = 0):
    """``(transport, moment, discrete, mass)`` sub-operators for group ``g``, pulled from the MGOp's own
    chain strategy -- the data+strategy layout has no per-axis properties on ``mgop`` itself. (The chain
    strategy has no scatter operator -- ``C_g = M T^-1 H D`` doesn't need one -- so the within-group
    scatter isn't available via this path; see :func:`full_single_group_strategy`.)"""
    chain = mgop.strategy.single_group_chain_strategy
    return chain.all_operators(SingleGroupData(mgop.data, g))


def full_single_group_strategy(mgop) -> ComposedSingleGroupStrategy:
    """A :class:`ComposedSingleGroupStrategy` (the full ``I - M T^-1 H D S`` operator) sharing the MGOp's
    chain strategy's sub-strategies, plus an ``IndexedScatter`` for the within-group scatter the chain
    strategy doesn't carry."""
    chain = mgop.strategy.single_group_chain_strategy
    fields = {f.name: getattr(chain, f.name) for f in dataclasses.fields(chain)}
    return ComposedSingleGroupStrategy(**fields, scatter_strategy=IndexedScatter())


def single_group_operator(mgop, g: int = 0) -> SingleGroupOperator:
    return SingleGroupOperator(SingleGroupData(mgop.data, g), full_single_group_strategy(mgop))


_REPRESENTATIVE_BATCH_SIZE = 1000   # matches the batch size used elsewhere (_FWD_OVERRIDES/_INV_OVERRIDES)


def default_multi_group_strategies() -> "Dict[str, Callable[[], object]]":
    """The hand-picked backend recipes from ``DefaultMultiGroupStrategy``, each as a zero-arg factory
    (a fresh strategy tree per call, since strategies are plain immutable ``eqx.Module`` trees) -- the
    fixed row set :class:`ComposedBench` sweeps, instead of an auto-discovered cross product."""
    return {
        "jax":            lambda: DefaultMultiGroupStrategy.default_multi_group_strategy(),
        "jax-batched":    lambda: DefaultMultiGroupStrategy.default_multi_group_strategy(
            batch_size=_REPRESENTATIVE_BATCH_SIZE),
        "pallas":         lambda: DefaultMultiGroupStrategy.default_multi_group_pallas_strategy(),
        "pallas-batched": lambda: DefaultMultiGroupStrategy.default_multi_group_pallas_strategy(
            batch_size=_REPRESENTATIVE_BATCH_SIZE),
        "warp":           lambda: DefaultMultiGroupStrategy.default_multi_group_warp_strategy(),
        "warp-batched":   lambda: DefaultMultiGroupStrategy.default_multi_group_warp_strategy(
            batch_size=_REPRESENTATIVE_BATCH_SIZE),
    }


@dataclass
class ComposedBench:
    """One whole-pipeline axis, swept over :func:`default_multi_group_strategies` rather than an
    auto-discovered per-field sweep. ``extract`` pulls the benchmarked operator out of the built MGOP;
    ``make_call(op) -> (fn, args)`` produces the timed callable + its inputs (default: ``mv_call``).
    """
    name: str
    extract: Callable
    group: str = "cheap"
    make_call: Callable = mv_call
    strategies: Dict[str, Callable] = field(default_factory=default_multi_group_strategies)

    def run(self, sd, qs) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        table = LiveTable(self.name, list(self.strategies))
        for label, make_strategy in self.strategies.items():
            table.start(label)

            def build_fn(make=make_strategy):
                mgop = create_multi_group_operator(sd, qs, make())
                return self.make_call(self.extract(mgop))

            m = measure(build_fn)
            table.ok(m.peak_mb, m.t_ms) if m.status == "ok" else table.skip(m.reason)
            out[label] = m.__dict__
        return out


@dataclass
class Bench:
    """One operator axis swept over its discovered strategy variants.

    ``chain_field``/``top_level``/``lift`` control how a discovered instance is composed into the default
    multi-group strategy tree, see :func:`compose_chain_strategy`. ``extract`` pulls the benchmarked
    operator out of the built MGOP. ``make_call(op) -> (fn, args)`` produces the timed callable + its
    inputs (default: ``mv_call``).
    """
    name: str
    base: Type
    chain_field: str
    extract: Callable
    top_level: bool = False
    lift: Callable = lambda inst: inst
    group: str = "cheap"
    make_call: Callable = mv_call
    instantiate: bool = True   # False -> the axis value is the concrete *class*, not an instance
    overrides: Dict[Type, List[object]] = field(default_factory=dict)

    def _compose(self, instance):
        return compose_chain_strategy(self.chain_field, instance, self.lift, self.top_level)

    def run(self, sd, qs) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        variants = discover_variants(self.base, self.overrides, self.instantiate)
        table = LiveTable(self.name, [v.label for v in variants])
        for v in variants:
            table.start(v.label)
            if v.instance is None:
                table.skip(v.skip_reason)
                out[v.label] = Measurement.skip(v.skip_reason).__dict__
                continue

            def build_fn(strat=v.instance):
                mgop = create_multi_group_operator(sd, qs, self._compose(strat))
                return self.make_call(self.extract(mgop))

            m = measure(build_fn)
            table.ok(m.peak_mb, m.t_ms) if m.status == "ok" else table.skip(m.reason)
            out[v.label] = m.__dict__
        return out
