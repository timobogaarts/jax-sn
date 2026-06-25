"""
Bench-runner helpers. A ``bench_*.py`` module declares one or more :class:`Bench` objects (in a module
list named ``BENCHES``); ``run.py`` discovers those modules and runs every Bench. Each Bench owns one
operator axis: it discovers the strategy variants, builds the operator via the single uniform path
(``create_multi_group_operator(sd, qs, **{kwarg: strategy})`` then pull the sub-operator), and measures.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Type

import jax.numpy as jnp

from _common import Measurement, measure, LiveTable
from _discovery import discover_variants, Variant
from jax_sn.operators.multi_group_operator.multi_group_operator_creation import create_multi_group_operator


def ones_in(op) -> jnp.ndarray:
    return jnp.ones(op.in_structure().shape)


def mv_call(op) -> Tuple[Callable, tuple]:
    """The standard single-application benchmark: ``op.mv(ones)``."""
    return (lambda o, x: o.mv(x), (op, ones_in(op)))


def inv_call(op) -> Tuple[Callable, tuple]:
    """The inverse-sweep benchmark: ``op.inv(ones)``."""
    return (lambda o, b: o.inv(b), (op, ones_in(op)))


@dataclass
class Bench:
    """One operator axis swept over its discovered strategy variants.

    ``extract`` pulls the benchmarked operator out of a built MGOP (default: ``getattr(mgop, attr)``).
    ``make_call(op) -> (fn, args)`` produces the timed callable + its inputs (default: ``mv_call``).
    """
    name: str
    base: Type
    kwarg: str
    attr: str = ""
    group: str = "cheap"
    make_call: Callable = mv_call
    extract: Callable = None
    instantiate: bool = True   # False -> the axis value is the concrete *class* (e.g. scatter)
    overrides: Dict[Type, List[object]] = field(default_factory=dict)

    def _extract(self, mgop):
        return self.extract(mgop) if self.extract is not None else getattr(mgop, self.attr)

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
                mgop = create_multi_group_operator(sd, qs, **{self.kwarg: strat})
                return self.make_call(self._extract(mgop))

            m = measure(build_fn)
            table.ok(m.peak_mb, m.t_ms) if m.status == "ok" else table.skip(m.reason)
            out[v.label] = m.__dict__
        return out
