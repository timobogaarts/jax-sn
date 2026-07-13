"""
Autodiscovery engine for the perf-test suite.

The benchmarked set of strategies is **never hand-listed as a matrix**. It is derived from the package:

  1. every *concrete* subclass of a strategy base (``MassStrategy``, ``SingleAngleInversionStrategy``, ...) that is
     importable on this machine is a candidate -- so optional backends (pallas, warp) appear only when
     their module imports (no GPU/warp -> the subclass simply never registers);
  2. a candidate that builds but is *incompatible* with the current problem (e.g. a warp tetrahedron
     kernel on a non-tet mesh) self-classifies as a skip via the package's own ``check_compatible`` /
     build-time assertions -- caught by ``_common.measure``, not predicted by a ``requires=(...)`` table.

The one piece that cannot be discovered is a *representative constructor parameter* (a batch size, a
level cap): default-construction benchmarks e.g. ``BatchedForward()`` == non-batched. Benches supply a
small per-class ``overrides`` map for those; it is local and tiny, not a cross-product.
"""
import dataclasses
from typing import Dict, List, Tuple, Type

import equinox as eqx


def ensure_optional_backends() -> None:
    """Import the optional kernel modules so their strategy subclasses register for discovery.

    Guarded: a missing GPU/warp backend just means those subclasses never appear (the desired
    behaviour -- discovery excludes what the machine cannot run).
    """
    try:
        import warp
        warp.config.quiet = True   # silence the init banner + per-module load lines, which otherwise
        #                            print to stdout mid-measurement and corrupt the live aligned rows
    except Exception:
        pass
    for mod in (
        "jax_sn.operators.transport_operator.implementations.transport_operator_pallas",
        "jax_sn.operators.transport_operator.implementations.transport_operator_warp",
        "jax_sn.operators.mass_operator.mass_operator_warp",
    ):
        try:
            __import__(mod)
        except Exception:
            pass


def concrete_subclasses(base: Type) -> List[Type]:
    """All importable concrete subclasses of ``base`` (recursive, abstract bases filtered out)."""
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


def label(inst) -> str:
    """``ClassName`` plus any set scalar init fields, e.g. ``BatchedForward(1000)``."""
    name = type(inst).__name__
    extras = []
    if dataclasses.is_dataclass(inst):
        for f in dataclasses.fields(inst):
            try:
                v = getattr(inst, f.name)
            except Exception:
                continue
            if isinstance(v, (int, float, str, bool)) and v is not None:
                extras.append(str(v))
    return f"{name}({','.join(extras)})" if extras else name


@dataclasses.dataclass
class Variant:
    """One discovered strategy instance to benchmark, or a skip with a reason."""
    label: str
    instance: object = None
    skip_reason: str = ""


def discover_variants(base: Type, overrides: Dict[Type, List[object]] = None,
                      instantiate: bool = True) -> List[Variant]:
    """Discovered variants for ``base``.

    ``instantiate=True`` (strategy axes): each variant's ``instance`` is a default-constructed strategy.
    ``instantiate=False`` (a *class*-valued axis like ``gather_scatter_operator_cls``): each variant's
    ``instance`` is the concrete subclass itself.

    ``overrides`` maps a concrete class to an explicit list of instances to benchmark (replacing the
    default-constructed one) -- used to inject representative batch sizes / level caps.
    """
    overrides = overrides or {}
    variants: List[Variant] = []
    for cls in concrete_subclasses(base):
        if cls in overrides:
            for inst in overrides[cls]:
                variants.append(Variant(label(inst), inst))
            continue
        if not instantiate:
            variants.append(Variant(cls.__name__, cls))
            continue
        try:
            inst = cls()                       # all current strategies default-construct
        except Exception as e:
            variants.append(Variant(cls.__name__, skip_reason=f"no default ctor: {e}"))
            continue
        variants.append(Variant(label(inst), inst))
    return variants


def compatible_pairs(
    inv_variants: List[Variant], lift_variants: List[Variant], check
) -> List[Tuple[Variant, Variant]]:
    """The (inversion, lifting) pairs that pass ``check(inv, lift) -> bool`` (raises => incompatible).

    Used by the multiangle bench: the warp liftings assert a matching warp inversion in their
    ``check_compatible``, so the valid combos are *pruned from the full product*, not declared.
    """
    pairs = []
    for inv in inv_variants:
        if inv.instance is None:
            continue
        for lift in lift_variants:
            if lift.instance is None:
                continue
            try:
                if check(inv.instance, lift.instance):
                    pairs.append((inv, lift))
            except Exception:
                pass  # incompatible pair -> pruned
    return pairs
