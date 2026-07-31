'''
Module to handle flat dependent modules in a relatively simple way.
In essense, a derived module is a graph (a sweep data depends on face normals which depend on vertices), and so we use some graph tools to create the graph,
check that what you are modifying makes sense, etc.
Used for parametrizing the jax_sn modules.

[Claude generated most of this!]
'''

from __future__ import annotations

import dataclasses
import graphlib
import inspect
from typing import Any, Callable, ClassVar, Dict, FrozenSet, Mapping, NamedTuple, Tuple

import equinox as eqx


__all__ = [
    "MissingQuantityError",
    "direct_dependencies",
    "primitives_of",
    "predecessor_graph",
    "transitive_dependencies",
    "topological_order",
    "Graph",
    "DerivedModule",
]


class MissingQuantityError(AttributeError):
    '''Raised when a quantity cannot be derived because a primitive it depends on is ``None``.'''



def direct_dependencies(fn: Callable) -> Tuple[str, ...]:
    '''
    A derivation's dependencies are its parameter names, as with pytest fixtures.
    '''
    return tuple(inspect.signature(fn).parameters)


def primitives_of(derivations: Mapping[str, Callable]) -> FrozenSet[str]:
    '''
    Required input primitives for derivation registry.
    '''
    depended = {dep for fn in derivations.values() for dep in direct_dependencies(fn)}
    return frozenset(depended - set(derivations))


def predecessor_graph(derivations: Mapping[str, Callable]) -> Dict[str, FrozenSet[str]]:
    '''
    The registry as a ``{node: predecessors}`` mapping, the form :mod:`graphlib` expects.
    '''
    return {name: frozenset(direct_dependencies(fn)) for name, fn in derivations.items()}


def transitive_dependencies(derivations: Mapping[str, Callable], name: str) -> FrozenSet[str]:
    '''
    Everything reachable from ``name``, primitives included and ``name`` itself excluded.
    '''
    graph = predecessor_graph(derivations)
    graphlib.TopologicalSorter(graph).prepare()     # rejects cycles
    found: set = set()
    queue = list(graph.get(name, ()))
    while queue:
        n = queue.pop()
        if n in found:
            continue
        found.add(n)
        queue.extend(graph.get(n, ()))
    return frozenset(found)


def topological_order(derivations: Mapping[str, Callable]) -> Tuple[str, ...]:
    '''
    The derived quantities, ordered so each follows its dependencies. Primitives are excluded.
    Raises :exc:`graphlib.CycleError` if the registry is cyclic.
    '''
    order = graphlib.TopologicalSorter(predecessor_graph(derivations)).static_order()
    return tuple(n for n in order if n in derivations)


# ---------------------------------------------------------------------------------------------
# The module base.
# ---------------------------------------------------------------------------------------------

class Graph(NamedTuple):
    '''
    What a class's storage policy implies. All derived from the registry plus which quantities the
    class declared as fields; see :meth:`DerivedModule.graph`.
    '''

    quantities     : FrozenSet[str]                    # everything with a derivation
    primitives     : FrozenSet[str]                    # everything without one
    stored         : FrozenSet[str]                    # quantities held as fields
    derived        : FrozenSet[str]                    # quantities recomputed on access
    frozen         : FrozenSet[str]                    # primitives a stored quantity depends on
    free           : FrozenSet[str]                    # primitives safe to replace
    deps           : Mapping[str, FrozenSet[str]]      # quantity -> its direct dependencies
    primitive_deps : Mapping[str, FrozenSet[str]]      # quantity -> primitives it needs, ignoring
                                                       #   storage: what staleness follows from
    required_deps  : Mapping[str, FrozenSet[str]]      # quantity -> primitives it needs *here*,
                                                       #   stopping at stored quantities

    def __str__(self):
        return "\n".join(
            f"{k:8s} {sorted(getattr(self, k))}" for k in ("stored", "derived", "frozen", "free")
        )


def _missing_primitives(instance: "DerivedModule", name: str) -> FrozenSet[str]:
    '''Primitives ``name`` needs that are ``None`` on ``instance``.'''
    graph = type(instance).graph()
    if name in graph.stored:
        return frozenset()
    return frozenset(
        p for p in graph.required_deps.get(name, ()) if getattr(instance, p, None) is None
    )


def _evaluate(instance: "DerivedModule", name: str, memo: Dict[str, Any]) -> Any:
    '''
    Resolve ``name``, reusing anything already in ``memo``.

    The registry is a DAG, not a tree: a quantity with several consumers would otherwise be rebuilt
    once per consumer, and the cost multiplies along every diamond. ``memo`` is per top-level access
    and dies with it, so nothing outlives the trace it was built in.
    '''
    if name in memo:
        return memo[name]
    if name in type(instance).graph().derived:
        fn  = instance.DERIVATIONS[name]
        val = fn(*(_evaluate(instance, p, memo) for p in direct_dependencies(fn)))
    else:
        val = getattr(instance, name)       # primitive, or a quantity this class stores as a field
    memo[name] = val
    return val


def _make_accessor(name: str, fn: Callable) -> property:
    '''
    The accessor injected for a quantity a subclass chose not to store. Injection (rather than, say,
    ``__getattr__``) is what lets an ``eqx.AbstractVar`` declaration count as implemented.
    '''
    params = direct_dependencies(fn)

    def _get(self, _name=name):
        missing = _missing_primitives(self, _name)
        if missing:
            raise MissingQuantityError(
                f"Cannot derive '{_name}' on {type(self).__name__}: it requires "
                f"{sorted(missing)}, which {'is' if len(missing) == 1 else 'are'} None."
            )
        return _evaluate(self, _name, {})

    _get.__name__ = name
    _get.__qualname__ = name
    _get.__doc__ = (
        f"Derived from {list(params)}. Recomputed on every access (though only once per access, "
        f"however many consumers it has); see ``{fn.__module__}.{fn.__qualname__}``."
    )
    _get._derived_accessor = True       # tag, so _check_not_shadowed can recognise it
    return property(_get)


def _check_not_shadowed(cls: type, name: str):
    '''
    A field cannot override an inherited accessor (a ``property`` has no setter, so ``__init__``
    fails). Both causes are the same mistake: a class that derives ``name`` sits above one that
    stores it.
    '''
    for base in cls.__mro__[1:]:
        attr = base.__dict__.get(name)
        if attr is None:
            continue
        if isinstance(attr, property) and getattr(attr.fget, "_derived_accessor", False):
            raise TypeError(
                f"{cls.__name__} stores '{name}' as a field, but {base.__name__} derives it, and a "
                f"field cannot override an accessor. Either declare {base.__name__} with "
                f"abstract=True (so nothing is injected there and it becomes safe to inherit from), "
                f"or make {cls.__name__} a sibling of {base.__name__} rather than a subclass."
            )
        return          # some nearer base defines it another way: not ours to police


class DerivedModule(eqx.Module):
    '''
    Base for modules built from a derivation registry.

    Subclasses set :attr:`DERIVATIONS` to a ``{quantity: function}`` mapping whose functions take
    their dependencies as named parameters. A concrete subclass declares as fields the primitives
    plus whichever derived quantities it wants to store; accessors for the rest are injected
    automatically.

    Declare intermediate classes ``abstract=True``::

        class Interface(DerivedModule, abstract=True): ...

    which suppresses that injection. Since it is an injected accessor that makes a class unsafe to
    inherit from -- a field cannot override one -- an abstract class *is* safe to inherit from, so
    shared fields and methods can live there and be declared once.

    Attributes
    ----------
    DERIVATIONS : ClassVar[Mapping[str, Callable]]
        The derivation registry. Inherited by subclasses unless overridden.
    '''

    DERIVATIONS: ClassVar[Mapping[str, Callable]] = {}

    def __init_subclass__(cls, abstract : bool = False, **kwargs):
        super().__init_subclass__(**kwargs)
        if abstract:
            return
        declared = set(getattr(cls, "__annotations__", {}))
        for name, fn in cls.DERIVATIONS.items():
            if name in declared:
                _check_not_shadowed(cls, name)
                continue
            if name in cls.__dict__:
                continue
            setattr(cls, name, _make_accessor(name, fn))

    # -- graph -------------------------------------------------------------------------------

    @classmethod
    def graph(cls) -> Graph:
        '''
        This class's storage policy and what it implies. Memoised on the class; it cannot be built
        in ``__init_subclass__`` because ``dataclasses.fields`` is only populated after equinox's
        metaclass has finished.
        '''
        cached = cls.__dict__.get("_graph")       # __dict__, so a subclass never sees a parent's
        if cached is not None:
            return cached

        derivations = cls.DERIVATIONS
        quantities = frozenset(derivations)
        primitives = primitives_of(derivations)
        try:
            fields = frozenset(f.name for f in dataclasses.fields(cls))
        except TypeError:      # not a dataclass yet (abstract class still being built)
            fields = frozenset()

        stored = fields & quantities
        deps = predecessor_graph(derivations)
        primitive_deps = {
            name: transitive_dependencies(derivations, name) & primitives for name in quantities
        }
        frozen = frozenset().union(*(primitive_deps[n] for n in stored)) if stored else frozenset()

        # what is actually needed to compute each quantity on *this* class: a stored quantity is
        # read as a field, so it shields everything upstream of it.
        required: Dict[str, FrozenSet[str]] = {}

        def require(name: str) -> FrozenSet[str]:
            if name in required:
                return required[name]
            if name in stored:
                required[name] = frozenset()
                return required[name]
            required[name] = frozenset()          # guards the recursion; cycles already rejected
            out: set = set()
            for dep in deps[name]:
                if dep in primitives:
                    out.add(dep)
                elif dep not in stored:
                    out |= require(dep)
            required[name] = frozenset(out)
            return required[name]

        for name in quantities:
            require(name)

        cached = Graph(quantities, primitives, stored, quantities - stored,
                       frozen, primitives - frozen, deps, primitive_deps, required)
        type.__setattr__(cls, "_graph", cached)
        return cached

    def available_quantities(self) -> FrozenSet[str]:
        '''
        The quantities this instance can actually produce: a quantity is unavailable when a
        primitive it needs is ``None`` (e.g. angular quantities on a builder with no quadrature set).
        '''
        return frozenset(
            n for n in type(self).graph().quantities if not _missing_primitives(self, n)
        )

    def updated(self, **primitives: Any) -> "DerivedModule":
        '''
        Replace primitives, keeping everything else. Only ``graph().free`` primitives may be
        replaced; everything derived from them follows automatically.
        '''
        cls, graph = type(self), type(self).graph()

        unknown = set(primitives) - graph.primitives
        if unknown:
            if unknown & graph.quantities:
                name = sorted(unknown & graph.quantities)[0]
                raise ValueError(
                    f"'{name}' is a quantity, not a primitive; it follows from "
                    f"{sorted(graph.primitive_deps[name])}. Update those instead."
                )
            raise ValueError(f"Unknown primitive(s) {sorted(unknown)} for {cls.__name__}. "
                             f"Known primitives: {sorted(graph.primitives)}.")

        frozen = set(primitives) & graph.frozen
        if frozen:
            culprits = sorted(q for q in graph.stored if graph.primitive_deps[q] & frozen)
            raise ValueError(
                f"{cls.__name__} cannot update {sorted(frozen)}: it stores {culprits}, derived "
                f"from {'it' if len(frozen) == 1 else 'them'}. Free to update: "
                f"{sorted(graph.free)}. Use a class that derives {culprits} instead, or "
                f"equinox.tree_at directly if you accept the inconsistency."
            )

        values = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        return cls(**{**values, **primitives})

    def materialise(self, cls: type) -> "DerivedModule":
        '''
        Convert to another class over the same registry, storing whatever that class stores.
        Promoting pays a repeated derivation once; demoting frees more primitives to update.
        Raises :exc:`MissingQuantityError` if the target stores something unavailable here.
        '''
        return cls(**{f.name: getattr(self, f.name) for f in dataclasses.fields(cls)})
