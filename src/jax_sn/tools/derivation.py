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
import sys
from typing import (Any, Callable, ClassVar, Dict, FrozenSet, Mapping, NamedTuple, Tuple,
                    get_type_hints)

import equinox as eqx
import jax


__all__ = [
    "MissingQuantityError",        
    "Graph",
    "DerivedModule",
]


class MissingQuantityError(AttributeError):
    '''Raised when a quantity cannot be derived because a primitive it depends on is ``None``.'''



def _direct_dependencies(fn: Callable) -> Tuple[str, ...]:
    '''
    A derivation's dependencies are its parameter names.
    '''
    return tuple(inspect.signature(fn).parameters)


def _primitives_of(derivations: Mapping[str, Callable]) -> FrozenSet[str]:
    '''
    Required input primitives for derivation registry.
    '''
    depended = {dep for fn in derivations.values() for dep in _direct_dependencies(fn)}
    return frozenset(depended - set(derivations))


def _predecessor_graph(derivations: Mapping[str, Callable]) -> Dict[str, FrozenSet[str]]:
    '''
    Map each quantity and its direct dependencies.
    '''
    return {name: frozenset(_direct_dependencies(fn)) for name, fn in derivations.items()}


def _transitive_dependencies(derivations: Mapping[str, Callable], name: str) -> FrozenSet[str]:
    '''
    Everything reachable from ``name``, primitives included and ``name`` itself excluded.
    '''
    graph = _predecessor_graph(derivations)
    graphlib.TopologicalSorter(graph).prepare()     # rejects cycles
    found = set()
    queue = list(graph.get(name, ()))
    while queue:
        n = queue.pop()
        if n in found:
            continue
        found.add(n)
        queue.extend(graph.get(n, ()))
    return frozenset(found)


def _topological_order(derivations: Mapping[str, Callable]) -> Tuple[str, ...]:
    '''
    The derived quantities, ordered so each follows its dependencies. Primitives are excluded.
    Raises :exc:`graphlib.CycleError` if the registry is cyclic.
    '''
    order = graphlib.TopologicalSorter(_predecessor_graph(derivations)).static_order()
    return tuple(n for n in order if n in derivations)


def _frontier(derivations: Mapping[str, Callable],
              projections: Mapping[str, Mapping[str, Callable]],
              frozen: FrozenSet[str]) -> FrozenSet[str]:
    '''
    The quantities to store when ``frozen`` cannot change: those fully determined by ``frozen``,
    minus those already readable out of one that is stored.

    Note the second clause is projection-reachability, not dependency. A quantity another stored one
    merely *consumed* is not thereby recoverable -- ``topology_build`` is built from
    ``face_template`` but does not hold it -- so dropping it would silently trade a field read for a
    recomputation.
    '''
    primitives = _primitives_of(derivations)
    unknown = set(frozen) - primitives
    if unknown:
        raise ValueError(f"Unknown primitive(s) {sorted(unknown)}. "
                         f"Known primitives: {sorted(primitives)}.")
    determined = {
        q for q in derivations
        if _transitive_dependencies(derivations, q) & primitives <= set(frozen)
    }
    stored: set = set()
    for name in reversed(_topological_order(derivations)):      # most derived first
        if name in determined and name not in _projection_paths(projections, frozenset(stored)):
            stored.add(name)
    return frozenset(stored)


def _projection_paths(projections: Mapping[str, Mapping[str, Callable]],
                      stored: FrozenSet[str]) -> Dict[str, Tuple[str, Callable]]:
    '''
    Quantities recoverable from a stored one, each mapped to the single step that gets it there.
    Breadth-first from ``stored``, so every path bottoms out in a field read and cannot recurse.
    '''
    paths: Dict[str, Tuple[str, Callable]] = {}
    queue = list(stored)
    while queue:
        src = queue.pop(0)
        for name, sources in projections.items():
            if name in stored or name in paths or src not in sources:
                continue
            paths[name] = (src, sources[src])
            queue.append(name)
    return paths

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
    projected      : FrozenSet[str]                    # quantities read back out of a stored one
    derived        : FrozenSet[str]                    # quantities recomputed on access
    frozen         : FrozenSet[str]                    # primitives a stored quantity depends on
    free           : FrozenSet[str]                    # primitives safe to replace
    deps           : Mapping[str, FrozenSet[str]]      # quantity -> its direct dependencies
    primitive_deps : Mapping[str, FrozenSet[str]]      # quantity -> primitives it needs, ignoring
                                                       #   storage: what staleness follows from
    required_deps  : Mapping[str, FrozenSet[str]]      # quantity -> primitives it needs *here*,
                                                       #   stopping at stored/projected quantities
    projections    : Mapping[str, Tuple[str, Callable]]  # projected quantity -> (source, extractor)

    def __str__(self):
        return "\n".join(
            f"{k:10s} {sorted(getattr(self, k))}"
            for k in ("stored", "projected", "derived", "frozen", "free")
        )

    __repr__ = __str__          # the default tuple repr is unreadable, and this is for reading


def _missing_primitives(instance: "DerivedModule", name: str) -> FrozenSet[str]:
    '''Primitives ``name`` needs that are ``None`` on ``instance``.'''
    graph = type(instance).graph()
    if name in graph.stored or name in graph.projected:
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
    graph = type(instance).graph()
    if name in graph.stored or name not in graph.quantities:
        val = getattr(instance, name)       # primitive, or a quantity this class stores as a field
    elif name in graph.projections:
        source, extract = graph.projections[name]
        val = extract(_evaluate(instance, source, memo))
    else:
        fn  = instance.DERIVATIONS[name]
        val = fn(*(_evaluate(instance, p, memo) for p in _direct_dependencies(fn)))
    if name in instance.NON_DIFFERENTIABLE:
        # Here rather than in the derivation, so that no derivation *can* forget it, and so that it
        # holds on the stored and projected paths too.
        val = jax.lax.stop_gradient(val)
    memo[name] = val
    return val


def _make_accessor(name: str, fn: Callable) -> property:
    '''
    The accessor injected for a quantity a subclass chose not to store. Injection (rather than, say,
    ``__getattr__``) is what lets an ``eqx.AbstractVar`` declaration count as implemented.
    '''
    params = _direct_dependencies(fn)

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
        f"Derived from {list(params)}; see ``{fn.__module__}.{fn.__qualname__}``. Recomputed on "
        f"every access (though only once per access, however many consumers it has) unless a "
        f"stored quantity projects to it."
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


_GENERATED: Dict[Tuple[type, FrozenSet[str]], type] = {}


def _stored_annotation(fn: Callable) -> Any:
    '''The field type for a stored quantity: its derivation's return annotation.'''
    ann = inspect.signature(fn).return_annotation
    if ann is inspect.Signature.empty or isinstance(ann, str):
        try:
            ann = get_type_hints(fn).get("return", Any)
        except Exception:
            ann = Any
    return ann


def _generated(base: type, stored: FrozenSet[str]) -> type:
    '''
    The class over ``base``'s registry storing exactly ``stored``.

    Memoised on ``(base, stored)``, and deliberately so: a class object is part of a pytree's
    treedef, so two structurally identical but distinct classes would compare unequal, retrace and
    break :func:`jax.tree_util.tree_map` across them.
    '''
    key = (base, frozenset(stored))
    if key in _GENERATED:
        return _GENERATED[key]

    suffix = "_".join(sorted(stored)) if stored else "nothing"
    name   = f"{base.__name__}_storing_{suffix}"
    cls    = type(name, (base,), {
        "__annotations__": {n: _stored_annotation(base.DERIVATIONS[n]) for n in sorted(stored)},
        "__module__"     : base.__module__,
        "__qualname__"   : name,
        "__doc__"        : f"{base.__name__} storing {sorted(stored)}; generated by "
                           f"{base.__name__}.freezing/varying.",
    })
    _GENERATED[key] = cls
    setattr(sys.modules[base.__module__], name, cls)      # so the class is picklable
    return cls


class _DerivedModuleMeta(type(eqx.Module)):
    '''
    Instantiating an abstract class builds the policy that stores nothing.

    An abstract class cannot hold quantities itself -- a field cannot override an injected accessor,
    which is the whole reason the rungs are abstract -- but "store nothing, keep every primitive
    updatable" is the obvious reading of ``Rung(...)``, and making the user spell ``.freezing()`` for
    it is noise. Only instantiation is intercepted; class creation still goes through equinox.
    '''

    def __call__(cls, *args, **kwargs):
        if cls.__dict__.get("_derivation_abstract") and cls.DERIVATIONS:
            return cls.freezing()(*args, **kwargs)
        return super().__call__(*args, **kwargs)


class DerivedModule(eqx.Module, metaclass=_DerivedModuleMeta):
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
    PROJECTIONS: ClassVar[Mapping[str, Mapping[str, Callable]]] = {}
    NON_DIFFERENTIABLE: ClassVar[FrozenSet[str]] = frozenset()

    def __init_subclass__(cls, abstract : bool = False, **kwargs):
        super().__init_subclass__(**kwargs)
        type.__setattr__(cls, "_derivation_abstract", abstract)
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
        primitives = _primitives_of(derivations)
        try:
            fields = frozenset(f.name for f in dataclasses.fields(cls))
        except TypeError:      # not a dataclass yet (abstract class still being built)
            fields = frozenset()

        stored = fields & quantities
        deps = _predecessor_graph(derivations)
        primitive_deps = {
            name: _transitive_dependencies(derivations, name) & primitives for name in quantities
        }
        frozen = frozenset().union(*(primitive_deps[n] for n in stored)) if stored else frozenset()

        cls._check_projections()
        projections = _projection_paths(cls.PROJECTIONS, stored)
        projected = frozenset(projections)
        shielded = stored | projected

        # what is actually needed to compute each quantity on *this* class: a stored quantity is
        # read as a field and a projected one is read out of a stored one, so both shield
        # everything upstream of them.
        required: Dict[str, FrozenSet[str]] = {}

        def require(name: str) -> FrozenSet[str]:
            if name in required:
                return required[name]
            if name in shielded:
                required[name] = frozenset()
                return required[name]
            required[name] = frozenset()          # guards the recursion; cycles already rejected
            out: set = set()
            for dep in deps[name]:
                if dep in primitives:
                    out.add(dep)
                elif dep not in shielded:
                    out |= require(dep)
            required[name] = frozenset(out)
            return required[name]

        for name in quantities:
            require(name)

        cached = Graph(quantities, primitives, stored, projected, quantities - shielded,
                       frozen, primitives - frozen, deps, primitive_deps, required, projections)
        type.__setattr__(cls, "_graph", cached)
        return cached

    @classmethod
    def _check_projections(cls):
        '''A projection extracts a quantity from another quantity; both ends must be in the registry.'''
        for name, sources in cls.PROJECTIONS.items():
            bad = ({name} | set(sources)) - set(cls.DERIVATIONS)
            if bad:
                raise ValueError(
                    f"{cls.__name__}.PROJECTIONS refers to {sorted(bad)}, which "
                    f"{'is' if len(bad) == 1 else 'are'} not in DERIVATIONS. A projection reads one "
                    f"quantity out of another; primitives are already fields."
                )
        bad = frozenset(cls.NON_DIFFERENTIABLE) - set(cls.DERIVATIONS)
        if bad:
            raise ValueError(
                f"{cls.__name__}.NON_DIFFERENTIABLE names {sorted(bad)}, which "
                f"{'is' if len(bad) == 1 else 'are'} not in DERIVATIONS. Only a derived quantity can "
                f"be marked; a primitive is the caller's to stop_gradient."
            )

    # -- generated storage policies ------------------------------------------------------------

    @classmethod
    def frontier(cls, *frozen : str) -> FrozenSet[str]:
        '''The quantities worth storing if ``frozen`` never changes.'''
        cls._check_projections()
        return _frontier(cls.DERIVATIONS, cls.PROJECTIONS, frozenset(frozen))

    @classmethod
    def freezing(cls, *frozen : str) -> type:
        '''
        A subclass storing everything that ``frozen`` determines. Freezing a primitive no quantity
        is fully determined by is a no-op; the result's ``graph().frozen`` says what was achieved.
        '''
        return _generated(cls._generation_base(), cls.frontier(*frozen))

    @classmethod
    def storing(cls, *quantities : str) -> type:
        '''
        A subclass storing exactly ``quantities``. Unlike :meth:`freezing`, this states the storage
        directly rather than deriving it, so it can hold something a frontier would not pick -- use
        it to seed an instance from data you already have, then :meth:`materialise` into the policy
        you want.
        '''
        unknown = set(quantities) - set(cls.DERIVATIONS)
        if unknown:
            raise ValueError(f"Unknown quantities {sorted(unknown)} for {cls.__name__}. "
                             f"Known quantities: {sorted(cls.DERIVATIONS)}.")
        return _generated(cls._generation_base(), frozenset(quantities))

    @classmethod
    def varying(cls, *free : str) -> type:
        '''A subclass that keeps ``free`` updatable and stores as much of the rest as it can.'''
        primitives = _primitives_of(cls.DERIVATIONS)
        unknown = set(free) - primitives
        if unknown:
            raise ValueError(f"Unknown primitive(s) {sorted(unknown)}. "
                             f"Known primitives: {sorted(primitives)}.")
        return cls.freezing(*(primitives - set(free)))

    @classmethod
    def _generation_base(cls) -> type:
        '''
        The nearest ``abstract=True`` ancestor. A concrete class has injected accessors, and a field
        cannot override one, so only an abstract class is safe to generate from.
        '''
        for base in cls.__mro__:
            if base.__dict__.get("_derivation_abstract"):
                return base
        raise TypeError(
            f"{cls.__name__} has no abstract ancestor to generate from. Declare the class carrying "
            f"DERIVATIONS with `abstract=True`."
        )

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
                f"{sorted(graph.free)}. Move to "
                f"`{cls._generation_base().__name__}.varying({', '.join(map(repr, sorted(graph.free | frozen)))})` "
                f"first, or use equinox.tree_at directly if you accept the inconsistency."
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
