"""
Shared helpers for the example performance scripts (`perf_suite.py`, `refactor_perf_verify.py`).

Self-contained measurement (no dependency on internal `jax_sn.tools.memory_analysis` signatures) so
the perf scripts stay portable:

- ``peak_mb(f, *args)``  -- predicted peak device memory (MB) from the *compiler* (no execution, so it
  never OOMs and works even for configs too large to actually run).
- ``time_ms(f, *args)``  -- steady-state wall time (min-of-N, the standard low-noise estimator) after a
  warm-up; returns ``None`` on out-of-memory.
- ``compile_ms(f, *args)`` -- one-shot lower+compile time.
- ``build_problem(...)`` -- a breeding-blanket multigroup slab (downscattering -> lower-triangular
  solvable), parametrized by dim / degree / resolution / quadrature.
"""
import time
import jax
import jax.numpy as jnp
import equinox as eqx

from jax_sn.domain.cross_sections import CrossSectionData
from jax_sn.domain import Domain
from jax_sn.solution_domain import SolutionDomain, BasixLagrangianSimplex
from jax_sn.domain.domain_generation import create_cartesian_setup
import jax_sn

from breeding_blanket_setup import bb_data_dict, base_region_sizes, base_resolutions

# Exceptions that mean "didn't fit / couldn't run" rather than "bug".
try:
    from jax.errors import JaxRuntimeError as _JaxRuntimeError  # newer jax
except Exception:  # pragma: no cover
    _JaxRuntimeError = RuntimeError
_OOM_MARKERS = ("RESOURCE_EXHAUSTED", "out of memory", "OUT_OF_MEMORY")


def _is_oom(exc: Exception) -> bool:
    return any(m in str(exc) for m in _OOM_MARKERS)


def peak_mb(f, *args):
    """Predicted peak device memory (MB) of ``eqx.filter_jit(f)`` compiled on ``args``.

    Uses the compiler's memory analysis (no execution), so it is near-deterministic and cannot OOM.
    Returns a string marker on failure.
    """
    try:
        compiled = eqx.filter_jit(f).lower(*args).compile()
        return compiled.compiled.memory_analysis().peak_memory_in_bytes / 1024**2
    except Exception as e:
        return f"err:{type(e).__name__}"


def compile_ms(f, *args):
    """One-shot lower+compile wall time (ms)."""
    try:
        t0 = time.perf_counter()
        eqx.filter_jit(f).lower(*args).compile()
        return (time.perf_counter() - t0) * 1000.0
    except Exception:
        return None


def time_ms(f, *args, number=5, repeats=3):
    """Steady-state wall time (ms): min over ``repeats`` blocks of ``number`` calls, after one warm-up.

    min-of-N rejects interference (other GPU work, clock transients). Returns ``None`` on OOM.
    """
    fj = eqx.filter_jit(f)
    try:
        jax.block_until_ready(fj(*args))  # warm up (compile + first run)
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            for _ in range(number):
                out = fj(*args)
            jax.block_until_ready(out)
            best = min(best, (time.perf_counter() - t0) / number)
        return best * 1000.0
    except Exception as e:
        if _is_oom(e):
            return None
        raise


def build_problem(dim=3, degree=1, n_other=3, res_mult=1, tn_order=3):
    """Breeding-blanket multigroup slab. Returns (solution_domain, quadrature_set)."""
    data = bb_data_dict()
    xs = {n: CrossSectionData(total=jnp.array(data[n][0]), scattering=jnp.array(data[n][1])) for n in data}
    names = ['Plasma', 'TungstenArmor', 'FirstWall', 'BreedingZone',
             'BackSupportingStructure', 'VVWallFront', 'VVShield', 'VVWallBack']
    xs_regions = [xs[n] for n in names]
    res = [max(1, r * res_mult) for r in base_resolutions()]
    region_size_res = list(zip(base_region_sizes(), res))
    other_dims = [[(5.0, n_other)]] * (dim - 1)
    source_eg0 = [1 if i == 0 else 0 for i in range(len(names))]

    verts, conn, cross_sections, _ = create_cartesian_setup(
        dim, [region_size_res] + other_dims, xs_regions, source_eg0, 0)
    element = BasixLagrangianSimplex(degree, dim)
    domain = Domain.from_mesh_and_cross_sections((verts, conn), element.face_template, cross_sections)
    solution_domain = SolutionDomain.from_element_and_domain(element, domain)
    quad = jax_sn.quadrature_set.create_tn_quadrature_set(tn_order)
    return solution_domain, quad
