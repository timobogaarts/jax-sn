'''
Temporary CI diagnostic for the parametrization notebook's gradient cell, which times out
(>600 s) on a hosted runner while taking ~3 s locally on the same package versions.

Reproduces that cell standalone and reports *where* the time goes:
- ``jax_log_compiles`` prints one line per XLA compilation with its duration, so a slow
  compile is visible as such rather than as a hang.
- ``faulthandler.dump_traceback_later(60, repeat=True)`` dumps a Python stack every 60 s to
  the real stderr, which the runner's log captures. That separates the three cases: stuck in
  ``backend_compile`` (XLA), stuck in a runtime execute, or stuck in Python (e.g. the level
  computation looping host-side).

Run it before the docs build:  python tools/debug_parametrization_cell.py
'''

import faulthandler
import os
import platform
import sys
import time

faulthandler.dump_traceback_later(60, repeat=True, file=sys.__stderr__)

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_log_compiles", True)

import equinox as eqx
import jax.numpy as jnp
import jax_sbgeom as jsb
from jax_sbgeom.flux_surfaces.miller_tokamak import MillerTokamak

from jax_sn.domain import CrossSectionData, CrossSectionsIndexed, Domain
from jax_sn.operator_creation.builders import MultiGroupBuilder
from jax_sn.operator_creation.default_strategies import DefaultMultiGroupStrategy
from jax_sn.quadrature_set import create_tn_quadrature_set
from jax_sn.solution_domain import BasixLagrangianSimplex


def _log(msg):
    print(f"[debug] {msg}", flush=True)


_log(f"python {platform.python_version()}  jax {jax.__version__}  devices {jax.devices()}")
_log(f"os.cpu_count()={os.cpu_count()}  affinity={len(os.sched_getaffinity(0))}")
_log(f"XLA_FLAGS={os.environ.get('XLA_FLAGS')!r}")


def timed(label, fn):
    t = time.perf_counter()
    out = fn()
    jax.block_until_ready(out)
    _log(f"{label}: {time.perf_counter() - t:.1f} s")
    return out


# --- notebook setup cells, verbatim in effect -------------------------------------------
n_s, n_theta = 4, 7
tokamak = MillerTokamak(rminor=1.0, rmajor=1.2, elongation=0.5, triangularity=0.2)
mesh = jsb.flux_surfaces.flux_surface_meshing.mesh_poloidal_plane(
    tokamak, jnp.linspace(0.0, 1.0, n_s), 0.0, n_theta)
mesh = (jnp.stack([mesh[0][:, 0], mesh[0][:, 2]], axis=-1), mesh[1])

water = CrossSectionData(
    total=jnp.array([15.2090, 24.4140]),
    scattering=jnp.array([[[5.5069, 0.0], [6.6227, 11.4080]],
                          [[1.158567, 0.0], [1.1234767, 2.4733]]]))
iron = CrossSectionData(
    total=jnp.array([30.3110, 26.8760]),
    scattering=jnp.array([[[18.2500, 0.0], [6.45294, 21.3570]],
                          [[4.36, 0.0], [-0.003508, 2.3457]]]))
void = CrossSectionData(total=jnp.array([1e-10, 1e-10]), scattering=jnp.zeros((2, 2, 2)))

n_elements = mesh[1].shape[0]
indices = jnp.concatenate([jnp.full((n_elements // 3,), 0),
                           jnp.full((n_elements // 3,), 1),
                           jnp.full((n_elements - 2 * (n_elements // 3),), 2)])
cross_sections_indexed = CrossSectionsIndexed(
    indices=indices, cross_section_data=CrossSectionData.from_list([water, iron, void]))

element = BasixLagrangianSimplex(2, 2)
quadrature_set = create_tn_quadrature_set(3)
domain = Domain.from_mesh_and_cross_sections(mesh, element.face_template, cross_sections_indexed)
_log(f"{n_elements} elements, {quadrature_set.angles.shape[0]} angles, n_basis={element.n_basis}")

mg_builder = MultiGroupBuilder(
    vertices=domain.vertices,
    connectivity=domain.connectivity,
    element=element,
    cross_sections_indexed=domain.cross_sections_indexed,
    quadrature_set=quadrature_set,
    strategy=DefaultMultiGroupStrategy.default_multi_group_strategy())

varying_vertices = mg_builder.varied("vertices")


def _updated_mg(theta, f_theta, builder):
    return builder.updated(vertices=f_theta(theta, builder.domain.vertices))


@eqx.filter_jit
def objective(theta, f_theta, builder):
    updated = _updated_mg(theta, f_theta, builder)
    rhs = jnp.ones_like(updated.multi_group_operator.in_structure())
    return jnp.sum(updated.multi_group_operator.mv(rhs))


f_theta = lambda theta, old_vertices: old_vertices.at[-1].add(theta)
_grad_objective = eqx.filter_jit(jax.grad(objective, argnums=0))

theta = jnp.array([0.1, 0.04])

# --- the cell that times out, one step at a time ----------------------------------------
timed("objective trace+compile+run (1st call)",
      lambda: objective(theta, f_theta, varying_vertices))
timed("objective run (2nd call, cached)",
      lambda: objective(theta + 1e-5, f_theta, varying_vertices))
timed("grad trace+compile+run (1st call)",
      lambda: _grad_objective(theta, f_theta, varying_vertices))

faulthandler.cancel_dump_traceback_later()
_log("done")
