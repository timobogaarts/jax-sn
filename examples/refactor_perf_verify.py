"""
Verify + benchmark the data-ownership refactor against the pre-refactor state.

This exercises (almost) everything the `operator = setup + spec + strategy` refactor touched -- MGOp
construction, the derived sub-operators (transport / mass / gather-scatter / moment / discrete), the
single-group operator, transpose, the parametrised reconstruction (vertices + cross sections + sweep
rebuild), and a grad through the lower-triangular solve -- and reports peak memory + wall time for each.

It is written against the API that is identical on the pre-refactor commit and HEAD
(`create_multi_group_operator(sd, qs)`, the public operator methods, the parametrisation), and brings
its own memory/timing measurement, so the SAME file can be run on both states for a direct diff:

    # new (HEAD), uses the editable install:
    python examples/refactor_perf_verify.py --degree 1 --n-other 3

    # old baseline, from a worktree at the pre-refactor commit:
    git worktree add /tmp/jaxsn-base <pre-refactor-sha>
    cp examples/refactor_perf_verify.py /tmp/jaxsn-base/examples/
    PYTHONPATH=/tmp/jaxsn-base/src python /tmp/jaxsn-base/examples/refactor_perf_verify.py --degree 1 --n-other 3

Vary --degree / --res-mult / --n-other / --tn-order to push the number of DOFs (some effects only show
up at higher DOF counts).
"""
import argparse
import time
import jax
import jax.numpy as jnp
import equinox as eqx
import lineax as lx

jax.config.update("jax_enable_x64", True)

from jax_sn.domain.cross_sections import CrossSectionData
from jax_sn.domain import Domain
from jax_sn.solution_domain import SolutionDomain, BasixLagrangianSimplex
from jax_sn.domain.domain_generation import create_cartesian_setup
from jax_sn.operators.multi_group_operator.multi_group_operator_creation import create_multi_group_operator
from jax_sn.operators.multi_group_operator import MultiGroupLowerTriangularBlockSolver
from jax_sn.parametrisation import MultigroupOperatorParametrisedVerticesXS
import jax_sn

from breeding_blanket_setup import bb_data_dict, base_region_sizes, base_resolutions


# --------------------------------------------------------------------------------------------------
# Measurement helpers (self-contained: no dependency on internal helper signatures)
# --------------------------------------------------------------------------------------------------
def peak_mb(f, *args):
    """Peak device memory (MB) of `eqx.filter_jit(f)` lowered+compiled on `args`."""
    try:
        compiled = eqx.filter_jit(f).lower(*args).compile()
        return compiled.compiled.memory_analysis().peak_memory_in_bytes / 1024**2
    except Exception as e:
        return f"n/a ({type(e).__name__})"


def time_ms(f, *args, number=5):
    """Mean wall time (ms) of `eqx.filter_jit(f)` over `number` calls after one warm-up."""
    fj = eqx.filter_jit(f)
    jax.block_until_ready(fj(*args))  # warm up (compile + first run)
    t0 = time.perf_counter()
    for _ in range(number):
        out = fj(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / number * 1000.0


def bench(label, f, *args, number=5):
    mem = peak_mb(f, *args)
    t = time_ms(f, *args, number=number)
    mem_s = f"{mem:8.1f}" if isinstance(mem, float) else f"{mem:>8}"
    print(f"  {label:<34} peak {mem_s} MB   {t:9.2f} ms")
    return mem, t


# --------------------------------------------------------------------------------------------------
# Setup (breeding-blanket multigroup slab, downscattering -> lower-triangular solvable)
# --------------------------------------------------------------------------------------------------
def build_mgop(dim, degree, n_other, tn_order, res_mult):
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
    mgop = create_multi_group_operator(solution_domain, quad).convert_to_lower_block_triangular()
    return mgop


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=3)
    p.add_argument("--degree", type=int, default=1)
    p.add_argument("--n-other", type=int, default=3, help="transverse resolution per non-x dim")
    p.add_argument("--res-mult", type=int, default=1, help="multiplier on the base x-resolutions")
    p.add_argument("--tn-order", type=int, default=3)
    p.add_argument("--number", type=int, default=5, help="timed repetitions per op")
    p.add_argument("--skip-grad", action="store_true", help="skip the (slow) solve+grad section")
    args = p.parse_args()

    print("devices:", jax.devices())
    mgop = build_mgop(args.dim, args.degree, args.n_other, args.tn_order, args.res_mult)

    n_g, n_a = mgop.n_energy_groups, mgop.n_angles
    n_e, n_b, n_m = mgop.n_elements, mgop.n_basis, mgop.n_moments
    discrete_dofs = n_g * n_a * n_e * n_b
    print("\n=== problem ===")
    print(f"  dim {args.dim}  degree {args.degree}  groups {n_g}  angles {n_a}  "
          f"elements {n_e}  moments {n_m}  basis {n_b}")
    print(f"  discrete DOFs (g*a*e*b) = {discrete_dofs:,}   moment DOFs (g*m*e*b) = {n_g*n_m*n_e*n_b:,}")

    # device sanity: confirm operator arrays and an mv result live on the GPU
    x = jnp.ones(mgop.in_structure().shape)
    y = jax.block_until_ready(mgop.mv(x))
    print("  mv result device:", list(y.devices()))

    sgop = mgop.to_single_group_operator(0)
    top = mgop.transport_operator
    x_sg = jnp.ones(sgop.in_structure().shape)
    x_t = jnp.ones(top.in_structure().shape)

    print("\n=== sub-operators (forward / transpose / inverse) ===")
    bench("MGOp.mv",                 lambda op, v: op.mv(v),                 mgop,  x,    number=args.number)
    bench("MGOp.transpose().mv",     lambda op, v: op.transpose().mv(v),    mgop,  x,    number=args.number)
    bench("SingleGroupOp.mv",        lambda op, v: op.mv(v),                 sgop,  x_sg, number=args.number)
    bench("transport.mv",            lambda op, v: op.mv(v),                 top,   x_t,  number=args.number)
    bench("transport.inv",           lambda op, v: op.inv(v),                top,   x_t,  number=args.number)
    bench("transport.transpose().inv", lambda op, v: op.transpose().inv(v), top,   x_t,  number=args.number)
    bench("mass.mv",                 lambda op, v: op.mv(v),                 mgop.mass_operator,    x_t,  number=args.number)
    bench("moment.mv",               lambda op, v: op.mv(v),                 mgop.moment_operator,  x_t,  number=args.number)
    bench("discrete.mv",             lambda op, v: op.mv(v),                 mgop.discrete_operator, jnp.ones(mgop.discrete_operator.in_structure().shape), number=args.number)

    print("\n=== parametrised reconstruction (vertices + cross sections + sweep rebuild) ===")
    theta = MultigroupOperatorParametrisedVerticesXS.theta_base(mgop)
    bench("reconstruct_operator",
          lambda op, th: MultigroupOperatorParametrisedVerticesXS.reconstruct_operator(op, th),
          mgop, theta, number=args.number)

    if not args.skip_grad:
        print("\n=== lower-triangular solve + grad through reconstruction ===")
        solver = MultiGroupLowerTriangularBlockSolver(lx.BiCGStab(1e-10, 1e-10, max_steps=200))
        rhs = jnp.ones(mgop.in_structure().shape)

        def solve_sum(th, op, b):
            m = MultigroupOperatorParametrisedVerticesXS.reconstruct_operator(op, th)
            return jnp.sum(lx.linear_solve(m, b, solver).value)

        bench("solve (lower-tri)", solve_sum, theta, mgop, rhs, number=max(2, args.number // 2))
        vg = lambda th, op, b: jax.value_and_grad(solve_sum)(th, op, b)
        bench("value_and_grad(solve)", vg, theta, mgop, rhs, number=max(2, args.number // 2))

    print("\nDONE")


if __name__ == "__main__":
    main()
