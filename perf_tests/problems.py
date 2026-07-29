"""
Named problem sizes for the perf-test suite. ``build(name)`` returns ``(solution_domain, quadrature_set,
info)``. Sizes mirror the two regimes of the old suite: a high element count for the single-application
operators, a smaller one (fewer angles) for the iterative solves.
"""
from _create_problem import build_problem

# dim / degree / n_other (element count) / res_mult / tn_order (2->32 angles, 3->72)
PROBLEMS = {
    "mv-d1-60k":  dict(dim=3, degree=1, n_other=20, res_mult=1, tn_order=3),
    "mv-d2-60k":  dict(dim=3, degree=2, n_other=20, res_mult=1, tn_order=3),
    "mv-d3-60k":  dict(dim=3, degree=3, n_other=20, res_mult=1, tn_order=3),
    "solve-d1-10k": dict(dim=3, degree=1, n_other=8, res_mult=1, tn_order=2),
    "tiny-d1":    dict(dim=3, degree=1, n_other=2, res_mult=1, tn_order=2),  # smoke tests
}

# Default problem per bench group.
DEFAULT_MV = "mv-d1-60k"
DEFAULT_SOLVE = "solve-d1-10k"


def build(name: str):
    params = PROBLEMS[name]
    sd, qs = build_problem(**params)
    info = (f"{name}: dim {params['dim']} deg {params['degree']} "
            f"elems {sd.domain.n_elements} angles {qs.angles.shape[0]} "
            f"groups {sd.domain.cross_sections_indexed.n_energy_groups}")
    return sd, qs, info
