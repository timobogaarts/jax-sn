# Performance check — data-ownership refactor (Stages B–C.2b)

*Run 2026-06-24 on this machine (NVIDIA RTX 3080, 10 GB; jaxsbgeom conda env, float64).*

## Question

Did the `operator = setup + spec + strategy` refactor — in particular **Stage C.2b**, which moved
`angle` out of `SweepData` and made the transport sweep kernels take it as an explicit argument —
regress transport-sweep performance?

Stages B / B.1 / C.1 / C.2 only change pytree *structure* (sub-operators became derived `@property`
views built outside `jit`); they don't alter any traced computation. **C.2b is the only change that
touches the compute kernels**, so the transport inverse-sweep benchmark (`examples/time_pallas.py`,
degree 0–3 tetrahedron, 60 000 elements) is the relevant target — it exercises exactly the kernels
that changed (`transport_inverse_sweep_complete_vmap` / `_batched`, their pallas + multi-angle
variants).

## Method

Apples-to-apples on the *same machine, same session* (to remove driver/thermal differences from the
numbers recorded in the script comments, which were taken at an earlier date):

- **Baseline**: pre-refactor commit `88bcff9`, run from a throwaway git worktree with
  `PYTHONPATH` pointed at its `src` (verified it loaded the old code — `BasicSweepData` still had an
  `angle` field).
- **Refactored**: current `HEAD` (`656752a`, Stage C.2b). Two full runs; the table shows their mean.

Each number is `timeit` mean of 5 calls after a warm-up call (`block_until_ready`).

## Results (ms)

| Case (deg, kernel) | Baseline `88bcff9` | Refactored (mean) | Δ |
|---|---:|---:|---:|
| 0  vmap                | 1870.7 | 1885.2 | +0.8% |
| 0  batched             | 1860.3 | 1888.3 | +1.5% |
| 0  vmap · pallas       |   45.3 |   47.2 | +4.2% |
| 0  batched · pallas    |   47.7 |   47.8 | +0.3% |
| 1  vmap                |  633.1 |  651.4 | +2.9% |
| 1  batched             |  633.6 |  657.1 | +3.7% |
| 1  vmap · pallas       |   80.8 |   81.3 | +0.7% |
| 1  batched · pallas    |   72.9 |   73.4 | +0.7% |
| 2  vmap                |  731.7 |  739.6 | +1.1% |
| 2  batched             |  711.9 |  731.0 | +2.7% |
| 2  vmap · pallas       |  186.6 |  186.6 |  0.0% |
| 2  batched · pallas    |  189.1 |  189.9 | +0.4% |
| 3  vmap                |  757.6 |  756.6 | −0.1% |
| 3  batched             |  790.0 |  765.6 | −3.1% |
| 3  vmap · pallas       |  505.6 |  495.2 | −2.1% |
| 3  batched · pallas    |  527.0 |  504.6 | −4.2% |
| 1  multi-angle batched | 1489.0 | 1521.5 | +2.2% |
| 1  multi-angle pallas  |  184.2 |  182.8 | −0.7% |
| 2  multi-angle batched | 4921.0 | 5028.6 | +2.2% |
| 2  multi-angle pallas  | 3796.1 | 3777.6 | −0.5% |
| 3  multi-angle batched | 4646.3 | 4398.0 | −5.3% |
| 3  multi-angle pallas  | 3220.7 | 3216.0 | −0.1% |

## Conclusion: no regression

- Every kernel is within **±5%** of baseline, with the differences **scattered around zero** (some
  faster, some slower) — there is no systematic slowdown. The spread matches the run-to-run jitter
  between the two refactored runs themselves (e.g. deg-0 vmap measured 1868 vs 1902 ms = 1.8% apart),
  which is expected GPU benchmark noise (clock boost / thermals / memory fragmentation), not an
  algorithmic effect.
- This is the expected outcome: C.2b passes `angle` as a separate kernel argument instead of reading
  it off `sweep_data.angle` — the *same* arrays reach the *same* inner functions, so the compiled
  computation is unchanged.

### Note on the degree-2 multi-angle number

The degree-2 multi-angle case is ~2.7× the value recorded in the old script comments (≈1883 ms) and
is even slower than degree 3 — but **the pre-refactor baseline reproduces it exactly** (4921 ms vs the
refactored 5029 ms). So it is a property of this machine's current state (likely memory pressure on
the 10 GB card and/or a different JAX/driver version than when the comments were written), **not** a
regression introduced by the refactor.

### Scope

`time_pallas.py` is the benchmark that targets the changed kernels and still runs against the current
API (after a one-line update of its `base_parameters` to pass `angle`). The other perf scripts
(`time_warp*.py`, `single_group_perf_mem_analysis.py`, the pareto/fw_cadis notebooks) rely on the
older `single_angle_transport_operator_cls=` argument to `create_multi_group_operator`, removed by the
earlier transport-operator strategy refactor (pre-dating this work), so they were not run here.
