"""
Shared utilities for the modular perf-test suite (``perf_tests/``): the measurement primitives (peak
memory / wall time, OOM-safe), golden dump/compare, and a thin record type.

This folder is **not** part of the installed package -- it is a set of standalone scripts (run from
inside ``perf_tests/`` so the ``_*`` helpers and ``bench_*`` modules import each other), each measuring
one operator axis. It is also self-contained: the problem builder + cross-section data live in
``_create_problem.py``, the autodiscovery engine in ``_discovery.py``, the orchestrator in ``run.py``.

Measurement primitives (self-contained, no dependency on internal memory-analysis signatures):
  - ``peak_mb(f, *args)``  -- predicted peak device memory (MB) from the *compiler* (no execution, so it
    never OOMs and works even for configs too large to actually run).
  - ``time_ms(f, *args)``  -- steady-state wall time (min-of-N) after a warm-up; ``None`` on out-of-memory.
  - ``compile_ms(f, *args)`` -- one-shot lower+compile time.
"""
import os
import time
import json
from dataclasses import dataclass, asdict
from typing import Optional, Union

import jax
import equinox as eqx

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")


# --------------------------------------------------------------------------------------------------
# Measurement primitives
# --------------------------------------------------------------------------------------------------
def _is_oom(exc: Exception) -> bool:
    """True for "didn't fit / couldn't run" rather than a bug."""
    s = str(exc).lower()
    return ("resource_exhausted" in s or "out of memory" in s or "exhausted" in s
            or "out_of_memory" in s)


def safe_run(fn, *args):
    """Run + block on ``fn(*args)``. Returns (result, None) or (None, 'OOM'); re-raises non-OOM errors."""
    try:
        return jax.block_until_ready(fn(*args)), None
    except Exception as e:
        if _is_oom(e):
            return None, "OOM"
        raise


def peak_mb(f, *args):
    """Predicted peak device memory (MB) of ``eqx.filter_jit(f)`` compiled on ``args`` (no execution,
    so it is near-deterministic and cannot OOM). Returns a string marker on failure."""
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


def time_ms(f, *args, target_total_s=0.5, min_repeats=2, max_repeats=8,
            single_run_s=2.0, warmup_only_s=8.0, min_block_s=0.02):
    """Adaptive steady-state wall time (ms). Returns ``None`` on OOM.

    A warm-up (compile + first run) is always needed, so it is timed and reused. The number of measured
    repeats is then sized from a single probe instead of fixed, so we don't re-run a slow op pointlessly
    nor under-sample a fast one:

    - **warm-up alone is huge** (>= ``warmup_only_s``): report the warm-up time itself -- at this scale
      the compile is a negligible fraction and a second run would just add seconds (1 execution total).
    - **one run is long** (probe >= ``single_run_s``): the probe *is* the measurement (warm-up + probe,
      2 executions) -- min-of-N buys nothing on a multi-second, low-variance op.
    - **otherwise**: batch fast calls into ~``min_block_s`` blocks and take min-of-N over enough repeats
      to fill ~``target_total_s`` of timed work, clamped to ``[min_repeats, max_repeats]``.
    """
    fj = eqx.filter_jit(f)
    try:
        t0 = time.perf_counter()
        jax.block_until_ready(fj(*args))                 # warm up: compile + first execution
        t_warm = time.perf_counter() - t0
        if t_warm >= warmup_only_s:
            return t_warm * 1000.0                       # re-running isn't worth the seconds

        t0 = time.perf_counter()
        jax.block_until_ready(fj(*args))                 # one clean post-warm-up probe
        t_call = time.perf_counter() - t0
        if t_call >= single_run_s:
            return t_call * 1000.0                        # long + low-variance: one timed run is plenty

        number = max(1, int(min_block_s / t_call))       # batch fast ops so a block is reliably timeable
        repeats = min(max_repeats, max(min_repeats, int(target_total_s / (t_call * number))))

        best = t_call                                    # the probe is the first steady-state sample
        for _ in range(repeats - 1):
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


# --------------------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------------------
@dataclass
class Measurement:
    """One measured (peak memory, wall time) point, or a skip with a reason."""
    status: str                         # "ok" | "skip"
    peak_mb: Optional[Union[float, str]] = None
    t_ms: Optional[float] = None
    reason: str = ""                    # populated when status == "skip"

    @classmethod
    def skip(cls, reason: str) -> "Measurement":
        return cls(status="skip", reason=reason)


def short_exc(exc: Exception) -> str:
    s = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return s[:100]


def measure(build_fn) -> Measurement:
    """``build_fn() -> (callable, args_tuple)``; returns peak memory + adaptive wall time.

    Build/compile failures classify the point as a skip (with the package's own error text), so an
    unavailable backend or an incompatible strategy never aborts the sweep -- it self-documents.
    """
    try:
        f, args = build_fn()
    except Exception as e:  # construction / compatibility failure -> self-classifying skip
        return Measurement.skip(short_exc(e))
    return Measurement(status="ok", peak_mb=peak_mb(f, *args), t_ms=time_ms(f, *args))


# --------------------------------------------------------------------------------------------------
# Live aligned output: print each row's label first (so the running variant is visible), then complete
# the same line in-place with the aligned result once measured -- no separate end-of-bench table.
# --------------------------------------------------------------------------------------------------
def _cell_peak(peak) -> str:
    return f"{peak:8.1f} M" if isinstance(peak, (int, float)) else f"{str(peak):>8} M"


def _cell_time(t) -> str:
    return f"{t:8.2f} ms" if isinstance(t, (int, float)) else f"{'OOM':>8} ms"


class LiveTable:
    """A bench's result block, printed live. Construct with every row label up front (so the label
    column is aligned), then per row: ``start(label)`` before measuring, ``ok``/``skip`` after."""

    def __init__(self, name: str, labels, extra_header: str = ""):
        self.w = max(24, max((len(s) for s in labels), default=24))
        print(f"\n  {name}   (peak MB / time ms{extra_header})", flush=True)

    def start(self, label: str):
        print(f"    {label:<{self.w}}   ", end="", flush=True)

    def ok(self, peak, t, suffix: str = ""):
        print(f"{_cell_peak(peak)}  {_cell_time(t)}{suffix}", flush=True)

    def skip(self, reason: str):
        print(f"skip: {reason}", flush=True)


# --------------------------------------------------------------------------------------------------
# Golden I/O  (one file per machine, holding every bench it was recorded with)
# --------------------------------------------------------------------------------------------------
def golden_path(name: str) -> str:
    return os.path.join(GOLDEN_DIR, name + ".json")


def load_golden(name: str) -> dict:
    with open(golden_path(name)) as fh:
        return json.load(fh)


def save_golden(name: str, results: dict) -> str:
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    path = golden_path(name)
    existing = {}
    if os.path.exists(path):
        with open(path) as fh:
            existing = json.load(fh)
    existing.update(results)                      # merge: re-running one bench updates only its block
    with open(path, "w") as fh:
        json.dump(existing, fh, indent=2, default=str)
    return path


def compare_golden(results: dict, golden: dict, mem_tol: float = 0.05, time_tol: float = 0.25):
    """Return (fails, warns). Peak-memory drift is a fail; time drift is advisory."""
    fails, warns = [], []
    for bench, block in results.items():
        gblock = golden.get(bench, {})
        for label, m in block.items():
            if m.get("status") != "ok":
                continue
            gm = gblock.get(label)
            if not gm or gm.get("status") != "ok":
                continue
            cur_m, ref_m = m.get("peak_mb"), gm.get("peak_mb")
            if isinstance(cur_m, (int, float)) and isinstance(ref_m, (int, float)) and ref_m > 0:
                d = (cur_m - ref_m) / ref_m
                if abs(d) > mem_tol:
                    fails.append(f"{bench}/{label} peak_mb {ref_m:.1f}->{cur_m:.1f} ({d:+.0%})")
            cur_t, ref_t = m.get("t_ms"), gm.get("t_ms")
            if isinstance(cur_t, (int, float)) and isinstance(ref_t, (int, float)) and ref_t > 0:
                d = (cur_t - ref_t) / ref_t
                if abs(d) > time_tol:
                    warns.append(f"{bench}/{label} t_ms {ref_t:.1f}->{cur_t:.1f} ({d:+.0%})")
    return fails, warns


def measurement_to_dict(m: Measurement) -> dict:
    return asdict(m)
