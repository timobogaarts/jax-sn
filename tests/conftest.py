# tests/conftest.py
import os
import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--jax-platform",
        action="store",
        default=None,
        help="Set JAX platform (cpu/gpu/tpu) before import"
    )

    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )    
    parser.addoption(
        "--runpallas",
        action="store_true",
        default=False,
        help="Run Pallas tests"
    )

def pytest_configure(config):
    platform = config.getoption("--jax-platform")
    if platform:
        os.environ["JAX_PLATFORM_NAME"] = platform

    # Configure JAX via environment variables *before* any test module imports jax.
    # pytest_configure runs after option parsing but before collection imports the
    # test modules, so this is the safe place to set import-time JAX config
    # (same pattern as --jax-platform above).
    #
    # 1. x64 globally: nearly every test file already calls
    #    jax.config.update("jax_enable_x64", True); setting it here covers the few
    #    that don't (e.g. test_transport_operator.py) and keeps the compilation
    #    cache keyed on a single, consistent dtype configuration.
    # 2. Persistent compilation cache: XLA compilation dominates test runtime, and
    #    without a persistent cache every `pytest` run recompiles from cold. This
    #    caches compiled executables across runs under <repo>/.jax_cache.
    os.environ.setdefault("JAX_ENABLE_X64", "True")

    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, ".jax_cache"))
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", cache_dir)
    # Cache small kernels too: the transport/operator tests emit many sub-0.5s
    # vmapped kernels that aggregate to most of the runtime (caching them takes a
    # warm transport-property run from ~34s to ~17s). A 10ms floor keeps those
    # real kernels while skipping trivial primitives (broadcast/convert/reshape,
    # ~1ms to compile) that would otherwise flood the cache from eager code paths
    # (operator setup, un-jitted interpolate) without any warm-run benefit.
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0.01")
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")



def pytest_runtest_setup(item):
    # Check if the test has the "slow" marker
    if "slow" in item.keywords and not item.config.getoption("--runslow"):
        pytest.skip("Skipping slow test, use --runslow to run")
    if "pallas" in item.keywords and not item.config.getoption("--runpallas"):
        pytest.skip("Skipping Pallas test, use --runpallas to run")