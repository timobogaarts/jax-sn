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
    parser.addoption(
        "--reference-mode",
        action="store",
        default=None,
        choices=["live", "frozen", "freeze"],
        help="Source for operator-property reference values (see operator_util reference-values block): "
             "frozen=frozen bundle only, error if missing (default); live=compute from the pivot; "
             "freeze=compute and (re)write the bundles. Sets JAXSN_REFERENCE.",
    )

def pytest_configure(config):
    platform = config.getoption("--jax-platform")
    if platform:
        os.environ["JAX_PLATFORM_NAME"] = platform

    reference_mode = config.getoption("--reference-mode")
    if reference_mode:
        os.environ["JAXSN_REFERENCE"] = reference_mode

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
    



def pytest_runtest_setup(item):
    # Check if the test has the "slow" marker
    if "slow" in item.keywords and not item.config.getoption("--runslow"):
        pytest.skip("Skipping slow test, use --runslow to run")
    if "pallas" in item.keywords and not item.config.getoption("--runpallas"):
        pytest.skip("Skipping Pallas test, use --runpallas to run")

# ---------------------------------------------------------------------------------
# Breeding Blanket base setup (re-used to limit the number of setup compilations)
# ---------------------------------------------------------------------------------

import pytest

from bb_setup import _setup_bb_args, _get_bb_args
DIM_DEGREES = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (3, 1), (3, 2)]

@pytest.fixture(params=DIM_DEGREES, scope="session")
def bb_mgop_dim_degree(request):
    return request.param

from functools import lru_cache


@lru_cache(maxsize=None)
def _build_bb_mgop(dim, degree):
    '''Session-wide memoised BB build, keyed by (dim, degree).

    Cached using lru to avoid pytest limitations (will not be too many dim,degree pairs anyway)
    '''
    args = _get_bb_args(dim, degree)    
    return (*_setup_bb_args(*args), args)

@pytest.fixture(scope='session')
def bb_mgop(bb_mgop_dim_degree):
    '''(MultiGroupOperatorScan, source_total, solution_domain, bb_args); built once per (dim, degree).'''
    return _build_bb_mgop(*bb_mgop_dim_degree)

@pytest.fixture(scope='session')
def bb_sgop(bb_mgop):
    '''(SingleGroupOperator[group 0], source_total, solution_domain, bb_args), sharing bb_mgop.'''
    mgop, source_total, solution_domain, args = bb_mgop
    return mgop.to_single_group_operator(0), source_total, solution_domain, args
