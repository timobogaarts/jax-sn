'''
Stored reference data for tests that need an external-solver ground truth.

The tests load the frozen ``.npz`` bundles from this directory via :func:`load`;
the working tree contains no reference generator (it depends on an external
neutron-transport solver). To regenerate, ``regenerate.py`` checks out the pinned
historical commit that still holds the generator, runs it, and copies the bundles
back here::

    python tests/reference_data/regenerate.py

Each bundle is a dict of named arrays keyed by a canonical string (see the
``key_*`` helpers). Bundles are committed (the global ``*.npz`` gitignore is
overridden for this directory) so the suite runs with no external dependency.

Provenance note: most bundles are external-solver ground truth. The self-pinned exceptions
-- frozen to jax-sn's own current output, i.e. regression pins rather than independent-oracle
accuracy tests -- are ``bb__d3`` (the external solver tetrahedralises the 3D slab differently,
so a like-for-like comparison isn't possible), ``transport_sweep__*`` (a localized single-angle
sweep guard; external accuracy of the sweep is anchored by ``bb__*``), ``moment__*`` (the
first-collision moment vector on the BB problem, guarding the higher-l projection that ``bb__*``'s
scalar-flux oracle can't see), ``fem__*`` (frozen from the retired in-tree FEM oracle) and
``fwcadis__*`` (a round-trip regression pin on the FW-CADIS forward + continuous-adjoint pipeline).
These are excluded from ``regenerate.py`` so re-running it never overwrites them.
'''
import os
import numpy as onp

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def _path(name: str) -> str:
    return os.path.join(DATA_DIR, name + '.npz')


def save(name: str, **arrays) -> None:
    '''Save a bundle of named arrays under ``name`` (called only by the generator).'''
    onp.savez(_path(name), **{k: onp.asarray(v) for k, v in arrays.items()})


def load(name: str) -> dict:
    '''Load a stored bundle as a dict of numpy arrays.

    Raises a clear error (pointing at the generator) if the bundle is missing,
    which happens when a test requests a parameter combination that has not been
    generated yet.
    '''
    path = _path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing reference data bundle '{name}' ({path}). "
            f"Regenerate with:  python -m tests.reference_data._generate"
        )
    with onp.load(path) as data:
        return {k: data[k] for k in data.files}


# --- canonical keys ---------------------------------------------------------

def key_lm(tn_set: int, l_max: int) -> str:
    return f'lm__tn{tn_set}_lmax{l_max}'


def key_transport(dim: int, degree: int) -> str:
    '''Mesh + source + cross sections for the moment test and test_local_real_inversion.'''
    return f'transport__d{dim}_deg{degree}'


def key_transport_sweep(dim: int) -> str:
    '''Single-angle vmap sweep on the BB domain, self-pinned to the current implementation
    (regression guard, not external truth -- see the provenance note above).'''
    return f'transport_sweep__d{dim}'


def key_moment(dim: int) -> str:
    '''Full spherical-harmonic moment vector of the first-collision flux on the BB problem,
    self-pinned to the current implementation (regression guard on the higher-l moment projection;
    BB's own oracle key_bb only checks the scalar l=0 flux -- see the provenance note above).'''
    return f'moment__d{dim}'


def key_bb(dim: int) -> str:
    return f'bb__d{dim}'


def key_fw_cadis(dim: int, degree: int) -> str:
    '''FW-CADIS importance map (scalar-flux moment) interpolated along the BB domain diagonal,
    self-pinned to jax-sn's own current output -- a round-trip regression guard on the forward +
    continuous-adjoint solve pipeline, not external ground truth (see the provenance note above).'''
    return f'fwcadis__d{dim}_deg{degree}'


def key_fem(element: str, degree: int) -> str:
    '''FEM volume (aij/ii/bij), face-outflow (cij) and face-inflow (dij) matrices for a simplex element.

    aij/ii/bij/cij are frozen from the retired in-tree reference implementation (old_generate_basis.py),
    an independent oracle for the current basix-based matrices. dij (the per-permutation face-connection
    matrices) was never emitted by that oracle, so it is self-pinned to the current implementation
    (regression guard); it is not external/independent ground truth.
    '''
    return f'fem__{element}_deg{degree}'
