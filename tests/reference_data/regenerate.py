'''
Regenerate the frozen reference data.

The working tree is intentionally free of the reference *generator* (which depends
on an external neutron-transport solver). Instead, the generator lives at a pinned
historical commit; this script checks that commit out into a throwaway git worktree,
runs the generator there (with the external solver installed), and copies the
produced ``.npz`` bundles back into this directory.

Usage
-----
    python tests/reference_data/regenerate.py [--ref <git-ref>]

``--ref`` defaults to ``FREEZE_REF`` below. If the branch history is ever rewritten
(rebase/squash) so that commit becomes unreachable, pass a ref that still contains
``tests/reference_data/_generate.py`` (or update ``FREEZE_REF``).

Note: regeneration reproduces the reference for the problem setups *as they were at
that commit*. If a test's problem definition changes, the generator must be updated
at a new commit and ``FREEZE_REF`` pointed at it.

The directly-swept references reproduce byte-for-byte; the iteratively-solved ones
(breeding blanket, anisotropic) can differ at the solver-tolerance level (~1e-21),
which is far below the test tolerances, so those bundle files may show a tiny diff.
'''
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

# Commit that contains tests/reference_data/_generate.py and the solver wiring.
FREEZE_REF = 'af53db06fdedcfc4b34e0011882083f1b22d84c2'

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = subprocess.check_output(['git', '-C', HERE, 'rev-parse', '--show-toplevel']).decode().strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ref', default=FREEZE_REF, help='git ref containing _generate.py')
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix='jaxsn-refgen-')
    worktree = os.path.join(tmp, 'wt')
    try:
        subprocess.check_call(['git', '-C', REPO, 'worktree', 'add', '--detach', worktree, args.ref])

        generator = os.path.join(worktree, 'tests', 'reference_data', '_generate.py')
        if not os.path.exists(generator):
            raise SystemExit(
                f"ref {args.ref} has no tests/reference_data/_generate.py; "
                f"pass --ref pointing at a commit that does."
            )
        subprocess.check_call([sys.executable, generator])

        # Bundles that must NOT be pulled from the external-solver generator: `bb__d3` is
        # self-pinned to jax-sn's own output (the external solver meshes 3D differently), and
        # the `anisotropic__*` bundles were retired when that test folded into the BB fixture.
        SKIP = {'bb__d3.npz'}
        SKIP_PREFIXES = ('anisotropic__',)

        produced = os.path.join(worktree, 'tests', 'reference_data')
        n = 0
        for fname in os.listdir(produced):
            if not fname.endswith('.npz'):
                continue
            if fname in SKIP or fname.startswith(SKIP_PREFIXES):
                print(f"Skipping {fname} (self-pinned / retired; not overwritten from generator)")
                continue
            shutil.copy2(os.path.join(produced, fname), os.path.join(HERE, fname))
            n += 1
        print(f"Copied {n} regenerated .npz bundles into {HERE}")
    finally:
        subprocess.call(['git', '-C', REPO, 'worktree', 'remove', '--force', worktree])
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
