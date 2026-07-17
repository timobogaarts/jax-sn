"""
Whole-machinery aggregate matvecs -- ``MGOp.mv`` / ``MGOp.Tmv`` / ``SGOp.mv`` -- swept over the hand-picked
default composed strategies (see ``_bench.default_multi_group_strategies``: ``DefaultMultiGroupStrategy``'s
jax/pallas/warp recipes, batched and non), not an auto-discovered per-field sweep -- these exercise the
whole pipeline, so an internally-consistent backend combo (e.g. a Warp transport lift paired with
``WarpMass``, not a default ``LaxMass``) matters more than covering every strategy x strategy cross
product. These are the per-iteration reference costs a solve is implicitly built from, kept as their own
bench so the whole-operator numbers from the old suite are preserved.
"""
from _bench import ComposedBench, mv_call, ones_in, single_group_operator


def tmv_call(mgop):
    return (lambda o, x: o.transpose().mv(x), (mgop, ones_in(mgop)))


BENCHES = [
    ComposedBench(name="MGOp.mv", extract=lambda m: m),
    ComposedBench(name="MGOp.Tmv", extract=lambda m: m, make_call=tmv_call),
    ComposedBench(name="SGOp.mv", extract=single_group_operator),
]

if __name__ == "__main__":
    from run import run_module
    run_module(__name__)
