"""
Whole-machinery aggregate matvecs -- ``MGOp.mv`` / ``MGOp.Tmv`` / ``SGOp.mv`` -- swept over the
transport-affecting axis (``InversionStrategy``, the dominant cost of a full apply). These are the
per-iteration reference costs a solve is implicitly built from, kept as their own bench so the
whole-operator numbers from the old suite are preserved.
"""
from _bench import Bench, mv_call, ones_in
from bench_transport import _INV_OVERRIDES
from jax_sn.operators.transport_operator import InversionStrategy


def tmv_call(mgop):
    return (lambda o, x: o.transpose().mv(x), (mgop, ones_in(mgop)))


BENCHES = [
    Bench(name="MGOp.mv", base=InversionStrategy, kwarg="inversion_strategy",
          extract=lambda m: m, overrides=_INV_OVERRIDES),
    Bench(name="MGOp.Tmv", base=InversionStrategy, kwarg="inversion_strategy",
          extract=lambda m: m, make_call=tmv_call, overrides=_INV_OVERRIDES),
    Bench(name="SGOp.mv", base=InversionStrategy, kwarg="inversion_strategy",
          extract=lambda m: m.to_single_group_operator(0), overrides=_INV_OVERRIDES),
]

if __name__ == "__main__":
    from run import run_module
    run_module(__name__)
