"""
Single-angle transport operator. Two axes:

  - ``transport.forward`` : ``op.mv`` swept over every ``SingleAngleForwardStrategy``.
  - ``transport.inv``     : ``op.inv`` (the inverse sweep) swept over every ``SingleAngleInversionStrategy``.

Each discovered single-angle strategy is lifted to the multi-angle form the chain expects
(``_bench.lift_forward``/``lift_inversion``) before the operator is built, then indexed back down to a
single angle for the call. Discovery auto-derives the right toposort per inversion
(``default_sweep_data_type``); pallas/warp inversions appear only if importable and self-skip (via
``check_compatible``) on an incompatible mesh. The ``overrides`` inject a representative batch size, since
default-construction gives ``batch_size=None`` (== non-batched).
"""
from _bench import Bench, inv_call, lift_forward, lift_inversion, single_group_operators
from jax_sn.operators.transport_operator import (
    SingleAngleForwardStrategy, SingleAngleInversionStrategy, BatchedForward, BatchedSweep,
)

# Representative batched params (default-construct would benchmark the non-batched batch_size=None).
_FWD_OVERRIDES = {BatchedForward: [BatchedForward(1000)]}
_INV_OVERRIDES = {BatchedSweep: [BatchedSweep(1000)]}
try:
    from jax_sn.operators.transport_operator import BatchedPallasSweep
    _INV_OVERRIDES[BatchedPallasSweep] = [BatchedPallasSweep(1000)]
except ImportError:
    pass


def single_angle(mgop):
    return single_group_operators(mgop)[0][0]


BENCHES = [
    Bench(name="transport.forward", base=SingleAngleForwardStrategy, chain_field="transport_strategy",
          lift=lift_forward, extract=single_angle, overrides=_FWD_OVERRIDES),
    Bench(name="transport.inv", base=SingleAngleInversionStrategy, chain_field="transport_inversion_strategy",
          lift=lift_inversion, extract=single_angle, make_call=inv_call, overrides=_INV_OVERRIDES),
]

if __name__ == "__main__":
    from run import run_module
    run_module(__name__)
