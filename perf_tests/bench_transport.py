"""
Single-angle transport operator. Two axes:

  - ``transport.forward`` : ``op.mv`` swept over every ``ForwardStrategy``.
  - ``transport.inv``     : ``op.inv`` (the inverse sweep) swept over every ``InversionStrategy``.

Discovery auto-derives the right toposort per inversion (``default_sweep_data_type``); pallas/warp
inversions appear only if importable and self-skip (via ``check_compatible``) on an incompatible mesh.
The ``overrides`` inject a representative batch size, since default-construction gives ``batch_size=None``
(== non-batched).
"""
from _bench import Bench, inv_call
from jax_sn.operators.transport_operator import (
    ForwardStrategy, InversionStrategy, BatchedForward, BatchedSweep,
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
    return mgop.transport_operator[0]


BENCHES = [
    Bench(name="transport.forward", base=ForwardStrategy, kwarg="forward_strategy",
          extract=single_angle, overrides=_FWD_OVERRIDES),
    Bench(name="transport.inv", base=InversionStrategy, kwarg="inversion_strategy",
          extract=single_angle, make_call=inv_call, overrides=_INV_OVERRIDES),
]

if __name__ == "__main__":
    from run import run_module
    run_module(__name__)
