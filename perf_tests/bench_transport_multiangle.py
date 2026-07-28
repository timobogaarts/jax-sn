"""
Multi-angle transport inverse: ``transport_operator.inv`` swept over the hand-picked default composed
strategies (see ``_bench.default_multi_group_strategies``: ``DefaultMultiGroupStrategy``'s jax/pallas/warp
recipes, batched and non). Each recipe pairs its own internally-consistent transport lift (and mass
strategy), rather than combining a per-strategy ``SingleAngleInversionStrategy`` discovery with an ad hoc
lift + default -- the same reasoning as ``bench_machinery.py``. To sweep an off-default pairing explicitly
(e.g. ``ScanAnglesInversion`` serialisation), use ``dev_bench.py --lifting``.
"""
from _bench import ComposedBench, inv_call, single_group_operators


def multi_angle(mgop):
    return single_group_operators(mgop)[0]


BENCHES = [ComposedBench(name="transport.inv.multiangle", extract=multi_angle, make_call=inv_call)]

if __name__ == "__main__":
    from run import run_module
    run_module(__name__)
