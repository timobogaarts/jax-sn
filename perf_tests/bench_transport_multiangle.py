"""
Multi-angle transport inverse: ``transport_operator.inv`` swept over every ``SingleAngleInversionStrategy``, each
using its *natural* multi-angle angle inversion. ``create_multi_group_operator`` auto-resolves that
(``VmapAngles`` over a vmappable kernel, the fused native warp lift otherwise), so there is no
(strategy x lifting) product to enumerate and prune any more -- the one footgun (a vmap over an opaque
kernel) is now unconstructible rather than flagged. To benchmark a non-default pairing (e.g.
``ScanAngles`` serialisation) explicitly, use ``dev_bench.py --lifting``.
"""
from dataclasses import dataclass, field

from _common import measure, LiveTable
from _bench import inv_call
from _discovery import discover_variants
from bench_transport import _INV_OVERRIDES
from jax_sn.operators.transport_operator import SingleAngleInversionStrategy, BatchedForward
from jax_sn.operators.multi_group_operator.multi_group_operator_creation import create_multi_group_operator


@dataclass
class MultiAngleInvBench:
    name: str = "transport.inv.multiangle"
    group: str = "cheap"
    inv_overrides: dict = field(default_factory=lambda: dict(_INV_OVERRIDES))

    def run(self, sd, qs):
        out = {}
        invs = [i for i in discover_variants(SingleAngleInversionStrategy, self.inv_overrides) if i.instance is not None]
        table = LiveTable(self.name, [i.label for i in invs])
        for inv in invs:
            table.start(inv.label)

            def build_fn(i=inv.instance):
                # multi_angle_inversion_strategy left to auto-resolution: each strategy's natural multi-angle lift.
                mgop = create_multi_group_operator(sd, qs, forward_strategy=BatchedForward(1000), inversion_strategy=i)
                return inv_call(mgop.transport_operator)

            m = measure(build_fn)
            table.ok(m.peak_mb, m.t_ms) if m.status == "ok" else table.skip(m.reason)
            out[inv.label] = m.__dict__
        return out


BENCHES = [MultiAngleInvBench()]

if __name__ == "__main__":
    from run import run_module
    run_module(__name__)
