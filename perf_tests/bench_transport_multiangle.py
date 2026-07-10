"""
Multi-angle transport inverse: ``transport_operator.inv`` over the (InversionStrategy x InversionAngleLifting)
product. This is the one axis that is genuinely a combination -- and the valid combinations are *not* the
full product: the warp liftings require a matching warp inversion. We don't encode that table; we
enumerate the product and let the operator's own construction-time ``check_compatible`` reject the bad
pairs (caught by ``measure`` as a self-documenting skip), so it stays correct as kernels are added.
"""
from dataclasses import dataclass, field

from _common import measure, LiveTable
from _bench import inv_call
from _discovery import discover_variants, serializes
from bench_transport import _INV_OVERRIDES
from jax_sn.operators.transport_operator import (
    InversionStrategy, InversionAngleLifting, BatchedForward,
)
from jax_sn.operators.multi_group_operator.multi_group_operator_creation import create_multi_group_operator


@dataclass
class MultiAngleInvBench:
    name: str = "transport.inv.multiangle"
    group: str = "cheap"
    inv_overrides: dict = field(default_factory=lambda: dict(_INV_OVERRIDES))

    def run(self, sd, qs):
        out = {}
        invs = [i for i in discover_variants(InversionStrategy, self.inv_overrides) if i.instance is not None]
        lifts = [l for l in discover_variants(InversionAngleLifting) if l.instance is not None]
        # Full product, minus the vmap-over-opaque-kernel combos that would silently serialise
        # (e.g. VmapInversionAngles x warp). Incompatible combos are *not* filtered here -- they self-skip via
        # the operator's own check_compatible, caught by ``measure``.
        pairs = [(i, l) for i in invs for l in lifts if not serializes(l.instance, i.instance)]
        table = LiveTable(self.name, [f"{i.label} + {l.label}" for i, l in pairs])
        for inv, lift in pairs:
            label = f"{inv.label} + {lift.label}"
            table.start(label)

            def build_fn(i=inv.instance, l=lift.instance):
                mgop = create_multi_group_operator(
                    sd, qs, forward_strategy=BatchedForward(1000),
                    inversion_strategy=i, inversion_lifting=l)
                return inv_call(mgop.transport_operator)

            m = measure(build_fn)
            table.ok(m.peak_mb, m.t_ms) if m.status == "ok" else table.skip(m.reason)
            out[label] = m.__dict__
        return out


BENCHES = [MultiAngleInvBench()]

if __name__ == "__main__":
    from run import run_module
    run_module(__name__)
