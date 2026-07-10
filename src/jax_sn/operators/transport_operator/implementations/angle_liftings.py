
from ..transport_operator_interface import TransportOperator, ForwardAngleLifting, InversionAngleLifting
import equinox as eqx


def _map_over_angles(single_op: "TransportOperator", leading, fn):
    '''
    Lift a single-angle reduction over the angle axis by vmapping. The angle axis lives in both the
    ``angle`` and the sweep ordering, so we vmap over their leading axes together (and the leading axis
    of the input), rebuilding a single-angle operator per angle. The base operator therefore never
    needs to know how to vmap.
    '''
    def one_angle(angle_i, sweep_i, v_i):
        op_i = eqx.tree_at(lambda o: (o.angle, o.sweep_data), single_op, (angle_i, sweep_i))
        return fn(op_i, v_i)
    return eqx.filter_vmap(one_angle, in_axes=(0, 0, 0))(single_op.angle, single_op.sweep_data, leading)


class VmapForwardAngles(ForwardAngleLifting):
    '''Lift the forward apply by vmapping the single-angle ``mv`` over the angle axis.'''

    def forward(self, single_op: "TransportOperator", x):
        return _map_over_angles(single_op, x, lambda op, v: op.mv(v))


class VmapInversionAngles(InversionAngleLifting):
    '''Lift the inverse sweep by vmapping the single-angle ``_invert_internal`` over the angle axis.'''

    @classmethod
    def vmaps_inner(cls):
        return True   # lifts by vmapping the single-angle inverse

    def invert(self, single_op: "TransportOperator", b):
        return _map_over_angles(single_op, b, lambda op, v: op._invert_internal(v))
