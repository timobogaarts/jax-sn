"""Discrete (moment-to-discrete) operator: ``discrete.mv`` swept over every ``DiscreteStrategy``."""
from _bench import Bench
from jax_sn.operators.moment_operator import DiscreteStrategy

BENCHES = [
    Bench(name="discrete.mv", base=DiscreteStrategy, kwarg="discrete_strategy", attr="discrete_operator"),
]

if __name__ == "__main__":
    from run import run_module
    run_module(__name__)
