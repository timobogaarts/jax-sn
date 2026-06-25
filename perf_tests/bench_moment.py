"""Moment operator: ``moment.mv`` swept over every discovered ``MomentStrategy``."""
from _bench import Bench
from jax_sn.operators.moment_operator import MomentStrategy

BENCHES = [
    Bench(name="moment.mv", base=MomentStrategy, kwarg="moment_strategy", attr="moment_operator"),
]

if __name__ == "__main__":
    from run import run_module
    run_module(__name__)
