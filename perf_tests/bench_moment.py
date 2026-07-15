"""Moment operator: ``moment.mv`` swept over every discovered ``FluxMomentStrategy``."""
from _bench import Bench
from jax_sn.operators.moment_operator import FluxMomentStrategy

BENCHES = [
    Bench(name="moment.mv", base=FluxMomentStrategy, kwarg="moment_strategy", attr="moment_operator"),
]

if __name__ == "__main__":
    from run import run_module
    run_module(__name__)
