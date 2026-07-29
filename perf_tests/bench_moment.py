"""Moment operator: ``moment.mv`` swept over every discovered ``FluxMomentStrategy``."""
from _bench import Bench, single_group_operators
from jax_sn.operators.moment_operator import FluxMomentStrategy

BENCHES = [
    Bench(name="moment.mv", base=FluxMomentStrategy, chain_field="moment_strategy",
          extract=lambda mgop: single_group_operators(mgop)[1]),
]

if __name__ == "__main__":
    from run import run_module
    run_module(__name__)
