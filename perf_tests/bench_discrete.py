"""Discrete (moment-to-discrete) operator: ``discrete.mv`` swept over every ``FluxMomentStrategy``.

Shares its strategy base with ``moment.mv``: direction comes from the data, not the strategy."""
from _bench import Bench
from jax_sn.operators.moment_operator import FluxMomentStrategy

BENCHES = [
    Bench(name="discrete.mv", base=FluxMomentStrategy, kwarg="discrete_strategy", attr="discrete_operator"),
]

if __name__ == "__main__":
    from run import run_module
    run_module(__name__)
