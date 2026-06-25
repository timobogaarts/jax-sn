"""Mass operator: ``mass.mv`` swept over every discovered ``MassStrategy``."""
from _bench import Bench
from jax_sn.operators.mass_operator import MassStrategy

BENCHES = [
    Bench(name="mass.mv", base=MassStrategy, kwarg="mass_strategy", attr="mass_operator"),
]

if __name__ == "__main__":
    from run import run_module
    run_module(__name__)
