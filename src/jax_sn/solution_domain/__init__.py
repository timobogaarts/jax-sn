from .base_fem import Element, PrecomputedGeometricData, FEMMatrices
from .solution_domain import SolutionDomain, SpatialData, compute_setup_for_solution_domain
from .basix_fem import BasixLagrangianSimplex

__all__ = [
    "Element", "PrecomputedGeometricData", "FEMMatrices",
    "SolutionDomain", "SpatialData", "compute_setup_for_solution_domain",
    "BasixLagrangianSimplex", 
    ]