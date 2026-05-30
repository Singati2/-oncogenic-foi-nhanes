"""Unit tests for the catalytic ODE solver.

Reference test: for constant force of infection λ, the catalytic model
    dS/da = -λ · S(a),  S(0) = 1
has the analytical solution  S(a) = exp(-λ · a),  so seroprevalence  P(a) = 1 - exp(-λ · a).

Any numerical solver we ship must recover this to high precision.
"""

import math

import pytest


@pytest.mark.skip(reason="solver not yet implemented — see analysis/model/")
def test_constant_foi_matches_analytical() -> None:
    """Numerical solver must match 1 - exp(-λa) for constant λ."""
    lambda_ = 0.05
    ages = [10.0, 20.0, 40.0, 60.0]

    # from analysis.model.catalytic import solve_constant_foi
    # numerical = solve_constant_foi(lambda_=lambda_, ages=ages)
    # for a, p in zip(ages, numerical):
    #     assert math.isclose(p, 1 - math.exp(-lambda_ * a), abs_tol=1e-6)

    raise NotImplementedError
