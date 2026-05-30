"""Shared pytest fixtures and configuration.

Test categories (see project plan §8):
- unit: solver correctness, math identities
- recovery: parameter recovery from simulated data
- sbc: simulation-based calibration (nightly only, marked @pytest.mark.sbc)
- identifiability: prior-vs-posterior checks
- predictive: prior / posterior predictive checks
"""

import pytest


@pytest.fixture(scope="session")
def rng_seed() -> int:
    """Single canonical seed for reproducible randomness across tests."""
    return 20260530
