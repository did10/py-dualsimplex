"""Shared fixtures for the dualsimplex_py test suite."""

import subprocess

import numpy as np
import pandas as pd
import pytest

from dualsimplex_py import find_rscript


@pytest.fixture
def synthetic_X():
    """Small samples x genes DataFrame with one all-zero row and column.

    Deterministic (seeded RNG) so exact kept shapes are stable. Unit tests
    mock the R call; integration tests run the real solver on this matrix.
    """
    rng = np.random.default_rng(0)
    n_samples, n_genes = 12, 20
    data = rng.integers(0, 100, size=(n_samples, n_genes)).astype(float)
    data[data < 30] = 0.0  # non-negative, sparse-ish
    data[3, :] = 0.0       # all-zero sample row
    data[:, 5] = 0.0       # all-zero gene column
    return pd.DataFrame(
        data,
        index=[f"sample_{i}" for i in range(n_samples)],
        columns=[f"gene_{j}" for j in range(n_genes)],
    )


@pytest.fixture(scope="session")
def r_dualsimplex():
    """Session-scoped probe that R + the DualSimplex R package are usable.

    Skips integration tests when they are not.
    """
    rscript = find_rscript()
    if rscript is None:
        pytest.skip("Rscript not found (set DUALSIMPLEX_RSCRIPT or add R to PATH)")
    probe = (
        'suppressPackageStartupMessages(library(DualSimplex)); '
        'cat("DUALSIMPLEX_R_OK\\n")'
    )
    try:
        proc = subprocess.run(
            [rscript, "--vanilla", "-e", probe],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"DualSimplex R package not usable: {exc}")
    if proc.returncode != 0 or "DUALSIMPLEX_R_OK" not in proc.stdout:
        pytest.skip("DualSimplex R package not installed/usable in R")
    return rscript
