"""Integration tests: require R with the DualSimplex R package installed.

Skipped automatically when R/DualSimplex are not available (see conftest).
Use a fast custom optimization schedule so the suite runs in seconds.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.integration

from dualsimplex_py import DualSimplex

N_ITER = 400  # fast custom-optimization schedule for tests


def _model(**overrides):
    params = dict(
        n_components=3,
        random_state=123,
        optimization="custom",
        n_iterations=N_ITER,
        max_sinkhorn_iterations=100,
    )
    params.update(overrides)
    return DualSimplex(**params)


def test_fit_shapes_and_nonnegativity(synthetic_X, r_dualsimplex):
    model = _model()
    model.fit(synthetic_X)
    assert model.W_.shape == (20, 3)
    assert model.H_.shape == (3, 11)
    assert model.components_.shape == (3, 20)
    assert model.n_features_in_ == 20
    assert model.kept_samples_ == [f"sample_{i}" for i in range(12) if i != 3]
    assert (model.W_.values >= 0).all()
    assert (model.H_.values >= 0).all()


def test_fit_reproducible(synthetic_X, r_dualsimplex):
    def run():
        model = _model()
        model.fit(synthetic_X)
        return model.W_.values, model.H_.values

    W1, H1 = run()
    W2, H2 = run()
    np.testing.assert_allclose(W1, W2, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(H1, H2, rtol=1e-6, atol=1e-8)


def test_fit_transform(synthetic_X, r_dualsimplex):
    model = _model()
    props = model.fit_transform(synthetic_X)
    assert props.shape == (12, 3)
    assert np.isnan(props[3]).all()
    kept = [i for i in range(12) if i != 3]
    np.testing.assert_allclose(props[kept], model.H_.T.values, rtol=1e-6, atol=1e-8)


def test_transform_nnls(synthetic_X, r_dualsimplex):
    model = _model()
    model.fit(synthetic_X)
    props = model.transform(synthetic_X, method="nnls")
    assert props.shape == (12, 3)
    kept = [i for i in range(12) if i != 3]
    assert np.allclose(props[kept].sum(axis=1), 1.0, atol=1e-4)
    # the all-zero sample has nothing to project -> all-zero proportions
    assert np.allclose(props[3], 0.0)


def test_transform_dualsimplex_method(synthetic_X, r_dualsimplex):
    model = _model()
    model.fit(synthetic_X)
    props = model.transform(synthetic_X, method="dualsimplex")
    assert props.shape == (12, 3)
    kept = [i for i in range(12) if i != 3]
    assert np.isfinite(props[kept]).all()


def test_save_state_path(synthetic_X, r_dualsimplex, tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    model = _model(save_state_path=str(state_dir))
    model.fit(synthetic_X)
    assert state_dir.exists()
    assert any(state_dir.iterdir())


def test_env_var_rscript(synthetic_X, monkeypatch, r_dualsimplex):
    # NOTE: K=2 is avoided here because the upstream R package's
    # ``random_invertible`` initializer calls ``diag(scale_factors)`` on a
    # length-1 vector for K=2, which R treats as a scalar -> wrong-size
    # identity -> "non-conformable arguments" (K>=3 is fine).
    monkeypatch.setenv("DUALSIMPLEX_RSCRIPT", r_dualsimplex)
    model = _model(n_iterations=200)  # K=3
    model.fit(synthetic_X)
    assert model.W_.shape == (20, 3)
