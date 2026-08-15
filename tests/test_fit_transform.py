"""Tests for fit / fit_transform / transform / alignment / NNLS."""

import numpy as np
import pandas as pd
import pytest

from dualsimplex_py import DualSimplex


def _fake_solve(X):
    """Deterministic stand-in for the R solve: 2 components, 3 kept genes."""
    W = pd.DataFrame(
        [[1.0, 0.0], [0.5, 2.0], [0.0, 3.0]],
        index=[f"gene_{j}" for j in (0, 2, 4)],
        columns=["c0", "c1"],
    )
    samples = [f"sample_{i}" for i in range(12) if i != 3]
    H = pd.DataFrame(
        np.array(
            [
                [0.7, 0.3, 0.6, 0.4, 0.5, 0.5, 0.8, 0.2, 0.9, 0.1, 0.65],
                [0.3, 0.7, 0.4, 0.6, 0.5, 0.5, 0.2, 0.8, 0.1, 0.9, 0.35],
            ]
        ),
        index=["c0", "c1"],
        columns=samples,
    )
    return {
        "W": W,
        "H": H,
        "kept_features": list(W.index),
        "kept_sample_idx": np.array([0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]),
        "out_dir": "fake",
    }


@pytest.fixture
def model(monkeypatch, synthetic_X):
    m = DualSimplex(n_components=2, random_state=1)
    monkeypatch.setattr(m, "_solve", lambda X: _fake_solve(X))
    return m


def test_fit_attributes(model, synthetic_X):
    model.fit(synthetic_X)
    assert model.W_.shape == (20, 2)          # full feature space
    assert model.H_.shape == (2, 11)
    assert model.components_.shape == (2, 20)
    assert model.n_features_in_ == 20
    assert model.feature_names_in_ == [f"gene_{j}" for j in range(20)]
    assert model.kept_features_ == ["gene_0", "gene_2", "gene_4"]
    assert model.kept_samples_ == [f"sample_{i}" for i in range(12) if i != 3]
    assert model.n_samples_in_ == 12


def test_fit_zero_rows_for_dropped_genes(model, synthetic_X):
    model.fit(synthetic_X)
    assert (model.W_.loc["gene_5"].values == 0).all()   # all-zero gene
    assert (model.W_.loc["gene_1"].values == 0).all()   # dropped by fake solve
    assert model.W_.loc["gene_0"].tolist() == [1.0, 0.0]


def test_fit_transform_matches_H_transpose(model, synthetic_X):
    props = model.fit_transform(synthetic_X)
    assert props.shape == (12, 2)
    assert np.isnan(props[3]).all()           # dropped sample -> NaN
    kept = [i for i in range(12) if i != 3]
    np.testing.assert_allclose(props[kept], model.H_.T.values)


def test_transform_requires_fit(synthetic_X):
    with pytest.raises(ValueError, match="not fitted"):
        DualSimplex().transform(synthetic_X)


def test_transform_bad_method(model, synthetic_X):
    model.fit(synthetic_X)
    with pytest.raises(ValueError, match="Unknown method"):
        model.transform(synthetic_X, method="bogus")


def test_transform_feature_count_mismatch(model, synthetic_X):
    model.fit(synthetic_X)
    with pytest.raises(ValueError, match="features"):
        model.transform(synthetic_X.drop(columns=["gene_0"]), method="nnls")


def test_transform_missing_columns(model, synthetic_X):
    model.fit(synthetic_X)
    X_bad = synthetic_X.rename(columns={"gene_0": "other"})  # same count, wrong names
    with pytest.raises(ValueError, match="missing"):
        model.transform(X_bad, method="nnls")


def test_transform_nnls_recovery():
    rng = np.random.default_rng(3)
    W = np.abs(rng.normal(size=(50, 3)))
    W = W / W.sum(axis=0, keepdims=True)
    H_true = rng.dirichlet(np.ones(3), size=10)
    X = pd.DataFrame(
        H_true @ W.T,  # samples x genes
        index=[f"s{i}" for i in range(10)],
        columns=[f"g{i}" for i in range(50)],
    )
    model = DualSimplex(n_components=3)
    model.W_ = pd.DataFrame(
        W, index=[f"g{i}" for i in range(50)], columns=["0", "1", "2"]
    )
    model.components_ = W.T
    model.n_components_ = 3
    model.n_features_in_ = 50
    model.feature_names_in_ = [f"g{i}" for i in range(50)]
    props = model.transform(X, method="nnls")
    np.testing.assert_allclose(props, H_true, atol=1e-6)


def test_align_components_hungarian():
    rng = np.random.default_rng(4)
    W_fit = pd.DataFrame(
        rng.normal(size=(20, 3)), index=[f"g{i}" for i in range(20)]
    )
    W_new = W_fit[[1, 0, 2]].copy()          # permuted columns
    H_new = pd.DataFrame([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    model = DualSimplex(n_components=3)
    model.W_ = W_fit
    aligned = model._align_components(W_new, H_new)
    np.testing.assert_allclose(aligned[0], [0.3, 0.4])  # old column 1
    np.testing.assert_allclose(aligned[1], [0.1, 0.2])  # old column 0
    np.testing.assert_allclose(aligned[2], [0.5, 0.6])


def test_align_components_fallback_few_common():
    W_fit = pd.DataFrame([[1.0], [2.0]], index=["a", "b"])
    W_new = pd.DataFrame([[9.0], [8.0]], index=["x", "y"])  # no shared genes
    H_new = pd.DataFrame([[0.3, 0.7]])
    model = DualSimplex(n_components=1)
    model.W_ = W_fit
    aligned = model._align_components(W_new, H_new)
    np.testing.assert_allclose(aligned[0], [0.3, 0.7])
