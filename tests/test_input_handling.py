"""Tests for input validation, transposition, and zero/MAD filtering."""

import numpy as np
import pandas as pd
import pytest

from dualsimplex_py import DualSimplex


def test_validate_input_dataframe_stringifies_labels():
    X = pd.DataFrame(np.ones((3, 2)), index=[1, 2, 3], columns=[10, 20])
    out = DualSimplex()._validate_input(X)
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == ["1", "2", "3"]
    assert list(out.columns) == ["10", "20"]


def test_validate_input_ndarray():
    out = DualSimplex()._validate_input(np.ones((3, 2)))
    assert isinstance(out, np.ndarray)
    assert out.shape == (3, 2)


def test_validate_input_sparse():
    import scipy.sparse as sp

    out = DualSimplex()._validate_input(sp.csr_matrix(np.ones((3, 2))))
    assert isinstance(out, np.ndarray)
    assert out.shape == (3, 2)


def test_to_genes_x_samples_transpose_and_drop_zeros(synthetic_X):
    df, kept_mask = DualSimplex()._to_genes_x_samples(synthetic_X)
    assert df.shape == (19, 11)              # 12 samples - 1 zero, 20 genes - 1 zero
    assert df.index[5] == "gene_6"           # gene_5 was dropped
    assert df.columns[3] == "sample_4"       # sample_3 was dropped
    assert kept_mask.tolist() == [True] * 3 + [False] + [True] * 8
    assert (df.values >= 0).all()


def test_to_genes_x_samples_keeps_names(synthetic_X):
    df, _ = DualSimplex()._to_genes_x_samples(synthetic_X)
    assert set(df.index) == {f"gene_{j}" for j in range(20) if j != 5}
    assert set(df.columns) == {f"sample_{i}" for i in range(12) if i != 3}


def test_to_genes_x_samples_ndarray_default_names():
    df, kept = DualSimplex()._to_genes_x_samples(np.ones((4, 3)))
    assert df.index.tolist() == ["0", "1", "2"]
    assert df.columns.tolist() == ["0", "1", "2", "3"]
    assert kept.tolist() == [True, True, True, True]


def test_to_genes_x_samples_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        DualSimplex()._to_genes_x_samples(np.zeros((0, 5)))
    with pytest.raises(ValueError, match="non-empty"):
        DualSimplex()._to_genes_x_samples(np.zeros((5, 0)))


def test_to_genes_x_samples_all_zero_raises():
    with pytest.raises(ValueError, match="no non-zero"):
        DualSimplex()._to_genes_x_samples(np.zeros((3, 3)))


def test_log_mad_filter_drops_constant_genes():
    rng = np.random.default_rng(1)
    X = rng.integers(1, 100, size=(8, 6)).astype(float)
    X[:, 3] = 5.0   # constant column -> MAD 0 -> dropped
    X[0, :] = 0.0   # zero row dropped by the zero filter
    model = DualSimplex(log_mad_gt=0.0)
    df, _ = model._to_genes_x_samples(pd.DataFrame(X))
    assert "3" not in df.index
    assert df.shape[0] == 5  # 6 genes - constant one


def test_log_mad_filter_removes_all_raises():
    X = pd.DataFrame(np.ones((8, 6)))  # every gene constant -> all MAD 0
    model = DualSimplex(log_mad_gt=0.0)
    with pytest.raises(ValueError, match="log_mad_gt filter removed all features"):
        model._to_genes_x_samples(X)


def test_reorder_to_fit_features_reorders(synthetic_X):
    model = DualSimplex()
    fit_cols = [f"gene_{j}" for j in range(20)]
    model.feature_names_in_ = fit_cols
    shuffled = synthetic_X[list(reversed(fit_cols))]
    out = model._reorder_to_fit_features(shuffled)
    assert list(out.columns) == fit_cols


def test_reorder_to_fit_features_missing_raises(synthetic_X):
    model = DualSimplex()
    model.feature_names_in_ = [f"gene_{j}" for j in range(20)]
    with pytest.raises(ValueError, match="missing"):
        model._reorder_to_fit_features(synthetic_X.drop(columns=["gene_0"]))
