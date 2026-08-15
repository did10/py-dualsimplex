"""sklearn-style Python wrapper around the R DualSimplex deconvolution package.

DualSimplex (artyomovlab/dualsimplex, R6 class ``DualSimplexSolver``) factorizes
a non-negative genes x samples matrix into ``W`` (genes x K basis/signatures) and
``H`` (K x samples proportions). This wrapper exposes that solver through the
standard scikit-learn ``fit`` / ``transform`` / ``fit_transform`` API.

The R solver runs headlessly via ``Rscript`` in a subprocess -- no rpy2 needed.
The Rscript binary is located with ``shutil.which("Rscript")`` (or the
``DUALSIMPLEX_RSCRIPT`` environment variable); see ``find_rscript``.
Communication happens through CSV/JSON files in a temporary directory.

Input convention (sklearn-like): ``X`` is samples x features. Internally the
matrix is transposed to the genes x samples layout the R package expects.
"""

from __future__ import annotations

import importlib.resources
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, nnls
from scipy.stats import median_abs_deviation
from sklearn.base import BaseEstimator, TransformerMixin

def find_rscript() -> Optional[str]:
    """Locate an Rscript binary.

    Returns the value of the ``DUALSIMPLEX_RSCRIPT`` environment variable if
    set, otherwise the first ``Rscript`` on ``PATH`` (via ``shutil.which``),
    otherwise ``None``.
    """
    env_rscript = os.environ.get("DUALSIMPLEX_RSCRIPT")
    if env_rscript:
        return env_rscript
    return shutil.which("Rscript")


class DualSimplex(TransformerMixin, BaseEstimator):
    """Non-negative matrix factorization via the R DualSimplex solver.

    Fits ``X ~= W @ H`` with ``W`` (genes x K) and ``H`` (K x samples),
    subject to non-negativity and (soft) sum-to-one constraints on H.

    Parameters
    ----------
    n_components : int, default=5
        Rank K of the factorization (number of cell types / signatures).
    random_state : int, default=42
        Seed passed to R's ``set.seed`` before ``init_solution``.
    max_sinkhorn_iterations : int, default=300
        Passed to ``set_data``.
    max_dim : int or None, default=None
        Passed to ``set_data``. If None, defaults to ``min(n_samples, 30)``
        (raised to at least ``n_components``).
    sinkhorn_tol : float, default=1e-17
        Passed to ``set_data``.
    svd_method : str, default="svd"
        Passed to ``set_data``.
    linearize : bool, default=True
        Apply R's ``linearize_dataset`` (no-op for raw counts; converts
        log-scale data with ``2**x - 1``).
    log_mad_gt : float or None, default=None
        If set, features whose log2(x+1) median absolute deviation across
        samples is <= this threshold are dropped *before* calling R (a
        Python-side stand-in for R's ``basic_filter(log_mad_gt=...)``).
    plane_d_lt : float or None, default=None
        If set, applies R's ``distance_filter(plane_d_lt=...)`` and re-projects.
    zero_d_lt : float or None, default=None
        If set, applies R's ``distance_filter(zero_d_lt=...)`` and re-projects.
    initialization : str, default="random_invertible"
        R ``init_solution`` strategy. ``"random_invertible"`` (the package
        default) cannot fail; ``"random"`` sometimes fails with
        "ensure X and Omega are inverse".
    optimization : {"default", "custom"}, default="default"
        "default" runs R's ``default_optimization()`` (the paper's robust
        schedule, ~75k iterations). "custom" runs ``optim_solution`` with
        ``n_iterations`` and the ``coef_*`` settings below.
    n_iterations : int, default=10000
        Iterations for ``optimization="custom"``.
    optim_method : str, default="positivity"
        ``optim_config(method=...)`` for "custom" optimization.
    coef_der_X, coef_der_Omega : float, default=0.01
    coef_hinge_H, coef_hinge_W : float, default=0.5
        ``optim_config`` coefficients for "custom" optimization.
    reverse_sinkhorn_type : str, default="clean"
        Passed to ``finalize_solution`` ("clean" gives column-normalized W,
        row-normalized H).
    rscript : str or None, default=None
        Path to the Rscript binary. Defaults to the ``DUALSIMPLEX_RSCRIPT``
        environment variable, then to ``Rscript`` on ``PATH``.
    verbose : bool, default=False
        Stream R output to stdout.
    keep_temp : bool, default=False
        Keep the temporary work directory (printed when verbose).
    work_dir : str or None, default=None
        If set, use this directory instead of a temp dir (never deleted).
    save_state_path : str or None, default=None
        If set, also calls R's ``save_state(save_state_path)``.
    timeout : float or None, default=None
        Timeout (seconds) for the R subprocess.

    Attributes
    ----------
    W_ : pandas.DataFrame, genes x K
        Fitted basis/signatures in the full (post-filter) feature space.
    H_ : pandas.DataFrame, K x samples
        Training proportions.
    components_ : numpy.ndarray, K x genes
        Same as ``W_.T.values`` (sklearn convention).
    feature_names_in_ / sample_names_in_ : lists
        Feature and sample names seen at fit time.
    kept_features_ / kept_samples_ : lists
        Features / samples actually passed to R after filtering.
    n_features_in_ : int
    n_components_ : int
    """

    def __init__(
        self,
        n_components: int = 5,
        *,
        random_state: int = 42,
        max_sinkhorn_iterations: int = 300,
        max_dim: Optional[int] = None,
        sinkhorn_tol: float = 1e-17,
        svd_method: str = "svd",
        linearize: bool = True,
        log_mad_gt: Optional[float] = None,
        plane_d_lt: Optional[float] = None,
        zero_d_lt: Optional[float] = None,
        initialization: str = "random_invertible",
        optimization: str = "default",
        n_iterations: int = 10000,
        optim_method: str = "positivity",
        coef_der_X: float = 0.01,
        coef_der_Omega: float = 0.01,
        coef_hinge_H: float = 0.5,
        coef_hinge_W: float = 0.5,
        reverse_sinkhorn_type: str = "clean",
        rscript: Optional[str] = None,
        verbose: bool = False,
        keep_temp: bool = False,
        work_dir: Optional[str] = None,
        save_state_path: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.n_components = n_components
        self.random_state = random_state
        self.max_sinkhorn_iterations = max_sinkhorn_iterations
        self.max_dim = max_dim
        self.sinkhorn_tol = sinkhorn_tol
        self.svd_method = svd_method
        self.linearize = linearize
        self.log_mad_gt = log_mad_gt
        self.plane_d_lt = plane_d_lt
        self.zero_d_lt = zero_d_lt
        self.initialization = initialization
        self.optimization = optimization
        self.n_iterations = n_iterations
        self.optim_method = optim_method
        self.coef_der_X = coef_der_X
        self.coef_der_Omega = coef_der_Omega
        self.coef_hinge_H = coef_hinge_H
        self.coef_hinge_W = coef_hinge_W
        self.reverse_sinkhorn_type = reverse_sinkhorn_type
        self.rscript = rscript
        self.verbose = verbose
        self.keep_temp = keep_temp
        self.work_dir = work_dir
        self.save_state_path = save_state_path
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit(self, X, y=None):
        """Fit DualSimplex on ``X`` (samples x features).

        Runs the full R pipeline (``set_data`` -> ``project`` -> optional
        filters -> ``init_solution`` -> ``default_optimization`` /
        ``optim_solution`` -> ``finalize_solution``) in a subprocess.
        """
        X = self._validate_input(X)
        self._check_rank(X.shape)

        out = self._solve(X)

        W, H = out["W"], out["H"]                      # genes x K, K x samples
        full_features = list(self._feature_names(X))
        kept_features = out["kept_features"]
        kept_samples = np.asarray(self._sample_names(X))[out["kept_sample_idx"]].tolist()

        # Re-expand W to the full feature space (dropped genes -> 0 rows).
        W_full = pd.DataFrame(0.0, index=full_features, columns=W.columns)
        W_full.loc[kept_features] = W.values

        self.W_ = W_full
        self.H_ = H
        self.components_ = W_full.values.T             # K x genes
        self.n_components_ = int(self.n_components)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = full_features
        self.sample_names_in_ = list(self._sample_names(X))
        self.kept_features_ = kept_features
        self.kept_samples_ = kept_samples
        self.kept_sample_idx_ = out["kept_sample_idx"]
        self.n_samples_in_ = X.shape[0]
        self.last_out_dir_ = out["out_dir"]
        return self

    def transform(self, X, method: str = "dualsimplex",
                  normalize_proportions: bool = True) -> np.ndarray:
        """Project new samples onto the fitted basis.

        Parameters
        ----------
        X : samples x features (same features as fit).
        method : {"dualsimplex", "nnls"}, default="dualsimplex"
            "dualsimplex": re-runs the full R pipeline on ``X`` and returns
            proportions, with its K components re-ordered (Hungarian matching
            on W-column correlations) to match the fitted basis.
            "nnls": fast per-sample non-negative least squares against the
            fitted ``W_`` (no R call).
        normalize_proportions : bool, default=True
            For method="nnls": divide each row by its sum so rows sum to ~1.

        Returns
        -------
        numpy.ndarray, samples x K, in the fitted component order.
        """
        if not hasattr(self, "W_"):
            raise ValueError("This DualSimplex instance is not fitted yet. "
                             "Call 'fit' before 'transform'.")
        if method not in ("dualsimplex", "nnls"):
            raise ValueError(f"Unknown method {method!r}; use 'dualsimplex' or 'nnls'.")

        X = self._validate_input(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features but was fitted with "
                f"{self.n_features_in_}."
            )
        X = self._reorder_to_fit_features(X)

        if method == "nnls":
            return self._nnls_transform(X, normalize_proportions)

        out = self._solve(X)
        H_aligned = self._align_components(out["W"], out["H"]).T  # kept x K
        result = np.full((X.shape[0], self.n_components), np.nan)
        result[out["kept_sample_idx"]] = H_aligned
        return result

    def fit_transform(self, X, y=None, **fit_params) -> np.ndarray:
        """Fit and return the training proportions ``H_.T`` (samples x K)."""
        self.fit(X, y, **fit_params)
        result = np.full((X.shape[0], self.n_components), np.nan)
        result[self.kept_sample_idx_] = self.H_.T.values
        return result

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _solve(self, X) -> dict:
        """Run the R pipeline on ``X`` (samples x features) and return outputs."""
        genes_df, kept_row_mask = self._to_genes_x_samples(X)

        if self.n_components > genes_df.shape[0]:
            raise ValueError(
                f"n_components={self.n_components} > {genes_df.shape[0]} "
                f"non-zero features after filtering."
            )

        rscript = self._resolve_rscript()
        driver = self._driver_path()

        if self.work_dir is not None:
            out_dir = self.work_dir
            Path(out_dir).mkdir(parents=True, exist_ok=True)
        else:
            out_dir = tempfile.mkdtemp(prefix="dualsimplex_")

        config = self._build_config(genes_df.shape[1])
        genes_df.to_csv(os.path.join(out_dir, "data.csv"))
        with open(os.path.join(out_dir, "config.json"), "w") as fh:
            json.dump(config, fh, indent=2)

        cmd = [rscript, "--vanilla", str(driver), os.path.join(out_dir, "config.json"), out_dir]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )

        if proc.returncode != 0:
            tail = (proc.stderr or "")[-4000:] or (proc.stdout or "")[-4000:]
            raise RuntimeError(
                f"DualSimplex R run failed (exit {proc.returncode}):\n{tail}"
            )

        if self.verbose:
            if proc.stdout:
                print(proc.stdout)
            if proc.stderr:
                print(proc.stderr, file=sys.stderr)

        W = pd.read_csv(os.path.join(out_dir, "W.csv"), index_col=0)
        H = pd.read_csv(os.path.join(out_dir, "H.csv"), index_col=0)
        # R row/col names come back through CSV as strings; pandas would
        # parse numeric labels as int64. Force str everywhere so keys stay
        # consistent with the Python-side feature/sample names.
        W.index = [str(i) for i in W.index]
        H.index = [str(i) for i in H.index]
        H.columns = [str(c) for c in H.columns]
        kept_features = [
            str(f)
            for f in pd.read_csv(
                os.path.join(out_dir, "kept_features.csv"),
                dtype={"feature": str},
            )["feature"]
        ]

        kept_sample_idx = np.where(kept_row_mask)[0]

        if not self.keep_temp and self.work_dir is None:
            shutil.rmtree(out_dir, ignore_errors=True)

        return {
            "W": W,
            "H": H,
            "kept_features": kept_features,
            "kept_sample_idx": kept_sample_idx,
            "out_dir": out_dir,
        }

    def _align_components(self, W_new: pd.DataFrame, H_new: pd.DataFrame) -> np.ndarray:
        """Reorder H_new rows so component k matches fitted component k."""
        K = self.n_components
        if W_new.shape[1] != K:
            raise ValueError(
                f"transform found {W_new.shape[1]} components but the fit "
                f"used {K}."
            )
        common = [g for g in self.W_.index if g in W_new.index]
        if len(common) >= 5:
            a = self.W_.loc[common].values      # genes x K
            b = W_new.loc[common].values        # genes x K
            corr = np.corrcoef(a, b, rowvar=False)[:K, K:]
            corr = np.nan_to_num(corr, nan=0.0)
            cost = 1.0 - np.abs(corr)
            row_ind, col_ind = linear_sum_assignment(cost)
            order = np.empty(K, dtype=int)
            order[row_ind] = col_ind
        else:
            order = np.arange(K)
        return H_new.values[order]              # K x samples, fit order

    def _nnls_transform(self, X, normalize: bool) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(float)
        W = self.W_.values                      # genes x K
        H = np.empty((X.shape[0], self.n_components), dtype=float)
        for i, x in enumerate(X):
            h, _ = nnls(W, x)
            if normalize and h.sum() > 0:
                h = h / h.sum()
            H[i] = h
        return H

    # ------------------------------------------------------------------ #
    # Data plumbing
    # ------------------------------------------------------------------ #
    def _validate_input(self, X):
        if hasattr(X, "toarray"):               # sparse
            X = X.toarray()
        if isinstance(X, pd.DataFrame):
            # Coerce labels to str so integer column names (e.g. from
            # pd.DataFrame(ndarray)) round-trip through the CSV bridge.
            X = X.copy()
            X.columns = [str(c) for c in X.columns]
            X.index = [str(i) for i in X.index]
            return X
        arr = np.asarray(X, dtype=float)
        return arr

    @staticmethod
    def _feature_names(X):
        if isinstance(X, pd.DataFrame):
            return [str(c) for c in X.columns]
        return [str(i) for i in range(X.shape[1])]

    @staticmethod
    def _sample_names(X):
        if isinstance(X, pd.DataFrame):
            return [str(i) for i in X.index]
        return [str(i) for i in range(X.shape[0])]

    def _reorder_to_fit_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reorder X columns to match the fitted feature order."""
        if isinstance(X, pd.DataFrame):
            fit_cols = self.feature_names_in_
            if list(X.columns) == fit_cols:
                return X
            if set(X.columns) == set(fit_cols):
                return X[fit_cols]
            missing = [c for c in fit_cols if c not in X.columns]
            raise ValueError(
                f"X columns do not match fit features; missing {missing[:5]}"
            )
        return X

    def _to_genes_x_samples(self, X):
        """Transpose to genes x samples, drop all-zero rows/cols, optional MAD filter.

        Returns (DataFrame genes x samples with kept names, kept_row_mask) where
        kept_row_mask indexes the *original* samples (columns of X).
        """
        feature_names = self._feature_names(X)
        sample_names = self._sample_names(X)

        if isinstance(X, pd.DataFrame):
            arr = X.values.astype(float)
        else:
            arr = np.asarray(X, dtype=float)

        n_genes, n_samples = arr.shape[1], arr.shape[0]
        if n_samples == 0 or n_genes == 0:
            raise ValueError("X must be non-empty.")

        row_sums = arr.sum(axis=1)              # per sample
        col_sums = arr.sum(axis=0)              # per gene (features)
        kept_samples = row_sums > 0
        kept_genes = col_sums > 0

        if kept_genes.sum() == 0 or kept_samples.sum() == 0:
            raise ValueError("X has no non-zero rows/columns after filtering.")

        genes_idx = np.where(kept_genes)[0]
        samples_idx = np.where(kept_samples)[0]
        sub = arr[np.ix_(samples_idx, genes_idx)].T     # genes x samples

        # Optional MAD-based feature filter (log2(x+1) space, like R's mad()).
        if self.log_mad_gt is not None:
            log1p = np.log2(sub + 1.0)
            mads = median_abs_deviation(
                log1p, axis=1, scale=1.4826, nan_policy="omit"
            )
            mad_keep = mads > self.log_mad_gt
            if mad_keep.sum() == 0:
                raise ValueError("log_mad_gt filter removed all features.")
            sub = sub[mad_keep]
            genes_idx = genes_idx[mad_keep]

        kept_gene_names = [feature_names[i] for i in genes_idx]
        kept_sample_names = [sample_names[i] for i in samples_idx]

        df = pd.DataFrame(sub, index=kept_gene_names, columns=kept_sample_names)
        kept_row_mask = kept_samples
        return df, kept_row_mask

    def _build_config(self, n_samples: int) -> dict:
        max_dim = self.max_dim
        if max_dim is None:
            max_dim = min(n_samples, 30)
        max_dim = max(int(max_dim), int(self.n_components))
        return {
            "k": int(self.n_components),
            "max_sinkhorn_iterations": int(self.max_sinkhorn_iterations),
            "max_dim": int(max_dim),
            "sinkhorn_tol": float(self.sinkhorn_tol),
            "svd_method": self.svd_method,
            "linearize": bool(self.linearize),
            "plane_d_lt": self.plane_d_lt,
            "zero_d_lt": self.zero_d_lt,
            "seed": int(self.random_state),
            "initialization": self.initialization,
            "optimization": self.optimization,
            "n_iterations": int(self.n_iterations),
            "optim_method": self.optim_method,
            "coef_der_X": float(self.coef_der_X),
            "coef_der_Omega": float(self.coef_der_Omega),
            "coef_hinge_H": float(self.coef_hinge_H),
            "coef_hinge_W": float(self.coef_hinge_W),
            "reverse_sinkhorn_type": self.reverse_sinkhorn_type,
            "save_state": self.save_state_path,
        }

    def _check_rank(self, shape: tuple):
        if self.n_components < 1:
            raise ValueError("n_components must be >= 1.")
        if shape[0] < self.n_components:
            raise ValueError(
                f"n_components={self.n_components} > n_samples={shape[0]}."
            )

    # ------------------------------------------------------------------ #
    # R environment plumbing
    # ------------------------------------------------------------------ #
    def _resolve_rscript(self) -> str:
        rscript = self.rscript or find_rscript()
        if rscript is None:
            raise FileNotFoundError(
                "No Rscript found. Install R with the DualSimplex R package and "
                "either add Rscript to PATH, set the DUALSIMPLEX_RSCRIPT "
                "environment variable, or pass rscript=... to the constructor."
            )
        if not os.path.exists(rscript):
            raise FileNotFoundError(
                f"Rscript not found at {rscript!r}. Check the path, or set "
                f"rscript=... / DUALSIMPLEX_RSCRIPT to an existing Rscript "
                f"binary in the environment that has DualSimplex installed."
            )
        return rscript

    @staticmethod
    def _driver_path():
        return importlib.resources.files("dualsimplex_py").joinpath(
            "_r", "dualsimplex_fit.R"
        )
