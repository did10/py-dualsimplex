# dualsimplex-py

> ⚠️ **AI-generated code notice**
>
> This project was developed with the assistance of an AI coding assistant
> (GitHub Copilot) and reviewed by a human maintainer. It is a thin, tested
> wrapper around the upstream R package — the underlying algorithm is from
> [artyomovlab/DualSimplex](https://github.com/artyomovlab/dualsimplex).
> Please report any issues you find, and verify results on your own data.

[![CI](https://github.com/did10/py-dualsimplex/actions/workflows/ci.yml/badge.svg)](https://github.com/did10/py-dualsimplex/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A minimal, scikit-learn–style Python wrapper around the R
[DualSimplex](https://github.com/artyomovlab/dualsimplex) deconvolution
package. No plotting, no R state management — just `fit`, `transform`,
`fit_transform`, and the solver configuration up front.

DualSimplex factorizes a non-negative **genes x samples** matrix `X` into

```
X ≈ W · H
```

with `W` (genes x K, the basis/signature matrix) and `H` (K x samples,
per-sample proportions). Note: DualSimplex's `"clean"` finalize step does
**not** enforce exact sum-to-one on H — normalize the rows yourself when
you need true proportions:

```python
props = model.fit_transform(X)                    # samples x K
props = props / props.sum(axis=1, keepdims=True)  # optional row normalize
```

## How it works

- The R solver runs headlessly in a subprocess (`Rscript --vanilla`), passing
  the data as CSV and a JSON config, and reading `W`/`H` back as CSV.
- No `rpy2` required — just an `Rscript` binary with the `DualSimplex` R
  package installed.

## Prerequisites

**Python side** — Python >= 3.10. The scientific stack (`numpy`, `pandas`,
`scipy`, `scikit-learn`) is installed automatically with the package.

**R side** — any R installation with the `DualSimplex` R package:

```r
install.packages("devtools")
devtools::install_github("artyomovlab/DualSimplex")
```

That's it — the wrapper finds `Rscript` automatically (see below).

## Install

```bash
pip install git+https://github.com/did10/py-dualsimplex
```

For development:

```bash
git clone https://github.com/did10/py-dualsimplex
cd py-dualsimplex
pip install -e ".[dev]"
```

## Rscript configuration

The wrapper needs a working `Rscript` binary, resolved in this order:

1. the `rscript=` constructor parameter,
2. the `DUALSIMPLEX_RSCRIPT` environment variable,
3. `Rscript` found on `PATH`.

It raises a clear error if none is found. If `Rscript` is already on your
`PATH` (e.g. after `conda activate`-ing your R environment), no configuration
is needed at all.

If Python and R live in different environments (a common setup), point the
wrapper at the right `Rscript`:

```python
model = DualSimplex(n_components=5, rscript="/path/to/Rscript")
```

or, without touching your code:

```bash
export DUALSIMPLEX_RSCRIPT=/path/to/Rscript
```

Note: the R process must also be able to load its R libraries (`R_LIBS`,
site packages, etc.). If R was installed via conda, activating that
environment in the same shell usually handles this.

## Usage

```python
import pandas as pd
from dualsimplex_py import DualSimplex

# X: samples x features (sklearn convention). Internally transposed to
# genes x samples, exactly like the upstream DualSimplex pipeline.
X = pd.read_csv(".../mixed_bulks_counts.csv", index_col=0).T   # samples x genes

model = DualSimplex(n_components=5, random_state=1234)
model.fit(X)

W = model.W_                    # genes x K   (basis signatures)
H = model.H_                    # K x samples (training proportions)
props_train = model.fit_transform(X)   # samples x K, = H.T

# Project held-out samples onto the fitted basis (full R re-run, components
# re-ordered to match the fit via Hungarian matching on W correlation):
props_test = model.transform(X_held_out)          # samples x K

# ...or fast NNLS projection (no R call):
props_test_fast = model.transform(X_held_out, method="nnls")
```

### Reproducing the upstream `DualSimplex.R` defaults

```python
model = DualSimplex(
    n_components=5,
    random_state=1234,             # matches set.seed(1234)
    max_sinkhorn_iterations=300,
    max_dim=None,                  # -> min(n_samples, 30), like the script
    sinkhorn_tol=1e-17,
    initialization="random_invertible",
    optimization="default",        # dso$default_optimization()
    reverse_sinkhorn_type="clean", # dso$finalize_solution("clean")
)
```

Notes:

- `"random_invertible"` (the package default) constructs X and Omega as exact
  inverses and cannot fail; `"random"` can fail with *"ensure X and Omega are
  inverse"*.
- Known upstream quirk: with `n_components=2`, `"random_invertible"` can
  crash inside R with *"non-conformable arguments"* because R's `diag()`
  treats the length-1 scale vector as a scalar and builds an identity of the
  wrong size (the upstream bug affects any K=2 run). K>=3 is unaffected; for
  a 2-component fit, retry with a different `random_state` or use
  `optimization="custom"` with `initialization="random"`.
- `optimization="custom"` uses `optim_solution(n_iterations, optim_config(...))`
  instead of the paper's default schedule (useful for quick smoke tests,
  e.g. `n_iterations=2000`).
- All-zero rows/columns are dropped before R. A MAD-based gene filter can be
  applied Python-side via `log_mad_gt=` (R's `basic_filter` has fragile
  default gene-name filters, so it is not exposed).
- `plane_d_lt` / `zero_d_lt` forward to R's `distance_filter` (+ re-project).

## Configuration summary

| Parameter | Default | Meaning |
|---|---|---|
| `n_components` | 5 | Rank K |
| `random_state` | 42 | R `set.seed` |
| `max_sinkhorn_iterations` | 300 | `set_data` |
| `max_dim` | None | `set_data` (`min(n_samples, 30)`) |
| `sinkhorn_tol` | 1e-17 | `set_data` |
| `svd_method` | "svd" | `set_data` |
| `linearize` | True | `linearize_dataset` |
| `log_mad_gt` | None | Python-side gene MAD filter |
| `plane_d_lt` / `zero_d_lt` | None | R `distance_filter` |
| `initialization` | "random_invertible" | `init_solution` |
| `optimization` | "default" | `default_optimization` \| `optim_solution` |
| `n_iterations` | 10000 | custom mode iterations |
| `optim_method` / `coef_*` | positivity / 0.01 / 0.5 | custom `optim_config` |
| `reverse_sinkhorn_type` | "clean" | `finalize_solution` |
| `rscript` | `DUALSIMPLEX_RSCRIPT` env var, then `Rscript` on `PATH` | Rscript binary |
| `verbose` / `keep_temp` / `work_dir` | False | diagnostics |
| `save_state_path` | None | also call R `save_state` |
| `timeout` | None | subprocess timeout (s) |

## Development & testing

Unit tests do **not** require R:

```bash
python -m pytest -m "not integration"
```

Integration tests run the real R solver and are skipped automatically when R
is unavailable:

```bash
DUALSIMPLEX_RSCRIPT=/path/to/Rscript python -m pytest -m integration
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Citation

If you use this wrapper in your work, please cite the original DualSimplex
algorithm:

> Non-negative matrix factorization and deconvolution as dual simplex problem
> — Denis Kleverov, Ekaterina Aladyeva, Alexey Serdyukov, Maxim Artyomov,
> bioRxiv 2024.04.09.588652; doi: https://doi.org/10.1101/2024.04.09.588652

## License

[MIT](LICENSE) — copyright (c) 2026 did10.
