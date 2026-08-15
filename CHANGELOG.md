# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-15

Initial public release.

### Added
- sklearn-style `DualSimplex` estimator (`fit` / `transform` / `fit_transform`).
- Headless R subprocess driver (`_r/dualsimplex_fit.R`) — no rpy2 needed.
- Portable Rscript resolution: `rscript=` parameter > `DUALSIMPLEX_RSCRIPT`
  env var > `Rscript` on `PATH`.
- Python-side gene MAD filter (`log_mad_gt=`) and fast NNLS projection
  (`transform(..., method="nnls")`).
- pytest test suite (unit tests run without R; R-backed integration tests are
  marked `integration` and skipped automatically when R is unavailable).
- GitHub Actions CI (unit tests on Python 3.10–3.13).
- MIT license.
