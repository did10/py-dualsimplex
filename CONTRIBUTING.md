# Contributing

Thanks for considering contributing to `dualsimplex-py`!

> **Note:** This project was developed with the assistance of an AI coding
> assistant. Contributions and reviews by humans are still very welcome.

## Development setup

```bash
git clone https://github.com/did10/py-dualsimplex
cd py-dualsimplex
python -m pip install -e ".[dev]"
```

## Running the tests

Unit tests do **not** require R:

```bash
python -m pytest -m "not integration"
```

Integration tests exercise the real R `DualSimplex` solver and are skipped
automatically when R is not available:

```bash
# R (with the DualSimplex R package) on PATH:
python -m pytest -m integration

# ...or point the wrapper at a specific Rscript:
DUALSIMPLEX_RSCRIPT=/path/to/Rscript python -m pytest -m integration
```

## Before submitting a pull request

- Run the full unit suite: `python -m pytest -m "not integration"`.
- Run integration tests if you have R available.
- Keep `CHANGELOG.md` up to date (add an entry under an "Unreleased" section).
- Follow the existing code style: PEP 8, type hints, and scikit-learn
  estimator conventions (`fit` / `transform` / `fit_transform`, stored
  attributes ending with `_`).
