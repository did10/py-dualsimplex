"""dualsimplex_py: sklearn-style Python wrapper for the R DualSimplex solver."""

from ._wrapper import DualSimplex, find_rscript

__all__ = ["DualSimplex", "find_rscript"]
__version__ = "0.1.0"
