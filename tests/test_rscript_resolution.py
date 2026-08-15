"""Tests for Rscript resolution: param > env var > PATH > clear errors."""

import os

import pytest

from dualsimplex_py import DualSimplex, find_rscript


def _make_executable(path):
    path.write_text("#!/bin/sh\necho fake-rscript\n")
    path.chmod(0o755)
    return str(path)


def test_find_rscript_prefers_env_var(monkeypatch, tmp_path):
    env_script = _make_executable(tmp_path / "env_Rscript")
    _make_executable(tmp_path / "Rscript")
    monkeypatch.setenv("DUALSIMPLEX_RSCRIPT", env_script)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert find_rscript() == env_script


def test_find_rscript_falls_back_to_path(monkeypatch, tmp_path):
    path_script = _make_executable(tmp_path / "Rscript")
    monkeypatch.delenv("DUALSIMPLEX_RSCRIPT", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert find_rscript() == path_script


def test_find_rscript_returns_none(monkeypatch, tmp_path):
    monkeypatch.delenv("DUALSIMPLEX_RSCRIPT", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert find_rscript() is None


def test_explicit_rscript_param_wins(monkeypatch, tmp_path):
    explicit = _make_executable(tmp_path / "explicit_Rscript")
    env_script = _make_executable(tmp_path / "env_Rscript")
    monkeypatch.setenv("DUALSIMPLEX_RSCRIPT", env_script)
    assert DualSimplex(rscript=explicit)._resolve_rscript() == explicit


def test_resolve_raises_when_not_found(monkeypatch, tmp_path):
    monkeypatch.delenv("DUALSIMPLEX_RSCRIPT", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="No Rscript found"):
        DualSimplex()._resolve_rscript()


def test_resolve_raises_for_missing_explicit_path(tmp_path):
    model = DualSimplex(rscript=str(tmp_path / "does_not_exist"))
    with pytest.raises(FileNotFoundError, match="does_not_exist"):
        model._resolve_rscript()


def test_resolve_uses_env_var(monkeypatch, tmp_path):
    env_script = _make_executable(tmp_path / "env_Rscript")
    monkeypatch.setenv("DUALSIMPLEX_RSCRIPT", env_script)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert DualSimplex()._resolve_rscript() == env_script
