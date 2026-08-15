"""Tests for the _solve() CSV bridge, using a mocked R subprocess."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from dualsimplex_py import DualSimplex


class FakeProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _write_outputs(out_dir, W=None, H=None, kept=None):
    W = pd.DataFrame(
        W if W is not None else [[1.0, 0.0], [0.5, 2.0], [0.0, 3.0]],
        index=["g0", "g1", "g2"],
        columns=["c0", "c1"],
    )
    H = pd.DataFrame(
        H if H is not None else [[0.7, 0.3], [0.2, 0.8]],
        index=["c0", "c1"],
        columns=["s0", "s1"],
    )
    W.to_csv(os.path.join(out_dir, "W.csv"))
    H.to_csv(os.path.join(out_dir, "H.csv"))
    pd.DataFrame({"feature": kept if kept is not None else ["g0", "g1", "g2"]}).to_csv(
        os.path.join(out_dir, "kept_features.csv"), index=False
    )


@pytest.fixture
def rscript_stub(tmp_path):
    """A real, executable dummy Rscript so resolution succeeds without R."""
    p = tmp_path / "Rscript"
    p.write_text("#!/bin/sh\necho fake-rscript\n")
    p.chmod(0o755)
    return str(p)


@pytest.fixture(autouse=True)
def _set_env_rscript(rscript_stub, monkeypatch):
    monkeypatch.setenv("DUALSIMPLEX_RSCRIPT", rscript_stub)


@pytest.fixture
def mock_solve(monkeypatch):
    """Replace subprocess.run with one that writes canned R outputs.

    Records the command, out_dir, config.json, and data.csv for assertions.
    """
    calls = {}

    def _run(cmd, **kwargs):
        out_dir = cmd[-1]
        _write_outputs(out_dir)
        calls["cmd"] = cmd
        calls["out_dir"] = out_dir
        with open(os.path.join(out_dir, "config.json")) as fh:
            calls["config"] = json.load(fh)
        calls["data"] = pd.read_csv(os.path.join(out_dir, "data.csv"), index_col=0)
        return FakeProc()

    monkeypatch.setattr("dualsimplex_py._wrapper.subprocess.run", _run)
    return calls


def test_solve_parses_outputs(synthetic_X, mock_solve):
    out = DualSimplex(n_components=2, random_state=1)._solve(synthetic_X)
    assert out["W"].shape == (3, 2)
    assert out["H"].shape == (2, 2)
    assert out["kept_features"] == ["g0", "g1", "g2"]
    assert list(out["W"].index) == ["g0", "g1", "g2"]  # str coercion
    assert list(out["H"].index) == ["c0", "c1"]
    assert len(out["kept_sample_idx"]) == 11


def test_solve_writes_config_and_data(synthetic_X, mock_solve):
    DualSimplex(n_components=2, random_state=1)._solve(synthetic_X)
    cfg = mock_solve["config"]
    assert cfg["k"] == 2
    assert cfg["seed"] == 1
    assert cfg["max_dim"] == 11  # min(11 kept samples, 30)
    data = mock_solve["data"]
    assert data.shape == (19, 11)  # genes x samples after zero-drop
    assert "gene_5" not in data.index
    assert "sample_3" not in data.columns


def test_solve_raises_on_r_failure(synthetic_X, monkeypatch):
    def _run(cmd, **kwargs):
        return FakeProc(returncode=1, stderr="boom: something broke")

    monkeypatch.setattr("dualsimplex_py._wrapper.subprocess.run", _run)
    with pytest.raises(RuntimeError, match="boom: something broke"):
        DualSimplex(n_components=2).fit(synthetic_X)


def test_solve_cleans_temp_dir_by_default(synthetic_X, mock_solve):
    out = DualSimplex(n_components=2)._solve(synthetic_X)
    assert not os.path.exists(out["out_dir"])


def test_solve_keeps_temp_dir_when_requested(synthetic_X, mock_solve):
    out = DualSimplex(n_components=2, keep_temp=True)._solve(synthetic_X)
    assert os.path.exists(out["out_dir"])
    assert os.path.exists(os.path.join(out["out_dir"], "W.csv"))


def test_solve_uses_work_dir(synthetic_X, mock_solve, tmp_path):
    work_dir = tmp_path / "ds_work"
    out = DualSimplex(n_components=2, work_dir=str(work_dir))._solve(synthetic_X)
    assert out["out_dir"] == str(work_dir)
    assert work_dir.exists()
    # work_dir is never deleted even without keep_temp
    assert os.path.exists(os.path.join(out["out_dir"], "W.csv"))


def test_solve_passes_timeout(synthetic_X, monkeypatch):
    seen = {}

    def _run(cmd, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        _write_outputs(cmd[-1])
        return FakeProc()

    monkeypatch.setattr("dualsimplex_py._wrapper.subprocess.run", _run)
    DualSimplex(n_components=2, timeout=42.0)._solve(synthetic_X)
    assert seen["timeout"] == 42.0
