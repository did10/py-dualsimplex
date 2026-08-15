"""Tests for the JSON config passed to the R driver."""

import json

from dualsimplex_py import DualSimplex, __version__, find_rscript


def test_public_api_exports():
    assert __version__ == "0.1.0"
    assert callable(find_rscript)


def test_build_config_has_all_keys():
    cfg = DualSimplex(n_components=4, random_state=7)._build_config(n_samples=50)
    expected = {
        "k",
        "max_sinkhorn_iterations",
        "max_dim",
        "sinkhorn_tol",
        "svd_method",
        "linearize",
        "plane_d_lt",
        "zero_d_lt",
        "seed",
        "initialization",
        "optimization",
        "n_iterations",
        "optim_method",
        "coef_der_X",
        "coef_der_Omega",
        "coef_hinge_H",
        "coef_hinge_W",
        "reverse_sinkhorn_type",
        "save_state",
    }
    assert set(cfg) == expected


def test_build_config_values():
    model = DualSimplex(
        n_components=3,
        random_state=99,
        max_sinkhorn_iterations=50,
        sinkhorn_tol=1e-9,
        svd_method="irlba",
        linearize=False,
        plane_d_lt=0.05,
        zero_d_lt=None,
        initialization="random",
        optimization="custom",
        n_iterations=200,
        optim_method="theta",
        coef_der_X=0.1,
        coef_der_Omega=0.2,
        coef_hinge_H=0.3,
        coef_hinge_W=0.4,
        reverse_sinkhorn_type="clean",
        save_state_path="/tmp/state",
    )
    cfg = model._build_config(n_samples=10)
    assert cfg["k"] == 3
    assert cfg["seed"] == 99
    assert cfg["max_sinkhorn_iterations"] == 50
    assert cfg["sinkhorn_tol"] == 1e-9
    assert cfg["svd_method"] == "irlba"
    assert cfg["linearize"] is False
    assert cfg["plane_d_lt"] == 0.05
    assert cfg["zero_d_lt"] is None
    assert cfg["initialization"] == "random"
    assert cfg["optimization"] == "custom"
    assert cfg["n_iterations"] == 200
    assert cfg["optim_method"] == "theta"
    assert cfg["coef_der_X"] == 0.1
    assert cfg["coef_der_Omega"] == 0.2
    assert cfg["coef_hinge_H"] == 0.3
    assert cfg["coef_hinge_W"] == 0.4
    assert cfg["reverse_sinkhorn_type"] == "clean"
    assert cfg["save_state"] == "/tmp/state"


def test_max_dim_default_and_floor():
    model = DualSimplex(n_components=5)
    assert model._build_config(n_samples=50)["max_dim"] == 30  # min(50, 30)
    assert model._build_config(n_samples=10)["max_dim"] == 10  # min(10, 30)
    assert model._build_config(n_samples=3)["max_dim"] == 5    # floored to k


def test_config_is_json_serializable():
    cfg = DualSimplex()._build_config(n_samples=20)
    json.dumps(cfg)  # must not raise


def test_config_uses_defaults():
    cfg = DualSimplex()._build_config(n_samples=20)
    assert cfg["optimization"] == "default"
    assert cfg["initialization"] == "random_invertible"
    assert cfg["reverse_sinkhorn_type"] == "clean"
    assert cfg["save_state"] is None
    assert cfg["max_sinkhorn_iterations"] == 300
    assert cfg["sinkhorn_tol"] == 1e-17
