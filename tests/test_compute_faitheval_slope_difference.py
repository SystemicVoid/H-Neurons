"""Tests for scripts/compute_faitheval_slope_difference.py."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _clear_compute_module() -> None:
    sys.modules.pop("compute_faitheval_slope_difference", None)
    sys.modules.pop("scripts.compute_faitheval_slope_difference", None)


def test_module_imports_with_scripts_dir_on_sys_path(monkeypatch: pytest.MonkeyPatch):
    _clear_compute_module()
    monkeypatch.syspath_prepend(str(SCRIPTS_DIR))

    module = importlib.import_module("compute_faitheval_slope_difference")

    assert module.__file__ is not None
    assert Path(module.__file__).resolve() == (
        SCRIPTS_DIR / "compute_faitheval_slope_difference.py"
    )


def test_compute_neuron_vs_random_raises_when_control_dir_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_compute_module()
    monkeypatch.syspath_prepend(str(SCRIPTS_DIR))
    module = importlib.import_module("compute_faitheval_slope_difference")

    neuron_ids = ["item_a", "item_b"]
    neuron_traj = np.array([[True, False], [False, True]], dtype=bool)
    control_traj = np.array([[False, False], [True, True]], dtype=bool)

    def fake_load_trajectories(data_dir: Path, alphas: list[float]):
        if data_dir == module.NEURON_DIR:
            return neuron_ids, neuron_traj
        return neuron_ids, control_traj

    def fake_paired_bootstrap_slope_difference(*args, **kwargs):
        return {
            "slope_difference_pp_per_alpha": {"estimate": 1.0},
            "slope_a_pp_per_alpha": 1.0,
            "slope_b_pp_per_alpha": 0.0,
        }

    existing_dirs = {
        module.CONTROL_BASE / seed_name
        for seed_name in module.CONTROL_SEEDS
        if seed_name != "seed_4_unconstrained"
    }

    monkeypatch.setattr(module, "ALPHAS", [0.0, 1.0])
    monkeypatch.setattr(module, "load_trajectories", fake_load_trajectories)
    monkeypatch.setattr(
        module,
        "paired_bootstrap_slope_difference",
        fake_paired_bootstrap_slope_difference,
    )
    monkeypatch.setattr(Path, "exists", lambda path: path in existing_dirs)

    with pytest.raises(
        ValueError, match="missing control directories: seed_4_unconstrained"
    ):
        module.compute_neuron_vs_random(
            n_resamples=10,
            seed=42,
            permutation_resamples=20,
        )


def test_compute_neuron_vs_random_raises_when_item_overlap_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_compute_module()
    monkeypatch.syspath_prepend(str(SCRIPTS_DIR))
    module = importlib.import_module("compute_faitheval_slope_difference")

    neuron_ids = ["item_a", "item_b"]
    neuron_traj = np.array([[True, False], [False, True]], dtype=bool)
    aligned_ids = ["item_b", "item_a"]
    aligned_traj = np.array([[False, False], [True, True]], dtype=bool)
    partial_ids = ["item_a"]
    partial_traj = np.array([[True, True]], dtype=bool)

    def fake_load_trajectories(data_dir: Path, alphas: list[float]):
        if data_dir == module.NEURON_DIR:
            return neuron_ids, neuron_traj
        if data_dir == module.CONTROL_BASE / "seed_2_layer_matched":
            return partial_ids, partial_traj
        return aligned_ids, aligned_traj

    def fake_paired_bootstrap_slope_difference(*args, **kwargs):
        return {
            "slope_difference_pp_per_alpha": {"estimate": 1.0},
            "slope_a_pp_per_alpha": 1.0,
            "slope_b_pp_per_alpha": 0.0,
        }

    existing_dirs = {
        module.CONTROL_BASE / seed_name for seed_name in module.CONTROL_SEEDS
    }

    monkeypatch.setattr(module, "ALPHAS", [0.0, 1.0])
    monkeypatch.setattr(module, "load_trajectories", fake_load_trajectories)
    monkeypatch.setattr(
        module,
        "paired_bootstrap_slope_difference",
        fake_paired_bootstrap_slope_difference,
    )
    monkeypatch.setattr(Path, "exists", lambda path: path in existing_dirs)

    with pytest.raises(
        ValueError,
        match="control trajectories without full item overlap: seed_2_layer_matched",
    ):
        module.compute_neuron_vs_random(
            n_resamples=10,
            seed=42,
            permutation_resamples=20,
        )
