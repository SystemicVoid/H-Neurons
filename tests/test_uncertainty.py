import numpy as np
import pytest

from scripts.uncertainty import (
    paired_bootstrap_binary_rate_difference,
    paired_bootstrap_curve_effects,
    paired_bootstrap_slope_difference,
    stratified_bootstrap_classifier_metrics,
    wilson_interval,
)


def test_wilson_interval_is_bounded_and_contains_point_estimate():
    interval = wilson_interval(7, 10)
    point_estimate = 0.7

    assert 0.0 <= interval.lower <= point_estimate <= interval.upper <= 1.0


def test_stratified_bootstrap_classifier_metrics_is_reproducible():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.05, 0.20, 0.65, 0.60, 0.75, 0.95])

    first = stratified_bootstrap_classifier_metrics(
        y_true,
        y_prob,
        n_resamples=200,
        seed=7,
    )
    second = stratified_bootstrap_classifier_metrics(
        y_true,
        y_prob,
        n_resamples=200,
        seed=7,
    )

    assert first == second
    assert first["metrics"]["accuracy"]["estimate"] == 5 / 6
    assert first["metrics"]["balanced_accuracy"]["estimate"] == pytest.approx(5 / 6)


def test_paired_bootstrap_curve_effects_reports_positive_delta_for_increasing_curve():
    trajectories = np.array(
        [
            [0, 0, 1],
            [0, 1, 1],
            [0, 0, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    )
    alphas = np.array([0.0, 1.0, 2.0])

    summary = paired_bootstrap_curve_effects(
        trajectories,
        alphas,
        n_resamples=500,
        seed=3,
    )

    assert summary["delta_0_to_max_pp"]["estimate"] > 0
    assert summary["slope_pp_per_alpha"]["estimate"] > 0


def test_noop_to_max_delta_uses_correct_baseline():
    trajectories = np.array(
        [
            [0, 0, 1],
            [0, 1, 1],
            [0, 0, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    )
    alphas = np.array([0.0, 1.0, 2.0])

    summary = paired_bootstrap_curve_effects(
        trajectories,
        alphas,
        noop_alpha=1.0,
        n_resamples=500,
        seed=3,
    )

    # noop-to-max should equal (rate at alpha=2) - (rate at alpha=1)
    rates = trajectories.mean(axis=0)
    expected = (rates[2] - rates[1]) * 100.0
    assert summary["delta_noop_to_max_pp"]["estimate"] == pytest.approx(expected)
    assert summary["delta_noop_to_max_pp"]["noop_alpha"] == 1.0
    # noop-to-max must be smaller than 0-to-max (ablation recovery excluded)
    assert (
        summary["delta_noop_to_max_pp"]["estimate"]
        < summary["delta_0_to_max_pp"]["estimate"]
    )
    ci = summary["delta_noop_to_max_pp"]["ci"]
    assert ci["lower"] <= summary["delta_noop_to_max_pp"]["estimate"] <= ci["upper"]


def test_without_noop_alpha_omits_field():
    trajectories = np.array([[0, 1], [1, 1]], dtype=bool)
    alphas = np.array([0.0, 1.0])

    summary = paired_bootstrap_curve_effects(
        trajectories, alphas, n_resamples=100, seed=1
    )

    assert "delta_noop_to_max_pp" not in summary


def test_invalid_noop_alpha_raises():
    trajectories = np.array([[0, 1], [1, 1]], dtype=bool)
    alphas = np.array([0.0, 1.0])

    with pytest.raises(ValueError, match="noop_alpha=99.0"):
        paired_bootstrap_curve_effects(
            trajectories, alphas, noop_alpha=99.0, n_resamples=100, seed=1
        )


def test_paired_bootstrap_binary_rate_difference_reports_expected_point_estimate():
    baseline = np.array([0, 0, 1, 1], dtype=bool)
    comparison = np.array([0, 1, 1, 1], dtype=bool)

    summary = paired_bootstrap_binary_rate_difference(
        baseline,
        comparison,
        n_resamples=500,
        seed=11,
    )

    assert summary["estimate_pp"] == 25.0
    assert (
        summary["ci_pp"]["lower"] <= summary["estimate_pp"] <= summary["ci_pp"]["upper"]
    )


def test_slope_difference_detects_known_divergence():
    # Condition A has increasing compliance; condition B is flat
    rng = np.random.default_rng(99)
    n = 200
    alphas = np.array([0.0, 1.0, 2.0, 3.0])
    traj_a = np.zeros((n, 4), dtype=bool)
    traj_b = np.zeros((n, 4), dtype=bool)
    for j, a in enumerate(alphas):
        traj_a[:, j] = rng.random(n) < (0.5 + 0.1 * a)
        traj_b[:, j] = rng.random(n) < 0.5

    result = paired_bootstrap_slope_difference(
        traj_a, traj_b, alphas, n_resamples=500, seed=7
    )

    diff = result["slope_difference_pp_per_alpha"]
    assert diff["estimate"] > 0
    assert diff["ci"]["lower"] > 0  # CI should exclude zero


def test_slope_difference_near_zero_for_identical_conditions():
    rng = np.random.default_rng(42)
    n = 100
    alphas = np.array([0.0, 1.0, 2.0])
    traj = (rng.random((n, 3)) < 0.5).astype(bool)

    result = paired_bootstrap_slope_difference(
        traj, traj, alphas, n_resamples=500, seed=3
    )

    assert result["slope_difference_pp_per_alpha"]["estimate"] == pytest.approx(0.0)


def test_slope_difference_permutation_test_null():
    # Identical conditions → permutation p should be large
    rng = np.random.default_rng(42)
    n = 80
    alphas = np.array([0.0, 1.0, 2.0])
    traj = (rng.random((n, 3)) < 0.5).astype(bool)

    result = paired_bootstrap_slope_difference(
        traj, traj, alphas, n_resamples=200, seed=5, permutation_resamples=500
    )

    assert result["permutation_test"]["p_value"] >= 0.3


def test_slope_difference_shape_mismatch_raises():
    traj_a = np.ones((10, 3), dtype=bool)
    traj_b = np.ones((8, 3), dtype=bool)
    alphas = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="same shape"):
        paired_bootstrap_slope_difference(traj_a, traj_b, alphas)
