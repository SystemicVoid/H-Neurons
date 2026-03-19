import numpy as np
import pytest

from scripts.uncertainty import (
    paired_bootstrap_binary_rate_difference,
    paired_bootstrap_curve_effects,
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
