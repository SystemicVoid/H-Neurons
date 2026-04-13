from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import norm
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


DEFAULT_CONFIDENCE = 0.95
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 42


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float
    level: float
    method: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower": float(self.lower),
            "upper": float(self.upper),
            "level": float(self.level),
            "method": self.method,
        }


def percentile_interval(
    samples: np.ndarray,
    confidence: float = DEFAULT_CONFIDENCE,
    *,
    method: str,
) -> Interval:
    alpha = 1.0 - confidence
    lower, upper = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return Interval(
        lower=float(lower), upper=float(upper), level=confidence, method=method
    )


def wilson_interval(
    successes: int,
    total: int,
    confidence: float = DEFAULT_CONFIDENCE,
) -> Interval:
    if total < 0:
        raise ValueError("total must be non-negative")
    if successes < 0 or successes > total:
        raise ValueError("successes must be between 0 and total")
    if total == 0:
        return Interval(lower=0.0, upper=0.0, level=confidence, method="wilson")

    p = successes / total
    z = norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    z2 = z**2
    denom = 1.0 + z2 / total
    centre = p + z2 / (2.0 * total)
    radius = z * np.sqrt((p * (1.0 - p) / total) + (z2 / (4.0 * total**2)))
    lower = max(0.0, (centre - radius) / denom)
    upper = min(1.0, (centre + radius) / denom)
    return Interval(
        lower=float(lower), upper=float(upper), level=confidence, method="wilson"
    )


def build_rate_summary(
    successes: int,
    total: int,
    confidence: float = DEFAULT_CONFIDENCE,
    *,
    method: str = "wilson",
    count_key: str = "count",
    total_key: str = "n",
) -> dict[str, Any]:
    if total == 0:
        estimate = 0.0
        interval = Interval(lower=0.0, upper=0.0, level=confidence, method=method)
    else:
        estimate = successes / total
        if method != "wilson":
            raise ValueError(f"Unsupported rate CI method: {method}")
        interval = wilson_interval(successes, total, confidence)
    return {
        "estimate": float(estimate),
        count_key: int(successes),
        total_key: int(total),
        "ci": interval.to_dict(),
    }


def classifier_metric_values(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(roc_auc_score(y_true, y_prob)),
    }


def stratified_bootstrap_classifier_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
    confidence: float = DEFAULT_CONFIDENCE,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have matching shapes")

    y_pred = (y_prob >= threshold).astype(int)
    points = classifier_metric_values(y_true, y_pred, y_prob)

    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError(
            "Classifier bootstrap requires both positive and negative labels"
        )

    rng = np.random.default_rng(seed)
    metric_samples = {name: np.empty(n_resamples, dtype=float) for name in points}

    for sample_idx in range(n_resamples):
        resampled_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        resampled_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        indices = np.concatenate([resampled_pos, resampled_neg])
        sample_true = y_true[indices]
        sample_prob = y_prob[indices]
        sample_pred = (sample_prob >= threshold).astype(int)
        sample_metrics = classifier_metric_values(sample_true, sample_pred, sample_prob)
        for name, value in sample_metrics.items():
            metric_samples[name][sample_idx] = value

    metrics: dict[str, Any] = {}
    for name, point in points.items():
        interval = percentile_interval(
            metric_samples[name],
            confidence,
            method="bootstrap_percentile_stratified",
        )
        metrics[name] = {
            "estimate": float(point),
            "ci": interval.to_dict(),
        }

    return {
        "threshold": float(threshold),
        "bootstrap": {
            "n_resamples": int(n_resamples),
            "seed": int(seed),
            "confidence": float(confidence),
            "resampling": "stratified_by_label",
            "interval": "percentile",
        },
        "metrics": metrics,
    }


def paired_bootstrap_binary_rate_difference(
    baseline: np.ndarray,
    comparison: np.ndarray,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    baseline = np.asarray(baseline, dtype=bool)
    comparison = np.asarray(comparison, dtype=bool)
    if baseline.shape != comparison.shape:
        raise ValueError("baseline and comparison must have matching shapes")
    if baseline.ndim != 1:
        raise ValueError("baseline and comparison must be one-dimensional")

    n_items = len(baseline)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=float)
    for sample_idx in range(n_resamples):
        indices = rng.choice(n_items, size=n_items, replace=True)
        samples[sample_idx] = (
            comparison[indices].mean() - baseline[indices].mean()
        ) * 100.0

    interval = percentile_interval(
        samples,
        confidence,
        method="bootstrap_percentile_paired",
    )
    return {
        "estimate_pp": float((comparison.mean() - baseline.mean()) * 100.0),
        "ci_pp": interval.to_dict(),
        "bootstrap": {
            "n_resamples": int(n_resamples),
            "seed": int(seed),
            "confidence": float(confidence),
            "resampling": "paired_by_sample_id",
            "interval": "percentile",
        },
    }


def paired_bootstrap_continuous_mean_difference(
    baseline: np.ndarray,
    comparison: np.ndarray,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    baseline = np.asarray(baseline, dtype=float)
    comparison = np.asarray(comparison, dtype=float)
    if baseline.shape != comparison.shape:
        raise ValueError("baseline and comparison must have matching shapes")
    if baseline.ndim != 1:
        raise ValueError("baseline and comparison must be one-dimensional")

    n_items = len(baseline)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=float)
    for sample_idx in range(n_resamples):
        indices = rng.choice(n_items, size=n_items, replace=True)
        samples[sample_idx] = (
            comparison[indices].mean() - baseline[indices].mean()
        ) * 100.0

    interval = percentile_interval(
        samples,
        confidence,
        method="bootstrap_percentile_paired_continuous",
    )
    return {
        "estimate_pp": float((comparison.mean() - baseline.mean()) * 100.0),
        "ci_pp": interval.to_dict(),
        "bootstrap": {
            "n_resamples": int(n_resamples),
            "seed": int(seed),
            "confidence": float(confidence),
            "resampling": "paired_by_sample_id",
            "interval": "percentile",
        },
    }


def paired_bootstrap_curve_effects(
    trajectories: np.ndarray,
    alphas: np.ndarray,
    *,
    noop_alpha: float | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    trajectories = np.asarray(trajectories, dtype=bool)
    alphas = np.asarray(alphas, dtype=float)
    if trajectories.ndim != 2:
        raise ValueError(
            "trajectories must be a 2D array of shape (n_samples, n_alphas)"
        )
    if trajectories.shape[1] != len(alphas):
        raise ValueError("trajectories second dimension must match alphas length")

    noop_idx: int | None = None
    if noop_alpha is not None:
        matches = np.flatnonzero(np.isclose(alphas, noop_alpha))
        if len(matches) != 1:
            raise ValueError(
                f"noop_alpha={noop_alpha} must match exactly one entry in alphas; "
                f"found {len(matches)} matches"
            )
        noop_idx = int(matches[0])

    rates = trajectories.mean(axis=0)
    slope = np.polyfit(alphas, rates * 100.0, 1)[0]
    delta = (rates[-1] - rates[0]) * 100.0
    noop_delta = (rates[-1] - rates[noop_idx]) * 100.0 if noop_idx is not None else None

    n_samples = trajectories.shape[0]
    rng = np.random.default_rng(seed)
    slope_samples = np.empty(n_resamples, dtype=float)
    delta_samples = np.empty(n_resamples, dtype=float)
    noop_delta_samples = (
        np.empty(n_resamples, dtype=float) if noop_idx is not None else None
    )
    for sample_idx in range(n_resamples):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        sample_rates = trajectories[indices].mean(axis=0)
        slope_samples[sample_idx] = np.polyfit(alphas, sample_rates * 100.0, 1)[0]
        delta_samples[sample_idx] = (sample_rates[-1] - sample_rates[0]) * 100.0
        if noop_delta_samples is not None:
            noop_delta_samples[sample_idx] = (
                sample_rates[-1] - sample_rates[noop_idx]
            ) * 100.0

    result = {
        "rates": rates.tolist(),
        "delta_0_to_max_pp": {
            "estimate": float(delta),
            "ci": percentile_interval(
                delta_samples,
                confidence,
                method="bootstrap_percentile_paired",
            ).to_dict(),
        },
        "slope_pp_per_alpha": {
            "estimate": float(slope),
            "ci": percentile_interval(
                slope_samples,
                confidence,
                method="bootstrap_percentile_paired",
            ).to_dict(),
        },
        "bootstrap": {
            "n_resamples": int(n_resamples),
            "seed": int(seed),
            "confidence": float(confidence),
            "resampling": "paired_by_sample_id",
            "interval": "percentile",
        },
    }

    if noop_alpha is not None:
        # noop_idx, noop_delta, noop_delta_samples are guaranteed set when noop_alpha is
        assert noop_delta is not None and noop_delta_samples is not None
        result["delta_noop_to_max_pp"] = {
            "estimate": float(noop_delta),
            "noop_alpha": float(noop_alpha),
            "ci": percentile_interval(
                noop_delta_samples,
                confidence,
                method="bootstrap_percentile_paired",
            ).to_dict(),
        }

    return result


def paired_bootstrap_slope_difference(
    trajectories_a: np.ndarray,
    trajectories_b: np.ndarray,
    alphas: np.ndarray,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    permutation_resamples: int = 0,
) -> dict[str, Any]:
    """Bootstrap CI on the slope difference (A minus B) for paired trajectory data.

    Both trajectory matrices must have the same items in the same order,
    enabling paired resampling that preserves item-level covariance.

    When *permutation_resamples* > 0, a permutation test shuffles condition
    labels per item to build a null distribution of slope differences.
    """
    trajectories_a = np.asarray(trajectories_a, dtype=bool)
    trajectories_b = np.asarray(trajectories_b, dtype=bool)
    alphas = np.asarray(alphas, dtype=float)

    if trajectories_a.shape != trajectories_b.shape:
        raise ValueError("trajectories_a and trajectories_b must have the same shape")
    if trajectories_a.ndim != 2:
        raise ValueError("trajectories must be 2D arrays (n_samples, n_alphas)")
    if trajectories_a.shape[1] != len(alphas):
        raise ValueError("trajectories second dimension must match alphas length")

    n_samples = trajectories_a.shape[0]

    # Direct OLS on the difference trajectory: slope(a-b) == slope(a) - slope(b)
    diff_traj = trajectories_a.astype(float) - trajectories_b.astype(float)
    x_centered = alphas - alphas.mean()
    ss_x = float((x_centered**2).sum())

    def _ols_slope(rates_pp: np.ndarray) -> float:
        return float((x_centered * (rates_pp - rates_pp.mean())).sum() / ss_x)

    # Point estimates
    rates_a = trajectories_a.mean(axis=0) * 100.0
    rates_b = trajectories_b.mean(axis=0) * 100.0
    slope_a = _ols_slope(rates_a)
    slope_b = _ols_slope(rates_b)
    diff_rates = diff_traj.mean(axis=0) * 100.0
    observed_diff = _ols_slope(diff_rates)

    # Paired bootstrap
    rng = np.random.default_rng(seed)
    diff_samples = np.empty(n_resamples, dtype=float)
    for sample_idx in range(n_resamples):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        dr = diff_traj[idx].mean(axis=0) * 100.0
        diff_samples[sample_idx] = _ols_slope(dr)

    ci = percentile_interval(
        diff_samples, confidence, method="bootstrap_percentile_paired"
    )

    result: dict[str, Any] = {
        "slope_a_pp_per_alpha": float(slope_a),
        "slope_b_pp_per_alpha": float(slope_b),
        "slope_difference_pp_per_alpha": {
            "estimate": float(observed_diff),
            "ci": ci.to_dict(),
        },
        "bootstrap": {
            "n_resamples": int(n_resamples),
            "seed": int(seed),
            "confidence": float(confidence),
            "resampling": "paired_by_sample_id",
            "interval": "percentile",
        },
    }

    # Permutation test: flip condition labels per item via sign flip on diff
    if permutation_resamples > 0:
        perm_rng = np.random.default_rng(seed + 1)
        n_extreme = 0
        for _ in range(permutation_resamples):
            signs = np.where(perm_rng.random(n_samples) < 0.5, -1.0, 1.0)
            dr = (signs[:, None] * diff_traj).mean(axis=0) * 100.0
            if _ols_slope(dr) >= observed_diff:
                n_extreme += 1
        result["permutation_test"] = {
            "p_value": float((n_extreme + 1) / (permutation_resamples + 1)),
            "n_extreme": int(n_extreme),
            "n_permutations": int(permutation_resamples),
            "alternative": "one_sided_greater",
        }

    return result
