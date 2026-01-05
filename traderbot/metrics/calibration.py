"""Probability calibration metrics and utilities.

Provides:
- Reliability curve (calibration curve)
- Brier score
- Expected Calibration Error (ECE)
- Platt scaling
- Isotonic regression calibration
- Optimal threshold finding
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import optimize
from scipy.interpolate import interp1d

from traderbot.config import get_config
from traderbot.logging_setup import get_logger

logger = get_logger("metrics.calibration")


@dataclass
class CalibrationResult:
    """Result of calibration analysis."""

    brier_score: float
    ece: float
    optimal_threshold: float
    reliability_curve: dict[str, list[float]]
    n_samples: int
    platt_params: dict[str, float] = field(default_factory=dict)
    isotonic_mapping: dict[str, list[float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "brier_score": self.brier_score,
            "ece": self.ece,
            "optimal_threshold": self.optimal_threshold,
            "reliability_curve": self.reliability_curve,
            "n_samples": self.n_samples,
            "platt_params": self.platt_params,
            "isotonic_mapping": self.isotonic_mapping,
        }


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute reliability (calibration) curve.

    Groups predictions into bins by predicted probability,
    then computes actual positive rate in each bin.

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_prob: Predicted probabilities [0, 1].
        n_bins: Number of bins. Defaults to config.

    Returns:
        Tuple of (bin_centers, fraction_positives, bin_counts).
    """
    config = get_config()
    n_bins = n_bins or config.calibration.n_bins

    y_true = np.asarray(y_true).flatten()
    y_prob = np.asarray(y_prob).flatten()

    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fraction_positives = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        # Find samples in this bin
        if i == n_bins - 1:
            # Last bin includes right edge
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        else:
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])

        bin_counts[i] = np.sum(mask)

        if bin_counts[i] > 0:
            fraction_positives[i] = np.mean(y_true[mask])
        else:
            fraction_positives[i] = np.nan

    return bin_centers, fraction_positives, bin_counts


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier score (mean squared error of probabilities).

    Brier score ranges from 0 (perfect) to 1 (worst).

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_prob: Predicted probabilities [0, 1].

    Returns:
        Brier score.
    """
    y_true = np.asarray(y_true).flatten()
    y_prob = np.asarray(y_prob).flatten()

    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int | None = None,
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE is the weighted average of |accuracy - confidence| across bins.

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_prob: Predicted probabilities [0, 1].
        n_bins: Number of bins. Defaults to config.

    Returns:
        ECE value.
    """
    bin_centers, fraction_positives, bin_counts = reliability_curve(y_true, y_prob, n_bins)

    total_samples = np.sum(bin_counts)
    if total_samples == 0:
        return 0.0

    ece = 0.0
    for i in range(len(bin_centers)):
        if bin_counts[i] > 0:
            # |fraction_positives - bin_center| weighted by bin count
            calibration_error = abs(fraction_positives[i] - bin_centers[i])
            ece += (bin_counts[i] / total_samples) * calibration_error

    return float(ece)


def platt_scaling(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, float]:
    """Fit Platt scaling parameters.

    Platt scaling fits a sigmoid: P(y=1|f) = 1 / (1 + exp(A*f + B))
    where f = log(p / (1-p)) is the logit of the original probability.

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_prob: Predicted probabilities [0, 1].

    Returns:
        Tuple of (A, B) parameters for sigmoid calibration.
    """
    y_true = np.asarray(y_true).flatten()
    y_prob = np.asarray(y_prob).flatten()

    # Clip probabilities to avoid log(0)
    eps = 1e-7
    y_prob = np.clip(y_prob, eps, 1 - eps)

    # Convert to logits
    logits = np.log(y_prob / (1 - y_prob))

    # Fit sigmoid: minimize negative log-likelihood
    def neg_log_likelihood(params: tuple[float, float]) -> float:
        a, b = params
        p = 1 / (1 + np.exp(a * logits + b))
        p = np.clip(p, eps, 1 - eps)
        ll = y_true * np.log(p) + (1 - y_true) * np.log(1 - p)
        return -np.sum(ll)

    # Initialize with identity mapping
    result = optimize.minimize(
        neg_log_likelihood,
        x0=[1.0, 0.0],
        method="L-BFGS-B",
    )

    return float(result.x[0]), float(result.x[1])


def apply_platt_scaling(
    y_prob: np.ndarray,
    a: float,
    b: float,
) -> np.ndarray:
    """Apply Platt scaling calibration.

    Args:
        y_prob: Original predicted probabilities.
        a: Platt scaling A parameter.
        b: Platt scaling B parameter.

    Returns:
        Calibrated probabilities.
    """
    eps = 1e-7
    y_prob = np.clip(y_prob, eps, 1 - eps)
    logits = np.log(y_prob / (1 - y_prob))
    calibrated = 1 / (1 + np.exp(a * logits + b))
    return calibrated


def isotonic_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit isotonic regression calibration.

    Isotonic regression finds a monotonically increasing mapping
    from predicted probabilities to calibrated probabilities.

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_prob: Predicted probabilities [0, 1].

    Returns:
        Tuple of (x_values, y_values) for the isotonic mapping.
    """
    y_true = np.asarray(y_true).flatten()
    y_prob = np.asarray(y_prob).flatten()

    # Sort by predicted probability
    order = np.argsort(y_prob)
    y_prob_sorted = y_prob[order]
    y_true_sorted = y_true[order]

    # Pool Adjacent Violators Algorithm (PAVA)
    n = len(y_true_sorted)
    weights = np.ones(n)
    values = y_true_sorted.astype(float)

    # Merge pools that violate monotonicity
    i = 0
    while i < n - 1:
        if values[i] > values[i + 1]:
            # Merge pools
            merged_value = (weights[i] * values[i] + weights[i + 1] * values[i + 1]) / (
                weights[i] + weights[i + 1]
            )
            merged_weight = weights[i] + weights[i + 1]

            values[i] = merged_value
            weights[i] = merged_weight

            # Remove i+1
            values = np.delete(values, i + 1)
            weights = np.delete(weights, i + 1)
            y_prob_sorted = np.delete(y_prob_sorted, i + 1)
            n -= 1

            # Check previous pool
            if i > 0:
                i -= 1
        else:
            i += 1

    return y_prob_sorted, values


def apply_isotonic_calibration(
    y_prob: np.ndarray,
    x_mapping: np.ndarray,
    y_mapping: np.ndarray,
) -> np.ndarray:
    """Apply isotonic regression calibration.

    Args:
        y_prob: Original predicted probabilities.
        x_mapping: X values from isotonic fit.
        y_mapping: Y values from isotonic fit.

    Returns:
        Calibrated probabilities.
    """
    # Create interpolation function
    if len(x_mapping) < 2:
        return y_prob

    # Extend to [0, 1] range
    x_extended = np.concatenate([[0], x_mapping, [1]])
    y_extended = np.concatenate([[y_mapping[0]], y_mapping, [y_mapping[-1]]])

    interp_func = interp1d(
        x_extended,
        y_extended,
        kind="linear",
        bounds_error=False,
        fill_value=(y_mapping[0], y_mapping[-1]),
    )

    return interp_func(y_prob)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> float:
    """Find optimal probability threshold for classification.

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_prob: Predicted probabilities [0, 1].
        metric: Metric to optimize ("f1", "accuracy", "youden").

    Returns:
        Optimal threshold value.
    """
    y_true = np.asarray(y_true).flatten()
    y_prob = np.asarray(y_prob).flatten()

    # Try thresholds from unique probability values
    thresholds = np.unique(y_prob)
    if len(thresholds) > 100:
        thresholds = np.linspace(0, 1, 101)

    best_threshold = 0.5
    best_score = -np.inf

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        if metric == "f1":
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            if precision + recall > 0:
                score = 2 * precision * recall / (precision + recall)
            else:
                score = 0

        elif metric == "accuracy":
            score = (tp + tn) / len(y_true)

        elif metric == "youden":
            # Youden's J statistic = sensitivity + specificity - 1
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = sensitivity + specificity - 1

        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return float(best_threshold)


def compute_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int | None = None,
    optimize_threshold: bool = False,
) -> CalibrationResult:
    """Compute comprehensive calibration metrics.

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_prob: Predicted probabilities [0, 1].
        n_bins: Number of bins for reliability curve.
        optimize_threshold: Whether to find optimal threshold.

    Returns:
        CalibrationResult with all metrics.
    """
    config = get_config()
    n_bins = n_bins or config.calibration.n_bins
    optimize_threshold = optimize_threshold or config.calibration.opt_threshold

    y_true = np.asarray(y_true).flatten()
    y_prob = np.asarray(y_prob).flatten()

    # Basic metrics
    brier = brier_score(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob, n_bins)

    # Reliability curve
    bin_centers, fraction_positives, bin_counts = reliability_curve(y_true, y_prob, n_bins)

    reliability = {
        "bin_centers": bin_centers.tolist(),
        "fraction_positives": [float(x) if not np.isnan(x) else None for x in fraction_positives],
        "bin_counts": bin_counts.tolist(),
    }

    # Optimal threshold
    if optimize_threshold:
        opt_threshold = find_optimal_threshold(y_true, y_prob, metric="f1")
    else:
        opt_threshold = config.calibration.proba_threshold

    # Platt scaling
    try:
        a, b = platt_scaling(y_true, y_prob)
        platt_params = {"a": a, "b": b}
    except Exception as e:
        logger.warning(f"Platt scaling failed: {e}")
        platt_params = {}

    # Isotonic calibration
    try:
        x_iso, y_iso = isotonic_calibration(y_true, y_prob)
        isotonic_mapping = {
            "x": x_iso.tolist(),
            "y": y_iso.tolist(),
        }
    except Exception as e:
        logger.warning(f"Isotonic calibration failed: {e}")
        isotonic_mapping = {}

    return CalibrationResult(
        brier_score=brier,
        ece=ece,
        optimal_threshold=opt_threshold,
        reliability_curve=reliability,
        n_samples=len(y_true),
        platt_params=platt_params,
        isotonic_mapping=isotonic_mapping,
    )
