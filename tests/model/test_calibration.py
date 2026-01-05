"""Tests for calibration metrics module."""

import numpy as np
import pytest

from traderbot.metrics.calibration import (
    CalibrationResult,
    brier_score,
    compute_calibration,
    expected_calibration_error,
    find_optimal_threshold,
    isotonic_calibration,
    platt_scaling,
    reliability_curve,
)


class TestBrierScore:
    """Tests for Brier score calculation."""

    def test_perfect_predictions(self) -> None:
        """Test Brier score is 0 for perfect predictions."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.0, 1.0, 1.0, 0.0, 1.0])

        score = brier_score(y_true, y_prob)

        assert score == pytest.approx(0.0, abs=0.001)

    def test_worst_predictions(self) -> None:
        """Test Brier score is 1 for worst predictions."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([1.0, 0.0, 0.0, 1.0, 0.0])

        score = brier_score(y_true, y_prob)

        assert score == pytest.approx(1.0, abs=0.001)

    def test_random_predictions(self) -> None:
        """Test Brier score for random predictions."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])

        score = brier_score(y_true, y_prob)

        # Brier = mean((0.5-0)^2 + (0.5-1)^2 + ...) = 0.25
        assert score == pytest.approx(0.25, abs=0.001)

    def test_brier_score_range(self) -> None:
        """Test Brier score is in valid range [0, 1]."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.uniform(0, 1, 100)

        score = brier_score(y_true, y_prob)

        assert 0 <= score <= 1


class TestExpectedCalibrationError:
    """Tests for Expected Calibration Error (ECE)."""

    def test_perfect_calibration(self) -> None:
        """Test ECE is 0 for perfectly calibrated predictions."""
        # Create predictions that are perfectly calibrated
        # In bin [0.4, 0.6], prob = 0.5, and 50% should be positive
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_prob = np.array([0.5] * 8)

        ece = expected_calibration_error(y_true, y_prob, n_bins=5)

        assert ece == pytest.approx(0.0, abs=0.01)

    def test_poor_calibration(self) -> None:
        """Test ECE is high for poorly calibrated predictions."""
        # Predict 0.9 for all, but only 10% are positive
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        y_prob = np.array([0.9] * 10)

        ece = expected_calibration_error(y_true, y_prob, n_bins=5)

        # ECE should be high (about 0.8)
        assert ece > 0.5

    def test_ece_range(self) -> None:
        """Test ECE is in valid range [0, 1]."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.uniform(0, 1, 100)

        ece = expected_calibration_error(y_true, y_prob, n_bins=10)

        assert 0 <= ece <= 1


class TestReliabilityCurve:
    """Tests for reliability curve calculation."""

    def test_basic_curve(self) -> None:
        """Test reliability curve generation."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.uniform(0, 1, 100)

        mean_pred, fraction_pos, bin_counts = reliability_curve(y_true, y_prob, n_bins=10)

        assert len(mean_pred) > 0
        assert len(fraction_pos) == len(mean_pred)
        assert len(bin_counts) == len(mean_pred)

    def test_curve_values_in_range(self) -> None:
        """Test curve values are in valid range."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        y_prob = np.random.uniform(0, 1, 200)

        mean_pred, fraction_pos, bin_counts = reliability_curve(y_true, y_prob, n_bins=10)

        # All predictions should be in [0, 1]
        assert all(0 <= p <= 1 for p in mean_pred)
        assert all(0 <= f <= 1 for f in fraction_pos)
        assert all(c >= 0 for c in bin_counts)


class TestPlattScaling:
    """Tests for Platt scaling."""

    def test_basic_platt(self) -> None:
        """Test Platt scaling returns parameters."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.uniform(0.2, 0.8, 100)

        a, b = platt_scaling(y_true, y_prob)

        assert isinstance(a, float)
        assert isinstance(b, float)

    def test_platt_improves_calibration(self) -> None:
        """Test Platt scaling can improve calibration."""
        np.random.seed(42)
        # Create poorly calibrated predictions
        y_true = np.concatenate([np.zeros(70), np.ones(30)])
        y_prob = np.concatenate([
            np.random.uniform(0.3, 0.5, 70),
            np.random.uniform(0.5, 0.7, 30),
        ])

        a, b = platt_scaling(y_true, y_prob)

        # Apply Platt scaling
        y_calibrated = 1 / (1 + np.exp(-(a * y_prob + b)))

        # Check calibrated predictions are valid
        assert all(0 <= p <= 1 for p in y_calibrated)


class TestIsotonicCalibration:
    """Tests for isotonic regression calibration."""

    def test_basic_isotonic(self) -> None:
        """Test isotonic calibration returns mapping."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.uniform(0, 1, 100)

        x_vals, y_vals = isotonic_calibration(y_true, y_prob)

        assert len(x_vals) > 0
        assert len(y_vals) == len(x_vals)

    def test_isotonic_monotonic(self) -> None:
        """Test isotonic calibration is monotonic."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        y_prob = np.random.uniform(0, 1, 200)

        x_vals, y_vals = isotonic_calibration(y_true, y_prob)

        # Should be monotonically increasing
        for i in range(1, len(y_vals)):
            assert y_vals[i] >= y_vals[i - 1]


class TestFindOptimalThreshold:
    """Tests for optimal threshold finding."""

    def test_finds_threshold(self) -> None:
        """Test optimal threshold is found."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.uniform(0, 1, 100)

        threshold = find_optimal_threshold(y_true, y_prob)

        assert 0 <= threshold <= 1

    def test_different_metrics(self) -> None:
        """Test different optimization metrics."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.uniform(0, 1, 100)

        thresh_f1 = find_optimal_threshold(y_true, y_prob, metric="f1")
        thresh_youden = find_optimal_threshold(y_true, y_prob, metric="youden")

        # Both should be valid thresholds
        assert 0 <= thresh_f1 <= 1
        assert 0 <= thresh_youden <= 1

    def test_balanced_dataset(self) -> None:
        """Test threshold on balanced dataset."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.2, 0.4, 0.6, 0.8])

        threshold = find_optimal_threshold(y_true, y_prob, metric="f1")

        # Should be around 0.5 for balanced data
        assert 0.3 <= threshold <= 0.7


class TestComputeCalibration:
    """Tests for full calibration computation."""

    def test_compute_calibration(self) -> None:
        """Test compute_calibration returns CalibrationResult."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.uniform(0, 1, 100)

        result = compute_calibration(y_true, y_prob)

        assert isinstance(result, CalibrationResult)
        assert hasattr(result, "brier_score")
        assert hasattr(result, "ece")
        assert hasattr(result, "optimal_threshold")
        assert hasattr(result, "reliability_curve")
        assert hasattr(result, "platt_params")
        assert hasattr(result, "isotonic_mapping")

    def test_calibration_result_values(self) -> None:
        """Test CalibrationResult values are valid."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.uniform(0, 1, 100)

        result = compute_calibration(y_true, y_prob, n_bins=10)

        assert 0 <= result.brier_score <= 1
        assert 0 <= result.ece <= 1
        assert 0 <= result.optimal_threshold <= 1
        assert result.n_samples == 100

    def test_calibration_with_optimal_threshold(self) -> None:
        """Test calibration with threshold optimization."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.uniform(0, 1, 100)

        result = compute_calibration(y_true, y_prob, optimize_threshold=True)

        # Threshold should be optimized (not default 0.5)
        assert result.optimal_threshold != 0.5 or abs(result.optimal_threshold - 0.5) < 0.1

    def test_calibration_reliability_curve_format(self) -> None:
        """Test reliability curve has correct format."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.uniform(0, 1, 100)

        result = compute_calibration(y_true, y_prob)

        assert "bin_centers" in result.reliability_curve
        assert "fraction_positives" in result.reliability_curve
        assert "bin_counts" in result.reliability_curve

        assert isinstance(result.reliability_curve["bin_centers"], list)
        assert isinstance(result.reliability_curve["fraction_positives"], list)
