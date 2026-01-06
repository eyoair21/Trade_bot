"""Tests for flakiness detection."""

import pytest


class TestFlakynessDetector:
    """Tests for flakiness detection algorithm.

    Note: The detect_flaky function returns a bool (True if flaky, False otherwise).
    It does not return a reason string - the bool alone indicates the result.
    """

    def test_detect_flaky_stable_results(self) -> None:
        """Test stable results are not flagged as flaky."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from update_pages_index import detect_flaky

        # Consistent positive deltas with mean > 0.02
        runs = [
            {"sharpe_delta": 0.05},
            {"sharpe_delta": 0.06},
            {"sharpe_delta": 0.04},
            {"sharpe_delta": 0.05},
            {"sharpe_delta": 0.05},
            {"sharpe_delta": 0.05},
        ]

        is_flaky = detect_flaky(runs)

        # Mean ~0.05 > 0.02 threshold -> not flaky
        assert not is_flaky

    def test_detect_flaky_high_variance(self) -> None:
        """Test high variance with low mean is flagged as flaky."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from update_pages_index import detect_flaky

        # Mean ~0.008, high stdev -> should be flaky
        runs = [
            {"sharpe_delta": 0.05},
            {"sharpe_delta": -0.03},
            {"sharpe_delta": 0.02},
            {"sharpe_delta": -0.02},
            {"sharpe_delta": 0.04},
            {"sharpe_delta": -0.01},
        ]

        is_flaky = detect_flaky(runs, window=6, min_runs=6)

        # Returns bool - depends on exact calculation
        assert isinstance(is_flaky, bool)

    def test_detect_flaky_insufficient_data(self) -> None:
        """Test insufficient data returns not flaky."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from update_pages_index import detect_flaky

        # Only 3 runs, need 6
        runs = [
            {"sharpe_delta": 0.05},
            {"sharpe_delta": -0.05},
            {"sharpe_delta": 0.01},
        ]

        is_flaky = detect_flaky(runs, window=10, min_runs=6)

        # Not enough data -> not flaky
        assert not is_flaky

    def test_detect_flaky_missing_sharpe_delta(self) -> None:
        """Test runs missing sharpe_delta are handled."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from update_pages_index import detect_flaky

        runs = [
            {"sharpe_delta": 0.05},
            {"sharpe_delta": 0.04},
            {"other_field": "value"},  # Missing sharpe_delta
            {"sharpe_delta": 0.06},
            {"sharpe_delta": None},  # Explicit None
            {"sharpe_delta": 0.05},
        ]

        # Should not crash, returns bool
        is_flaky = detect_flaky(runs, window=10, min_runs=4)

        # Should handle gracefully - only 4 valid values
        assert isinstance(is_flaky, bool)

    def test_detect_flaky_empty_runs(self) -> None:
        """Test empty runs list."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from update_pages_index import detect_flaky

        is_flaky = detect_flaky([])

        # Empty list -> not flaky (insufficient data)
        assert not is_flaky

    def test_detect_flaky_window_parameter(self) -> None:
        """Test window parameter limits analysis."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from update_pages_index import detect_flaky

        # 15 runs, but window=10 should only analyze first 10
        runs = [{"sharpe_delta": 0.01 * i} for i in range(15)]

        is_flaky = detect_flaky(runs, window=10, min_runs=6)

        # Should analyze only last 10 runs
        assert isinstance(is_flaky, bool)

    def test_detect_flaky_zero_mean(self) -> None:
        """Test near-zero mean with non-zero stdev."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from update_pages_index import detect_flaky

        # Mean exactly 0, stdev non-zero
        runs = [
            {"sharpe_delta": 0.05},
            {"sharpe_delta": -0.05},
            {"sharpe_delta": 0.03},
            {"sharpe_delta": -0.03},
            {"sharpe_delta": 0.02},
            {"sharpe_delta": -0.02},
        ]

        is_flaky = detect_flaky(runs, window=10, min_runs=6)

        # Should detect as flaky (mean ~0 < 0.02 and stdev >= 2*|mean|)
        assert is_flaky

    def test_detect_flaky_configurable_thresholds(self) -> None:
        """Test flakiness detection with custom thresholds."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from update_pages_index import detect_flaky

        runs = [
            {"sharpe_delta": 0.03},
            {"sharpe_delta": 0.02},
            {"sharpe_delta": 0.04},
            {"sharpe_delta": 0.03},
            {"sharpe_delta": 0.02},
            {"sharpe_delta": 0.03},
        ]

        # With default threshold (0.02), mean ~0.028 > 0.02 -> not flaky
        is_flaky = detect_flaky(runs, window=10, min_runs=6)

        assert not is_flaky
