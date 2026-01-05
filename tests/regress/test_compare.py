"""Tests for traderbot.metrics.compare module."""

import json
import tempfile
from pathlib import Path

import pytest

from traderbot.metrics.compare import (
    BaselineData,
    CurrentData,
    PerfBudget,
    compare_results,
    create_new_baseline,
    generate_baseline_diff,
    generate_regression_report,
    load_baseline,
    load_current_data,
    load_perf_budget,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def sample_budget() -> PerfBudget:
    """Create a sample performance budget."""
    return PerfBudget(
        metric="sharpe",
        mode="max",
        min_success_rate=0.75,
        max_p90_elapsed_s=60.0,
        max_sharpe_drop=0.05,
        epsilon_abs=1e-6,
    )


@pytest.fixture
def sample_baseline() -> BaselineData:
    """Create a sample baseline."""
    return BaselineData(
        git_sha="abc1234",
        created_utc="2026-01-01T00:00:00+00:00",
        metric="sharpe",
        mode="max",
        timing={"p50": 10.0, "p90": 20.0},
        leaderboard=[
            {"rank": 1, "run_idx": 0, "avg_oos_sharpe": 1.5},
            {"rank": 2, "run_idx": 1, "avg_oos_sharpe": 1.2},
        ],
        summary={
            "best_metric": 1.5,
            "success_rate": 1.0,
            "total_runs": 10,
        },
    )


@pytest.fixture
def sample_current_pass() -> CurrentData:
    """Create current data that should pass comparison."""
    return CurrentData(
        best_metric=1.55,  # Better than baseline
        success_rate=0.9,
        total_runs=10,
        timing={"p50": 8.0, "p90": 18.0},  # Faster than baseline
        leaderboard=[
            {"rank": 1, "run_idx": 2, "avg_oos_sharpe": 1.55},
            {"rank": 2, "run_idx": 3, "avg_oos_sharpe": 1.3},
        ],
    )


@pytest.fixture
def sample_current_fail_metric() -> CurrentData:
    """Create current data that fails metric check."""
    return CurrentData(
        best_metric=1.4,  # Worse than baseline (drop > 0.05)
        success_rate=0.9,
        total_runs=10,
        timing={"p50": 8.0, "p90": 18.0},
        leaderboard=[
            {"rank": 1, "run_idx": 2, "avg_oos_sharpe": 1.4},
        ],
    )


@pytest.fixture
def sample_current_fail_timing() -> CurrentData:
    """Create current data that fails timing check."""
    return CurrentData(
        best_metric=1.55,
        success_rate=0.9,
        total_runs=10,
        timing={"p50": 50.0, "p90": 100.0},  # P90 exceeds budget
        leaderboard=[
            {"rank": 1, "run_idx": 2, "avg_oos_sharpe": 1.55},
        ],
    )


@pytest.fixture
def sample_current_fail_success_rate() -> CurrentData:
    """Create current data that fails success rate check."""
    return CurrentData(
        best_metric=1.55,
        success_rate=0.5,  # Below min_success_rate
        total_runs=10,
        timing={"p50": 8.0, "p90": 18.0},
        leaderboard=[
            {"rank": 1, "run_idx": 2, "avg_oos_sharpe": 1.55},
        ],
    )


# ============================================================
# Test PerfBudget
# ============================================================


class TestPerfBudget:
    """Tests for PerfBudget dataclass."""

    def test_default_values(self):
        """Test default values for PerfBudget."""
        budget = PerfBudget()
        assert budget.metric == "sharpe"
        assert budget.mode == "max"
        assert budget.min_success_rate == 0.75
        assert budget.max_p90_elapsed_s == 60.0
        assert budget.max_sharpe_drop == 0.05
        assert budget.epsilon_abs == 1e-6

    def test_custom_values(self):
        """Test custom values for PerfBudget."""
        budget = PerfBudget(
            metric="total_return",
            mode="min",
            min_success_rate=0.9,
            max_p90_elapsed_s=30.0,
            max_sharpe_drop=0.1,
            epsilon_abs=1e-9,
        )
        assert budget.metric == "total_return"
        assert budget.mode == "min"
        assert budget.min_success_rate == 0.9
        assert budget.max_p90_elapsed_s == 30.0


# ============================================================
# Test BaselineData
# ============================================================


class TestBaselineData:
    """Tests for BaselineData dataclass."""

    def test_creation(self, sample_baseline: BaselineData):
        """Test baseline data creation."""
        assert sample_baseline.git_sha == "abc1234"
        assert sample_baseline.summary["best_metric"] == 1.5
        assert sample_baseline.summary["success_rate"] == 1.0
        assert sample_baseline.timing["p90"] == 20.0


# ============================================================
# Test CurrentData
# ============================================================


class TestCurrentData:
    """Tests for CurrentData dataclass."""

    def test_creation(self, sample_current_pass: CurrentData):
        """Test current data creation."""
        assert sample_current_pass.best_metric == 1.55
        assert sample_current_pass.success_rate == 0.9
        assert len(sample_current_pass.leaderboard) == 2


# ============================================================
# Test compare_results
# ============================================================


class TestCompareResults:
    """Tests for compare_results function."""

    def test_pass_scenario(
        self,
        sample_current_pass: CurrentData,
        sample_baseline: BaselineData,
        sample_budget: PerfBudget,
    ):
        """Test comparison that should pass all checks."""
        verdict = compare_results(sample_current_pass, sample_baseline, sample_budget)
        assert verdict.passed is True
        assert verdict.metric_passed is True
        assert verdict.timing_passed is True
        assert verdict.success_rate_passed is True

    def test_fail_metric_drop(
        self,
        sample_current_fail_metric: CurrentData,
        sample_baseline: BaselineData,
        sample_budget: PerfBudget,
    ):
        """Test comparison that fails metric drop check."""
        verdict = compare_results(
            sample_current_fail_metric, sample_baseline, sample_budget
        )
        assert verdict.passed is False
        assert verdict.metric_passed is False

    def test_fail_timing(
        self,
        sample_current_fail_timing: CurrentData,
        sample_baseline: BaselineData,
        sample_budget: PerfBudget,
    ):
        """Test comparison that fails timing check."""
        verdict = compare_results(
            sample_current_fail_timing, sample_baseline, sample_budget
        )
        assert verdict.passed is False
        assert verdict.timing_passed is False

    def test_fail_success_rate(
        self,
        sample_current_fail_success_rate: CurrentData,
        sample_baseline: BaselineData,
        sample_budget: PerfBudget,
    ):
        """Test comparison that fails success rate check."""
        verdict = compare_results(
            sample_current_fail_success_rate, sample_baseline, sample_budget
        )
        assert verdict.passed is False
        assert verdict.success_rate_passed is False

    def test_multiple_failures(
        self,
        sample_baseline: BaselineData,
        sample_budget: PerfBudget,
    ):
        """Test comparison with multiple failures."""
        current = CurrentData(
            best_metric=1.3,  # Fails metric drop
            success_rate=0.5,  # Fails success rate
            total_runs=10,
            timing={"p50": 50.0, "p90": 100.0},  # Fails timing
            leaderboard=[],
        )
        verdict = compare_results(current, sample_baseline, sample_budget)
        assert verdict.passed is False
        assert verdict.metric_passed is False
        assert verdict.timing_passed is False
        assert verdict.success_rate_passed is False

    def test_verdict_messages(
        self,
        sample_current_pass: CurrentData,
        sample_baseline: BaselineData,
        sample_budget: PerfBudget,
    ):
        """Test that verdict contains appropriate messages."""
        verdict = compare_results(sample_current_pass, sample_baseline, sample_budget)
        assert len(verdict.messages) > 0


# ============================================================
# Test load functions
# ============================================================


class TestLoadFunctions:
    """Tests for load_* functions."""

    def test_load_perf_budget_from_file(self):
        """Test loading performance budget from YAML file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("""
metric: sharpe
mode: max
min_success_rate: 0.8
max_p90_elapsed_s: 45.0
max_sharpe_drop: 0.03
epsilon_abs: 1e-8
""")
            f.flush()
            budget = load_perf_budget(Path(f.name))

        assert budget.metric == "sharpe"
        assert budget.min_success_rate == 0.8
        assert budget.max_p90_elapsed_s == 45.0
        assert budget.max_sharpe_drop == 0.03

    def test_load_baseline_from_file(self):
        """Test loading baseline from JSON file."""
        baseline_data = {
            "git_sha": "test123",
            "created_utc": "2026-01-01T00:00:00+00:00",
            "metric": "sharpe",
            "mode": "max",
            "leaderboard": [{"rank": 1, "avg_oos_sharpe": 1.0}],
            "timing": {"p50": 10.0, "p90": 20.0},
            "summary": {
                "best_metric": 1.0,
                "success_rate": 1.0,
                "total_runs": 1,
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(baseline_data, f)
            f.flush()
            baseline = load_baseline(Path(f.name))

        assert baseline.git_sha == "test123"
        assert baseline.summary["best_metric"] == 1.0
        assert baseline.timing["p90"] == 20.0

    def test_load_current_data_from_sweep(self, tmp_path: Path):
        """Test loading current data from sweep directory."""
        # Create mock sweep output
        all_results = [
            {
                "_status": "success",
                "_run_idx": 0,
                "_elapsed_seconds": 15.0,
                "avg_oos_sharpe": 1.2,
            },
            {
                "_status": "success",
                "_run_idx": 1,
                "_elapsed_seconds": 18.0,
                "avg_oos_sharpe": 1.5,
            },
            {
                "_status": "error",
                "_run_idx": 2,
                "_elapsed_seconds": 5.0,
            },
        ]

        sweep_meta = {"metric": "sharpe", "mode": "max"}

        (tmp_path / "all_results.json").write_text(json.dumps(all_results))
        (tmp_path / "sweep_meta.json").write_text(json.dumps(sweep_meta))

        current = load_current_data(tmp_path)

        # No leaderboard.csv, so best_metric comes from default
        assert current.success_rate == pytest.approx(2 / 3)
        assert current.total_runs == 3


# ============================================================
# Test generate functions
# ============================================================


class TestGenerateFunctions:
    """Tests for generate_* functions."""

    def test_generate_regression_report(
        self,
        sample_current_pass: CurrentData,
        sample_baseline: BaselineData,
        sample_budget: PerfBudget,
    ):
        """Test regression report generation."""
        verdict = compare_results(sample_current_pass, sample_baseline, sample_budget)
        report = generate_regression_report(
            verdict, sample_current_pass, sample_baseline, sample_budget
        )

        assert "Regression Report" in report
        assert "PASS" in report or "FAIL" in report
        assert "sharpe" in report.lower() or "Sharpe" in report

    def test_generate_baseline_diff(
        self,
        sample_current_pass: CurrentData,
        sample_baseline: BaselineData,
        sample_budget: PerfBudget,
    ):
        """Test baseline diff generation."""
        verdict = compare_results(sample_current_pass, sample_baseline, sample_budget)
        diff = generate_baseline_diff(verdict, sample_current_pass, sample_baseline)

        assert "deltas" in diff
        assert "metric" in diff["deltas"]
        assert "passed" in diff


# ============================================================
# Test create_new_baseline
# ============================================================


class TestCreateNewBaseline:
    """Tests for create_new_baseline function."""

    def test_create_baseline(self, sample_current_pass: CurrentData):
        """Test creating new baseline from current data."""
        baseline = create_new_baseline(sample_current_pass, "newsha123")

        assert baseline["git_sha"] == "newsha123"
        assert baseline["summary"]["best_metric"] == sample_current_pass.best_metric
        assert baseline["summary"]["success_rate"] == sample_current_pass.success_rate
        assert "created_utc" in baseline

    def test_baseline_includes_leaderboard(self, sample_current_pass: CurrentData):
        """Test that baseline includes leaderboard."""
        baseline = create_new_baseline(sample_current_pass, "sha456")
        assert "leaderboard" in baseline
        assert len(baseline["leaderboard"]) == len(sample_current_pass.leaderboard)


# ============================================================
# Test with fixture files
# ============================================================


class TestWithFixtureFiles:
    """Tests using fixture files from tests/fixtures/."""

    @pytest.fixture
    def fixtures_dir(self) -> Path:
        """Get path to fixtures directory."""
        return Path(__file__).parent.parent / "fixtures"

    def test_load_baseline_pass_fixture(self, fixtures_dir: Path):
        """Test loading baseline_pass.json fixture."""
        baseline_path = fixtures_dir / "baseline_pass.json"
        if baseline_path.exists():
            baseline = load_baseline(baseline_path)
            assert baseline.git_sha == "abc1234"
            assert baseline.summary["best_metric"] == 1.5

    def test_load_perf_budget_strict_fixture(self, fixtures_dir: Path):
        """Test loading strict performance budget fixture."""
        budget_path = fixtures_dir / "perf_budget_strict.yaml"
        if budget_path.exists():
            budget = load_perf_budget(budget_path)
            assert budget.min_success_rate == 0.99
            assert budget.max_p90_elapsed_s == 5.0

    def test_load_perf_budget_relaxed_fixture(self, fixtures_dir: Path):
        """Test loading relaxed performance budget fixture."""
        budget_path = fixtures_dir / "perf_budget_relaxed.yaml"
        if budget_path.exists():
            budget = load_perf_budget(budget_path)
            assert budget.min_success_rate == 0.5
            assert budget.max_sharpe_drop == 1.0


# ============================================================
# Test edge cases
# ============================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_leaderboard(self, sample_baseline: BaselineData, sample_budget: PerfBudget):
        """Test handling of empty leaderboard."""
        current = CurrentData(
            best_metric=0.0,
            success_rate=0.0,
            total_runs=0,
            timing={"p50": 0.0, "p90": 0.0},
            leaderboard=[],
        )
        verdict = compare_results(current, sample_baseline, sample_budget)
        # Should fail due to success rate
        assert verdict.passed is False

    def test_identical_results(
        self,
        sample_baseline: BaselineData,
        sample_budget: PerfBudget,
    ):
        """Test when current equals baseline exactly."""
        current = CurrentData(
            best_metric=sample_baseline.summary["best_metric"],
            success_rate=sample_baseline.summary["success_rate"],
            total_runs=sample_baseline.summary["total_runs"],
            timing=sample_baseline.timing.copy(),
            leaderboard=sample_baseline.leaderboard.copy(),
        )
        verdict = compare_results(current, sample_baseline, sample_budget)
        assert verdict.passed is True

    def test_epsilon_tolerance(self, sample_baseline: BaselineData):
        """Test epsilon tolerance in metric comparison."""
        budget = PerfBudget(
            max_sharpe_drop=0.0,  # No drop allowed
            epsilon_abs=0.1,  # But with 0.1 tolerance
        )
        # Current is 0.05 below baseline (1.5 - 0.05 = 1.45)
        # Since max_sharpe_drop is 0, this should fail unless epsilon helps
        # But epsilon is for determinism, not metric comparison
        current = CurrentData(
            best_metric=sample_baseline.summary["best_metric"] - 0.05,
            success_rate=1.0,
            total_runs=10,
            timing={"p50": 10.0, "p90": 20.0},
            leaderboard=[],
        )
        verdict = compare_results(current, sample_baseline, budget)
        # Will fail because drop is 0.05 > max_sharpe_drop of 0.0
        assert verdict.metric_passed is False


# ============================================================
# Test malformed input handling
# ============================================================


class TestMalformedInputs:
    """Tests for handling malformed or edge-case inputs."""

    def test_all_results_empty_list(self, tmp_path: Path):
        """Test handling of empty all_results.json."""
        (tmp_path / "all_results.json").write_text("[]")
        (tmp_path / "sweep_meta.json").write_text('{"metric": "sharpe", "mode": "max"}')

        current = load_current_data(tmp_path)
        assert current.total_runs == 0
        # Empty list means 0/0 = defaults to 1.0 (no failures occurred)
        assert current.success_rate == 1.0
        # best_metric should default to 0.0 when no successful runs
        assert current.best_metric == 0.0

    def test_all_results_all_failed_runs(self, tmp_path: Path):
        """Test handling when all runs failed."""
        results = [
            {"_status": "error", "_run_idx": 0, "_elapsed_seconds": 5.0},
            {"_status": "error", "_run_idx": 1, "_elapsed_seconds": 3.0},
        ]
        (tmp_path / "all_results.json").write_text(json.dumps(results))
        (tmp_path / "sweep_meta.json").write_text('{"metric": "sharpe", "mode": "max"}')

        current = load_current_data(tmp_path)
        assert current.total_runs == 2
        assert current.success_rate == 0.0
        # best_metric defaults to 0.0 when no successful runs
        assert current.best_metric == 0.0

    def test_all_results_with_nan_metrics(self, tmp_path: Path):
        """Test handling of NaN metric values in results."""
        import math

        results = [
            {"_status": "success", "_run_idx": 0, "_elapsed_seconds": 10.0, "avg_oos_sharpe": float("nan")},
            {"_status": "success", "_run_idx": 1, "_elapsed_seconds": 12.0, "avg_oos_sharpe": 1.5},
        ]
        (tmp_path / "all_results.json").write_text(json.dumps(results))
        (tmp_path / "sweep_meta.json").write_text('{"metric": "sharpe", "mode": "max"}')

        current = load_current_data(tmp_path)
        # Should handle NaN gracefully - may skip NaN values or use them
        # The exact behavior depends on implementation
        assert current.total_runs == 2
        assert current.success_rate == 1.0

    def test_all_results_with_inf_metrics(self, tmp_path: Path):
        """Test handling of Inf metric values in results."""
        results = [
            {"_status": "success", "_run_idx": 0, "_elapsed_seconds": 10.0, "avg_oos_sharpe": float("inf")},
            {"_status": "success", "_run_idx": 1, "_elapsed_seconds": 12.0, "avg_oos_sharpe": 1.5},
        ]
        (tmp_path / "all_results.json").write_text(json.dumps(results))
        (tmp_path / "sweep_meta.json").write_text('{"metric": "sharpe", "mode": "max"}')

        current = load_current_data(tmp_path)
        assert current.total_runs == 2

    def test_missing_sweep_meta(self, tmp_path: Path):
        """Test handling when sweep_meta.json is missing."""
        results = [
            {"_status": "success", "_run_idx": 0, "_elapsed_seconds": 10.0, "avg_oos_sharpe": 1.2},
        ]
        (tmp_path / "all_results.json").write_text(json.dumps(results))
        # No sweep_meta.json

        current = load_current_data(tmp_path)
        # Should use defaults (metric=sharpe, mode=max)
        assert current.total_runs == 1

    def test_malformed_baseline_missing_fields(self, tmp_path: Path):
        """Test handling baseline with missing fields."""
        baseline_data = {
            "git_sha": "test123",
            # missing created_utc, leaderboard, etc.
            "summary": {"best_metric": 1.0},
        }
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps(baseline_data))

        # Should handle gracefully with defaults
        baseline = load_baseline(baseline_path)
        assert baseline.git_sha == "test123"

    def test_perf_budget_with_string_numbers(self, tmp_path: Path):
        """Test loading perf_budget with string values for numbers."""
        budget_yaml = """
metric: sharpe
mode: max
min_success_rate: "0.75"
max_p90_elapsed_s: "60.0"
max_sharpe_drop: "0.05"
epsilon_abs: "1e-6"
"""
        budget_path = tmp_path / "budget.yaml"
        budget_path.write_text(budget_yaml)

        budget = load_perf_budget(budget_path)
        # Should coerce strings to floats
        assert budget.min_success_rate == 0.75
        assert budget.max_p90_elapsed_s == 60.0
        assert budget.epsilon_abs == 1e-6

    def test_results_missing_elapsed_seconds(self, tmp_path: Path):
        """Test handling results without _elapsed_seconds."""
        results = [
            {"_status": "success", "_run_idx": 0, "avg_oos_sharpe": 1.2},
            {"_status": "success", "_run_idx": 1, "avg_oos_sharpe": 1.5},
        ]
        (tmp_path / "all_results.json").write_text(json.dumps(results))
        (tmp_path / "sweep_meta.json").write_text('{"metric": "sharpe", "mode": "max"}')

        current = load_current_data(tmp_path)
        # Should handle missing timing gracefully
        assert current.total_runs == 2

    def test_baseline_round_trip(self, sample_current_pass: CurrentData, tmp_path: Path):
        """Test creating baseline and loading it back produces consistent data."""
        # Create baseline from current data
        baseline_dict = create_new_baseline(sample_current_pass, "testsha")

        # Write to file
        baseline_path = tmp_path / "baseline.json"
        with open(baseline_path, "w") as f:
            json.dump(baseline_dict, f)

        # Load it back
        loaded_baseline = load_baseline(baseline_path)

        # Verify round-trip consistency
        assert loaded_baseline.git_sha == "testsha"
        assert loaded_baseline.summary["best_metric"] == sample_current_pass.best_metric
        assert loaded_baseline.summary["success_rate"] == sample_current_pass.success_rate
        assert loaded_baseline.summary["total_runs"] == sample_current_pass.total_runs
        assert loaded_baseline.timing["p50"] == sample_current_pass.timing["p50"]
        assert loaded_baseline.timing["p90"] == sample_current_pass.timing["p90"]
