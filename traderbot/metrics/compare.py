"""Performance regression comparison utilities.

Compares current sweep results against a baseline and performance budget
to detect quality or speed regressions.
"""

import contextlib
import csv
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from traderbot.logging_setup import get_logger

logger = get_logger("metrics.compare")


@dataclass
class MetricBudget:
    """Budget configuration for a single metric."""

    name: str
    mode: str = "max"  # 'max' or 'min'
    max_drop: float | None = None  # Maximum drop from baseline (for 'max' mode metrics)
    min_value: float | None = None  # Minimum absolute value required
    max_value: float | None = None  # Maximum absolute value allowed (for 'min' mode like timing)
    epsilon: float = 0.0  # Tolerance for comparisons
    required: bool = True  # If True, failure on this metric fails overall check


@dataclass
class PerfBudget:
    """Performance budget thresholds."""

    metric: str = "sharpe"
    mode: str = "max"
    min_success_rate: float = 0.75
    max_p90_elapsed_s: float = 60.0
    max_sharpe_drop: float = 0.05
    epsilon_abs: float = 1e-6
    max_p50_elapsed_s: float | None = None
    # Per-metric epsilons for fine-grained tolerance control
    epsilon_metric: float | None = None  # Tolerance for metric comparisons
    epsilon_timing: float | None = None  # Tolerance for timing comparisons (seconds)
    # Multi-metric budget map (v0.6.0+)
    budgets: dict[str, MetricBudget] = field(default_factory=dict)


@dataclass
class BaselineData:
    """Baseline performance data."""

    git_sha: str
    created_utc: str
    metric: str
    mode: str
    leaderboard: list[dict[str, Any]]
    timing: dict[str, float]
    summary: dict[str, Any]


@dataclass
class CurrentData:
    """Current sweep performance data."""

    leaderboard: list[dict[str, Any]]
    timing: dict[str, float]
    success_rate: float
    total_runs: int
    best_metric: float
    determinism_result: dict[str, Any] | None = None
    used_fallback_leaderboard: bool = False
    used_fallback_timings: bool = False


@dataclass
class MetricVerdict:
    """Verdict for a single metric in multi-metric evaluation."""

    name: str
    passed: bool
    current_value: float
    baseline_value: float | None
    delta: float | None
    threshold: float | None
    epsilon: float
    mode: str
    message: str


@dataclass
class ComparisonVerdict:
    """Result of comparing current vs baseline."""

    passed: bool
    metric_passed: bool
    timing_passed: bool
    success_rate_passed: bool
    determinism_passed: bool | None  # None if skipped

    metric_delta: float
    timing_p50_delta: float
    timing_p90_delta: float
    success_rate_delta: float
    determinism_variance: float | None

    details: dict[str, Any] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)
    # Per-metric verdicts (v0.6.0+)
    metric_verdicts: list[MetricVerdict] = field(default_factory=list)


def _parse_metric_budgets(data: dict[str, Any]) -> dict[str, MetricBudget]:
    """Parse the budgets: map from YAML data.

    Args:
        data: Raw YAML data dictionary.

    Returns:
        Dictionary of metric name -> MetricBudget.
    """
    budgets_raw = data.get("budgets", {})
    if not budgets_raw:
        return {}

    result = {}
    for name, config in budgets_raw.items():
        if not isinstance(config, dict):
            continue

        result[name] = MetricBudget(
            name=name,
            mode=str(config.get("mode", "max")),
            max_drop=float(config["max_drop"]) if config.get("max_drop") is not None else None,
            min_value=float(config["min"]) if config.get("min") is not None else None,
            max_value=float(config["max"]) if config.get("max") is not None else None,
            epsilon=float(config.get("epsilon", 0.0)),
            required=bool(config.get("required", True)),
        )

    return result


def load_perf_budget(budget_path: Path) -> PerfBudget:
    """Load performance budget from YAML file.

    Supports both legacy single-metric format and new multi-metric budgets: map.

    Legacy format:
        metric: sharpe
        mode: max
        max_sharpe_drop: 0.05

    New format (v0.6.0+):
        budgets:
          sharpe:
            mode: max
            max_drop: 0.05
            epsilon: 0.01
          win_rate:
            mode: max
            min: 0.55
          p90_elapsed_s:
            mode: min
            max: 60
            epsilon: 2.0

    Args:
        budget_path: Path to perf_budget.yaml.

    Returns:
        PerfBudget instance.
    """
    with open(budget_path) as f:
        data = yaml.safe_load(f)

    # Parse multi-metric budgets if present
    budgets = _parse_metric_budgets(data)

    return PerfBudget(
        metric=str(data.get("metric", "sharpe")),
        mode=str(data.get("mode", "max")),
        min_success_rate=float(data.get("min_success_rate", 0.75)),
        max_p90_elapsed_s=float(data.get("max_p90_elapsed_s", 60.0)),
        max_sharpe_drop=float(data.get("max_sharpe_drop", 0.05)),
        epsilon_abs=float(data.get("epsilon_abs", 1e-6)),
        max_p50_elapsed_s=(
            float(data["max_p50_elapsed_s"])
            if data.get("max_p50_elapsed_s") is not None
            else None
        ),
        epsilon_metric=(
            float(data["epsilon_metric"])
            if data.get("epsilon_metric") is not None
            else None
        ),
        epsilon_timing=(
            float(data["epsilon_timing"])
            if data.get("epsilon_timing") is not None
            else None
        ),
        budgets=budgets,
    )


def load_baseline(baseline_path: Path) -> BaselineData:
    """Load baseline data from JSON file.

    Args:
        baseline_path: Path to baseline.json.

    Returns:
        BaselineData instance.
    """
    with open(baseline_path) as f:
        data = json.load(f)

    return BaselineData(
        git_sha=data.get("git_sha", "unknown"),
        created_utc=data.get("created_utc", ""),
        metric=data.get("metric", "sharpe"),
        mode=data.get("mode", "max"),
        leaderboard=data.get("leaderboard", []),
        timing=data.get("timing", {"p50": 0.0, "p90": 0.0}),
        summary=data.get("summary", {}),
    )


def load_current_data(sweep_dir: Path) -> CurrentData:
    """Load current sweep data from directory.

    Args:
        sweep_dir: Path to sweep output directory.

    Returns:
        CurrentData instance.
    """
    # Load sweep metadata for metric/mode info
    metric_name = "sharpe"
    mode = "max"
    sweep_meta_path = sweep_dir / "sweep_meta.json"
    if sweep_meta_path.exists():
        with open(sweep_meta_path) as f:
            _meta = json.load(f)
            metric_name = _meta.get("metric", "sharpe")
            mode = _meta.get("mode", "max")

    # Map metric names to result keys
    metric_key_map = {
        "sharpe": "avg_oos_sharpe",
        "total_return": "avg_oos_return_pct",
        "max_dd": "avg_oos_max_dd_pct",
    }
    metric_key = metric_key_map.get(metric_name, "avg_oos_sharpe")

    # Load leaderboard
    leaderboard_path = sweep_dir / "leaderboard.csv"
    leaderboard = []
    used_fallback_leaderboard = False
    if leaderboard_path.exists():
        with open(leaderboard_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry = {}
                for key, value in row.items():
                    try:
                        entry[key] = float(value)
                    except (ValueError, TypeError):
                        entry[key] = value
                leaderboard.append(entry)

    # Load timing data
    timing = {"p50": 0.0, "p90": 0.0}
    timing_path = sweep_dir / "timings.csv"
    used_fallback_timings = False
    if timing_path.exists():
        elapsed_times = []
        with open(timing_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                with contextlib.suppress(KeyError, ValueError):
                    elapsed_times.append(float(row["elapsed_s"]))

        if elapsed_times:
            timing["p50"] = float(np.percentile(elapsed_times, 50))
            timing["p90"] = float(np.percentile(elapsed_times, 90))

    # Load all results for success rate
    all_results_path = sweep_dir / "all_results.json"
    all_results = []
    success_rate = 1.0
    total_runs = 0
    if all_results_path.exists():
        with open(all_results_path) as f:
            all_results = json.load(f)
        total_runs = len(all_results)
        if total_runs > 0:
            success_count = sum(
                1 for r in all_results if r.get("_status") == "success"
            )
            success_rate = success_count / total_runs

    # Fallback: compute timing from all_results if timings.csv missing
    if not timing_path.exists() and all_results:
        used_fallback_timings = True
        elapsed = [
            r.get("_elapsed_seconds")
            for r in all_results
            if isinstance(r.get("_elapsed_seconds"), int | float)
        ]
        if elapsed:
            timing["p50"] = float(np.percentile(elapsed, 50))
            timing["p90"] = float(np.percentile(elapsed, 90))

    # Get best metric value
    best_metric = 0.0
    if leaderboard:
        # First row is best (leaderboard is sorted)
        best_metric = float(leaderboard[0].get(metric_key, 0.0))
    elif all_results:
        # Fallback: compute from all_results if leaderboard missing/empty
        used_fallback_leaderboard = True
        vals = [
            r.get(metric_key)
            for r in all_results
            if r.get("_status") == "success"
            and isinstance(r.get(metric_key), int | float)
        ]
        if vals:
            best_metric = max(vals) if mode == "max" else min(vals)

    # Load determinism result if available
    determinism_result = None
    determinism_path = sweep_dir / "determinism.json"
    if determinism_path.exists():
        with open(determinism_path) as f:
            determinism_result = json.load(f)

    return CurrentData(
        leaderboard=leaderboard,
        timing=timing,
        success_rate=success_rate,
        total_runs=total_runs,
        best_metric=best_metric,
        determinism_result=determinism_result,
        used_fallback_leaderboard=used_fallback_leaderboard,
        used_fallback_timings=used_fallback_timings,
    )


def _get_metric_values_from_current(
    current: CurrentData,
    baseline: BaselineData,
) -> dict[str, float]:
    """Extract metric values from current data for multi-metric evaluation.

    Args:
        current: Current sweep data.
        baseline: Baseline data for context.

    Returns:
        Dictionary of metric name -> current value.
    """
    values: dict[str, float] = {}

    # Primary metric (sharpe by default)
    values["sharpe"] = current.best_metric

    # Timing metrics
    values["p50_elapsed_s"] = current.timing["p50"]
    values["p90_elapsed_s"] = current.timing["p90"]

    # Success rate
    values["success_rate"] = current.success_rate

    # Extract from leaderboard if available (first row = best)
    if current.leaderboard:
        top = current.leaderboard[0]
        # Common metric aliases
        metric_aliases = {
            "avg_oos_sharpe": ["sharpe", "avg_oos_sharpe"],
            "avg_oos_return_pct": ["total_return", "return", "avg_oos_return_pct"],
            "avg_oos_max_dd_pct": ["max_dd", "drawdown", "avg_oos_max_dd_pct"],
            "win_rate": ["win_rate"],
        }
        for key, aliases in metric_aliases.items():
            if key in top:
                val = top[key]
                if isinstance(val, int | float):
                    for alias in aliases:
                        values[alias] = float(val)

    return values


def _get_baseline_value_for_metric(
    baseline: BaselineData,
    metric_name: str,
) -> float | None:
    """Get baseline value for a specific metric.

    Args:
        baseline: Baseline data.
        metric_name: Name of the metric.

    Returns:
        Baseline value or None if not found.
    """
    # Check summary first
    if metric_name == "sharpe":
        return baseline.summary.get("best_metric")
    if metric_name == "success_rate":
        return baseline.summary.get("success_rate")
    if metric_name == "p50_elapsed_s":
        return baseline.timing.get("p50")
    if metric_name == "p90_elapsed_s":
        return baseline.timing.get("p90")

    # Check leaderboard
    if baseline.leaderboard:
        top = baseline.leaderboard[0]
        # Try direct key
        if metric_name in top:
            val = top.get(metric_name)
            if isinstance(val, int | float):
                return float(val)
        # Try common mappings
        key_map = {
            "total_return": "avg_oos_return_pct",
            "return": "avg_oos_return_pct",
            "max_dd": "avg_oos_max_dd_pct",
            "drawdown": "avg_oos_max_dd_pct",
        }
        mapped_key = key_map.get(metric_name)
        if mapped_key and mapped_key in top:
            val = top.get(mapped_key)
            if isinstance(val, int | float):
                return float(val)

    return None


def _evaluate_metric_budget(
    metric_name: str,
    current_val: float,
    baseline_val: float | None,
    metric_budget: MetricBudget,
) -> tuple[bool, str]:
    """Evaluate a single metric against its budget.

    Args:
        metric_name: Name of the metric.
        current_val: Current value.
        baseline_val: Baseline value (may be None).
        metric_budget: Budget configuration for this metric.

    Returns:
        Tuple of (passed, message).
    """
    epsilon = metric_budget.epsilon

    # Check absolute min/max constraints first
    if metric_budget.min_value is not None:
        effective_min = metric_budget.min_value - epsilon
        if current_val < effective_min:
            return False, (
                f"{metric_name} below minimum: "
                f"{current_val:.4f} < {metric_budget.min_value:.4f}"
                + (f" (epsilon={epsilon})" if epsilon > 0 else "")
            )

    if metric_budget.max_value is not None:
        effective_max = metric_budget.max_value + epsilon
        if current_val > effective_max:
            return False, (
                f"{metric_name} exceeds maximum: "
                f"{current_val:.4f} > {metric_budget.max_value:.4f}"
                + (f" (epsilon={epsilon})" if epsilon > 0 else "")
            )

    # Check regression from baseline (max_drop)
    if metric_budget.max_drop is not None and baseline_val is not None:
        delta = current_val - baseline_val

        if metric_budget.mode == "max":
            # For 'max' metrics (like sharpe), regression is when current drops
            effective_delta = delta + epsilon  # Be lenient
            if effective_delta < -metric_budget.max_drop:
                return False, (
                    f"{metric_name} regression: "
                    f"drop={-delta:.4f} > max_drop={metric_budget.max_drop:.4f} "
                    f"(current={current_val:.4f}, baseline={baseline_val:.4f})"
                )
        else:  # min mode
            # For 'min' metrics (like timing), regression is when current increases
            effective_delta = delta - epsilon  # Be lenient
            if effective_delta > metric_budget.max_drop:
                return False, (
                    f"{metric_name} regression: "
                    f"increase={delta:.4f} > max_drop={metric_budget.max_drop:.4f} "
                    f"(current={current_val:.4f}, baseline={baseline_val:.4f})"
                )

    return True, f"{metric_name} OK"


def compare_results(
    current: CurrentData,
    baseline: BaselineData,
    budget: PerfBudget,
) -> ComparisonVerdict:
    """Compare current results against baseline and budget.

    Args:
        current: Current sweep data.
        baseline: Baseline data.
        budget: Performance budget thresholds.

    Returns:
        ComparisonVerdict with pass/fail status and details.
    """
    messages = []
    details = {}

    # Check metric: prefer perf_budget.yaml metric/mode over baseline
    if budget.metric != baseline.metric:
        messages.append(
            f"Warning: budget metric '{budget.metric}' differs from "
            f"baseline metric '{baseline.metric}'. Using budget metric."
        )

    baseline_best = baseline.summary.get("best_metric", 0.0)
    metric_delta = current.best_metric - baseline_best

    # Use per-metric epsilon if specified, otherwise use max_sharpe_drop
    metric_epsilon = budget.epsilon_metric if budget.epsilon_metric is not None else 0.0

    # For 'max' mode, regression is when current < baseline - threshold
    # For 'min' mode, regression is when current > baseline + threshold
    # Apply epsilon tolerance: ignore tiny deltas within epsilon range
    if budget.mode == "max":
        # Regression if drop exceeds threshold (accounting for epsilon noise)
        effective_delta = metric_delta + metric_epsilon  # Be lenient by epsilon
        metric_passed = effective_delta >= -budget.max_sharpe_drop
        if not metric_passed:
            messages.append(
                f"FAIL: Metric regression detected. "
                f"Current {budget.metric}={current.best_metric:.4f}, "
                f"baseline={baseline_best:.4f}, "
                f"drop={-metric_delta:.4f} > max_drop={budget.max_sharpe_drop}"
            )
    else:  # min mode
        effective_delta = metric_delta - metric_epsilon  # Be lenient by epsilon
        metric_passed = effective_delta <= budget.max_sharpe_drop
        if not metric_passed:
            messages.append(
                f"FAIL: Metric regression detected. "
                f"Current {budget.metric}={current.best_metric:.4f}, "
                f"baseline={baseline_best:.4f}, "
                f"increase={metric_delta:.4f} > max_drop={budget.max_sharpe_drop}"
            )

    details["metric"] = {
        "current": current.best_metric,
        "baseline": baseline_best,
        "delta": metric_delta,
        "threshold": budget.max_sharpe_drop,
        "epsilon": metric_epsilon,
        "passed": metric_passed,
    }

    # Check timing
    timing_p50_delta = current.timing["p50"] - baseline.timing.get("p50", 0.0)
    timing_p90_delta = current.timing["p90"] - baseline.timing.get("p90", 0.0)

    # Use epsilon_timing for tolerance on timing comparisons (absorbs CI noise)
    timing_epsilon = budget.epsilon_timing if budget.epsilon_timing is not None else 0.0

    # P90 check with epsilon tolerance
    timing_passed = current.timing["p90"] <= (budget.max_p90_elapsed_s + timing_epsilon)
    if not timing_passed:
        messages.append(
            f"FAIL: Timing budget exceeded. "
            f"P90={current.timing['p90']:.2f}s > max={budget.max_p90_elapsed_s}s"
            + (f" (epsilon={timing_epsilon}s)" if timing_epsilon > 0 else "")
        )

    # Optional P50 check with epsilon tolerance
    if budget.max_p50_elapsed_s is not None:
        p50_exceeded = current.timing["p50"] > (budget.max_p50_elapsed_s + timing_epsilon)
        if p50_exceeded:
            timing_passed = False
            messages.append(
                f"FAIL: P50 timing budget exceeded. "
                f"P50={current.timing['p50']:.2f}s > max={budget.max_p50_elapsed_s}s"
                + (f" (epsilon={timing_epsilon}s)" if timing_epsilon > 0 else "")
            )

    details["timing"] = {
        "current_p50": current.timing["p50"],
        "current_p90": current.timing["p90"],
        "baseline_p50": baseline.timing.get("p50", 0.0),
        "baseline_p90": baseline.timing.get("p90", 0.0),
        "delta_p50": timing_p50_delta,
        "delta_p90": timing_p90_delta,
        "budget_p90": budget.max_p90_elapsed_s,
        "epsilon": timing_epsilon,
        "passed": timing_passed,
    }

    # Check success rate
    baseline_success_rate = baseline.summary.get("success_rate", 1.0)
    success_rate_delta = current.success_rate - baseline_success_rate

    success_rate_passed = current.success_rate >= budget.min_success_rate
    if not success_rate_passed:
        messages.append(
            f"FAIL: Success rate too low. "
            f"Current={current.success_rate:.2%} < min={budget.min_success_rate:.2%}"
        )

    details["success_rate"] = {
        "current": current.success_rate,
        "baseline": baseline_success_rate,
        "delta": success_rate_delta,
        "threshold": budget.min_success_rate,
        "passed": success_rate_passed,
    }

    # Check determinism (if available)
    determinism_passed: bool | None = None
    determinism_variance: float | None = None

    if current.determinism_result is not None:
        determinism_variance = current.determinism_result.get("max_variance", 0.0)
        determinism_passed = determinism_variance <= budget.epsilon_abs

        if not determinism_passed:
            messages.append(
                f"FAIL: Determinism check failed. "
                f"Max variance={determinism_variance:.2e} > epsilon={budget.epsilon_abs:.2e}"
            )

        details["determinism"] = {
            "max_variance": determinism_variance,
            "epsilon": budget.epsilon_abs,
            "passed": determinism_passed,
            "runs": current.determinism_result.get("runs", []),
        }
    else:
        messages.append("INFO: Determinism check skipped (no determinism.json found)")
        details["determinism"] = {"skipped": True}

    # Process multi-metric budgets (v0.6.0+)
    metric_verdicts: list[MetricVerdict] = []
    budgets_all_passed = True

    if budget.budgets:
        # Map of metric names to their values in current data
        # This maps budget metric names to actual data keys
        metric_value_map = _get_metric_values_from_current(current, baseline)

        for metric_name, metric_budget in budget.budgets.items():
            current_val = metric_value_map.get(metric_name)
            baseline_val = _get_baseline_value_for_metric(baseline, metric_name)

            if current_val is None:
                # Metric not found in current data
                verdict_entry = MetricVerdict(
                    name=metric_name,
                    passed=not metric_budget.required,  # Pass if not required
                    current_value=0.0,
                    baseline_value=baseline_val,
                    delta=None,
                    threshold=metric_budget.max_drop or metric_budget.max_value,
                    epsilon=metric_budget.epsilon,
                    mode=metric_budget.mode,
                    message=f"Metric '{metric_name}' not found in current data",
                )
                metric_verdicts.append(verdict_entry)
                if metric_budget.required:
                    budgets_all_passed = False
                    messages.append(f"FAIL: Metric '{metric_name}' not found in current data")
                continue

            # Evaluate the metric based on its configuration
            passed_check, msg = _evaluate_metric_budget(
                metric_name, current_val, baseline_val, metric_budget
            )

            delta = current_val - baseline_val if baseline_val is not None else None
            verdict_entry = MetricVerdict(
                name=metric_name,
                passed=passed_check,
                current_value=current_val,
                baseline_value=baseline_val,
                delta=delta,
                threshold=metric_budget.max_drop or metric_budget.max_value or metric_budget.min_value,
                epsilon=metric_budget.epsilon,
                mode=metric_budget.mode,
                message=msg,
            )
            metric_verdicts.append(verdict_entry)

            if not passed_check and metric_budget.required:
                budgets_all_passed = False
                messages.append(f"FAIL: {msg}")

        details["budgets"] = {
            mv.name: {
                "passed": mv.passed,
                "current": mv.current_value,
                "baseline": mv.baseline_value,
                "delta": mv.delta,
                "threshold": mv.threshold,
                "epsilon": mv.epsilon,
                "mode": mv.mode,
            }
            for mv in metric_verdicts
        }

    # Overall verdict
    passed = metric_passed and timing_passed and success_rate_passed and budgets_all_passed
    if determinism_passed is False:
        passed = False

    if passed:
        messages.insert(0, "PASS: All regression checks passed")
    else:
        messages.insert(0, "FAIL: One or more regression checks failed")

    return ComparisonVerdict(
        passed=passed,
        metric_passed=metric_passed,
        timing_passed=timing_passed,
        success_rate_passed=success_rate_passed,
        determinism_passed=determinism_passed,
        metric_delta=metric_delta,
        timing_p50_delta=timing_p50_delta,
        timing_p90_delta=timing_p90_delta,
        success_rate_delta=success_rate_delta,
        determinism_variance=determinism_variance,
        details=details,
        messages=messages,
        metric_verdicts=metric_verdicts,
    )


def generate_regression_report(
    verdict: ComparisonVerdict,
    current: CurrentData,
    baseline: BaselineData,
    budget: PerfBudget,
) -> str:
    """Generate markdown regression report.

    Args:
        verdict: Comparison verdict.
        current: Current data.
        baseline: Baseline data.
        budget: Performance budget.

    Returns:
        Markdown report string.
    """
    lines = []

    # Header with badge
    status_badge = "✅ PASS" if verdict.passed else "❌ FAIL"
    lines.append(f"# Regression Report {status_badge}")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(UTC).isoformat()}")
    lines.append(f"**Baseline SHA:** `{baseline.git_sha}`")
    lines.append(f"**Baseline Created:** {baseline.created_utc}")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Check | Status | Current | Baseline | Delta | Threshold |")
    lines.append("|-------|--------|---------|----------|-------|-----------|")

    # Metric row
    metric_status = "✅" if verdict.metric_passed else "❌"
    lines.append(
        f"| {budget.metric.title()} | {metric_status} | "
        f"{current.best_metric:.4f} | "
        f"{baseline.summary.get('best_metric', 0.0):.4f} | "
        f"{verdict.metric_delta:+.4f} | "
        f"max drop: {budget.max_sharpe_drop} |"
    )

    # Timing P90 row
    timing_status = "✅" if verdict.timing_passed else "❌"
    lines.append(
        f"| Timing P90 | {timing_status} | "
        f"{current.timing['p90']:.2f}s | "
        f"{baseline.timing.get('p90', 0.0):.2f}s | "
        f"{verdict.timing_p90_delta:+.2f}s | "
        f"max: {budget.max_p90_elapsed_s}s |"
    )

    # Timing P50 row
    lines.append(
        f"| Timing P50 | - | "
        f"{current.timing['p50']:.2f}s | "
        f"{baseline.timing.get('p50', 0.0):.2f}s | "
        f"{verdict.timing_p50_delta:+.2f}s | "
        f"- |"
    )

    # Success rate row
    sr_status = "✅" if verdict.success_rate_passed else "❌"
    lines.append(
        f"| Success Rate | {sr_status} | "
        f"{current.success_rate:.2%} | "
        f"{baseline.summary.get('success_rate', 1.0):.2%} | "
        f"{verdict.success_rate_delta:+.2%} | "
        f"min: {budget.min_success_rate:.2%} |"
    )

    # Determinism row
    if verdict.determinism_passed is None:
        det_status = "⏭️ Skipped"
        det_variance = "N/A"
    elif verdict.determinism_passed:
        det_status = "✅"
        det_variance = f"{verdict.determinism_variance:.2e}"
    else:
        det_status = "❌"
        det_variance = f"{verdict.determinism_variance:.2e}"

    lines.append(
        f"| Determinism | {det_status} | "
        f"{det_variance} | - | - | "
        f"epsilon: {budget.epsilon_abs:.0e} |"
    )

    # Per-metric budget verdicts (v0.6.0+)
    if verdict.metric_verdicts:
        lines.append("")
        lines.append("### Per-Metric Budgets")
        lines.append("")
        lines.append("| Metric | Status | Current | Baseline | Delta | Threshold |")
        lines.append("|--------|--------|---------|----------|-------|-----------|")

        for mv in verdict.metric_verdicts:
            mv_status = "✅" if mv.passed else "❌"
            baseline_str = f"{mv.baseline_value:.4f}" if mv.baseline_value is not None else "N/A"
            delta_str = f"{mv.delta:+.4f}" if mv.delta is not None else "N/A"
            threshold_str = f"{mv.threshold:.4f}" if mv.threshold is not None else "N/A"
            lines.append(
                f"| {mv.name} | {mv_status} | "
                f"{mv.current_value:.4f} | {baseline_str} | {delta_str} | {threshold_str} |"
            )

    lines.append("")

    # Messages
    lines.append("## Details")
    lines.append("")
    for msg in verdict.messages:
        if msg.startswith("FAIL:"):
            lines.append(f"- ❌ {msg}")
        elif msg.startswith("PASS:"):
            lines.append(f"- ✅ {msg}")
        elif msg.startswith("INFO:"):
            lines.append(f"- ℹ️ {msg}")
        elif msg.startswith("Warning:"):
            lines.append(f"- ⚠️ {msg}")
        else:
            lines.append(f"- {msg}")

    lines.append("")

    # Leaderboard comparison (top 3)
    lines.append("## Top 3 Comparison")
    lines.append("")
    lines.append("### Current")
    lines.append("")
    lines.append("| Rank | Sharpe | Return % | Max DD % |")
    lines.append("|------|--------|----------|----------|")
    for entry in current.leaderboard[:3]:
        lines.append(
            f"| {int(entry.get('rank', 0))} | "
            f"{float(entry.get('avg_oos_sharpe', 0)):.4f} | "
            f"{float(entry.get('avg_oos_return_pct', 0)):.2f}% | "
            f"{float(entry.get('avg_oos_max_dd_pct', 0)):.2f}% |"
        )

    lines.append("")
    lines.append("### Baseline")
    lines.append("")
    lines.append("| Rank | Sharpe | Return % | Max DD % |")
    lines.append("|------|--------|----------|----------|")
    for entry in baseline.leaderboard[:3]:
        lines.append(
            f"| {entry.get('rank', 0)} | "
            f"{entry.get('avg_oos_sharpe', 0):.4f} | "
            f"{entry.get('avg_oos_return_pct', 0):.2f}% | "
            f"{entry.get('avg_oos_max_dd_pct', 0):.2f}% |"
        )

    lines.append("")

    return "\n".join(lines)


def generate_baseline_diff(
    verdict: ComparisonVerdict,
    current: CurrentData,
    baseline: BaselineData,
) -> dict[str, Any]:
    """Generate baseline diff JSON.

    Args:
        verdict: Comparison verdict.
        current: Current data.
        baseline: Baseline data.

    Returns:
        Dictionary for baseline_diff.json.
    """
    return {
        "generated_utc": datetime.now(UTC).isoformat(),
        "passed": verdict.passed,
        "baseline_sha": baseline.git_sha,
        "baseline_created": baseline.created_utc,
        "deltas": {
            "metric": verdict.metric_delta,
            "timing_p50": verdict.timing_p50_delta,
            "timing_p90": verdict.timing_p90_delta,
            "success_rate": verdict.success_rate_delta,
        },
        "verdicts": {
            "metric": verdict.metric_passed,
            "timing": verdict.timing_passed,
            "success_rate": verdict.success_rate_passed,
            "determinism": verdict.determinism_passed,
        },
        "current_summary": {
            "best_metric": current.best_metric,
            "timing_p50": current.timing["p50"],
            "timing_p90": current.timing["p90"],
            "success_rate": current.success_rate,
            "total_runs": current.total_runs,
        },
        "baseline_summary": baseline.summary,
        "details": verdict.details,
    }


def create_new_baseline(
    current: CurrentData,
    git_sha: str,
) -> dict[str, Any]:
    """Create a new baseline from current data.

    Args:
        current: Current sweep data.
        git_sha: Git SHA for the new baseline.

    Returns:
        Dictionary for baseline.json.
    """
    return {
        "git_sha": git_sha,
        "created_utc": datetime.now(UTC).isoformat(),
        "metric": "sharpe",
        "mode": "max",
        "leaderboard": current.leaderboard[:5],  # Top 5
        "timing": current.timing,
        "summary": {
            "best_metric": current.best_metric,
            "success_rate": current.success_rate,
            "total_runs": current.total_runs,
        },
    }


@dataclass
class VarianceEntry:
    """Entry in variance report for a single regressor."""

    run_idx: int
    metric_name: str
    values: list[float]
    mean: float
    std: float
    cv: float  # Coefficient of variation (std/mean)
    is_flaky: bool


def generate_variance_report(
    entries: list[VarianceEntry],
    threshold: float,
) -> dict[str, Any]:
    """Generate variance report from rerun analysis.

    Args:
        entries: List of variance entries from reruns.
        threshold: Variance threshold for flagging flaky results.

    Returns:
        Dictionary for variance_report.json.
    """
    flaky_count = sum(1 for e in entries if e.is_flaky)

    return {
        "generated_utc": datetime.now(UTC).isoformat(),
        "threshold": threshold,
        "total_entries": len(entries),
        "flaky_count": flaky_count,
        "flaky_rate": flaky_count / len(entries) if entries else 0.0,
        "entries": [
            {
                "run_idx": e.run_idx,
                "metric_name": e.metric_name,
                "values": e.values,
                "mean": e.mean,
                "std": e.std,
                "cv": e.cv,
                "is_flaky": e.is_flaky,
            }
            for e in entries
        ],
    }


def generate_variance_markdown(
    entries: list[VarianceEntry],
    threshold: float,
) -> str:
    """Generate markdown variance report from rerun analysis.

    Args:
        entries: List of variance entries from reruns.
        threshold: Variance threshold for flagging flaky results.

    Returns:
        Markdown report string.
    """
    lines = []
    lines.append("## Variance Analysis Report")
    lines.append("")
    lines.append(f"**Threshold (CV):** {threshold:.2f}")
    lines.append(f"**Total Entries:** {len(entries)}")

    flaky_entries = [e for e in entries if e.is_flaky]
    lines.append(f"**Flaky Entries:** {len(flaky_entries)}")
    lines.append("")

    if not entries:
        lines.append("No entries to analyze.")
        return "\n".join(lines)

    # Summary table
    lines.append("| Run Idx | Metric | Mean | Std | CV | Flaky |")
    lines.append("|---------|--------|------|-----|----|----|")

    for entry in entries:
        flaky_mark = "⚠️" if entry.is_flaky else "✅"
        lines.append(
            f"| {entry.run_idx} | {entry.metric_name} | "
            f"{entry.mean:.4f} | {entry.std:.4f} | {entry.cv:.4f} | {flaky_mark} |"
        )

    lines.append("")

    # Flaky entries detail
    if flaky_entries:
        lines.append("### Flaky Entries (CV > threshold)")
        lines.append("")
        for entry in flaky_entries:
            lines.append(f"- **Run {entry.run_idx}** ({entry.metric_name}): CV={entry.cv:.4f}")
            lines.append(f"  - Values: {[f'{v:.4f}' for v in entry.values]}")
        lines.append("")

    return "\n".join(lines)


def generate_html_report(
    verdict: ComparisonVerdict,
    current: CurrentData,
    baseline: BaselineData,
    budget: PerfBudget,
    provenance: dict | None = None,
) -> str:
    """Generate HTML regression report with dark/light toggle, status banner, and provenance footer.

    Args:
        verdict: Comparison verdict.
        current: Current data.
        baseline: Baseline data.
        budget: Performance budget.
        provenance: Optional provenance data for footer badge.

    Returns:
        HTML report string.
    """
    status_color = "#28a745" if verdict.passed else "#dc3545"
    status_text = "PASS" if verdict.passed else "FAIL"
    status_emoji = "✅" if verdict.passed else "❌"
    verdict_class = "pass" if verdict.passed else "fail"
    generated_utc = datetime.now(UTC).isoformat()

    # Extract provenance info for footer
    prov_sha = provenance.get("git_sha", "unknown") if provenance else "unknown"
    prov_timestamp = provenance.get("generated_utc", generated_utc) if provenance else generated_utc

    # Calculate summary stats for banner
    sharpe_delta = verdict.metric_delta
    timing_p90 = current.timing.get("p90", 0.0)
    total_runs = current.total_runs

    html = f"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Regression Report - {status_text}</title>
    <style>
        :root {{
            --bg-primary: #f8f9fa;
            --bg-card: white;
            --text-primary: #212529;
            --text-secondary: #6c757d;
            --border-color: #dee2e6;
            --header-bg: #f1f3f4;
            --shadow: rgba(0,0,0,0.1);
            --pass-color: #28a745;
            --fail-color: #dc3545;
            --pass-bg: #d4edda;
            --fail-bg: #f8d7da;
        }}
        [data-theme="dark"] {{
            --bg-primary: #1a1a2e;
            --bg-card: #16213e;
            --text-primary: #e4e4e7;
            --text-secondary: #a1a1aa;
            --border-color: #3f3f46;
            --header-bg: #1e3a5f;
            --shadow: rgba(0,0,0,0.3);
            --pass-color: #3fb950;
            --fail-color: #f85149;
            --pass-bg: #1a3d1a;
            --fail-bg: #4a1a1a;
        }}
        * {{
            transition: background-color 0.3s, color 0.3s, border-color 0.3s;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            padding-top: 120px;
            background: var(--bg-primary);
            color: var(--text-primary);
        }}
        .sticky-header {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background: var(--bg-card);
            padding: 10px 20px;
            box-shadow: 0 2px 4px var(--shadow);
            z-index: 1000;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .sticky-header h1 {{
            margin: 0;
            font-size: 1.2em;
        }}
        .theme-toggle {{
            background: var(--header-bg);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 6px 12px;
            cursor: pointer;
            font-size: 0.9em;
            color: var(--text-primary);
        }}
        .theme-toggle:hover {{
            opacity: 0.8;
        }}
        .verdict-banner {{
            position: fixed;
            top: 50px;
            left: 0;
            right: 0;
            padding: 8px 20px;
            text-align: center;
            font-weight: bold;
            font-size: 0.95em;
            z-index: 999;
            display: flex;
            justify-content: center;
            gap: 30px;
            align-items: center;
        }}
        .verdict-banner.pass {{
            background: var(--pass-bg);
            color: var(--pass-color);
            border-bottom: 2px solid var(--pass-color);
        }}
        .verdict-banner.fail {{
            background: var(--fail-bg);
            color: var(--fail-color);
            border-bottom: 2px solid var(--fail-color);
        }}
        .verdict-banner .stat {{
            display: inline-block;
        }}
        .verdict-banner .stat-label {{
            font-weight: normal;
            opacity: 0.8;
            margin-right: 4px;
        }}
        .badge {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 6px;
            background: {status_color};
            color: white;
            font-weight: bold;
            font-size: 1.2em;
        }}
        .badge-small {{
            padding: 4px 10px;
            font-size: 0.9em;
        }}
        .header {{
            background: var(--bg-card);
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px var(--shadow);
        }}
        .card {{
            background: var(--bg-card);
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px var(--shadow);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }}
        th, td {{
            text-align: left;
            padding: 10px;
            border-bottom: 1px solid var(--border-color);
        }}
        th {{
            background: var(--header-bg);
            font-weight: 600;
        }}
        .pass {{ color: var(--pass-color); }}
        .fail {{ color: var(--fail-color); }}
        .info {{ color: var(--text-secondary); }}
        h1, h2 {{
            margin-top: 0;
        }}
        .meta {{
            color: var(--text-secondary);
            font-size: 0.9em;
        }}
        code {{
            background: var(--header-bg);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'SF Mono', Consolas, monospace;
        }}
        .footer {{
            background: var(--bg-card);
            padding: 15px 20px;
            border-radius: 8px;
            margin-top: 30px;
            box-shadow: 0 1px 3px var(--shadow);
            text-align: center;
            font-size: 0.85em;
            color: var(--text-secondary);
        }}
        .footer-badge {{
            display: inline-block;
            background: var(--header-bg);
            padding: 4px 10px;
            border-radius: 4px;
            font-family: 'SF Mono', Consolas, monospace;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="sticky-header">
        <h1>{status_emoji} Regression: <span class="badge badge-small">{status_text}</span></h1>
        <button class="theme-toggle" onclick="toggleTheme()">Toggle Dark/Light</button>
    </div>

    <div class="verdict-banner {verdict_class}">
        <span class="stat"><span class="stat-label">Verdict:</span> {status_text}</span>
        <span class="stat"><span class="stat-label">Sharpe Δ:</span> {sharpe_delta:+.4f}</span>
        <span class="stat"><span class="stat-label">P90:</span> {timing_p90:.2f}s</span>
        <span class="stat"><span class="stat-label">Runs:</span> {total_runs}</span>
    </div>

    <div class="header">
        <h1>{status_emoji} Regression Report <span class="badge">{status_text}</span></h1>
        <p class="meta">
            <strong>Generated:</strong> {generated_utc}<br>
            <strong>Baseline SHA:</strong> <code>{baseline.git_sha}</code><br>
            <strong>Baseline Created:</strong> {baseline.created_utc}
        </p>
    </div>

    <div class="card">
        <h2>Summary</h2>
        <table>
            <thead>
                <tr>
                    <th>Check</th>
                    <th>Status</th>
                    <th>Current</th>
                    <th>Baseline</th>
                    <th>Delta</th>
                    <th>Threshold</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>{budget.metric.title()}</td>
                    <td class="{'pass' if verdict.metric_passed else 'fail'}">{'✅' if verdict.metric_passed else '❌'}</td>
                    <td>{current.best_metric:.4f}</td>
                    <td>{baseline.summary.get('best_metric', 0.0):.4f}</td>
                    <td>{verdict.metric_delta:+.4f}</td>
                    <td>max drop: {budget.max_sharpe_drop}</td>
                </tr>
                <tr>
                    <td>Timing P90</td>
                    <td class="{'pass' if verdict.timing_passed else 'fail'}">{'✅' if verdict.timing_passed else '❌'}</td>
                    <td>{current.timing['p90']:.2f}s</td>
                    <td>{baseline.timing.get('p90', 0.0):.2f}s</td>
                    <td>{verdict.timing_p90_delta:+.2f}s</td>
                    <td>max: {budget.max_p90_elapsed_s}s</td>
                </tr>
                <tr>
                    <td>Timing P50</td>
                    <td class="info">-</td>
                    <td>{current.timing['p50']:.2f}s</td>
                    <td>{baseline.timing.get('p50', 0.0):.2f}s</td>
                    <td>{verdict.timing_p50_delta:+.2f}s</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Success Rate</td>
                    <td class="{'pass' if verdict.success_rate_passed else 'fail'}">{'✅' if verdict.success_rate_passed else '❌'}</td>
                    <td>{current.success_rate:.2%}</td>
                    <td>{baseline.summary.get('success_rate', 1.0):.2%}</td>
                    <td>{verdict.success_rate_delta:+.2%}</td>
                    <td>min: {budget.min_success_rate:.2%}</td>
                </tr>
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>Details</h2>
        <ul>
"""

    for msg in verdict.messages:
        if msg.startswith("FAIL:"):
            html += f'            <li class="fail">❌ {msg}</li>\n'
        elif msg.startswith("PASS:"):
            html += f'            <li class="pass">✅ {msg}</li>\n'
        elif msg.startswith("INFO:"):
            html += f'            <li class="info">ℹ️ {msg}</li>\n'
        elif msg.startswith("Warning:"):
            html += f'            <li class="info">⚠️ {msg}</li>\n'
        else:
            html += f"            <li>{msg}</li>\n"

    html += """        </ul>
    </div>
"""

    # Per-metric budget verdicts (v0.6.0+)
    if verdict.metric_verdicts:
        html += """
    <div class="card">
        <h2>Per-Metric Budgets</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Status</th>
                    <th>Current</th>
                    <th>Baseline</th>
                    <th>Delta</th>
                    <th>Threshold</th>
                </tr>
            </thead>
            <tbody>
"""
        for mv in verdict.metric_verdicts:
            status_class = "pass" if mv.passed else "fail"
            status_icon = "✅" if mv.passed else "❌"
            baseline_str = f"{mv.baseline_value:.4f}" if mv.baseline_value is not None else "N/A"
            delta_str = f"{mv.delta:+.4f}" if mv.delta is not None else "N/A"
            threshold_str = f"{mv.threshold:.4f}" if mv.threshold is not None else "N/A"
            html += f"""                <tr>
                    <td>{mv.name}</td>
                    <td class="{status_class}">{status_icon}</td>
                    <td>{mv.current_value:.4f}</td>
                    <td>{baseline_str}</td>
                    <td>{delta_str}</td>
                    <td>{threshold_str}</td>
                </tr>
"""
        html += """            </tbody>
        </table>
    </div>
"""

    # Footer with provenance badge
    html += f"""
    <div class="footer">
        <span class="footer-badge">Built from {prov_sha[:7] if len(prov_sha) > 7 else prov_sha} at {prov_timestamp}</span>
        <br><br>
        TraderBot Regression Report &bull; <a href="https://github.com/eyoair21/Trade_Bot">GitHub</a>
    </div>

    <script>
        function toggleTheme() {{
            const html = document.documentElement;
            const currentTheme = html.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            html.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
        }}

        // Load saved theme preference
        (function() {{
            const savedTheme = localStorage.getItem('theme');
            if (savedTheme) {{
                document.documentElement.setAttribute('data-theme', savedTheme);
            }} else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {{
                document.documentElement.setAttribute('data-theme', 'dark');
            }}
        }})();
    </script>
</body>
</html>
"""
    return html


def generate_provenance_json(
    current: CurrentData,
    git_sha: str = "unknown",
) -> dict[str, Any]:
    """Generate provenance JSON documenting data sources.

    Provenance schema v1:
    - schema_version: "1"
    - git_sha: Git commit SHA
    - generated_utc: ISO timestamp
    - data_sources: Map of data type to source file
    - fallbacks_used: Map of data type to whether fallback was used
    - notes: List of notes

    Args:
        current: Current data with fallback flags.
        git_sha: Git commit SHA (default: "unknown").

    Returns:
        Dictionary for provenance.json.
    """
    return {
        "schema_version": "1",
        "git_sha": git_sha,
        "generated_utc": datetime.now(UTC).isoformat(),
        "data_sources": {
            "leaderboard": "all_results.json" if current.used_fallback_leaderboard else "leaderboard.csv",
            "timings": "all_results.json" if current.used_fallback_timings else "timings.csv",
        },
        "fallbacks_used": {
            "leaderboard": current.used_fallback_leaderboard,
            "timings": current.used_fallback_timings,
        },
        "notes": [],
    }


def generate_summary_json(
    run_id: str,
    verdict: ComparisonVerdict,
    current: CurrentData,
    baseline: BaselineData | None = None,
    git_sha: str = "unknown",
) -> dict[str, Any]:
    """Generate summary.json for per-run summary card.

    Schema v1:
    - run_id: Run identifier (e.g., "12345-abc1234")
    - verdict: "PASS" or "FAIL"
    - sharpe_delta: Change in Sharpe ratio from baseline
    - trades_delta: Change in total trades from baseline (if available)
    - timing_p90: P90 timing in seconds
    - git_sha: Git commit SHA
    - generated_utc: ISO timestamp

    Args:
        run_id: Run identifier string.
        verdict: ComparisonVerdict with pass/fail status.
        current: Current sweep data.
        baseline: Optional baseline data for computing deltas.
        git_sha: Git commit SHA.

    Returns:
        Dictionary for summary.json.
    """
    # Calculate trades delta if baseline available
    trades_delta: int | None = None
    if baseline is not None:
        baseline_trades = baseline.summary.get("total_trades", 0)
        current_trades = current.total_runs
        trades_delta = current_trades - baseline_trades

    return {
        "schema_version": "1",
        "run_id": run_id,
        "verdict": "PASS" if verdict.passed else "FAIL",
        "sharpe_delta": round(verdict.metric_delta, 6),
        "trades_delta": trades_delta,
        "timing_p90": round(current.timing.get("p90", 0.0), 3),
        "git_sha": git_sha,
        "generated_utc": datetime.now(UTC).isoformat(),
    }
