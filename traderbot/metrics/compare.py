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
class PerfBudget:
    """Performance budget thresholds."""

    metric: str = "sharpe"
    mode: str = "max"
    min_success_rate: float = 0.75
    max_p90_elapsed_s: float = 60.0
    max_sharpe_drop: float = 0.05
    epsilon_abs: float = 1e-6
    max_p50_elapsed_s: float | None = None


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


def load_perf_budget(budget_path: Path) -> PerfBudget:
    """Load performance budget from YAML file.

    Args:
        budget_path: Path to perf_budget.yaml.

    Returns:
        PerfBudget instance.
    """
    with open(budget_path) as f:
        data = yaml.safe_load(f)

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

    # For 'max' mode, regression is when current < baseline - threshold
    # For 'min' mode, regression is when current > baseline + threshold
    if budget.mode == "max":
        metric_passed = metric_delta >= -budget.max_sharpe_drop
        if not metric_passed:
            messages.append(
                f"FAIL: Metric regression detected. "
                f"Current {budget.metric}={current.best_metric:.4f}, "
                f"baseline={baseline_best:.4f}, "
                f"drop={-metric_delta:.4f} > max_drop={budget.max_sharpe_drop}"
            )
    else:  # min mode
        metric_passed = metric_delta <= budget.max_sharpe_drop
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
        "passed": metric_passed,
    }

    # Check timing
    timing_p50_delta = current.timing["p50"] - baseline.timing.get("p50", 0.0)
    timing_p90_delta = current.timing["p90"] - baseline.timing.get("p90", 0.0)

    timing_passed = current.timing["p90"] <= budget.max_p90_elapsed_s
    if not timing_passed:
        messages.append(
            f"FAIL: Timing budget exceeded. "
            f"P90={current.timing['p90']:.2f}s > max={budget.max_p90_elapsed_s}s"
        )

    # Optional P50 check
    if budget.max_p50_elapsed_s is not None and current.timing["p50"] > budget.max_p50_elapsed_s:
        timing_passed = False
        messages.append(
            f"FAIL: P50 timing budget exceeded. "
            f"P50={current.timing['p50']:.2f}s > max={budget.max_p50_elapsed_s}s"
        )

    details["timing"] = {
        "current_p50": current.timing["p50"],
        "current_p90": current.timing["p90"],
        "baseline_p50": baseline.timing.get("p50", 0.0),
        "baseline_p90": baseline.timing.get("p90", 0.0),
        "delta_p50": timing_p50_delta,
        "delta_p90": timing_p90_delta,
        "budget_p90": budget.max_p90_elapsed_s,
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

    # Overall verdict
    passed = metric_passed and timing_passed and success_rate_passed
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
