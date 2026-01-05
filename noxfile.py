"""Nox automation for TraderBot development tasks."""

import json
from pathlib import Path

import nox

# Default sessions to run
nox.options.sessions = ["lint", "test"]


@nox.session(python=["3.10", "3.11", "3.12", "3.13"])
def test(session: nox.Session) -> None:
    """Run the test suite with coverage."""
    session.install("pytest", "pytest-cov")
    session.install("-e", ".")
    session.run(
        "pytest",
        "--cov=traderbot",
        "--cov-report=term-missing",
        "--cov-fail-under=70",
        "-q",
        *session.posargs,
    )


@nox.session(python="3.11")
def lint(session: nox.Session) -> None:
    """Run linting with ruff and black."""
    session.install("ruff", "black")
    session.run("ruff", "check", ".")
    session.run("black", "--check", ".")


@nox.session(python="3.11")
def typecheck(session: nox.Session) -> None:
    """Run type checking with mypy."""
    session.install("mypy", "pandas-stubs", "types-python-dateutil")
    session.install("-e", ".")
    session.run("mypy", "traderbot")


@nox.session(python=False)
def smoke(session: nox.Session) -> None:
    """Run a minimal CLI smoke test without virtualenv.

    This creates minimal test fixtures and runs the regression CLI
    to verify basic functionality works.
    """
    root = Path(".")

    # Create sweep directory with minimal data
    sweep = root / "runs" / "sweeps" / "ci_smoke"
    sweep.mkdir(parents=True, exist_ok=True)

    (sweep / "sweep_meta.json").write_text(
        json.dumps({"metric": "sharpe", "mode": "max"})
    )
    (sweep / "all_results.json").write_text(
        json.dumps(
            [
                {
                    "_status": "success",
                    "_run_idx": 0,
                    "_elapsed_seconds": 11.0,
                    "avg_oos_sharpe": 1.1,
                },
                {
                    "_status": "success",
                    "_run_idx": 1,
                    "_elapsed_seconds": 13.0,
                    "avg_oos_sharpe": 1.4,
                },
            ]
        )
    )

    # Create perf budget
    (root / "sweeps").mkdir(exist_ok=True)
    (root / "sweeps" / "perf_budget.yaml").write_text(
        "metric: sharpe\n"
        "mode: max\n"
        "min_success_rate: 0.5\n"
        "max_p90_elapsed_s: 300.0\n"
        "max_sharpe_drop: 1.0\n"
        "epsilon_abs: 1e-6\n"
    )

    # Create baseline
    (root / "benchmarks").mkdir(exist_ok=True)
    (root / "benchmarks" / "baseline.json").write_text(
        json.dumps(
            {
                "git_sha": "abc1234",
                "created_utc": "2026-01-01T00:00:00+00:00",
                "metric": "sharpe",
                "mode": "max",
                "leaderboard": [],
                "timing": {"p50": 100, "p90": 200},
                "summary": {"best_metric": 1.0, "success_rate": 1.0, "total_runs": 1},
            }
        )
    )

    # Run regression CLI
    session.run(
        "python",
        "-m",
        "traderbot.cli.regress",
        "compare",
        "--no-emoji",
        "--current",
        str(sweep),
        "--baseline",
        "benchmarks/baseline.json",
        "--budget",
        "sweeps/perf_budget.yaml",
        "--out",
        str(sweep / "report.md"),
    )

    session.log("Smoke test passed!")


@nox.session(python=False)
def clean(session: nox.Session) -> None:
    """Clean up generated files and caches."""
    import shutil

    paths_to_remove = [
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".coverage",
        "htmlcov",
        ".nox",
        "dist",
        "build",
        "*.egg-info",
    ]

    for pattern in paths_to_remove:
        for path in Path(".").glob(pattern):
            session.log(f"Removing {path}")
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
