"""Run manifest for reproducibility tracking.

Provides a single source of truth for all run parameters and configuration.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunManifest:
    """Manifest capturing all information needed to reproduce a run."""

    run_id: str
    git_sha: str
    seed: int
    params: dict[str, Any]
    universe: list[str]
    start_date: str
    end_date: str
    n_splits: int
    is_ratio: float
    sizer: str
    sizer_params: dict[str, Any] = field(default_factory=dict)
    data_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "run_id": self.run_id,
            "git_sha": self.git_sha,
            "seed": self.seed,
            "params": self.params,
            "universe": self.universe,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "n_splits": self.n_splits,
            "is_ratio": self.is_ratio,
            "sizer": self.sizer,
            "sizer_params": self.sizer_params,
            "data_digest": self.data_digest,
        }


def create_run_manifest(
    *,
    run_id: str,
    git_sha: str,
    seed: int,
    start_date: str,
    end_date: str,
    universe: list[str],
    n_splits: int,
    is_ratio: float,
    sizer: str,
    sizer_params: dict[str, Any],
    all_cli_params: dict[str, Any],
) -> RunManifest:
    """Create a run manifest from parameters.

    Args:
        run_id: Unique run identifier (timestamp-based).
        git_sha: Git commit SHA.
        seed: Random seed used.
        start_date: Start date string.
        end_date: End date string.
        universe: List of tickers.
        n_splits: Number of walk-forward splits.
        is_ratio: In-sample ratio.
        sizer: Position sizer type.
        sizer_params: Position sizer parameters.
        all_cli_params: All CLI parameters after applying defaults.

    Returns:
        RunManifest instance.
    """
    # Compute data digest (simple SHA256 of universe + date range)
    data_str = f"{','.join(sorted(universe))}|{start_date}|{end_date}"
    data_digest = hashlib.sha256(data_str.encode()).hexdigest()[:16]

    return RunManifest(
        run_id=run_id,
        git_sha=git_sha,
        seed=seed,
        params=all_cli_params,
        universe=universe,
        start_date=start_date,
        end_date=end_date,
        n_splits=n_splits,
        is_ratio=is_ratio,
        sizer=sizer,
        sizer_params=sizer_params,
        data_digest=data_digest,
    )


def to_jsonable(manifest: RunManifest) -> dict[str, Any]:
    """Convert RunManifest to JSON-serializable dictionary.

    Args:
        manifest: RunManifest instance.

    Returns:
        JSON-safe dictionary.
    """
    return manifest.to_dict()

