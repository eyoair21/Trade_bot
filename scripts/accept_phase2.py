"""Phase 2 acceptance check script.

Verifies that Phase 2 achieves ≥2% improvement in total_reward over baseline.
"""

import argparse
import json
import sys
from pathlib import Path


def load_metrics(results_path: Path) -> dict:
    """Load metrics from results.json."""
    with open(results_path) as f:
        return json.load(f)


def check_acceptance(
    baseline_metrics: dict,
    current_metrics: dict,
    threshold: float = 1.02,
) -> tuple[bool, str]:
    """Check if current run meets acceptance criteria.
    
    Args:
        baseline_metrics: Baseline run metrics
        current_metrics: Current run metrics
        threshold: Minimum improvement factor (1.02 = 2%)
    
    Returns:
        Tuple of (pass, message)
    """
    baseline_reward = baseline_metrics.get("total_reward", 0.0)
    current_reward = current_metrics.get("total_reward", 0.0)
    
    if baseline_reward == 0:
        return False, "Baseline total_reward is 0 or missing"
    
    improvement = current_reward / baseline_reward
    
    passed = improvement >= threshold
    
    message = f"""
Phase 2 Acceptance Check
========================

Baseline total_reward:  {baseline_reward:.2f}
Current total_reward:   {current_reward:.2f}
Improvement factor:     {improvement:.4f} ({(improvement - 1) * 100:.2f}%)
Required threshold:     {threshold:.4f} ({(threshold - 1) * 100:.2f}%)

Status: {'✅ PASS' if passed else '❌ FAIL'}
"""
    
    return passed, message


def main() -> int:
    """Run acceptance check."""
    parser = argparse.ArgumentParser(description="Phase 2 acceptance check")
    parser.add_argument(
        "--baseline",
        type=str,
        default="runs/baseline/results.json",
        help="Path to baseline results.json",
    )
    parser.add_argument(
        "--current",
        type=str,
        default="runs/phase2/results.json",
        help="Path to current results.json",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.02,
        help="Minimum improvement threshold (1.02 = 2%)",
    )
    
    args = parser.parse_args()
    
    baseline_path = Path(args.baseline)
    current_path = Path(args.current)
    
    # Check files exist
    if not baseline_path.exists():
        print(f"❌ Baseline file not found: {baseline_path}")
        return 1
    
    if not current_path.exists():
        print(f"❌ Current file not found: {current_path}")
        return 1
    
    # Load metrics
    try:
        baseline_metrics = load_metrics(baseline_path)
        current_metrics = load_metrics(current_path)
    except Exception as e:
        print(f"❌ Error loading metrics: {e}")
        return 1
    
    # Check acceptance
    passed, message = check_acceptance(baseline_metrics, current_metrics, args.threshold)
    
    print(message)
    
    # Additional metrics comparison
    print("\nAdditional Metrics:")
    print("-" * 50)
    
    for metric in ["sharpe", "sortino", "max_dd", "total_trades"]:
        baseline_val = baseline_metrics.get(metric, 0.0)
        current_val = current_metrics.get(metric, 0.0)
        
        if baseline_val != 0:
            change = ((current_val - baseline_val) / abs(baseline_val)) * 100
            symbol = "↑" if change > 0 else "↓"
            print(f"{metric:15} {baseline_val:10.3f} → {current_val:10.3f} ({symbol} {abs(change):.1f}%)")
        else:
            print(f"{metric:15} {baseline_val:10.3f} → {current_val:10.3f}")
    
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

