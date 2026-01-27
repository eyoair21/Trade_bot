"""Purged + Embargo time-series splits to prevent information leakage.

Based on "Advances in Financial Machine Learning" by Marcos López de Prado.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass
class SplitConfig:
    """Configuration for purged+embargo splits.
    
    Args:
        n_splits: Number of train/test splits
        test_size: Fraction of data in test set (0-1)
        embargo_days: Days to embargo after test end (buffer)
        purge_days: Days to purge before test start (label overlap)
    """
    n_splits: int
    test_size: float
    embargo_days: int = 0
    purge_days: int = 0


class PurgedEmbargoSplit:
    """Time-series splitter with purging and embargo.
    
    Ensures no information leakage between train and test by:
    - Purging: Remove train samples whose labels overlap with test period
    - Embargo: Add buffer after test to prevent look-ahead
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: float = 0.2,
        embargo_days: int = 5,
        purge_days: int = 1,
    ):
        """Initialize splitter.
        
        Args:
            n_splits: Number of sequential splits
            test_size: Fraction of data in each test set
            embargo_days: Days to embargo after test end
            purge_days: Days to purge before test start
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.embargo_days = embargo_days
        self.purge_days = purge_days
    
    def split(
        self,
        dates: pd.DatetimeIndex,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices with purge and embargo.
        
        Args:
            dates: DatetimeIndex of sample dates
        
        Yields:
            Tuples of (train_indices, test_indices)
        """
        n_samples = len(dates)
        test_samples = int(n_samples * self.test_size)
        
        # Create sequential splits
        for i in range(self.n_splits):
            # Test window slides forward
            test_start = int(i * test_samples)
            test_end = min(test_start + test_samples, n_samples)
            
            if test_end >= n_samples:
                break
            
            # Get test dates
            test_dates_start = dates[test_start]
            test_dates_end = dates[test_end - 1]
            
            # Apply purge and embargo
            train_indices = self._get_train_indices(
                dates=dates,
                test_start_date=test_dates_start,
                test_end_date=test_dates_end,
                test_start_idx=test_start,
                test_end_idx=test_end,
            )
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices
    
    def _get_train_indices(
        self,
        dates: pd.DatetimeIndex,
        test_start_date: pd.Timestamp,
        test_end_date: pd.Timestamp,
        test_start_idx: int,
        test_end_idx: int,
    ) -> np.ndarray:
        """Compute train indices with purge and embargo applied.
        
        Args:
            dates: All dates in dataset
            test_start_date: First date in test set
            test_end_date: Last date in test set
            test_start_idx: Index of test start
            test_end_idx: Index of test end
        
        Returns:
            Array of train indices
        """
        # Start with all indices before test
        train_mask = np.ones(len(dates), dtype=bool)
        
        # Exclude test period
        train_mask[test_start_idx:test_end_idx] = False
        
        # Apply purge: remove samples whose labels overlap with test
        if self.purge_days > 0:
            purge_cutoff = test_start_date - timedelta(days=self.purge_days)
            purge_mask = dates < purge_cutoff
            train_mask = train_mask & purge_mask
        
        # Apply embargo: remove samples after test within embargo window
        if self.embargo_days > 0:
            embargo_cutoff = test_end_date + timedelta(days=self.embargo_days)
            embargo_mask = (dates < test_start_date) | (dates > embargo_cutoff)
            train_mask = train_mask & embargo_mask
        
        # Return indices where mask is True
        return np.where(train_mask)[0]


def save_splits_info(
    splits: list[tuple[np.ndarray, np.ndarray]],
    dates: pd.DatetimeIndex,
    output_path: str,
) -> None:
    """Save split information to JSON for reproducibility.
    
    Args:
        splits: List of (train_indices, test_indices) tuples
        dates: DatetimeIndex of all dates
        output_path: Path to save JSON file
    """
    import json
    
    splits_info = []
    for i, (train_idx, test_idx) in enumerate(splits):
        info = {
            "split_id": i,
            "train_start": str(dates[train_idx[0]]) if len(train_idx) > 0 else None,
            "train_end": str(dates[train_idx[-1]]) if len(train_idx) > 0 else None,
            "train_samples": len(train_idx),
            "test_start": str(dates[test_idx[0]]) if len(test_idx) > 0 else None,
            "test_end": str(dates[test_idx[-1]]) if len(test_idx) > 0 else None,
            "test_samples": len(test_idx),
        }
        splits_info.append(info)
    
    with open(output_path, "w") as f:
        json.dump({"splits": splits_info}, f, indent=2)

