"""Tests for purged+embargo splits."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from traderbot.splits.purged import PurgedEmbargoSplit, save_splits_info


def test_purged_embargo_split_basic():
    """Test basic split functionality."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    
    splitter = PurgedEmbargoSplit(n_splits=3, test_size=0.2, embargo_days=0, purge_days=0)
    
    splits = list(splitter.split(dates))
    
    assert len(splits) == 3
    
    for train_idx, test_idx in splits:
        # Train and test should not overlap
        assert len(set(train_idx) & set(test_idx)) == 0
        # Test should be contiguous
        assert len(test_idx) == int(100 * 0.2)


def test_purged_embargo_split_with_embargo():
    """Test split with embargo period."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    
    splitter = PurgedEmbargoSplit(n_splits=2, test_size=0.2, embargo_days=5, purge_days=0)
    
    splits = list(splitter.split(dates))
    
    for train_idx, test_idx in splits:
        test_end_date = dates[test_idx[-1]]
        embargo_cutoff = test_end_date + timedelta(days=5)
        
        # No train samples in embargo window (after test end)
        train_dates = dates[train_idx]
        train_after_test = train_dates[train_dates > test_end_date]
        assert all(train_after_test > embargo_cutoff)


def test_purged_embargo_split_with_purge():
    """Test split with purge period."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    
    splitter = PurgedEmbargoSplit(n_splits=2, test_size=0.2, embargo_days=0, purge_days=3)
    
    splits = list(splitter.split(dates))
    
    for train_idx, test_idx in splits:
        test_start_date = dates[test_idx[0]]
        purge_cutoff = test_start_date - timedelta(days=3)
        
        # No train samples in purge window (before test start)
        train_dates = dates[train_idx]
        train_before_test = train_dates[train_dates < test_start_date]
        assert all(train_before_test < purge_cutoff)


def test_save_splits_info():
    """Test saving split information."""
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    
    splitter = PurgedEmbargoSplit(n_splits=2, test_size=0.3)
    splits = list(splitter.split(dates))
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "splits.json"
        save_splits_info(splits, dates, str(output_path))
        
        assert output_path.exists()
        
        # Verify content
        import json
        with open(output_path) as f:
            data = json.load(f)
        
        assert "splits" in data
        assert len(data["splits"]) == len(splits)


def test_purged_embargo_no_leakage():
    """Test that purge and embargo prevent leakage."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    
    splitter = PurgedEmbargoSplit(n_splits=2, test_size=0.2, embargo_days=3, purge_days=2)
    
    splits = list(splitter.split(dates))
    
    for train_idx, test_idx in splits:
        test_start = dates[test_idx[0]]
        test_end = dates[test_idx[-1]]
        
        train_dates = dates[train_idx]
        
        # No train samples in [test_start - purge, test_end + embargo]
        forbidden_start = test_start - timedelta(days=2)
        forbidden_end = test_end + timedelta(days=3)
        
        forbidden_train = train_dates[
            (train_dates >= forbidden_start) & (train_dates <= forbidden_end)
        ]
        
        assert len(forbidden_train) == 0

