#!/usr/bin/env python3
"""Train PatchTST model on OHLCV data.

Builds features from Parquet files, performs time-based train/val split,
trains the PatchTST model, and exports to TorchScript.

Outputs:
- models/patchtst.ts (TorchScript model)
- runs/{timestamp}/train_patchtst.json (training metrics)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Check for PyTorch availability
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: PyTorch is required for training. Install with: pip install torch")
    sys.exit(1)

from traderbot.config import get_config
from traderbot.data.adapters.parquet_local import ParquetLocalAdapter
from traderbot.features.ta import compute_model_features
from traderbot.logging_setup import get_logger
from traderbot.model.patchtst import PatchTSTConfig, PatchTSTModel, export_torchscript

logger = get_logger("scripts.train_patchtst")

# Defaults
DEFAULT_DATA_DIR = "data/ohlcv"
DEFAULT_MODEL_PATH = "models/patchtst.ts"
DEFAULT_RUNS_DIR = "runs"
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-4
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_SEED = 42


class TimeSeriesDataset(Dataset):
    """Dataset for time series sequences with labels."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ):
        """Initialize dataset.

        Args:
            features: Feature array [n_samples, lookback, n_features]
            labels: Label array [n_samples]
        """
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).float().unsqueeze(1)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_prepare_data(
    data_dir: Path,
    feature_names: list[str],
    lookback: int,
    horizon: int = 1,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load OHLCV data and prepare features/labels.

    Args:
        data_dir: Directory containing parquet files.
        feature_names: List of feature names to compute.
        lookback: Number of lookback periods.
        horizon: Prediction horizon (default 1 = next day).
        start_date: Optional start date filter (YYYY-MM-DD).
        end_date: Optional end date filter (YYYY-MM-DD).

    Returns:
        Tuple of (features, labels) arrays.
    """
    adapter = ParquetLocalAdapter(data_dir=data_dir)
    tickers = [f.stem for f in data_dir.glob("*.parquet")]

    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for ticker in tickers:
        try:
            df = adapter.load(ticker)

            # Apply date filtering if specified
            if start_date is not None or end_date is not None:
                df["date"] = pd.to_datetime(df["date"])
                if start_date is not None:
                    df = df[df["date"] >= start_date]
                if end_date is not None:
                    df = df[df["date"] <= end_date]
                df = df.reset_index(drop=True)

            if len(df) < lookback + horizon + 10:
                continue

            # Compute features
            feature_dict = compute_model_features(df, feature_names)

            # Stack features into array
            feature_arrays = []
            for name in feature_names:
                arr = feature_dict[name].values if name in feature_dict else np.zeros(len(df))
                feature_arrays.append(arr)

            # Shape: [time, n_features]
            stacked = np.column_stack(feature_arrays)

            # Create labels: 1 if close[t+horizon] > close[t], else 0
            close = df["close"].values
            labels = (close[horizon:] > close[:-horizon]).astype(np.float32)

            # Pad labels to match feature length
            labels = np.concatenate([np.zeros(horizon), labels])

            # Create sequences
            for i in range(lookback, len(stacked) - horizon):
                # Features: [lookback, n_features]
                seq_features = stacked[i - lookback : i]

                # Handle NaN values
                if np.isnan(seq_features).any():
                    # Forward fill then backward fill
                    seq_df = pd.DataFrame(seq_features)
                    seq_df = seq_df.ffill().bfill().fillna(0)
                    seq_features = seq_df.values

                # Label for this sequence
                seq_label = labels[i]

                all_features.append(seq_features)
                all_labels.append(seq_label)

        except Exception as e:
            logger.warning(f"Error processing {ticker}: {e}")
            continue

    if not all_features:
        raise ValueError("No valid data found in data directory")

    features = np.array(all_features, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.float32)

    logger.info(f"Loaded {len(features)} samples from {len(tickers)} tickers")
    return features, labels


def time_based_split(
    features: np.ndarray,
    labels: np.ndarray,
    val_split: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data by time (earlier data for train, later for val).

    Args:
        features: Feature array [n_samples, lookback, n_features]
        labels: Label array [n_samples]
        val_split: Fraction of data for validation.

    Returns:
        Tuple of (train_features, train_labels, val_features, val_labels)
    """
    n_samples = len(features)
    split_idx = int(n_samples * (1 - val_split))

    train_features = features[:split_idx]
    train_labels = labels[:split_idx]
    val_features = features[split_idx:]
    val_labels = labels[split_idx:]

    return train_features, train_labels, val_features, val_labels


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch.

    Returns:
        Tuple of (loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        predictions = (outputs > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validate model.

    Returns:
        Tuple of (loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * features.size(0)
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(
    data_dir: Path,
    model_path: Path,
    runs_dir: Path,
    feature_names: list[str],
    lookback: int,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LR,
    val_split: float = DEFAULT_VAL_SPLIT,
    seed: int = DEFAULT_SEED,
    config_overrides: dict | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Train PatchTST model.

    Args:
        data_dir: Directory containing OHLCV parquet files.
        model_path: Output path for TorchScript model.
        runs_dir: Directory for run logs.
        feature_names: List of feature names.
        lookback: Lookback period.
        epochs: Number of training epochs.
        batch_size: Batch size.
        learning_rate: Learning rate.
        val_split: Validation split fraction.
        seed: Random seed.
        config_overrides: Optional overrides for PatchTSTConfig.
        start_date: Optional start date filter (YYYY-MM-DD).
        end_date: Optional end date filter (YYYY-MM-DD).

    Returns:
        Dict with training metrics.
    """
    set_seed(seed)

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    date_info = ""
    if start_date or end_date:
        date_info = f" (filtered: {start_date or '*'} to {end_date or '*'})"
    logger.info(f"Loading data from {data_dir}{date_info}")

    features, labels = load_and_prepare_data(
        data_dir=data_dir,
        feature_names=feature_names,
        lookback=lookback,
        start_date=start_date,
        end_date=end_date,
    )

    # Split data
    train_features, train_labels, val_features, val_labels = time_based_split(
        features, labels, val_split
    )

    logger.info(f"Train samples: {len(train_features)}, Val samples: {len(val_features)}")

    # Create datasets and dataloaders
    train_dataset = TimeSeriesDataset(train_features, train_labels)
    val_dataset = TimeSeriesDataset(val_features, val_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # Create model config
    config_dict = {
        "lookback": lookback,
        "n_features": len(feature_names),
    }
    if config_overrides:
        config_dict.update(config_overrides)

    config = PatchTSTConfig(**config_dict)
    model = PatchTSTModel(config)

    # Device (CPU-only as per spec)
    device = torch.device("cpu")
    torch.set_num_threads(1)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_val_loss = float("inf")
    best_val_acc = 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    logger.info(f"Starting training for {epochs} epochs")

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

    # Export to TorchScript
    model_path.parent.mkdir(parents=True, exist_ok=True)
    export_torchscript(model, model_path)
    logger.info(f"Saved TorchScript model to {model_path}")

    # Prepare metrics
    metrics = {
        "timestamp": timestamp,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "val_split": val_split,
        "lookback": lookback,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "start_date": start_date,
        "end_date": end_date,
        "train_samples": len(train_features),
        "val_samples": len(val_features),
        "final_train_loss": history["train_loss"][-1],
        "final_train_acc": history["train_acc"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_val_acc": history["val_acc"][-1],
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "model_config": {
            "lookback": config.lookback,
            "n_features": config.n_features,
            "patch_size": config.patch_size,
            "stride": config.stride,
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "n_layers": config.n_layers,
            "d_ff": config.d_ff,
            "dropout": config.dropout,
            "output_dim": config.output_dim,
        },
        "history": history,
        "model_path": str(model_path),
    }

    # Save metrics
    metrics_path = run_dir / "train_patchtst.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved training metrics to {metrics_path}")

    return metrics


def main(args: list[str] | None = None) -> int:
    """Main entry point.

    Args:
        args: Command line arguments.

    Returns:
        Exit code.
    """
    config = get_config()

    parser = argparse.ArgumentParser(description="Train PatchTST model on OHLCV data")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing OHLCV parquet files (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Output path for TorchScript model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"Directory for run logs (default: {DEFAULT_RUNS_DIR})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LR,
        help=f"Learning rate (default: {DEFAULT_LR})",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=DEFAULT_VAL_SPLIT,
        help=f"Validation split fraction (default: {DEFAULT_VAL_SPLIT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=config.model.lookback,
        help=f"Lookback period (default: {config.model.lookback})",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=",".join(config.model.features),
        help=f"Comma-separated feature names (default: {','.join(config.model.features)})",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date filter for training data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date filter for training data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path for TorchScript model (overrides --model-path)",
    )

    parsed = parser.parse_args(args)

    feature_names = [f.strip() for f in parsed.features.split(",")]

    # --out overrides --model-path
    model_path = parsed.out if parsed.out else parsed.model_path

    print("Training PatchTST model")
    print(f"  Data directory: {parsed.data_dir}")
    print(f"  Model output: {model_path}")
    print(f"  Runs directory: {parsed.runs_dir}")
    print(f"  Epochs: {parsed.epochs}")
    print(f"  Batch size: {parsed.batch_size}")
    print(f"  Learning rate: {parsed.learning_rate}")
    print(f"  Val split: {parsed.val_split}")
    print(f"  Seed: {parsed.seed}")
    print(f"  Lookback: {parsed.lookback}")
    print(f"  Features: {feature_names}")
    if parsed.start_date:
        print(f"  Start date: {parsed.start_date}")
    if parsed.end_date:
        print(f"  End date: {parsed.end_date}")
    print()

    try:
        metrics = train_model(
            data_dir=parsed.data_dir,
            model_path=model_path,
            runs_dir=parsed.runs_dir,
            feature_names=feature_names,
            lookback=parsed.lookback,
            epochs=parsed.epochs,
            batch_size=parsed.batch_size,
            learning_rate=parsed.learning_rate,
            val_split=parsed.val_split,
            seed=parsed.seed,
            start_date=parsed.start_date,
            end_date=parsed.end_date,
        )

        print()
        print("Training completed!")
        print(f"  Final val loss: {metrics['final_val_loss']:.4f}")
        print(f"  Final val accuracy: {metrics['final_val_acc']:.4f}")
        print(f"  Best val loss: {metrics['best_val_loss']:.4f}")
        print(f"  Best val accuracy: {metrics['best_val_acc']:.4f}")
        print(f"  Model saved to: {metrics['model_path']}")

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
