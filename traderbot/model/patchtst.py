"""PatchTST model for multivariate time series forecasting.

Implements a simplified PatchTST architecture for probability-of-uptrend prediction.
Supports TorchScript export for CPU inference.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Try to import torch, but provide graceful fallback
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from traderbot.logging_setup import get_logger

logger = get_logger("model.patchtst")


@dataclass
class PatchTSTConfig:
    """Configuration for PatchTST model."""

    lookback: int = 32
    n_features: int = 6
    patch_size: int = 8
    stride: int = 4
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1
    output_dim: int = 1  # prob_up


def _check_torch() -> None:
    """Check if PyTorch is available."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for PatchTST model. " "Install with: pip install torch"
        )


if TORCH_AVAILABLE:

    class PatchEmbedding(nn.Module):
        """Patch embedding layer for time series."""

        def __init__(
            self,
            n_features: int,
            patch_size: int,
            stride: int,
            d_model: int,
        ):
            super().__init__()
            self.patch_size = patch_size
            self.stride = stride
            self.n_features = n_features

            # Linear projection for each patch
            self.projection = nn.Linear(patch_size * n_features, d_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Create patch embeddings from input.

            Args:
                x: Input tensor [batch, lookback, n_features]

            Returns:
                Patch embeddings [batch, n_patches, d_model]
            """
            batch_size, seq_len, n_features = x.shape

            # Unfold to create patches
            # [batch, n_patches, patch_size, n_features]
            patches = x.unfold(1, self.patch_size, self.stride)

            # Reshape to [batch, n_patches, patch_size * n_features]
            n_patches = patches.shape[1]
            patches = patches.reshape(batch_size, n_patches, -1)

            # Project to d_model
            return self.projection(patches)

    class TransformerEncoderBlock(nn.Module):
        """Single transformer encoder block."""

        def __init__(
            self,
            d_model: int,
            n_heads: int,
            d_ff: int,
            dropout: float,
        ):
            super().__init__()

            self.attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through encoder block.

            Args:
                x: Input tensor [batch, seq_len, d_model]

            Returns:
                Output tensor [batch, seq_len, d_model]
            """
            # Self-attention with residual
            attn_out, _ = self.attention(x, x, x)
            x = self.norm1(x + attn_out)

            # Feed-forward with residual
            ff_out = self.ff(x)
            x = self.norm2(x + ff_out)

            return x

    class PatchTSTModel(nn.Module):
        """PatchTST model for time series classification.

        Accepts 3D tensor [batch, lookback, n_features] and returns
        per-symbol probability of uptrend in [0, 1].
        """

        def __init__(self, config: PatchTSTConfig | None = None):
            super().__init__()

            if config is None:
                config = PatchTSTConfig()

            self.config = config

            # Calculate number of patches
            n_patches = (config.lookback - config.patch_size) // config.stride + 1

            # Patch embedding
            self.patch_embed = PatchEmbedding(
                n_features=config.n_features,
                patch_size=config.patch_size,
                stride=config.stride,
                d_model=config.d_model,
            )

            # Positional encoding (learnable)
            self.pos_embed = nn.Parameter(torch.randn(1, n_patches, config.d_model) * 0.02)

            # Transformer encoder layers
            self.encoder_layers = nn.ModuleList(
                [
                    TransformerEncoderBlock(
                        d_model=config.d_model,
                        n_heads=config.n_heads,
                        d_ff=config.d_ff,
                        dropout=config.dropout,
                    )
                    for _ in range(config.n_layers)
                ]
            )

            # Classification head
            self.head = nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.output_dim),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: Input tensor [batch, lookback, n_features]

            Returns:
                Probability of uptrend [batch, 1] in [0, 1]
            """
            # Patch embedding
            x = self.patch_embed(x)

            # Add positional encoding
            x = x + self.pos_embed

            # Transformer encoder
            for layer in self.encoder_layers:
                x = layer(x)

            # Global average pooling over patches
            x = x.mean(dim=1)

            # Classification head
            return self.head(x)

        @torch.jit.export
        def predict(self, x: torch.Tensor) -> torch.Tensor:
            """Predict probability of uptrend.

            This method is exported for TorchScript.

            Args:
                x: Input tensor [batch, lookback, n_features]

            Returns:
                Probability tensor [batch, 1]
            """
            return self.forward(x)

else:
    # Stub class when PyTorch is not available
    class PatchTSTModel:  # type: ignore[no-redef]
        """Stub PatchTST model when PyTorch is not available."""

        def __init__(self, config: PatchTSTConfig | None = None):
            _check_torch()


def export_torchscript(
    model: Any,
    path: str | Path,
    example_input: Any | None = None,
) -> Path:
    """Export model to TorchScript for CPU inference.

    Args:
        model: PatchTSTModel instance.
        path: Output path for .ts file.
        example_input: Example input tensor for tracing. Auto-generated if None.

    Returns:
        Path to saved TorchScript model.
    """
    _check_torch()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Set to eval mode
    model.eval()

    # Set CPU inference optimization
    torch.set_num_threads(1)

    if example_input is None:
        # Create example input based on config
        config = model.config
        example_input = torch.randn(1, config.lookback, config.n_features)

    # Script the model (better than tracing for control flow)
    scripted = torch.jit.script(model)

    # Save
    scripted.save(str(path))
    logger.info(f"Exported TorchScript model to {path}")

    return path


def load_torchscript(path: str | Path) -> Any:
    """Load TorchScript model for CPU inference.

    Args:
        path: Path to .ts file.

    Returns:
        TorchScript model ready for inference.
    """
    _check_torch()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    # Set CPU inference optimization
    torch.set_num_threads(1)

    # Load model
    model = torch.jit.load(str(path), map_location="cpu")
    model.eval()

    logger.info(f"Loaded TorchScript model from {path}")
    return model


def create_feature_tensor(
    features: dict[str, np.ndarray],
    feature_names: list[str],
    lookback: int,
) -> np.ndarray:
    """Create feature tensor for model input.

    Args:
        features: Dict mapping feature name to array of values.
        feature_names: Ordered list of feature names to include.
        lookback: Number of lookback periods.

    Returns:
        Feature tensor [1, lookback, n_features]
    """
    _check_torch()

    n_features = len(feature_names)
    tensor = np.zeros((1, lookback, n_features), dtype=np.float32)

    for i, name in enumerate(feature_names):
        if name in features:
            values = features[name]
            # Take last `lookback` values, pad with zeros if needed
            if len(values) >= lookback:
                tensor[0, :, i] = values[-lookback:]
            else:
                tensor[0, -len(values) :, i] = values

    return tensor


def batch_inference(
    model: Any,
    batch_tensor: np.ndarray,
) -> np.ndarray:
    """Run batch inference on CPU.

    Args:
        model: TorchScript model.
        batch_tensor: Input tensor [batch, lookback, n_features]

    Returns:
        Predictions [batch, 1]
    """
    _check_torch()

    with torch.no_grad():
        input_tensor = torch.from_numpy(batch_tensor).float()
        output = model(input_tensor)
        result: np.ndarray = output.numpy()
        return result
