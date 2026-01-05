"""Tests for PatchTST model I/O and inference."""

import numpy as np
import pytest

# Skip all tests if PyTorch is not available
torch = pytest.importorskip("torch")

from traderbot.model.patchtst import (  # noqa: E402
    PatchTSTConfig,
    PatchTSTModel,
    batch_inference,
    create_feature_tensor,
    export_torchscript,
    load_torchscript,
)


class TestPatchTSTConfig:
    """Tests for PatchTSTConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = PatchTSTConfig()
        assert config.lookback == 32
        assert config.n_features == 6
        assert config.patch_size == 8
        assert config.stride == 4
        assert config.d_model == 64
        assert config.n_heads == 4
        assert config.n_layers == 2
        assert config.d_ff == 128
        assert config.dropout == 0.1
        assert config.output_dim == 1

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = PatchTSTConfig(
            lookback=64,
            n_features=10,
            d_model=128,
        )
        assert config.lookback == 64
        assert config.n_features == 10
        assert config.d_model == 128


class TestPatchTSTModel:
    """Tests for PatchTSTModel."""

    @pytest.fixture
    def model(self) -> PatchTSTModel:
        """Create a model for testing."""
        config = PatchTSTConfig(lookback=32, n_features=6)
        return PatchTSTModel(config)

    def test_model_creation(self, model: PatchTSTModel) -> None:
        """Test model can be created."""
        assert model is not None
        assert model.config.lookback == 32
        assert model.config.n_features == 6

    def test_forward_pass(self, model: PatchTSTModel) -> None:
        """Test forward pass produces correct output shape."""
        batch_size = 4
        x = torch.randn(batch_size, 32, 6)

        output = model(x)

        assert output.shape == (batch_size, 1)

    def test_output_range(self, model: PatchTSTModel) -> None:
        """Test output is in [0, 1] due to sigmoid."""
        x = torch.randn(10, 32, 6)

        output = model(x)

        assert (output >= 0).all()
        assert (output <= 1).all()

    def test_predict_method(self, model: PatchTSTModel) -> None:
        """Test predict method (TorchScript export)."""
        x = torch.randn(2, 32, 6)

        output = model.predict(x)

        assert output.shape == (2, 1)
        assert (output >= 0).all()
        assert (output <= 1).all()

    def test_eval_mode(self, model: PatchTSTModel) -> None:
        """Test model in eval mode produces consistent output."""
        model.eval()
        x = torch.randn(1, 32, 6)

        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        assert torch.allclose(out1, out2)


class TestTorchScriptExport:
    """Tests for TorchScript export/load."""

    @pytest.fixture
    def model(self) -> PatchTSTModel:
        """Create a model for testing."""
        config = PatchTSTConfig(lookback=32, n_features=6)
        return PatchTSTModel(config)

    def test_export_and_load(self, model: PatchTSTModel, tmp_path) -> None:
        """Test export to TorchScript and reload."""
        model_path = tmp_path / "test_model.ts"

        # Export
        export_torchscript(model, model_path)
        assert model_path.exists()

        # Load
        loaded = load_torchscript(model_path)
        assert loaded is not None

    def test_loaded_model_inference(self, model: PatchTSTModel, tmp_path) -> None:
        """Test loaded model produces same output."""
        model_path = tmp_path / "test_model.ts"
        model.eval()

        # Get original output
        x = torch.randn(1, 32, 6)
        with torch.no_grad():
            original_out = model(x)

        # Export and reload
        export_torchscript(model, model_path)
        loaded = load_torchscript(model_path)

        # Get loaded output
        with torch.no_grad():
            loaded_out = loaded(x)

        assert torch.allclose(original_out, loaded_out, atol=1e-5)

    def test_load_nonexistent_raises(self, tmp_path) -> None:
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_torchscript(tmp_path / "nonexistent.ts")


class TestFeatureTensor:
    """Tests for feature tensor creation."""

    def test_create_feature_tensor(self) -> None:
        """Test creating feature tensor from dict."""
        features = {
            "close_ret_1": np.array([0.01, 0.02, -0.01, 0.03]),
            "rsi_14": np.array([50.0, 55.0, 45.0, 60.0]),
        }
        feature_names = ["close_ret_1", "rsi_14"]
        lookback = 4

        tensor = create_feature_tensor(features, feature_names, lookback)

        assert tensor.shape == (1, 4, 2)
        assert tensor.dtype == np.float32

    def test_feature_tensor_padding(self) -> None:
        """Test feature tensor handles short data with padding."""
        features = {
            "close_ret_1": np.array([0.01, 0.02]),  # Only 2 values
        }
        feature_names = ["close_ret_1"]
        lookback = 4

        tensor = create_feature_tensor(features, feature_names, lookback)

        assert tensor.shape == (1, 4, 1)
        # First 2 values should be zero (padding)
        assert tensor[0, 0, 0] == 0.0
        assert tensor[0, 1, 0] == 0.0
        # Last 2 values should be from data
        assert tensor[0, 2, 0] == pytest.approx(0.01)
        assert tensor[0, 3, 0] == pytest.approx(0.02)

    def test_feature_tensor_missing_feature(self) -> None:
        """Test missing features are zeros."""
        features = {
            "close_ret_1": np.array([0.01, 0.02, 0.03, 0.04]),
        }
        feature_names = ["close_ret_1", "missing_feature"]
        lookback = 4

        tensor = create_feature_tensor(features, feature_names, lookback)

        assert tensor.shape == (1, 4, 2)
        # First feature should have data
        assert tensor[0, :, 0].sum() != 0
        # Missing feature should be zeros
        assert tensor[0, :, 1].sum() == 0


class TestBatchInference:
    """Tests for batch inference."""

    @pytest.fixture
    def model(self, tmp_path) -> None:
        """Create and export a model for testing."""
        config = PatchTSTConfig(lookback=32, n_features=6)
        model = PatchTSTModel(config)
        model_path = tmp_path / "test_model.ts"
        export_torchscript(model, model_path)
        return load_torchscript(model_path)

    def test_batch_inference(self, model) -> None:
        """Test batch inference returns correct shape."""
        batch = np.random.randn(8, 32, 6).astype(np.float32)

        predictions = batch_inference(model, batch)

        assert predictions.shape == (8, 1)

    def test_batch_inference_single(self, model) -> None:
        """Test batch inference with single sample."""
        batch = np.random.randn(1, 32, 6).astype(np.float32)

        predictions = batch_inference(model, batch)

        assert predictions.shape == (1, 1)
        assert 0 <= predictions[0, 0] <= 1

    def test_batch_inference_deterministic(self, model) -> None:
        """Test batch inference is deterministic."""
        batch = np.random.randn(4, 32, 6).astype(np.float32)

        pred1 = batch_inference(model, batch)
        pred2 = batch_inference(model, batch)

        np.testing.assert_array_almost_equal(pred1, pred2)
