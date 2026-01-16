"""Tests for configuration module."""

from pathlib import Path

import pytest

from traderbot.config import (
    BacktestConfig,
    Config,
    DataConfig,
    RiskConfig,
    StrategyConfig,
    _get_env_float,
    _get_env_int,
    _get_env_str,
    get_config,
    reset_config,
)


class TestEnvHelpers:
    """Tests for environment variable helpers."""

    def test_get_env_str_with_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test getting string from env."""
        monkeypatch.setenv("TEST_VAR", "test_value")
        assert _get_env_str("TEST_VAR", "default") == "test_value"

    def test_get_env_str_default(self) -> None:
        """Test string default when env not set."""
        assert _get_env_str("NONEXISTENT_VAR", "default") == "default"

    def test_get_env_float_with_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test getting float from env."""
        monkeypatch.setenv("TEST_FLOAT", "3.14")
        assert _get_env_float("TEST_FLOAT", 0.0) == pytest.approx(3.14)

    def test_get_env_float_invalid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test float default when env is invalid."""
        monkeypatch.setenv("TEST_FLOAT", "not_a_number")
        assert _get_env_float("TEST_FLOAT", 1.0) == 1.0

    def test_get_env_int_with_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test getting int from env."""
        monkeypatch.setenv("TEST_INT", "42")
        assert _get_env_int("TEST_INT", 0) == 42

    def test_get_env_int_invalid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test int default when env is invalid."""
        monkeypatch.setenv("TEST_INT", "not_a_number")
        assert _get_env_int("TEST_INT", 10) == 10


class TestDataConfig:
    """Tests for DataConfig."""

    def test_from_env_defaults(self) -> None:
        """Test DataConfig with defaults."""
        config = DataConfig.from_env()
        assert config.data_dir == Path("./data")
        assert config.ohlcv_dir == Path("./data/ohlcv")

    def test_from_env_custom(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test DataConfig with custom env vars."""
        monkeypatch.setenv("DATA_DIR", "/custom/data")
        monkeypatch.setenv("OHLCV_DIR", "/custom/ohlcv")

        config = DataConfig.from_env()
        assert config.data_dir == Path("/custom/data")
        assert config.ohlcv_dir == Path("/custom/ohlcv")


class TestBacktestConfig:
    """Tests for BacktestConfig."""

    def test_from_env_defaults(self) -> None:
        """Test BacktestConfig with defaults."""
        config = BacktestConfig.from_env()
        assert config.initial_capital == 100000.0
        assert config.commission_bps == 10.0
        assert config.slippage_bps == 5.0


class TestRiskConfig:
    """Tests for RiskConfig."""

    def test_from_env_defaults(self) -> None:
        """Test RiskConfig with defaults."""
        config = RiskConfig.from_env()
        assert config.max_position_pct == 0.10
        assert config.max_gross_exposure == 1.0
        assert config.daily_loss_limit_pct == 0.02
        assert config.max_drawdown_pct == 0.15


class TestStrategyConfig:
    """Tests for StrategyConfig."""

    def test_from_env_defaults(self) -> None:
        """Test StrategyConfig with defaults."""
        config = StrategyConfig.from_env()
        assert config.ema_fast_period == 12
        assert config.ema_slow_period == 26
        assert config.rsi_period == 14


class TestConfig:
    """Tests for main Config class."""

    def test_from_env(self) -> None:
        """Test Config.from_env creates all sub-configs."""
        config = Config.from_env()

        assert config.data is not None
        assert config.logging is not None
        assert config.backtest is not None
        assert config.risk is not None
        assert config.strategy is not None
        assert config.random_seed == 42
        assert config.runs_dir == Path("./runs")
        assert config.data.ohlcv_dir == Path("./data/ohlcv")
        assert config.reports_dir == Path("./reports")

    def test_reports_and_ohlcv_dirs_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Config should respect OHLCV_DIR and REPORTS_DIR environment variables."""
        monkeypatch.setenv("OHLCV_DIR", "/tmp/ohlcv")
        monkeypatch.setenv("REPORTS_DIR", "/tmp/reports")

        reset_config()
        config = Config.from_env()

        assert config.data.ohlcv_dir == Path("/tmp/ohlcv")
        assert config.reports_dir == Path("/tmp/reports")

    def test_config_is_frozen(self) -> None:
        """Test config is immutable (frozen dataclass)."""
        config = Config.from_env()

        with pytest.raises(AttributeError):  # FrozenInstanceError
            config.random_seed = 123  # type: ignore


class TestGetConfig:
    """Tests for get_config function."""

    def test_get_config_returns_config(self) -> None:
        """Test get_config returns Config instance."""
        config = get_config()
        assert isinstance(config, Config)

    def test_get_config_caches_result(self) -> None:
        """Test get_config returns same instance."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_reset_config(self) -> None:
        """Test reset_config clears cache."""
        config1 = get_config()
        reset_config()
        config2 = get_config()

        # Should be equal but different instances
        assert config1.random_seed == config2.random_seed
