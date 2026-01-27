"""Configuration management for TraderBot.

Loads configuration from environment variables with sane defaults.
Uses python-dotenv to load from .env file if present.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env file if present
load_dotenv()


def _get_env_float(key: str, default: float) -> float:
    """Get float from environment variable."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_env_int(key: str, default: int) -> int:
    """Get int from environment variable."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_env_str(key: str, default: str) -> str:
    """Get string from environment variable."""
    return os.getenv(key, default)


@dataclass(frozen=True)
class DataConfig:
    """Data-related configuration."""

    data_dir: Path
    ohlcv_dir: Path

    @classmethod
    def from_env(cls) -> "DataConfig":
        """Create DataConfig from environment variables."""
        data_dir = Path(_get_env_str("DATA_DIR", "./data"))
        ohlcv_dir = Path(_get_env_str("OHLCV_DIR", "./data/ohlcv"))
        return cls(data_dir=data_dir, ohlcv_dir=ohlcv_dir)


@dataclass(frozen=True)
class LoggingConfig:
    """Logging-related configuration."""

    level: str
    format: str

    @classmethod
    def from_env(cls) -> "LoggingConfig":
        """Create LoggingConfig from environment variables."""
        return cls(
            level=_get_env_str("LOG_LEVEL", "INFO"),
            format=_get_env_str("LOG_FORMAT", "json"),
        )


@dataclass(frozen=True)
class BacktestConfig:
    """Backtesting-related configuration."""

    initial_capital: float
    commission_bps: float
    slippage_bps: float

    @classmethod
    def from_env(cls) -> "BacktestConfig":
        """Create BacktestConfig from environment variables."""
        return cls(
            initial_capital=_get_env_float("DEFAULT_INITIAL_CAPITAL", 100000.0),
            commission_bps=_get_env_float("DEFAULT_COMMISSION_BPS", 10.0),
            slippage_bps=_get_env_float("DEFAULT_SLIPPAGE_BPS", 5.0),
        )


@dataclass(frozen=True)
class RiskConfig:
    """Risk management configuration."""

    max_position_pct: float
    max_gross_exposure: float
    daily_loss_limit_pct: float
    max_drawdown_pct: float

    @classmethod
    def from_env(cls) -> "RiskConfig":
        """Create RiskConfig from environment variables."""
        return cls(
            max_position_pct=_get_env_float("MAX_POSITION_PCT", 0.10),
            max_gross_exposure=_get_env_float("MAX_GROSS_EXPOSURE", 1.0),
            daily_loss_limit_pct=_get_env_float("DAILY_LOSS_LIMIT_PCT", 0.02),
            max_drawdown_pct=_get_env_float("MAX_DRAWDOWN_PCT", 0.15),
        )


@dataclass(frozen=True)
class StrategyConfig:
    """Strategy-related configuration."""

    ema_fast_period: int
    ema_slow_period: int
    rsi_period: int
    rsi_oversold: int
    rsi_overbought: int
    atr_period: int
    atr_stop_multiplier: float

    @classmethod
    def from_env(cls) -> "StrategyConfig":
        """Create StrategyConfig from environment variables."""
        return cls(
            ema_fast_period=_get_env_int("EMA_FAST_PERIOD", 12),
            ema_slow_period=_get_env_int("EMA_SLOW_PERIOD", 26),
            rsi_period=_get_env_int("RSI_PERIOD", 14),
            rsi_oversold=_get_env_int("RSI_OVERSOLD", 30),
            rsi_overbought=_get_env_int("RSI_OVERBOUGHT", 70),
            atr_period=_get_env_int("ATR_PERIOD", 14),
            atr_stop_multiplier=_get_env_float("ATR_STOP_MULTIPLIER", 2.0),
        )


@dataclass(frozen=True)
class ModelConfig:
    """Machine learning model configuration."""

    lookback: int
    features: tuple[str, ...]
    model_path: Path
    patch_size: int
    stride: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    dropout: float

    @classmethod
    def from_env(cls) -> "ModelConfig":
        """Create ModelConfig from environment variables."""
        features_str = _get_env_str(
            "MODEL_FEATURES",
            "close_ret_1,rsi_14,atr_14,vwap_gap,dvol_5,regime_vix",
        )
        features = tuple(f.strip() for f in features_str.split(","))

        return cls(
            lookback=_get_env_int("MODEL_LOOKBACK", 32),
            features=features,
            model_path=Path(_get_env_str("MODEL_PATH", "./models/patchtst.ts")),
            patch_size=_get_env_int("MODEL_PATCH_SIZE", 8),
            stride=_get_env_int("MODEL_STRIDE", 4),
            d_model=_get_env_int("MODEL_D_MODEL", 64),
            n_heads=_get_env_int("MODEL_N_HEADS", 4),
            n_layers=_get_env_int("MODEL_N_LAYERS", 2),
            d_ff=_get_env_int("MODEL_D_FF", 128),
            dropout=_get_env_float("MODEL_DROPOUT", 0.1),
        )


@dataclass(frozen=True)
class UniverseConfig:
    """Dynamic universe selection configuration."""

    max_symbols: int
    min_dollar_volume: float
    min_volatility: float
    lookback_days: int

    @classmethod
    def from_env(cls) -> "UniverseConfig":
        """Create UniverseConfig from environment variables."""
        return cls(
            max_symbols=_get_env_int("UNIVERSE_MAX_SYMBOLS", 30),
            min_dollar_volume=_get_env_float("UNIVERSE_MIN_DOLLAR_VOLUME", 20_000_000.0),
            min_volatility=_get_env_float("UNIVERSE_MIN_VOLATILITY", 0.15),
            lookback_days=_get_env_int("UNIVERSE_LOOKBACK_DAYS", 20),
        )


@dataclass(frozen=True)
class ExecutionConfig:
    """Execution cost configuration."""

    slippage_bps: float
    fee_per_share: float

    @classmethod
    def from_env(cls) -> "ExecutionConfig":
        """Create ExecutionConfig from environment variables."""
        return cls(
            slippage_bps=_get_env_float("EXECUTION_SLIPPAGE_BPS", 2.0),
            fee_per_share=_get_env_float("EXECUTION_FEE_PER_SHARE", 0.0005),
        )


@dataclass(frozen=True)
class SizingConfig:
    """Position sizing configuration."""

    sizer: str  # "fixed", "vol", "kelly"
    fixed_frac: float
    vol_target: float
    kelly_cap: float

    @classmethod
    def from_env(cls) -> "SizingConfig":
        """Create SizingConfig from environment variables."""
        return cls(
            sizer=_get_env_str("SIZING_SIZER", "fixed"),
            fixed_frac=_get_env_float("SIZING_FIXED_FRAC", 0.1),
            vol_target=_get_env_float("SIZING_VOL_TARGET", 0.20),
            kelly_cap=_get_env_float("SIZING_KELLY_CAP", 0.25),
        )


@dataclass(frozen=True)
class CalibrationConfig:
    """PatchTST calibration configuration."""

    proba_threshold: float
    opt_threshold: bool
    n_bins: int

    @classmethod
    def from_env(cls) -> "CalibrationConfig":
        """Create CalibrationConfig from environment variables."""
        return cls(
            proba_threshold=_get_env_float("CALIBRATION_PROBA_THRESHOLD", 0.5),
            opt_threshold=_get_env_str("CALIBRATION_OPT_THRESHOLD", "false").lower()
            == "true",
            n_bins=_get_env_int("CALIBRATION_N_BINS", 10),
        )


@dataclass(frozen=True)
class RewardConfig:
    """Reward function weights configuration."""

    lambda_dd: float  # Drawdown penalty weight
    tau_turnover: float  # Turnover penalty weight
    kappa_breach: float  # Risk breach penalty weight

    @classmethod
    def from_env(cls) -> "RewardConfig":
        """Create RewardConfig from environment variables."""
        return cls(
            lambda_dd=_get_env_float("REWARD_LAMBDA_DD", 0.2),
            tau_turnover=_get_env_float("REWARD_TAU_TURNOVER", 0.001),
            kappa_breach=_get_env_float("REWARD_KAPPA_BREACH", 0.5),
        )


@dataclass(frozen=True)
class Config:
    """Main configuration container."""

    data: DataConfig
    logging: LoggingConfig
    backtest: BacktestConfig
    risk: RiskConfig
    strategy: StrategyConfig
    model: ModelConfig
    universe: UniverseConfig
    execution: ExecutionConfig
    sizing: SizingConfig
    calibration: CalibrationConfig
    reward: RewardConfig
    random_seed: int
    runs_dir: Path

    @classmethod
    def from_env(cls) -> "Config":
        """Create Config from environment variables."""
        return cls(
            data=DataConfig.from_env(),
            logging=LoggingConfig.from_env(),
            backtest=BacktestConfig.from_env(),
            risk=RiskConfig.from_env(),
            strategy=StrategyConfig.from_env(),
            model=ModelConfig.from_env(),
            universe=UniverseConfig.from_env(),
            execution=ExecutionConfig.from_env(),
            sizing=SizingConfig.from_env(),
            calibration=CalibrationConfig.from_env(),
            reward=RewardConfig.from_env(),
            random_seed=_get_env_int("RANDOM_SEED", 42),
            runs_dir=Path(_get_env_str("RUNS_DIR", "./runs")),
        )


# Global configuration instance
_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None
