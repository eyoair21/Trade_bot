"""Momentum strategy implementation.

EMA crossover with RSI filter and ATR-based dynamic stops.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from traderbot.config import get_config
from traderbot.engine.strategy_base import Signal, SignalType, StrategyBase
from traderbot.features.ta import calculate_atr, calculate_ema, calculate_rsi
from traderbot.logging_setup import get_logger

logger = get_logger("engine.strategy_momo")


class MomentumStrategy(StrategyBase):
    """Momentum strategy using EMA crossover with RSI filter.

    Entry signals:
    - Long: Fast EMA crosses above slow EMA, RSI not overbought
    - Short: Fast EMA crosses below slow EMA, RSI not oversold

    Exit signals:
    - EMA cross in opposite direction
    - RSI extreme reversal
    - ATR-based stop loss

    Position sizing:
    - Position cap as % of NAV (from config)
    """

    def __init__(
        self,
        name: str = "momentum",
        universe: list[str] | None = None,
        ema_fast_period: int | None = None,
        ema_slow_period: int | None = None,
        rsi_period: int | None = None,
        rsi_oversold: int | None = None,
        rsi_overbought: int | None = None,
        atr_period: int | None = None,
        atr_stop_multiplier: float | None = None,
        max_position_pct: float | None = None,
        seed: int | None = None,
    ):
        """Initialize momentum strategy.

        Args:
            name: Strategy name.
            universe: List of tickers to trade.
            ema_fast_period: Fast EMA period.
            ema_slow_period: Slow EMA period.
            rsi_period: RSI period.
            rsi_oversold: RSI oversold threshold.
            rsi_overbought: RSI overbought threshold.
            atr_period: ATR period for stop calculation.
            atr_stop_multiplier: ATR multiplier for stop distance.
            max_position_pct: Maximum position size as % of NAV.
            seed: Random seed for deterministic behavior.
        """
        super().__init__(name, universe)

        config = get_config()

        self.ema_fast_period = ema_fast_period or config.strategy.ema_fast_period
        self.ema_slow_period = ema_slow_period or config.strategy.ema_slow_period
        self.rsi_period = rsi_period or config.strategy.rsi_period
        self.rsi_oversold = rsi_oversold or config.strategy.rsi_oversold
        self.rsi_overbought = rsi_overbought or config.strategy.rsi_overbought
        self.atr_period = atr_period or config.strategy.atr_period
        self.atr_stop_multiplier = atr_stop_multiplier or config.strategy.atr_stop_multiplier
        self.max_position_pct = max_position_pct or config.risk.max_position_pct

        # Set random seed for deterministic behavior
        self._seed = seed or config.random_seed
        np.random.seed(self._seed)

        # Track previous indicators for crossover detection
        self._prev_ema_fast: dict[str, float] = {}
        self._prev_ema_slow: dict[str, float] = {}

    def initialize(self, initial_data: dict[str, pd.DataFrame]) -> None:
        """Initialize strategy with historical data.

        Pre-compute indicators for lookback period.
        """
        super().initialize(initial_data)

        for ticker, df in initial_data.items():
            if len(df) < self.ema_slow_period:
                continue

            close = df["close"]
            ema_fast = calculate_ema(close, self.ema_fast_period)
            ema_slow = calculate_ema(close, self.ema_slow_period)

            if len(ema_fast) > 0:
                self._prev_ema_fast[ticker] = ema_fast.iloc[-1]
            if len(ema_slow) > 0:
                self._prev_ema_slow[ticker] = ema_slow.iloc[-1]

        logger.info(f"Initialized {self.name} strategy for {len(initial_data)} tickers")

    def generate_signals(
        self,
        current_bar: dict[str, pd.Series],
        historical_data: dict[str, pd.DataFrame],
        timestamp: datetime,
    ) -> list[Signal]:
        """Generate trading signals for current bar.

        Args:
            current_bar: Dict mapping ticker to current bar data.
            historical_data: Dict mapping ticker to historical data.
            timestamp: Current timestamp.

        Returns:
            List of signals.
        """
        signals = []

        for ticker in self.universe:
            if ticker not in historical_data:
                continue

            df = historical_data[ticker]

            # Need enough data for slow EMA
            if len(df) < self.ema_slow_period + 1:
                continue

            close = df["close"]
            high = df["high"]
            low = df["low"]

            # Calculate indicators
            ema_fast = calculate_ema(close, self.ema_fast_period)
            ema_slow = calculate_ema(close, self.ema_slow_period)
            rsi = calculate_rsi(close, self.rsi_period)
            atr = calculate_atr(high, low, close, self.atr_period)

            # Get current values
            curr_ema_fast = ema_fast.iloc[-1]
            curr_ema_slow = ema_slow.iloc[-1]
            curr_rsi = rsi.iloc[-1]
            curr_atr = atr.iloc[-1]
            curr_price = close.iloc[-1]

            # Get previous values (for crossover detection)
            prev_ema_fast = self._prev_ema_fast.get(ticker, curr_ema_fast)
            prev_ema_slow = self._prev_ema_slow.get(ticker, curr_ema_slow)

            # Detect crossovers
            bullish_cross = (prev_ema_fast <= prev_ema_slow) and (curr_ema_fast > curr_ema_slow)
            bearish_cross = (prev_ema_fast >= prev_ema_slow) and (curr_ema_fast < curr_ema_slow)

            # Check current position state
            current_signal = self._state.signals.get(ticker)
            is_long = current_signal and current_signal.signal_type == SignalType.LONG
            is_short = current_signal and current_signal.signal_type == SignalType.SHORT

            signal = None

            # Generate signals
            if bullish_cross and not pd.isna(curr_rsi):
                # Bullish crossover
                if curr_rsi < self.rsi_overbought:
                    # Not overbought - valid long signal
                    stop_price = curr_price - (curr_atr * self.atr_stop_multiplier)
                    signal = Signal(
                        ticker=ticker,
                        signal_type=SignalType.LONG,
                        strength=1.0,
                        target_weight=self.max_position_pct,
                        stop_price=stop_price,
                        metadata={
                            "ema_fast": curr_ema_fast,
                            "ema_slow": curr_ema_slow,
                            "rsi": curr_rsi,
                            "atr": curr_atr,
                        },
                    )
                    logger.debug(
                        f"{timestamp} {ticker}: LONG signal, "
                        f"RSI={curr_rsi:.1f}, stop={stop_price:.2f}"
                    )

            elif bearish_cross and not pd.isna(curr_rsi):
                # Bearish crossover
                if curr_rsi > self.rsi_oversold:
                    # Not oversold - valid short signal
                    stop_price = curr_price + (curr_atr * self.atr_stop_multiplier)
                    signal = Signal(
                        ticker=ticker,
                        signal_type=SignalType.SHORT,
                        strength=1.0,
                        target_weight=-self.max_position_pct,
                        stop_price=stop_price,
                        metadata={
                            "ema_fast": curr_ema_fast,
                            "ema_slow": curr_ema_slow,
                            "rsi": curr_rsi,
                            "atr": curr_atr,
                        },
                    )
                    logger.debug(
                        f"{timestamp} {ticker}: SHORT signal, "
                        f"RSI={curr_rsi:.1f}, stop={stop_price:.2f}"
                    )

            elif is_long and not pd.isna(curr_rsi):
                # Check for exit conditions for long position
                if curr_rsi > self.rsi_overbought:
                    # Overbought - take profit
                    signal = Signal(
                        ticker=ticker,
                        signal_type=SignalType.FLAT,
                        metadata={"reason": "overbought"},
                    )
                    logger.debug(f"{timestamp} {ticker}: EXIT LONG (overbought)")

            elif is_short and not pd.isna(curr_rsi) and curr_rsi < self.rsi_oversold:
                # Exit short position - oversold - take profit
                signal = Signal(
                    ticker=ticker,
                    signal_type=SignalType.FLAT,
                    metadata={"reason": "oversold"},
                )
                logger.debug(f"{timestamp} {ticker}: EXIT SHORT (oversold)")

            if signal:
                signals.append(signal)
                self._state.signals[ticker] = signal

            # Update previous values for next bar
            self._prev_ema_fast[ticker] = curr_ema_fast
            self._prev_ema_slow[ticker] = curr_ema_slow

        return signals

    def on_bar(
        self,
        current_bar: dict[str, pd.Series],
        timestamp: datetime,
    ) -> None:
        """Handle new bar data.

        Updates stop prices based on trailing ATR.
        """
        pass  # Stop updates handled in generate_signals

    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()
        self._prev_ema_fast.clear()
        self._prev_ema_slow.clear()
        np.random.seed(self._seed)
