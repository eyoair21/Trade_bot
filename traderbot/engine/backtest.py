"""Backtesting engine.

Coordinates data, strategy, broker, and risk management.
Optionally integrates PatchTST model for signal generation.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from traderbot.config import get_config
from traderbot.data.calendar import get_sessions_between
from traderbot.engine.broker_sim import BrokerSimulator, OrderSide
from traderbot.engine.position_sizing import PositionSizer, SizerType, calculate_volatility
from traderbot.engine.risk import RiskManager
from traderbot.engine.strategy_base import StrategyBase
from traderbot.features.ta import compute_model_features
from traderbot.logging_setup import get_logger

# Optional: PyTorch model support
try:
    from traderbot.model.patchtst import (
        batch_inference,
        create_feature_tensor,
        load_torchscript,
    )

    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

logger = get_logger("engine.backtest")


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    start_date: date
    end_date: date
    initial_capital: float
    final_equity: float
    total_return: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    equity_curve: pd.DataFrame
    trades: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)
    execution_costs: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_capital": self.initial_capital,
            "final_equity": self.final_equity,
            "total_return": self.total_return,
            "total_return_pct": self.total_return_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "metadata": self.metadata,
            "execution_costs": self.execution_costs,
        }


class BacktestEngine:
    """Main backtesting engine.

    Coordinates:
    - Data loading and iteration
    - Strategy signal generation
    - Order execution via broker simulator
    - Risk management checks
    - Performance tracking
    """

    def __init__(
        self,
        strategy: StrategyBase,
        broker: BrokerSimulator | None = None,
        risk_manager: RiskManager | None = None,
        seed: int = 42,
        model_path: str | Path | None = None,
        sizer: PositionSizer | str | None = None,
        proba_threshold: float = 0.5,
    ):
        """Initialize backtest engine.

        Args:
            strategy: Trading strategy to backtest.
            broker: Broker simulator. Created if not provided.
            risk_manager: Risk manager. Created if not provided.
            seed: Random seed for reproducibility.
            model_path: Optional path to TorchScript model for signal generation.
            sizer: Position sizer or sizer type string. Defaults to config.
            proba_threshold: Minimum probability to generate a signal (0-1).
        """
        self.strategy = strategy
        self.broker = broker or BrokerSimulator()
        self.risk_manager = risk_manager or RiskManager()
        self._seed = seed
        self._proba_threshold = proba_threshold

        # Setup position sizer
        if sizer is None:
            self._sizer = PositionSizer()
        elif isinstance(sizer, str):
            self._sizer = PositionSizer(sizer_type=sizer)
        else:
            self._sizer = sizer

        # Set seeds for reproducibility
        np.random.seed(seed)

        # State
        self._equity_history: list[dict] = []
        self._trade_history: list[dict] = []

        # Load model if provided
        self._model = None
        self._model_config = None
        if model_path is not None:
            if not MODEL_AVAILABLE:
                logger.warning("PyTorch not available, model will not be used")
            else:
                try:
                    self._model = load_torchscript(model_path)
                    config = get_config()
                    self._model_config = config.model
                    logger.info(f"Loaded model from {model_path}")
                except Exception as e:
                    logger.warning(f"Failed to load model: {e}")

    def run(
        self,
        data: dict[str, pd.DataFrame],
        start_date: date | str,
        end_date: date | str,
    ) -> BacktestResult:
        """Run backtest on provided data.

        Args:
            data: Dict mapping ticker to OHLCV DataFrame.
            start_date: Start date for backtest.
            end_date: End date for backtest.

        Returns:
            BacktestResult with performance metrics.
        """
        # Reset state
        self.broker.reset()
        self.risk_manager.reset()
        self.strategy.reset()
        self._equity_history.clear()
        self._trade_history.clear()

        np.random.seed(self._seed)

        # Convert dates
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date).date()
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date).date()

        # Get trading sessions
        sessions = get_sessions_between(start_date, end_date)

        if not sessions:
            logger.warning("No trading sessions in date range")
            return self._create_empty_result(start_date, end_date)

        # Prepare data - filter to date range
        filtered_data: dict[str, pd.DataFrame] = {}
        for ticker, df in data.items():
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
            filtered_data[ticker] = df[mask].reset_index(drop=True)

        # Initialize strategy
        self.strategy.universe = list(filtered_data.keys())
        self.strategy.initialize(filtered_data)

        initial_capital = self.broker._initial_capital

        logger.info(
            f"Starting backtest: {start_date} to {end_date}, "
            f"{len(sessions)} sessions, {len(filtered_data)} tickers"
        )

        # Main loop over sessions
        for session_date in sessions:
            self._process_session(session_date, filtered_data)

        # Calculate final results
        return self._calculate_results(start_date, end_date, initial_capital)

    def _process_session(
        self,
        session_date: date,
        data: dict[str, pd.DataFrame],
    ) -> None:
        """Process a single trading session.

        Args:
            session_date: Date of the session.
            data: All OHLCV data.
        """
        timestamp = datetime.combine(session_date, datetime.min.time())

        # Get current bar data
        current_bar: dict[str, pd.Series] = {}
        historical_data: dict[str, pd.DataFrame] = {}

        for ticker, df in data.items():
            mask = df["date"].dt.date <= session_date
            hist = df[mask]

            if len(hist) == 0:
                continue

            historical_data[ticker] = hist
            current_bar[ticker] = hist.iloc[-1]

        if not current_bar:
            return

        # Get current prices for calculations
        current_prices = {t: bar["close"] for t, bar in current_bar.items()}

        # Calculate current equity
        equity = self.broker.calculate_equity(current_prices)
        gross_exposure = self.broker.calculate_gross_exposure(current_prices)

        # Update risk manager
        self.risk_manager.update_nav(equity, session_date)

        # Check risk limits
        risk_check = self.risk_manager.run_all_checks(
            nav=equity,
            current_date=session_date,
            current_gross=gross_exposure,
        )

        if not risk_check.passed:
            logger.warning(f"Risk check failed: {risk_check.reason}")
            # Cancel all pending orders
            self.broker.cancel_all_orders()
            self._record_equity(session_date, equity, current_prices)
            return

        # Process pending orders from previous session
        bar_data = {
            t: pd.Series(
                {
                    "open": b["open"],
                    "high": b["high"],
                    "low": b["low"],
                    "close": b["close"],
                    "volume": b["volume"],
                }
            )
            for t, b in current_bar.items()
        }
        fills = self.broker.process_bar(bar_data, timestamp)

        # Record fills
        for fill in fills:
            qty = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity
            self.strategy.on_fill(fill.ticker, qty, fill.price)
            self._trade_history.append(
                {
                    "date": session_date.isoformat(),
                    "ticker": fill.ticker,
                    "side": fill.side.value,
                    "quantity": fill.quantity,
                    "price": fill.price,
                    "commission": fill.commission,
                }
            )

        # Get model predictions if available
        model_predictions = self.get_model_predictions(historical_data)
        if model_predictions:
            # Filter predictions by threshold
            filtered_predictions = {
                ticker: prob
                for ticker, prob in model_predictions.items()
                if prob >= self._proba_threshold or prob <= (1 - self._proba_threshold)
            }
            # Attach predictions to strategy for use in signal generation
            self.strategy.model_predictions = filtered_predictions
            # Store raw predictions for calibration analysis
            self.strategy.raw_model_predictions = model_predictions

        # Generate signals
        signals = self.strategy.generate_signals(current_bar, historical_data, timestamp)

        # Get current positions
        positions = {t: p.quantity for t, p in self.broker.get_positions().items()}

        # Check for stop orders
        stop_orders = self.strategy.get_stop_orders(current_prices, positions)
        for order in stop_orders:
            # Validate against risk
            self.broker.submit_order(order)

        # Convert signals to orders using position sizer
        orders = self._generate_orders_with_sizer(
            signals,
            current_prices,
            positions,
            equity,
            historical_data,
        )

        # Submit orders (risk-checked)
        for order in orders:
            order_value = current_prices.get(order.ticker, 0) * order.quantity
            risk_result = self.risk_manager.check_position_size(order.ticker, order_value, equity)

            if risk_result.passed:
                self.broker.submit_order(order)
            else:
                logger.debug(f"Order rejected by risk: {risk_result.reason}")

        # Call strategy on_bar
        self.strategy.on_bar(current_bar, timestamp)

        # Record equity
        equity = self.broker.calculate_equity(current_prices)
        self._record_equity(session_date, equity, current_prices)

    def get_model_predictions(
        self,
        historical_data: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        """Get model predictions for all tickers.

        Args:
            historical_data: Dict mapping ticker to historical OHLCV data.

        Returns:
            Dict mapping ticker to prob_up prediction [0, 1].
        """
        if self._model is None or self._model_config is None:
            return {}

        predictions = {}
        feature_names = list(self._model_config.features)
        lookback = self._model_config.lookback

        for ticker, df in historical_data.items():
            if len(df) < lookback:
                continue

            try:
                # Compute features
                feature_dict = compute_model_features(df, feature_names)

                # Convert to numpy arrays
                feature_arrays: dict[str, np.ndarray] = {
                    name: np.asarray(series.values) for name, series in feature_dict.items()
                }

                # Create tensor
                tensor = create_feature_tensor(feature_arrays, feature_names, lookback)

                # Run inference
                output = batch_inference(self._model, tensor)
                predictions[ticker] = float(output[0, 0])

            except Exception as e:
                logger.debug(f"Model prediction failed for {ticker}: {e}")
                continue

        return predictions

    def _generate_orders_with_sizer(
        self,
        signals: list,
        prices: dict[str, float],
        positions: dict[str, int],
        equity: float,
        historical_data: dict[str, pd.DataFrame],
    ) -> list:
        """Generate orders using position sizer.

        Args:
            signals: List of Signal objects from strategy.
            prices: Current prices by ticker.
            positions: Current positions by ticker.
            equity: Current portfolio equity.
            historical_data: Historical data for volatility calculation.

        Returns:
            List of Order objects.
        """
        from traderbot.engine.broker_sim import Order, OrderSide, OrderType
        from traderbot.engine.strategy_base import SignalType

        orders = []

        for signal in signals:
            ticker = signal.ticker
            if ticker not in prices:
                continue

            price = prices[ticker]
            current_pos = positions.get(ticker, 0)

            # Determine target direction from SignalType
            if signal.signal_type == SignalType.LONG:
                target_side = OrderSide.BUY
                signal_strength = signal.strength
            elif signal.signal_type == SignalType.SHORT:
                target_side = OrderSide.SELL
                signal_strength = signal.strength
            else:
                continue

            # Calculate volatility for vol-targeting sizer
            volatility = 0.0
            if ticker in historical_data:
                df = historical_data[ticker]
                if "close" in df.columns and len(df) > 20:
                    volatility = calculate_volatility(df["close"], window=20)

            # Get model prediction for Kelly sizer
            win_prob = 0.5
            if hasattr(self.strategy, "model_predictions"):
                win_prob = self.strategy.model_predictions.get(ticker, 0.5)

            # Calculate position size
            size_result = self._sizer.calculate_size(
                equity=equity,
                price=price,
                volatility=volatility,
                win_prob=win_prob,
                win_loss_ratio=1.5,  # Default assumption
            )

            target_shares = size_result.shares

            if target_shares == 0:
                continue

            # Determine order quantity based on current position
            if target_side == OrderSide.BUY:
                if current_pos >= target_shares:
                    continue  # Already at or above target
                order_qty = target_shares - max(current_pos, 0)
            else:
                if current_pos <= -target_shares:
                    continue  # Already at or below target (short)
                order_qty = target_shares + min(current_pos, 0)

            if order_qty <= 0:
                continue

            # Apply risk limit cap
            max_position_value = equity * self.risk_manager.limits.max_position_pct
            max_shares = int(max_position_value / price)
            order_qty = min(order_qty, max_shares)

            if order_qty <= 0:
                continue

            order = Order(
                ticker=ticker,
                side=target_side,
                quantity=order_qty,
                order_type=OrderType.MARKET,
            )
            orders.append(order)

        return orders

    def _record_equity(
        self,
        session_date: date,
        equity: float,
        prices: dict[str, float],
    ) -> None:
        """Record equity for the session."""
        self._equity_history.append(
            {
                "date": session_date,
                "equity": equity,
                "cash": self.broker.cash,
                "num_positions": len(
                    [p for p in self.broker.get_positions().values() if p.quantity != 0]
                ),
            }
        )

    def _calculate_results(
        self,
        start_date: date,
        end_date: date,
        initial_capital: float,
    ) -> BacktestResult:
        """Calculate backtest performance metrics."""
        if not self._equity_history:
            return self._create_empty_result(start_date, end_date)

        equity_df = pd.DataFrame(self._equity_history)
        equity_df["date"] = pd.to_datetime(equity_df["date"])
        equity_df = equity_df.set_index("date")

        # Basic metrics
        final_equity = equity_df["equity"].iloc[-1]
        total_return = final_equity - initial_capital
        total_return_pct = (final_equity / initial_capital - 1) * 100

        # Calculate returns
        equity_df["returns"] = equity_df["equity"].pct_change()

        # Sharpe ratio (annualized, assuming 252 trading days)
        mean_return = equity_df["returns"].mean()
        std_return = equity_df["returns"].std()
        if std_return > 0 and not np.isnan(std_return):
            sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Max drawdown
        rolling_max = equity_df["equity"].cummax()
        drawdown = equity_df["equity"] - rolling_max
        max_drawdown = drawdown.min()
        max_drawdown_pct = (
            (max_drawdown / rolling_max[drawdown.idxmin()]) * 100 if max_drawdown < 0 else 0.0
        )

        # Trade statistics
        total_trades = len(self._trade_history)

        # Calculate trade P&L (simplified - based on realized P&L)
        fills = self.broker.fills
        trade_pnls: list[float] = []
        position_costs: dict[str, tuple[int, float]] = {}  # ticker -> (qty, avg_cost)

        for fill in fills:
            ticker = fill.ticker
            qty = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity

            if ticker not in position_costs:
                position_costs[ticker] = (0, 0.0)

            prev_qty, prev_cost = position_costs[ticker]

            if (prev_qty > 0 and qty < 0) or (prev_qty < 0 and qty > 0):
                # Closing trade
                close_qty = min(abs(prev_qty), abs(qty))
                if prev_qty > 0:
                    pnl = (fill.price - prev_cost) * close_qty
                else:
                    pnl = (prev_cost - fill.price) * close_qty
                trade_pnls.append(pnl - fill.commission)

            new_qty = prev_qty + qty
            if new_qty != 0:
                new_cost = (
                    (abs(prev_qty) * prev_cost + abs(qty) * fill.price) / abs(new_qty)
                    if prev_qty * qty > 0
                    else fill.price
                )
                position_costs[ticker] = (new_qty, new_cost)
            else:
                position_costs[ticker] = (0, 0.0)

        winning_trades = len([p for p in trade_pnls if p > 0])
        losing_trades = len([p for p in trade_pnls if p < 0])
        win_rate = winning_trades / len(trade_pnls) * 100 if trade_pnls else 0.0

        gross_profit = sum(p for p in trade_pnls if p > 0)
        gross_loss = abs(sum(p for p in trade_pnls if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Collect execution costs from broker
        execution_costs = {
            "commission": self.broker.total_commission,
            "fees": self.broker.total_fees,
            "slippage": self.broker.total_slippage,
            "total": self.broker.total_execution_costs,
        }

        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_equity=final_equity,
            total_return=total_return,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor if profit_factor != float("inf") else 0.0,
            equity_curve=equity_df.reset_index(),
            execution_costs=execution_costs,
            trades=self._trade_history,
        )

    def _create_empty_result(
        self,
        start_date: date,
        end_date: date,
    ) -> BacktestResult:
        """Create empty result when no trading occurred."""
        initial = self.broker._initial_capital
        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial,
            final_equity=initial,
            total_return=0.0,
            total_return_pct=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_pct=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            equity_curve=pd.DataFrame(columns=["date", "equity", "cash"]),
            trades=[],
        )
