"""Risk management module.

Provides risk limits and checks for position sizing and exposure control.
"""

from dataclasses import dataclass
from datetime import date

from traderbot.config import get_config
from traderbot.logging_setup import get_logger

logger = get_logger("engine.risk")


@dataclass
class RiskLimits:
    """Risk limit configuration."""

    max_position_pct: float = 0.10  # Max position size as % of NAV
    max_gross_exposure: float = 1.0  # Max gross exposure as fraction of NAV
    daily_loss_limit_pct: float = 0.02  # Max daily loss as % of NAV
    max_drawdown_pct: float = 0.15  # Max drawdown from peak as % of NAV

    @classmethod
    def from_config(cls) -> "RiskLimits":
        """Create RiskLimits from config."""
        config = get_config()
        return cls(
            max_position_pct=config.risk.max_position_pct,
            max_gross_exposure=config.risk.max_gross_exposure,
            daily_loss_limit_pct=config.risk.daily_loss_limit_pct,
            max_drawdown_pct=config.risk.max_drawdown_pct,
        )


@dataclass
class RiskCheckResult:
    """Result of a risk check."""

    passed: bool
    reason: str = ""
    adjusted_size: float | None = None


@dataclass
class RiskState:
    """Current risk state tracking."""

    peak_nav: float = 0.0
    daily_start_nav: float = 0.0
    current_date: date | None = None
    circuit_breaker_triggered: bool = False


class RiskManager:
    """Risk manager for position sizing and exposure control.

    Tracks:
    - Per-position size limits
    - Gross exposure limits
    - Daily loss limits
    - Running max drawdown
    - Circuit breaker state
    """

    def __init__(self, limits: RiskLimits | None = None):
        """Initialize risk manager.

        Args:
            limits: Risk limits to use. Defaults to config values.
        """
        self.limits = limits or RiskLimits.from_config()
        self._state = RiskState()

    def reset(self) -> None:
        """Reset risk state."""
        self._state = RiskState()

    def update_nav(self, nav: float, current_date: date) -> None:
        """Update NAV and track peaks/daily starts.

        Args:
            nav: Current net asset value.
            current_date: Current date.
        """
        # Update peak
        if nav > self._state.peak_nav:
            self._state.peak_nav = nav

        # Check for new day
        if self._state.current_date != current_date:
            self._state.daily_start_nav = nav
            self._state.current_date = current_date

    def check_position_size(
        self,
        ticker: str,
        proposed_value: float,
        nav: float,
    ) -> RiskCheckResult:
        """Check if proposed position size is within limits.

        Args:
            ticker: Ticker symbol.
            proposed_value: Proposed position value (absolute).
            nav: Current NAV.

        Returns:
            RiskCheckResult with pass/fail and optional adjusted size.
        """
        if self._state.circuit_breaker_triggered:
            return RiskCheckResult(
                passed=False,
                reason="Circuit breaker triggered - trading halted",
            )

        if nav <= 0:
            return RiskCheckResult(passed=False, reason="Invalid NAV")

        max_position_value = nav * self.limits.max_position_pct
        position_pct = abs(proposed_value) / nav

        if position_pct > self.limits.max_position_pct:
            logger.warning(
                f"Position size limit: {ticker} proposed {position_pct:.1%} > "
                f"limit {self.limits.max_position_pct:.1%}"
            )
            return RiskCheckResult(
                passed=False,
                reason=f"Position exceeds {self.limits.max_position_pct:.0%} NAV limit",
                adjusted_size=max_position_value,
            )

        return RiskCheckResult(passed=True)

    def check_gross_exposure(
        self,
        current_gross: float,
        additional_value: float,
        nav: float,
    ) -> RiskCheckResult:
        """Check if adding to exposure would breach gross limits.

        Args:
            current_gross: Current gross exposure value.
            additional_value: Additional exposure being added.
            nav: Current NAV.

        Returns:
            RiskCheckResult.
        """
        if self._state.circuit_breaker_triggered:
            return RiskCheckResult(
                passed=False,
                reason="Circuit breaker triggered",
            )

        if nav <= 0:
            return RiskCheckResult(passed=False, reason="Invalid NAV")

        new_gross = current_gross + abs(additional_value)
        gross_ratio = new_gross / nav

        if gross_ratio > self.limits.max_gross_exposure:
            logger.warning(
                f"Gross exposure limit: {gross_ratio:.1%} > "
                f"limit {self.limits.max_gross_exposure:.1%}"
            )
            return RiskCheckResult(
                passed=False,
                reason=f"Gross exposure exceeds {self.limits.max_gross_exposure:.0%} limit",
            )

        return RiskCheckResult(passed=True)

    def check_daily_loss(self, current_nav: float) -> RiskCheckResult:
        """Check if daily loss limit has been breached.

        Args:
            current_nav: Current NAV.

        Returns:
            RiskCheckResult.
        """
        if self._state.daily_start_nav <= 0:
            return RiskCheckResult(passed=True)

        daily_pnl = current_nav - self._state.daily_start_nav
        daily_pnl_pct = daily_pnl / self._state.daily_start_nav

        if daily_pnl_pct < -self.limits.daily_loss_limit_pct:
            logger.warning(
                f"Daily loss limit breached: {daily_pnl_pct:.2%} < "
                f"-{self.limits.daily_loss_limit_pct:.2%}"
            )
            return RiskCheckResult(
                passed=False,
                reason=f"Daily loss exceeds {self.limits.daily_loss_limit_pct:.0%} limit",
            )

        return RiskCheckResult(passed=True)

    def check_drawdown(self, current_nav: float) -> RiskCheckResult:
        """Check if max drawdown limit has been breached.

        Args:
            current_nav: Current NAV.

        Returns:
            RiskCheckResult.
        """
        if self._state.peak_nav <= 0:
            return RiskCheckResult(passed=True)

        drawdown = (self._state.peak_nav - current_nav) / self._state.peak_nav

        if drawdown > self.limits.max_drawdown_pct:
            logger.warning(
                f"Max drawdown breached: {drawdown:.2%} > " f"{self.limits.max_drawdown_pct:.2%}"
            )
            return RiskCheckResult(
                passed=False,
                reason=f"Drawdown exceeds {self.limits.max_drawdown_pct:.0%} limit",
            )

        return RiskCheckResult(passed=True)

    def trigger_circuit_breaker(self, reason: str) -> None:
        """Trigger the circuit breaker to halt trading.

        Args:
            reason: Reason for triggering.
        """
        logger.error(f"Circuit breaker triggered: {reason}")
        self._state.circuit_breaker_triggered = True

    def reset_circuit_breaker(self) -> None:
        """Reset the circuit breaker."""
        logger.info("Circuit breaker reset")
        self._state.circuit_breaker_triggered = False

    @property
    def circuit_breaker_active(self) -> bool:
        """Check if circuit breaker is active."""
        return self._state.circuit_breaker_triggered

    @property
    def current_drawdown(self) -> float:
        """Get current drawdown from peak."""
        if self._state.peak_nav <= 0:
            return 0.0
        return 0.0  # Would need current NAV to calculate

    def run_all_checks(
        self,
        nav: float,
        current_date: date,
        proposed_ticker: str | None = None,
        proposed_value: float = 0.0,
        current_gross: float = 0.0,
    ) -> RiskCheckResult:
        """Run all risk checks.

        Args:
            nav: Current NAV.
            current_date: Current date.
            proposed_ticker: Optional ticker for position size check.
            proposed_value: Optional proposed position value.
            current_gross: Current gross exposure.

        Returns:
            RiskCheckResult (first failure or pass).
        """
        self.update_nav(nav, current_date)

        # Check circuit breaker
        if self._state.circuit_breaker_triggered:
            return RiskCheckResult(passed=False, reason="Circuit breaker active")

        # Check drawdown
        result = self.check_drawdown(nav)
        if not result.passed:
            self.trigger_circuit_breaker(result.reason)
            return result

        # Check daily loss
        result = self.check_daily_loss(nav)
        if not result.passed:
            return result

        # Check position size if proposed
        if proposed_ticker and proposed_value != 0:
            result = self.check_position_size(proposed_ticker, proposed_value, nav)
            if not result.passed:
                return result

            # Check gross exposure
            result = self.check_gross_exposure(current_gross, proposed_value, nav)
            if not result.passed:
                return result

        return RiskCheckResult(passed=True)
