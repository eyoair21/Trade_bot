"""Tests for risk management module."""

from datetime import date

import pytest

from traderbot.engine.risk import RiskLimits, RiskManager


class TestRiskLimits:
    """Tests for RiskLimits configuration."""

    def test_default_limits(self) -> None:
        """Test default limit values."""
        limits = RiskLimits()

        assert limits.max_position_pct == 0.10
        assert limits.max_gross_exposure == 1.0
        assert limits.daily_loss_limit_pct == 0.02
        assert limits.max_drawdown_pct == 0.15

    def test_custom_limits(self) -> None:
        """Test custom limit values."""
        limits = RiskLimits(
            max_position_pct=0.05,
            max_gross_exposure=0.8,
            daily_loss_limit_pct=0.01,
            max_drawdown_pct=0.10,
        )

        assert limits.max_position_pct == 0.05
        assert limits.max_gross_exposure == 0.8


class TestRiskManager:
    """Tests for RiskManager."""

    @pytest.fixture
    def risk_manager(self) -> RiskManager:
        """Create risk manager with test limits."""
        limits = RiskLimits(
            max_position_pct=0.10,
            max_gross_exposure=1.0,
            daily_loss_limit_pct=0.02,
            max_drawdown_pct=0.15,
        )
        return RiskManager(limits=limits)

    def test_position_size_within_limit(self, risk_manager: RiskManager) -> None:
        """Test position size check passes when within limit."""
        nav = 100000
        proposed_value = 5000  # 5% of NAV

        result = risk_manager.check_position_size("AAPL", proposed_value, nav)

        assert result.passed
        assert result.reason == ""

    def test_position_size_exceeds_limit(self, risk_manager: RiskManager) -> None:
        """Test position size check fails when exceeds limit."""
        nav = 100000
        proposed_value = 15000  # 15% of NAV (limit is 10%)

        result = risk_manager.check_position_size("AAPL", proposed_value, nav)

        assert not result.passed
        assert "10%" in result.reason
        assert result.adjusted_size == pytest.approx(10000)  # 10% of NAV

    def test_gross_exposure_within_limit(self, risk_manager: RiskManager) -> None:
        """Test gross exposure check passes when within limit."""
        nav = 100000
        current_gross = 50000
        additional = 30000  # Total would be 80%

        result = risk_manager.check_gross_exposure(current_gross, additional, nav)

        assert result.passed

    def test_gross_exposure_exceeds_limit(self, risk_manager: RiskManager) -> None:
        """Test gross exposure check fails when exceeds limit."""
        nav = 100000
        current_gross = 80000
        additional = 30000  # Total would be 110%

        result = risk_manager.check_gross_exposure(current_gross, additional, nav)

        assert not result.passed
        assert "100%" in result.reason

    def test_daily_loss_within_limit(self, risk_manager: RiskManager) -> None:
        """Test daily loss check passes when within limit."""
        # Set up daily start NAV
        risk_manager.update_nav(100000, date(2023, 1, 3))

        # Check with small loss (1%)
        result = risk_manager.check_daily_loss(99000)

        assert result.passed

    def test_daily_loss_exceeds_limit(self, risk_manager: RiskManager) -> None:
        """Test daily loss check fails when exceeds limit."""
        # Set up daily start NAV
        risk_manager.update_nav(100000, date(2023, 1, 3))

        # Check with large loss (3%, limit is 2%)
        result = risk_manager.check_daily_loss(97000)

        assert not result.passed
        assert "2%" in result.reason

    def test_drawdown_within_limit(self, risk_manager: RiskManager) -> None:
        """Test drawdown check passes when within limit."""
        # Set peak
        risk_manager.update_nav(100000, date(2023, 1, 3))

        # Check with small drawdown (10%)
        result = risk_manager.check_drawdown(90000)

        assert result.passed

    def test_drawdown_exceeds_limit(self, risk_manager: RiskManager) -> None:
        """Test drawdown check fails when exceeds limit."""
        # Set peak
        risk_manager.update_nav(100000, date(2023, 1, 3))

        # Check with large drawdown (20%, limit is 15%)
        result = risk_manager.check_drawdown(80000)

        assert not result.passed
        assert "15%" in result.reason

    def test_circuit_breaker_trigger(self, risk_manager: RiskManager) -> None:
        """Test circuit breaker triggering."""
        assert not risk_manager.circuit_breaker_active

        risk_manager.trigger_circuit_breaker("Test trigger")

        assert risk_manager.circuit_breaker_active

    def test_circuit_breaker_blocks_trading(self, risk_manager: RiskManager) -> None:
        """Test circuit breaker blocks all trading."""
        risk_manager.trigger_circuit_breaker("Test")

        result = risk_manager.check_position_size("AAPL", 1000, 100000)

        assert not result.passed
        assert "Circuit breaker" in result.reason

    def test_circuit_breaker_reset(self, risk_manager: RiskManager) -> None:
        """Test circuit breaker reset."""
        risk_manager.trigger_circuit_breaker("Test")
        assert risk_manager.circuit_breaker_active

        risk_manager.reset_circuit_breaker()
        assert not risk_manager.circuit_breaker_active

    def test_run_all_checks_passes(self, risk_manager: RiskManager) -> None:
        """Test run_all_checks when all checks pass."""
        result = risk_manager.run_all_checks(
            nav=100000,
            current_date=date(2023, 1, 3),
            proposed_ticker="AAPL",
            proposed_value=5000,
            current_gross=20000,
        )

        assert result.passed

    def test_run_all_checks_fails_on_position(self, risk_manager: RiskManager) -> None:
        """Test run_all_checks fails on position size."""
        result = risk_manager.run_all_checks(
            nav=100000,
            current_date=date(2023, 1, 3),
            proposed_ticker="AAPL",
            proposed_value=15000,  # Exceeds 10% limit
            current_gross=20000,
        )

        assert not result.passed

    def test_peak_tracking(self, risk_manager: RiskManager) -> None:
        """Test peak NAV tracking."""
        risk_manager.update_nav(100000, date(2023, 1, 3))
        risk_manager.update_nav(110000, date(2023, 1, 4))
        risk_manager.update_nav(105000, date(2023, 1, 5))

        # Peak should be 110000
        assert risk_manager._state.peak_nav == 110000

    def test_daily_start_tracking(self, risk_manager: RiskManager) -> None:
        """Test daily start NAV tracking."""
        risk_manager.update_nav(100000, date(2023, 1, 3))
        risk_manager.update_nav(105000, date(2023, 1, 3))  # Same day

        # Daily start should still be first value
        assert risk_manager._state.daily_start_nav == 100000

        # New day
        risk_manager.update_nav(110000, date(2023, 1, 4))
        assert risk_manager._state.daily_start_nav == 110000

    def test_reset(self, risk_manager: RiskManager) -> None:
        """Test reset clears all state."""
        risk_manager.update_nav(100000, date(2023, 1, 3))
        risk_manager.trigger_circuit_breaker("Test")

        risk_manager.reset()

        assert risk_manager._state.peak_nav == 0.0
        assert not risk_manager.circuit_breaker_active

    def test_invalid_nav(self, risk_manager: RiskManager) -> None:
        """Test handling of invalid NAV."""
        result = risk_manager.check_position_size("AAPL", 1000, 0)
        assert not result.passed
        assert "Invalid NAV" in result.reason

        result = risk_manager.check_position_size("AAPL", 1000, -100)
        assert not result.passed
