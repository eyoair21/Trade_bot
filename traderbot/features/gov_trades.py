"""Government/Politician trades signal.

Stub interface for tracking trades disclosed by politicians.
Future implementations could parse data from:
- SEC Form 4 filings
- Congressional trade disclosures
- Federal Reserve official disclosures
"""

from dataclasses import dataclass
from datetime import date
from typing import Literal


@dataclass
class GovTrade:
    """Represents a disclosed government official trade."""

    ticker: str
    trade_date: date
    disclosure_date: date
    official_name: str
    official_position: str
    trade_type: Literal["buy", "sell"]
    amount_range: tuple[float, float]  # (min, max) dollar range
    shares: int | None = None


def get_gov_trades(
    ticker: str,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[GovTrade]:
    """Get government official trades for a ticker.

    Stub implementation returning empty list.

    Args:
        ticker: Ticker symbol.
        start_date: Optional start of date range.
        end_date: Optional end of date range.

    Returns:
        List of GovTrade objects.
    """
    # Stub: no trades
    return []


def get_gov_trade_signal(
    ticker: str,
    as_of_date: date,
    lookback_days: int = 30,
) -> float:
    """Get aggregated signal from government trades.

    Stub implementation returning neutral signal.

    Args:
        ticker: Ticker symbol.
        as_of_date: Date to calculate signal for.
        lookback_days: Number of days to look back.

    Returns:
        Signal in range [-1.0, 1.0].
        Positive = net buying by officials.
        Negative = net selling by officials.
        0.0 = neutral / no trades.
    """
    # Stub: neutral signal
    return 0.0


def get_gov_trade_signals(
    tickers: list[str],
    as_of_date: date,
    lookback_days: int = 30,
) -> dict[str, float]:
    """Get government trade signals for multiple tickers.

    Args:
        tickers: List of ticker symbols.
        as_of_date: Date to calculate signals for.
        lookback_days: Number of days to look back.

    Returns:
        Dict mapping ticker to signal value.
    """
    return {ticker: get_gov_trade_signal(ticker, as_of_date, lookback_days) for ticker in tickers}


def get_active_officials() -> list[dict]:
    """Get list of tracked government officials.

    Stub implementation returning empty list.

    Returns:
        List of official info dicts.
    """
    return []
