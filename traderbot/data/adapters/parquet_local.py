"""Local Parquet data adapter.

Loads OHLCV data from local parquet files.
"""

from datetime import date, datetime
from pathlib import Path

import pandas as pd

from traderbot.config import get_config
from traderbot.logging_setup import get_logger

logger = get_logger("data.adapters.parquet_local")


class ParquetLocalAdapter:
    """Adapter for loading OHLCV data from local parquet files.

    Files are expected at: {data_dir}/{TICKER}.parquet

    Each parquet file should have columns:
    - date or timestamp (datetime)
    - open, high, low, close, volume
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        base_path: Path | None = None,
    ):
        """Initialize the adapter.

        Args:
            data_dir: Base directory for data. Defaults to config value.
            base_path: Alias for data_dir (for CLI compatibility).
        """
        # Support both data_dir and base_path for flexibility
        effective_dir = base_path or data_dir
        if effective_dir is None:
            config = get_config()
            effective_dir = config.data.ohlcv_dir
        self._data_dir = Path(effective_dir)
        self._cache: dict[str, pd.DataFrame] = {}

    def _get_file_path(self, ticker: str) -> Path:
        """Get the parquet file path for a ticker."""
        return self._data_dir / f"{ticker}.parquet"

    def load(
        self,
        ticker: str,
        start_date: date | datetime | str | None = None,
        end_date: date | datetime | str | None = None,
        as_of_date: date | datetime | str | None = None,
    ) -> pd.DataFrame:
        """Load OHLCV data for a ticker.

        Args:
            ticker: Ticker symbol.
            start_date: Optional start date filter (inclusive).
            end_date: Optional end date filter (inclusive).
            as_of_date: Optional point-in-time filter (no rows after this date).

        Returns:
            DataFrame with OHLCV data.

        Raises:
            FileNotFoundError: If the parquet file does not exist.
        """
        file_path = self._get_file_path(ticker)

        if not file_path.exists():
            logger.warning(f"Parquet file not found: {file_path}")
            raise FileNotFoundError(f"No data file for ticker {ticker}: {file_path}")

        # Check cache
        if ticker not in self._cache:
            logger.info(f"Loading data for {ticker} from {file_path}")
            df = pd.read_parquet(file_path)

            # Ensure date column exists and is datetime
            # Support both 'date' and 'timestamp' column names
            if "timestamp" in df.columns and "date" not in df.columns:
                df = df.rename(columns={"timestamp": "date"})

            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            elif df.index.name == "date":
                df = df.reset_index()
                df["date"] = pd.to_datetime(df["date"])
            elif df.index.name == "timestamp":
                df = df.reset_index()
                df = df.rename(columns={"timestamp": "date"})
                df["date"] = pd.to_datetime(df["date"])

            # Normalize timezone - remove UTC for consistent comparisons
            if df["date"].dt.tz is not None:
                df["date"] = df["date"].dt.tz_localize(None)

            # Sort by date
            df = df.sort_values("date").reset_index(drop=True)

            self._cache[ticker] = df

        df = self._cache[ticker].copy()

        # Apply date filters
        if start_date is not None:
            start_dt = self._to_datetime(start_date)
            df = df[df["date"] >= start_dt]

        if end_date is not None:
            end_dt = self._to_datetime(end_date, end_of_day=True)
            df = df[df["date"] <= end_dt]

        # Point-in-time filter (no lookahead)
        if as_of_date is not None:
            as_of_dt = self._to_datetime(as_of_date, end_of_day=True)
            df = df[df["date"] <= as_of_dt]

        return df.reset_index(drop=True)

    def load_multiple(
        self,
        tickers: list[str],
        start_date: date | datetime | str | None = None,
        end_date: date | datetime | str | None = None,
        as_of_date: date | datetime | str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Load OHLCV data for multiple tickers.

        Args:
            tickers: List of ticker symbols.
            start_date: Optional start date filter.
            end_date: Optional end date filter.
            as_of_date: Optional point-in-time filter.

        Returns:
            Dict mapping ticker to DataFrame.
        """
        result = {}
        for ticker in tickers:
            try:
                result[ticker] = self.load(
                    ticker,
                    start_date=start_date,
                    end_date=end_date,
                    as_of_date=as_of_date,
                )
            except FileNotFoundError:
                logger.warning(f"Skipping {ticker}: no data file found")
                continue

        return result

    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._cache.clear()

    @staticmethod
    def _to_datetime(d: date | datetime | str, end_of_day: bool = False) -> pd.Timestamp:
        """Convert to pandas Timestamp (timezone-naive for comparison)."""
        if isinstance(d, str):
            dt = datetime.fromisoformat(d)
        elif isinstance(d, date) and not isinstance(d, datetime):
            if end_of_day:
                dt = datetime.combine(d, datetime.max.time())
            else:
                dt = datetime.combine(d, datetime.min.time())
        else:
            dt = d

        # Return as pandas Timestamp for proper comparison with both
        # tz-naive and tz-aware datetime columns
        return pd.Timestamp(dt)
