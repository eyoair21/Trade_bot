"""Point-in-Time (PTI) data store.

Provides point-in-time data access to prevent lookahead bias.
"""

from datetime import date, datetime

import pandas as pd


class PTIStore:
    """Point-in-time data store for backtesting.

    Ensures data accessed is only what was available at a given point in time.
    """

    def __init__(self, data: pd.DataFrame, date_column: str = "date"):
        """Initialize PTI store.

        Args:
            data: DataFrame with a date column.
            date_column: Name of the date column.
        """
        self._data = data.copy()
        self._date_column = date_column

        # Ensure date column is datetime
        if date_column in self._data.columns:
            self._data[date_column] = pd.to_datetime(self._data[date_column])
            self._data = self._data.sort_values(date_column).reset_index(drop=True)

    def get_as_of(self, as_of_date: date | datetime | str) -> pd.DataFrame:
        """Get data available as of a specific date.

        Args:
            as_of_date: The point-in-time date.

        Returns:
            DataFrame with only rows on or before as_of_date.
        """
        if isinstance(as_of_date, str):
            as_of_date = datetime.fromisoformat(as_of_date)
        elif isinstance(as_of_date, date) and not isinstance(as_of_date, datetime):
            as_of_date = datetime.combine(as_of_date, datetime.max.time())

        mask = self._data[self._date_column] <= as_of_date
        return self._data[mask].copy()

    def get_range(
        self,
        start_date: date | datetime | str,
        end_date: date | datetime | str,
    ) -> pd.DataFrame:
        """Get data within a date range.

        Args:
            start_date: Start of range (inclusive).
            end_date: End of range (inclusive).

        Returns:
            DataFrame with rows within the date range.
        """
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        elif isinstance(start_date, date) and not isinstance(start_date, datetime):
            start_date = datetime.combine(start_date, datetime.min.time())

        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)
        elif isinstance(end_date, date) and not isinstance(end_date, datetime):
            end_date = datetime.combine(end_date, datetime.max.time())

        mask = (self._data[self._date_column] >= start_date) & (
            self._data[self._date_column] <= end_date
        )
        return self._data[mask].copy()

    @property
    def data(self) -> pd.DataFrame:
        """Get full data (for inspection only)."""
        return self._data.copy()
