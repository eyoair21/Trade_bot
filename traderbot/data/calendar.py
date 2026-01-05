"""Business calendar for trading sessions.

Handles weekends and US market holidays.
"""

from datetime import date, datetime, timedelta

# Static US holiday set (month, day) - fixed holidays only
# For holidays that fall on weekends, exchanges observe adjacent weekdays
US_HOLIDAYS: set[tuple[int, int]] = {
    (1, 1),  # New Year's Day
    (7, 4),  # Independence Day
    (12, 25),  # Christmas
}

# Thanksgiving is 4th Thursday of November - handled separately
# For simplicity, we use a static approximation (Nov 23-29 range)


def _get_thanksgiving(year: int) -> date:
    """Get Thanksgiving date for a given year (4th Thursday of November)."""
    # Find first day of November
    nov_first = date(year, 11, 1)
    # Find first Thursday (weekday 3)
    days_until_thursday = (3 - nov_first.weekday()) % 7
    first_thursday = nov_first + timedelta(days=days_until_thursday)
    # 4th Thursday is 3 weeks later
    return first_thursday + timedelta(weeks=3)


def _is_observed_holiday(d: date) -> bool:
    """Check if date is an observed US market holiday.

    Holidays falling on Saturday are observed Friday.
    Holidays falling on Sunday are observed Monday.
    """
    # Check Thanksgiving
    if d == _get_thanksgiving(d.year):
        return True

    # Check fixed holidays with weekend observation rules
    # Need to check both current year and next year for year-boundary holidays
    years_to_check = [d.year, d.year + 1]

    for year in years_to_check:
        for month, day in US_HOLIDAYS:
            holiday = date(year, month, day)

            # If holiday is on Saturday, Friday is observed
            if holiday.weekday() == 5:  # Saturday
                if d == holiday - timedelta(days=1):
                    return True
            # If holiday is on Sunday, Monday is observed
            elif holiday.weekday() == 6:  # Sunday
                if d == holiday + timedelta(days=1):
                    return True
            # Otherwise, the holiday itself
            elif d == holiday:
                return True

    return False


def _to_date(ts: date | datetime | str) -> date:
    """Convert timestamp to date object."""
    if isinstance(ts, datetime):
        return ts.date()
    if isinstance(ts, str):
        return datetime.fromisoformat(ts).date()
    return ts


def is_session(ts: date | datetime | str) -> bool:
    """Check if the given date is a trading session.

    A trading session is a weekday that is not a US market holiday.

    Args:
        ts: Date, datetime, or ISO format string.

    Returns:
        True if the date is a trading session, False otherwise.
    """
    d = _to_date(ts)

    # Weekend check
    if d.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False

    # Holiday check
    return not _is_observed_holiday(d)


def next_session(ts: date | datetime | str) -> date:
    """Get the next trading session after the given date.

    Args:
        ts: Date, datetime, or ISO format string.

    Returns:
        The next trading session date.
    """
    d = _to_date(ts)
    candidate = d + timedelta(days=1)

    # Find next session (limit iterations to avoid infinite loop)
    for _ in range(10):
        if is_session(candidate):
            return candidate
        candidate += timedelta(days=1)

    # Fallback (should not happen)
    return candidate


def prev_session(ts: date | datetime | str) -> date:
    """Get the previous trading session before the given date.

    Args:
        ts: Date, datetime, or ISO format string.

    Returns:
        The previous trading session date.
    """
    d = _to_date(ts)
    candidate = d - timedelta(days=1)

    # Find previous session (limit iterations to avoid infinite loop)
    for _ in range(10):
        if is_session(candidate):
            return candidate
        candidate -= timedelta(days=1)

    # Fallback (should not happen)
    return candidate


def get_sessions_between(start: date | datetime | str, end: date | datetime | str) -> list[date]:
    """Get all trading sessions between two dates (inclusive).

    Args:
        start: Start date.
        end: End date.

    Returns:
        List of trading session dates.
    """
    start_d = _to_date(start)
    end_d = _to_date(end)

    sessions = []
    current = start_d

    while current <= end_d:
        if is_session(current):
            sessions.append(current)
        current += timedelta(days=1)

    return sessions
