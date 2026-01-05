"""Tests for business calendar module."""

from datetime import date, datetime

from traderbot.data.calendar import (
    _get_thanksgiving,
    get_sessions_between,
    is_session,
    next_session,
    prev_session,
)


class TestIsSession:
    """Tests for is_session function."""

    def test_weekday_is_session(self) -> None:
        """Regular weekdays are trading sessions."""
        # Tuesday, Jan 3, 2023
        assert is_session(date(2023, 1, 3))
        # Wednesday
        assert is_session(date(2023, 1, 4))
        # Thursday
        assert is_session(date(2023, 1, 5))
        # Friday
        assert is_session(date(2023, 1, 6))

    def test_weekend_not_session(self) -> None:
        """Weekends are not trading sessions."""
        # Saturday
        assert not is_session(date(2023, 1, 7))
        # Sunday
        assert not is_session(date(2023, 1, 8))

    def test_new_years_day_not_session(self) -> None:
        """New Year's Day is not a session."""
        # Jan 1, 2024 is Monday
        assert not is_session(date(2024, 1, 1))

    def test_new_years_day_observed_friday(self) -> None:
        """New Year's Day falling on Saturday is observed Friday."""
        # Jan 1, 2022 is Saturday, so Dec 31, 2021 (Friday) is observed
        assert not is_session(date(2021, 12, 31))

    def test_new_years_day_observed_monday(self) -> None:
        """New Year's Day falling on Sunday is observed Monday."""
        # Jan 1, 2023 was Sunday, so Jan 2, 2023 (Monday) is observed
        assert not is_session(date(2023, 1, 2))

    def test_independence_day_not_session(self) -> None:
        """July 4th is not a session."""
        # July 4, 2023 is Tuesday
        assert not is_session(date(2023, 7, 4))

    def test_christmas_not_session(self) -> None:
        """Christmas is not a session."""
        # Dec 25, 2023 is Monday
        assert not is_session(date(2023, 12, 25))

    def test_thanksgiving_not_session(self) -> None:
        """Thanksgiving is not a session."""
        # Thanksgiving 2023 is Nov 23
        assert not is_session(date(2023, 11, 23))
        # Thanksgiving 2024 is Nov 28
        assert not is_session(date(2024, 11, 28))

    def test_accepts_datetime(self) -> None:
        """Function accepts datetime objects."""
        assert is_session(datetime(2023, 1, 3, 10, 30))

    def test_accepts_string(self) -> None:
        """Function accepts ISO format strings."""
        assert is_session("2023-01-03")


class TestNextSession:
    """Tests for next_session function."""

    def test_next_session_from_weekday(self) -> None:
        """Next session from weekday is next weekday."""
        # Tuesday -> Wednesday
        assert next_session(date(2023, 1, 3)) == date(2023, 1, 4)

    def test_next_session_skips_weekend(self) -> None:
        """Next session skips weekend."""
        # Friday -> Monday
        assert next_session(date(2023, 1, 6)) == date(2023, 1, 9)

    def test_next_session_from_saturday(self) -> None:
        """Next session from Saturday is Monday."""
        assert next_session(date(2023, 1, 7)) == date(2023, 1, 9)

    def test_next_session_skips_holiday(self) -> None:
        """Next session skips holidays."""
        # Dec 22, 2023 (Friday) -> Dec 26 (Tuesday, skipping Christmas Monday)
        assert next_session(date(2023, 12, 22)) == date(2023, 12, 26)


class TestPrevSession:
    """Tests for prev_session function."""

    def test_prev_session_from_weekday(self) -> None:
        """Previous session from weekday is previous weekday."""
        # Wednesday -> Tuesday
        assert prev_session(date(2023, 1, 4)) == date(2023, 1, 3)

    def test_prev_session_skips_weekend(self) -> None:
        """Previous session skips weekend."""
        # Monday -> Friday
        assert prev_session(date(2023, 1, 9)) == date(2023, 1, 6)

    def test_prev_session_from_sunday(self) -> None:
        """Previous session from Sunday is Friday."""
        assert prev_session(date(2023, 1, 8)) == date(2023, 1, 6)


class TestGetSessionsBetween:
    """Tests for get_sessions_between function."""

    def test_sessions_in_range(self) -> None:
        """Get sessions in a date range."""
        sessions = get_sessions_between(date(2023, 1, 3), date(2023, 1, 10))

        # Should be 6 sessions: Jan 3-6, 9-10
        assert len(sessions) == 6
        assert sessions[0] == date(2023, 1, 3)
        assert sessions[-1] == date(2023, 1, 10)

    def test_excludes_weekends(self) -> None:
        """Sessions exclude weekends."""
        sessions = get_sessions_between(date(2023, 1, 6), date(2023, 1, 9))

        # Should be just Friday and Monday
        assert len(sessions) == 2
        assert date(2023, 1, 7) not in sessions
        assert date(2023, 1, 8) not in sessions

    def test_empty_range(self) -> None:
        """Empty range returns empty list."""
        sessions = get_sessions_between(date(2023, 1, 10), date(2023, 1, 3))
        assert sessions == []


class TestThanksgiving:
    """Tests for Thanksgiving calculation."""

    def test_thanksgiving_2023(self) -> None:
        """Thanksgiving 2023 is Nov 23."""
        assert _get_thanksgiving(2023) == date(2023, 11, 23)

    def test_thanksgiving_2024(self) -> None:
        """Thanksgiving 2024 is Nov 28."""
        assert _get_thanksgiving(2024) == date(2024, 11, 28)

    def test_thanksgiving_2025(self) -> None:
        """Thanksgiving 2025 is Nov 27."""
        assert _get_thanksgiving(2025) == date(2025, 11, 27)
