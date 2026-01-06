"""Tests for sparkline SVG rendering."""

import pytest


class TestSparklineRenderer:
    """Tests for inline SVG sparkline generation."""

    def test_render_sparkline_basic(self) -> None:
        """Test basic sparkline rendering."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from update_pages_index import render_sparkline

        points = [1.0, 2.0, 3.0, 2.5, 4.0]
        svg = render_sparkline(points)

        # Check SVG structure
        assert svg.startswith("<svg")
        assert "</svg>" in svg
        assert 'role="img"' in svg
        assert "aria-label" in svg

    def test_render_sparkline_dimensions(self) -> None:
        """Test sparkline with custom dimensions."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from update_pages_index import render_sparkline

        points = [1.0, 2.0, 3.0]
        svg = render_sparkline(points, width=200, height=50)

        assert 'width="200"' in svg
        assert 'height="50"' in svg

    def test_render_sparkline_accessibility(self) -> None:
        """Test sparkline has proper accessibility attributes."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from update_pages_index import render_sparkline

        points = [1.0, 2.0, 3.0]
        svg = render_sparkline(points, label="Sharpe Delta Trend")

        assert 'aria-label="Sharpe Delta Trend"' in svg
        assert 'role="img"' in svg

    def test_render_sparkline_color_var(self) -> None:
        """Test sparkline uses CSS custom property for color."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from update_pages_index import render_sparkline

        points = [1.0, 2.0, 3.0]
        svg = render_sparkline(points, color_var="--accent-color")

        assert "var(--accent-color)" in svg

    def test_render_sparkline_empty_points(self) -> None:
        """Test sparkline with empty points list."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from update_pages_index import render_sparkline

        svg = render_sparkline([])

        # Should return empty or placeholder
        assert svg == "" or "<svg" in svg

    def test_render_sparkline_single_point(self) -> None:
        """Test sparkline with single point."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from update_pages_index import render_sparkline

        svg = render_sparkline([5.0])

        # Should handle gracefully
        assert svg == "" or "<svg" in svg

    def test_render_sparkline_negative_values(self) -> None:
        """Test sparkline with negative values."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from update_pages_index import render_sparkline

        points = [-0.05, 0.02, -0.03, 0.01, -0.01]
        svg = render_sparkline(points)

        assert "<svg" in svg
        assert "</svg>" in svg

    def test_render_sparkline_all_same_value(self) -> None:
        """Test sparkline when all values are the same."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from update_pages_index import render_sparkline

        points = [5.0, 5.0, 5.0, 5.0]
        svg = render_sparkline(points)

        # Should not crash on zero range
        assert "<svg" in svg or svg == ""

    def test_render_sparkline_tooltip(self) -> None:
        """Test sparkline has title element for tooltip."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from update_pages_index import render_sparkline

        points = [1.0, 2.0, 3.0]
        svg = render_sparkline(points, label="Test Trend")

        # Should have title for tooltip
        assert "<title>" in svg or 'aria-label="Test Trend"' in svg
