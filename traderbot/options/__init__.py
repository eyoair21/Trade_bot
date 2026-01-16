"""Options module placeholder.

This module provides placeholder data classes and stub implementations
for options chain analysis. Actual API integration (Polygon, Tradier)
will be added in a future sprint.

Current capabilities (placeholder):
- Data classes for option quotes, chains, IV, skew
- Stub loaders that read from sample JSON files
- Basic metrics computation placeholders

Next Sprint - API Integration:
- Polygon.io options chain endpoint
- Tradier options chain endpoint
- Real-time IV surface construction
- Greeks calculation
"""

from traderbot.options.placeholders import (
    IVSnapshot,
    OptionChainSummary,
    OptionQuote,
    SkewPoint,
    compute_basic_metrics,
    load_option_chain_sample,
)

__all__ = [
    "IVSnapshot",
    "OptionChainSummary",
    "OptionQuote",
    "SkewPoint",
    "compute_basic_metrics",
    "load_option_chain_sample",
]
