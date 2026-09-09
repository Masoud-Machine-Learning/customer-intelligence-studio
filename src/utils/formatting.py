"""Presentation helpers: number, percentage and delta formatting.

Formatting lives here rather than in page scripts so that every page renders the same
metric in exactly the same way.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any
from collections.abc import Sequence

from src.utils.config import LOWER_IS_BETTER

EN_DASH = "-"


def is_missing(value: Any) -> bool:
    """True for ``None`` and NaN, which both mean 'not available' in this app."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def format_int(value: Any) -> str:
    """Thousands-separated integer, or a dash when unavailable."""
    if is_missing(value):
        return EN_DASH
    return f"{int(round(float(value))):,}"


def format_number(value: Any, decimals: int = 1) -> str:
    """Fixed-decimal number with thousands separators."""
    if is_missing(value):
        return EN_DASH
    return f"{float(value):,.{decimals}f}"


def format_percent(value: Any, decimals: int = 1) -> str:
    """Format a value already expressed on a 0-100 scale."""
    if is_missing(value):
        return EN_DASH
    return f"{float(value):.{decimals}f}%"


def format_minutes(value: Any, decimals: int = 1) -> str:
    """Format a duration expressed in minutes."""
    if is_missing(value):
        return EN_DASH
    return f"{float(value):.{decimals}f} min"


def format_score(value: Any, decimals: int = 2) -> str:
    """Format a 1-5 satisfaction score."""
    if is_missing(value):
        return EN_DASH
    return f"{float(value):.{decimals}f} / 5"


def format_compact(value: Any) -> str:
    """Compact form for large counts (12,483 -> 12.5k)."""
    if is_missing(value):
        return EN_DASH
    number = float(value)
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.1f}M"
    if abs(number) >= 10_000:
        return f"{number / 1_000:.1f}k"
    return f"{number:,.0f}"


def percent_change(current: Any, previous: Any) -> float | None:
    """Relative change in percent, or ``None`` when it cannot be computed."""
    if is_missing(current) or is_missing(previous):
        return None
    base = float(previous)
    if base == 0:
        return None
    return (float(current) - base) / abs(base) * 100.0


def absolute_change(current: Any, previous: Any) -> float | None:
    """Absolute change, or ``None`` when either side is missing."""
    if is_missing(current) or is_missing(previous):
        return None
    return float(current) - float(previous)


def format_delta(
    current: Any,
    previous: Any,
    *,
    unit: str = "",
    decimals: int = 1,
    as_points: bool = False,
) -> str:
    """Human-readable absolute change against the previous period.

    ``as_points`` renders percentage-point changes ("+2.1 pp"), which is the correct
    comparison for rate metrics such as on-time performance.
    """
    diff = absolute_change(current, previous)
    if diff is None:
        return "no comparison available"
    sign = "+" if diff >= 0 else ""
    suffix = " pp" if as_points else (f" {unit}" if unit else "")
    return f"{sign}{diff:.{decimals}f}{suffix} vs previous period"


def delta_direction(metric_key: str, current: Any, previous: Any) -> str:
    """Classify a change as ``good``, ``bad`` or ``flat`` for colour coding."""
    diff = absolute_change(current, previous)
    if diff is None or abs(diff) < 1e-9:
        return "flat"
    improving = diff < 0 if metric_key in LOWER_IS_BETTER else diff > 0
    return "good" if improving else "bad"


def format_date_range(start: date, end: date) -> str:
    """Readable inclusive date range."""
    return f"{start:%d %b %Y} to {end:%d %b %Y}"


def humanise_list(items: Sequence[Any], conjunction: str = "and") -> str:
    """Join names into readable prose: 'A, B and C'."""
    cleaned = [str(item) for item in items if item is not None and str(item) != ""]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    return f"{', '.join(cleaned[:-1])} {conjunction} {cleaned[-1]}"


def ratio_phrase(numerator: Any, denominator: Any, decimals: int = 1) -> str:
    """Express a multiplier such as '2.1x higher', used by the insights engine."""
    if is_missing(numerator) or is_missing(denominator) or float(denominator) == 0:
        return EN_DASH
    return f"{float(numerator) / float(denominator):.{decimals}f}x"
