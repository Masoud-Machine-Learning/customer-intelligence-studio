"""Filter state and its translation into parameterised SQL.

This is deliberately business logic rather than UI code: the Streamlit widgets in
``src.ui.filters`` only build a :class:`FilterState`, and every analytics function
accepts one. That keeps filtering testable without a running Streamlit session and
guarantees each page filters identically.

An empty selection means "no restriction on this dimension", which is what users expect
from a multi-select that has been left alone.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Any, Iterable

# Filterable dimensions mapped to their column in the analytics view. Restricting the
# mapping here means a column name can never be injected from user input.
FILTER_COLUMNS: dict[str, str] = {
    "regions": "region",
    "transport_modes": "transport_mode",
    "customer_segments": "customer_segment",
    "service_lines": "service_line",
    "channels": "channel",
    "time_bands": "time_band",
    "journey_purposes": "journey_purpose",
    "age_groups": "age_group",
}


def _clean(values: Iterable[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(dict.fromkeys(str(v) for v in values if v is not None and str(v) != ""))


@dataclass(frozen=True)
class FilterState:
    """The analytical population currently selected by the user."""

    start_date: date
    end_date: date
    regions: tuple[str, ...] = ()
    transport_modes: tuple[str, ...] = ()
    customer_segments: tuple[str, ...] = ()
    service_lines: tuple[str, ...] = ()
    channels: tuple[str, ...] = ()
    time_bands: tuple[str, ...] = ()
    journey_purposes: tuple[str, ...] = ()
    age_groups: tuple[str, ...] = ()

    @classmethod
    def create(cls, start_date: date, end_date: date, **selections: Any) -> "FilterState":
        """Build a state, normalising list-like selections into tuples."""
        cleaned = {key: _clean(selections.get(key)) for key in FILTER_COLUMNS}
        if end_date < start_date:
            start_date, end_date = end_date, start_date
        return cls(start_date=start_date, end_date=end_date, **cleaned)

    # -- period helpers ---------------------------------------------------------------

    @property
    def period_days(self) -> int:
        """Inclusive length of the selected period in days."""
        return (self.end_date - self.start_date).days + 1

    def previous_period(self) -> tuple[date, date]:
        """The equal-length window immediately preceding the selection."""
        previous_end = self.start_date - timedelta(days=1)
        previous_start = previous_end - timedelta(days=self.period_days - 1)
        return previous_start, previous_end

    def for_previous_period(self) -> "FilterState":
        """Same dimensional filters, shifted onto the previous period."""
        start, end = self.previous_period()
        return replace(self, start_date=start, end_date=end)

    def with_dates(self, start_date: date, end_date: date) -> "FilterState":
        return replace(self, start_date=start_date, end_date=end_date)

    def without(self, *dimensions: str) -> "FilterState":
        """Drop selected dimensions, used when a chart must show all values of one."""
        updates = {dim: () for dim in dimensions if dim in FILTER_COLUMNS}
        return replace(self, **updates)

    # -- SQL ---------------------------------------------------------------------------

    def active_dimensions(self) -> dict[str, tuple[str, ...]]:
        """Only the dimensions the user has actually restricted."""
        return {key: getattr(self, key) for key in FILTER_COLUMNS if getattr(self, key)}

    def describe(self) -> str:
        """Short human-readable summary of the dimensional restrictions."""
        active = self.active_dimensions()
        if not active:
            return "all regions, modes and segments"
        parts = []
        for key, values in active.items():
            label = key.replace("_", " ")
            if len(values) <= 2:
                parts.append(f"{label}: {', '.join(values)}")
            else:
                parts.append(f"{len(values)} {label}")
        return "; ".join(parts)


def build_where(
    filters: FilterState | None,
    *,
    include_dates: bool = True,
    alias: str = "",
) -> tuple[str, list[Any]]:
    """Return a ``WHERE`` clause and its bound parameters.

    Values are always bound as parameters; only column names - which come from the
    fixed :data:`FILTER_COLUMNS` mapping - are interpolated.
    """
    if filters is None:
        return "", []

    prefix = f"{alias}." if alias else ""
    clauses: list[str] = []
    params: list[Any] = []

    if include_dates:
        clauses.append(f"{prefix}date BETWEEN ? AND ?")
        params.extend([filters.start_date, filters.end_date])

    for attribute, column in FILTER_COLUMNS.items():
        values = getattr(filters, attribute)
        if not values:
            continue
        placeholders = ", ".join("?" for _ in values)
        clauses.append(f"{prefix}{column} IN ({placeholders})")
        params.extend(values)

    if not clauses:
        return "", []
    return "WHERE " + "\n  AND ".join(clauses), params


def where_fragment(filters: FilterState | None, **kwargs: Any) -> tuple[str, list[Any]]:
    """Like :func:`build_where` but as an ``AND``-joined fragment without ``WHERE``."""
    clause, params = build_where(filters, **kwargs)
    if not clause:
        return "TRUE", []
    return clause[len("WHERE ") :], params
