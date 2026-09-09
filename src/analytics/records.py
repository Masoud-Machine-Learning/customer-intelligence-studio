"""Row-level record access for the interactive analytics table.

Detail tables are paged and sorted in SQL rather than by pulling every row into the
browser: the point of the table is to inspect the records behind a selected metric, not
to dump the dataset.
"""

from __future__ import annotations

import pandas as pd

from src.data.database import ANALYTICS_VIEW, DataSource
from src.data.filters import FilterState, build_where
from src.data.loader import run_query

DISPLAY_COLUMNS: tuple[str, ...] = (
    "journey_id",
    "date",
    "timestamp",
    "region",
    "transport_mode",
    "service_line",
    "customer_segment",
    "channel",
    "time_band",
    "journey_duration_minutes",
    "delay_minutes",
    "on_time",
    "customer_satisfaction",
    "complaint_flag",
    "complaint_category",
    "resolution_time_hours",
    "service_disruption",
    "journey_purpose",
    "age_group",
    "fare_type",
)

SORT_OPTIONS: dict[str, str] = {
    "Most recent": "timestamp DESC",
    "Longest delay": "delay_minutes DESC",
    "Lowest satisfaction": "customer_satisfaction ASC NULLS LAST",
    "Highest satisfaction": "customer_satisfaction DESC NULLS LAST",
    "Longest journey": "journey_duration_minutes DESC",
    "Slowest complaint resolution": "resolution_time_hours DESC NULLS LAST",
}

SUBSET_FILTERS: dict[str, str] = {
    "All journeys": "",
    "Complaints only": "complaint_flag",
    "Delayed journeys only": "NOT on_time",
    "Disrupted journeys only": "service_disruption",
    "Low satisfaction (1-2)": "customer_satisfaction <= 2",
    "Accessibility services used": "accessibility_service_used",
}


def fetch_records(
    source: DataSource,
    filters: FilterState,
    *,
    columns: tuple[str, ...] = DISPLAY_COLUMNS,
    sort: str = "Most recent",
    subset: str = "All journeys",
    search: str = "",
    limit: int = 500,
) -> pd.DataFrame:
    """Return the records behind the current selection.

    ``search`` matches journey ID, service line, region and complaint category. It is
    bound as a parameter, never concatenated into the SQL text.
    """
    selected = [c for c in columns if c in DISPLAY_COLUMNS] or list(DISPLAY_COLUMNS)
    where, params = build_where(filters)
    clauses: list[str] = []

    subset_clause = SUBSET_FILTERS.get(subset, "")
    if subset_clause:
        clauses.append(subset_clause)

    if search.strip():
        clauses.append(
            "(lower(journey_id) LIKE ? OR lower(service_line) LIKE ? "
            "OR lower(region) LIKE ? OR lower(coalesce(complaint_category, '')) LIKE ?)"
        )
        pattern = f"%{search.strip().lower()}%"
        params = [*params, pattern, pattern, pattern, pattern]

    connector = "AND" if where else "WHERE"
    extra = f"{connector} " + " AND ".join(clauses) if clauses else ""
    order_by = SORT_OPTIONS.get(sort, SORT_OPTIONS["Most recent"])

    sql = f"""
    SELECT {", ".join(selected)}
    FROM {ANALYTICS_VIEW}
    {where}
    {extra}
    ORDER BY {order_by}
    LIMIT ?
    """
    return run_query(source, sql, [*params, limit])


def count_records(source: DataSource, filters: FilterState, subset: str = "All journeys") -> int:
    """How many records match, so the table can say what it is showing a slice of."""
    where, params = build_where(filters)
    subset_clause = SUBSET_FILTERS.get(subset, "")
    connector = "AND" if where else "WHERE"
    extra = f"{connector} {subset_clause}" if subset_clause else ""
    frame = run_query(source, f"SELECT count(*) AS n FROM {ANALYTICS_VIEW} {where} {extra}", params)
    return int(frame.loc[0, "n"] or 0)
