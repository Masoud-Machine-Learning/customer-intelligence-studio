"""Time-series aggregation and profiling.

SQL does the grouping (one row per day/week/month, optionally per dimension value);
Pandas does the sequential work that SQL is clumsy at - rolling means, reindexing a
gap-free calendar and pivoting into chart-ready shapes.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.analytics.kpis import GROUPING_COLUMNS, METRICS, select_clause
from src.data.database import ANALYTICS_VIEW, DataSource
from src.data.filters import FilterState, build_where
from src.data.loader import run_query

# Grain -> the pre-computed column in v_journeys_analytics.
GRAIN_COLUMNS: dict[str, str] = {
    "day": "date",
    "week": "week_start",
    "month": "month_start",
}

DEFAULT_TREND_METRICS: tuple[str, ...] = (
    "total_journeys",
    "avg_satisfaction",
    "on_time_performance",
    "avg_delay_minutes",
    "complaint_rate",
)


def time_series(
    source: DataSource,
    filters: FilterState | None,
    grain: str = "day",
    keys: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Aggregate the metric set over time at the requested grain."""
    if grain not in GRAIN_COLUMNS:
        raise ValueError(f"Unsupported grain: {grain}")
    keys = tuple(keys or DEFAULT_TREND_METRICS)
    column = GRAIN_COLUMNS[grain]
    where, params = build_where(filters)
    sql = f"""
    SELECT
        CAST({column} AS DATE) AS period,
        {select_clause(keys)}
    FROM {ANALYTICS_VIEW}
    {where}
    GROUP BY 1
    ORDER BY 1
    """
    frame = run_query(source, sql, params)
    if frame.empty:
        return pd.DataFrame(columns=["period", *keys])
    frame["period"] = pd.to_datetime(frame["period"])
    return frame


def fill_calendar(frame: pd.DataFrame, grain: str = "day") -> pd.DataFrame:
    """Reindex a daily series onto a gap-free calendar.

    Days with no journeys must appear as gaps in a chart rather than being skipped,
    otherwise a demand collapse looks like a straight line.
    """
    if frame.empty or grain != "day":
        return frame
    calendar = pd.date_range(frame["period"].min(), frame["period"].max(), freq="D")
    return frame.set_index("period").reindex(calendar).rename_axis("period").reset_index()


def add_rolling(
    frame: pd.DataFrame,
    column: str,
    window: int = 7,
    suffix: str = "_rolling",
) -> pd.DataFrame:
    """Append a centred-on-trailing rolling mean, computed in Pandas."""
    if frame.empty or column not in frame.columns:
        return frame
    result = frame.copy()
    result[f"{column}{suffix}"] = (
        result[column].rolling(window=window, min_periods=max(2, window // 3)).mean()
    )
    return result


def series_by_dimension(
    source: DataSource,
    filters: FilterState | None,
    dimension: str,
    metric: str,
    grain: str = "week",
    top_n: int = 6,
) -> pd.DataFrame:
    """One time series per dimension value, limited to the busiest ``top_n`` values."""
    if dimension not in GROUPING_COLUMNS:
        raise ValueError(f"Unsupported grouping column: {dimension}")
    if metric not in METRICS:
        raise ValueError(f"Unknown metric: {metric}")
    if grain not in GRAIN_COLUMNS:
        raise ValueError(f"Unsupported grain: {grain}")

    column = GRAIN_COLUMNS[grain]
    where, params = build_where(filters)
    sql = f"""
    WITH ranked AS (
        SELECT {dimension} AS dimension_value, count(*) AS n
        FROM {ANALYTICS_VIEW}
        {where}
        GROUP BY 1
        ORDER BY n DESC
        LIMIT ?
    )
    SELECT
        CAST(j.{column} AS DATE) AS period,
        j.{dimension}            AS dimension_value,
        {METRICS[metric].sql}    AS value,
        count(*)                 AS journeys
    FROM {ANALYTICS_VIEW} j
    -- An inner join to the ranked CTE both limits the series count and drops NULLs.
    JOIN ranked r ON r.dimension_value = j.{dimension}
    {where}
    GROUP BY 1, 2
    ORDER BY 1, 2
    """
    # The WHERE clause is applied twice (once inside the CTE, once on the outer query),
    # so the bound parameters must be supplied twice as well.
    frame = run_query(source, sql, [*params, top_n, *params])
    if frame.empty:
        return pd.DataFrame(columns=["period", "dimension_value", "value", "journeys"])
    frame["period"] = pd.to_datetime(frame["period"])
    return frame


def hourly_profile(
    source: DataSource,
    filters: FilterState | None,
    keys: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Metric profile by hour of day, split by weekday and weekend."""
    keys = tuple(
        keys or ("total_journeys", "avg_delay_minutes", "on_time_performance", "avg_satisfaction")
    )
    where, params = build_where(filters)
    sql = f"""
    SELECT
        journey_hour AS hour,
        CASE WHEN is_weekend THEN 'Weekend' ELSE 'Weekday' END AS day_type,
        {select_clause(keys)}
    FROM {ANALYTICS_VIEW}
    {where}
    GROUP BY 1, 2
    ORDER BY 1, 2
    """
    return run_query(source, sql, params)


def weekday_profile(
    source: DataSource,
    filters: FilterState | None,
    keys: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Metric profile by day of week, ordered Monday to Sunday."""
    keys = tuple(
        keys
        or (
            "total_journeys",
            "avg_delay_minutes",
            "on_time_performance",
            "avg_satisfaction",
            "complaint_rate",
        )
    )
    where, params = build_where(filters)
    sql = f"""
    SELECT
        day_of_week,
        {select_clause(keys)}
    FROM {ANALYTICS_VIEW}
    {where}
    GROUP BY 1
    ORDER BY 1
    """
    frame = run_query(source, sql, params)
    if frame.empty:
        return frame
    # DuckDB's EXTRACT(dow) is 0=Sunday; present the week starting on Monday.
    names = {
        0: "Sunday",
        1: "Monday",
        2: "Tuesday",
        3: "Wednesday",
        4: "Thursday",
        5: "Friday",
        6: "Saturday",
    }
    order = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6,
    }
    frame["day_name"] = frame["day_of_week"].map(names)
    frame["sort_order"] = frame["day_name"].map(order)
    return frame.sort_values("sort_order").drop(columns=["sort_order"]).reset_index(drop=True)


def heatmap_matrix(
    source: DataSource,
    filters: FilterState | None,
    metric: str = "avg_delay_minutes",
) -> pd.DataFrame:
    """Day-of-week by hour matrix for a single metric, ready for a heatmap."""
    if metric not in METRICS:
        raise ValueError(f"Unknown metric: {metric}")
    where, params = build_where(filters)
    sql = f"""
    SELECT
        day_of_week,
        journey_hour AS hour,
        {METRICS[metric].sql} AS value,
        count(*) AS journeys
    FROM {ANALYTICS_VIEW}
    {where}
    GROUP BY 1, 2
    HAVING count(*) >= 5
    ORDER BY 1, 2
    """
    frame = run_query(source, sql, params)
    if frame.empty:
        return pd.DataFrame()
    names = {
        0: "Sunday",
        1: "Monday",
        2: "Tuesday",
        3: "Wednesday",
        4: "Thursday",
        5: "Friday",
        6: "Saturday",
    }
    frame["day_name"] = frame["day_of_week"].map(names)
    matrix = frame.pivot(index="day_name", columns="hour", values="value")
    ordered = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return matrix.reindex([d for d in ordered if d in matrix.index])


def period_over_period_by_dimension(
    source: DataSource,
    filters: FilterState,
    dimension: str,
    metric: str,
    min_journeys: int = 50,
) -> pd.DataFrame:
    """Compare one metric per dimension value across the current and previous periods.

    This single query is the backbone of the insights engine: it answers "which segment
    changed the most" without pulling any row-level data into Python.
    """
    if dimension not in GROUPING_COLUMNS:
        raise ValueError(f"Unsupported grouping column: {dimension}")
    if metric not in METRICS:
        raise ValueError(f"Unknown metric: {metric}")

    previous = filters.for_previous_period()
    current_where, current_params = build_where(filters)
    previous_where, previous_params = build_where(previous)
    expression = METRICS[metric].sql

    sql = f"""
    WITH current_period AS (
        SELECT {dimension} AS dimension_value,
               {expression} AS value,
               count(*)     AS journeys
        FROM {ANALYTICS_VIEW}
        {current_where}
        GROUP BY 1
    ),
    previous_period AS (
        SELECT {dimension} AS dimension_value,
               {expression} AS value,
               count(*)     AS journeys
        FROM {ANALYTICS_VIEW}
        {previous_where}
        GROUP BY 1
    )
    SELECT
        coalesce(c.dimension_value, p.dimension_value) AS dimension_value,
        c.value     AS current_value,
        p.value     AS previous_value,
        c.journeys  AS current_journeys,
        p.journeys  AS previous_journeys,
        c.value - p.value AS absolute_change,
        CASE WHEN p.value IS NULL OR p.value = 0 THEN NULL
             ELSE 100.0 * (c.value - p.value) / abs(p.value) END AS percent_change
    FROM current_period c
    FULL OUTER JOIN previous_period p ON c.dimension_value = p.dimension_value
    WHERE coalesce(c.dimension_value, p.dimension_value) IS NOT NULL
      AND coalesce(c.journeys, 0) >= ?
      AND coalesce(p.journeys, 0) >= ?
    ORDER BY abs(coalesce(c.value, 0) - coalesce(p.value, 0)) DESC
    """
    return run_query(
        source,
        sql,
        [*current_params, *previous_params, min_journeys, min_journeys],
    )
