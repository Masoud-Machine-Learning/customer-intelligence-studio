"""KPI definitions and period-over-period comparison.

Every headline metric in the application is defined exactly once, here, as a SQL
expression. Pages ask for KPIs by key and never write their own aggregation, so the
"on-time performance" shown on the Executive Overview is by construction the same
number the insights engine and the AI assistant reason about.

Aggregation happens in DuckDB. Pandas is used only to assemble the comparison table
once the database has returned one row per period.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections.abc import Iterable

import pandas as pd

from src.data.database import ANALYTICS_VIEW, DataSource
from src.data.filters import FilterState, build_where
from src.data.loader import run_query
from src.utils.config import LOWER_IS_BETTER
from src.utils.formatting import (
    absolute_change,
    delta_direction,
    format_int,
    format_minutes,
    format_number,
    format_percent,
    format_score,
    percent_change,
)


@dataclass(frozen=True)
class MetricSpec:
    """A single analytical metric: how to compute it and how to present it."""

    key: str
    label: str
    sql: str
    unit: str  # count | percent | score | minutes | hours
    description: str
    compare_as_points: bool = False

    @property
    def lower_is_better(self) -> bool:
        return self.key in LOWER_IS_BETTER


# Rates are computed with FILTER so that NULLs (missing satisfaction, an uploaded feed
# without complaint data) reduce the denominator instead of silently counting as zero.
METRICS: dict[str, MetricSpec] = {
    "total_journeys": MetricSpec(
        key="total_journeys",
        label="Total Journeys",
        sql="count(*)",
        unit="count",
        description="Journeys in the selected population.",
    ),
    "active_customers": MetricSpec(
        key="active_customers",
        label="Active Customers",
        sql="count(DISTINCT customer_id)",
        unit="count",
        description="Distinct customers who travelled at least once.",
    ),
    "avg_satisfaction": MetricSpec(
        key="avg_satisfaction",
        label="Average Satisfaction",
        sql="avg(customer_satisfaction)",
        unit="score",
        description="Mean 1-5 satisfaction score, excluding journeys with no score.",
    ),
    "on_time_performance": MetricSpec(
        key="on_time_performance",
        label="On-Time Performance",
        sql=(
            "100.0 * count(*) FILTER (WHERE on_time) "
            "/ nullif(count(*) FILTER (WHERE on_time IS NOT NULL), 0)"
        ),
        unit="percent",
        description="Share of journeys arriving within the on-time threshold.",
        compare_as_points=True,
    ),
    "avg_delay_minutes": MetricSpec(
        key="avg_delay_minutes",
        label="Average Delay",
        sql="avg(delay_minutes)",
        unit="minutes",
        description="Mean delay across all journeys, including on-time ones.",
    ),
    "p90_delay_minutes": MetricSpec(
        key="p90_delay_minutes",
        label="90th Percentile Delay",
        sql="quantile_cont(delay_minutes, 0.9)",
        unit="minutes",
        description="Delay experienced by the worst-affected 10% of journeys.",
    ),
    "complaint_rate": MetricSpec(
        key="complaint_rate",
        label="Complaint Rate",
        sql=(
            "100.0 * count(*) FILTER (WHERE complaint_flag) "
            "/ nullif(count(*) FILTER (WHERE complaint_flag IS NOT NULL), 0)"
        ),
        unit="percent",
        description="Share of journeys that generated a complaint.",
        compare_as_points=True,
    ),
    "disruption_rate": MetricSpec(
        key="disruption_rate",
        label="Disruption Rate",
        sql=(
            "100.0 * count(*) FILTER (WHERE service_disruption) "
            "/ nullif(count(*) FILTER (WHERE service_disruption IS NOT NULL), 0)"
        ),
        unit="percent",
        description="Share of journeys affected by a recorded service disruption.",
        compare_as_points=True,
    ),
    "avg_resolution_hours": MetricSpec(
        key="avg_resolution_hours",
        label="Avg Complaint Resolution",
        sql="avg(resolution_time_hours)",
        unit="hours",
        description="Mean hours to resolve a complaint.",
    ),
    "repeat_rate": MetricSpec(
        key="repeat_rate",
        label="Repeat Customer Share",
        sql=(
            "100.0 * count(*) FILTER (WHERE repeat_customer) "
            "/ nullif(count(*) FILTER (WHERE repeat_customer IS NOT NULL), 0)"
        ),
        unit="percent",
        description="Share of journeys made by repeat customers.",
        compare_as_points=True,
    ),
    "digital_rate": MetricSpec(
        key="digital_rate",
        label="Digital Interaction Rate",
        sql=(
            "100.0 * count(*) FILTER (WHERE digital_interaction) "
            "/ nullif(count(*) FILTER (WHERE digital_interaction IS NOT NULL), 0)"
        ),
        unit="percent",
        description="Share of journeys with an accompanying digital interaction.",
        compare_as_points=True,
    ),
    "accessibility_rate": MetricSpec(
        key="accessibility_rate",
        label="Accessibility Service Use",
        sql=(
            "100.0 * count(*) FILTER (WHERE accessibility_service_used) "
            "/ nullif(count(*) FILTER (WHERE accessibility_service_used IS NOT NULL), 0)"
        ),
        unit="percent",
        description="Share of journeys using an accessibility service.",
        compare_as_points=True,
    ),
}

HEADLINE_KPIS: tuple[str, ...] = (
    "total_journeys",
    "active_customers",
    "avg_satisfaction",
    "on_time_performance",
    "avg_delay_minutes",
    "complaint_rate",
)


@dataclass(frozen=True)
class KpiValue:
    """A metric evaluated for the current period and the comparison period."""

    spec: MetricSpec
    current: float | None
    previous: float | None

    @property
    def key(self) -> str:
        return self.spec.key

    @property
    def label(self) -> str:
        return self.spec.label

    @property
    def change(self) -> float | None:
        return absolute_change(self.current, self.previous)

    @property
    def change_pct(self) -> float | None:
        return percent_change(self.current, self.previous)

    @property
    def direction(self) -> str:
        return delta_direction(self.key, self.current, self.previous)

    def formatted(self) -> str:
        """Format the current value according to the metric's unit."""
        return format_metric(self.current, self.spec.unit)

    def formatted_delta(self) -> str:
        """Change against the previous period, in the right units."""
        change = self.change
        if change is None:
            return "no comparison period"
        sign = "+" if change >= 0 else ""
        if self.spec.compare_as_points:
            return f"{sign}{change:.1f} pp vs previous period"
        if self.spec.unit == "count":
            pct = self.change_pct
            pct_text = f" ({sign}{pct:.1f}%)" if pct is not None else ""
            return f"{sign}{change:,.0f}{pct_text} vs previous period"
        if self.spec.unit == "minutes":
            return f"{sign}{change:.2f} min vs previous period"
        if self.spec.unit == "hours":
            return f"{sign}{change:.1f} h vs previous period"
        return f"{sign}{change:.2f} vs previous period"

    def as_evidence(self) -> dict[str, Any]:
        """Machine-readable form handed to the insights engine and AI assistant."""
        return {
            "metric": self.key,
            "label": self.label,
            "unit": self.spec.unit,
            "current": None if self.current is None else round(float(self.current), 4),
            "previous": None if self.previous is None else round(float(self.previous), 4),
            "absolute_change": None if self.change is None else round(self.change, 4),
            "percent_change": None if self.change_pct is None else round(self.change_pct, 2),
            "direction": self.direction,
        }


def format_metric(value: Any, unit: str) -> str:
    """Format a raw metric value for display, given its unit."""
    if unit == "count":
        return format_int(value)
    if unit == "percent":
        return format_percent(value)
    if unit == "score":
        return format_score(value)
    if unit == "minutes":
        return format_minutes(value)
    if unit == "hours":
        return format_number(value, 1) + " h"
    return format_number(value, 2)


def select_clause(keys: Iterable[str]) -> str:
    return ",\n    ".join(f"{METRICS[key].sql} AS {key}" for key in keys)


def compute_metrics(
    source: DataSource,
    filters: FilterState | None,
    keys: Iterable[str] | None = None,
) -> dict[str, float | None]:
    """Evaluate metrics for one filtered population (single SQL round trip)."""
    keys = tuple(keys or METRICS.keys())
    where, params = build_where(filters)
    sql = f"""
    SELECT
    {select_clause(keys)}
    FROM {ANALYTICS_VIEW}
    {where}
    """
    frame = run_query(source, sql, params)
    if frame.empty:
        return dict.fromkeys(keys)
    row = frame.iloc[0]
    return {key: (None if pd.isna(row[key]) else float(row[key])) for key in keys}


def compare_periods(
    source: DataSource,
    filters: FilterState,
    keys: Iterable[str] | None = None,
) -> dict[str, KpiValue]:
    """Evaluate metrics for the selected period and the preceding equal-length window.

    The comparison window is derived from the selection itself, so "previous period"
    always means the same number of days immediately before the current range.
    """
    keys = tuple(keys or HEADLINE_KPIS)
    current = compute_metrics(source, filters, keys)
    previous = compute_metrics(source, filters.for_previous_period(), keys)
    return {
        key: KpiValue(spec=METRICS[key], current=current.get(key), previous=previous.get(key))
        for key in keys
    }


def kpi_summary_frame(kpis: dict[str, KpiValue]) -> pd.DataFrame:
    """Tabular view of a KPI set, used for exports and the evidence panel."""
    return pd.DataFrame(
        [
            {
                "Metric": kpi.label,
                "Current": kpi.formatted(),
                "Previous": format_metric(kpi.previous, kpi.spec.unit),
                "Change": kpi.formatted_delta().replace(" vs previous period", ""),
                "Direction": kpi.direction,
            }
            for kpi in kpis.values()
        ]
    )


def population_summary(source: DataSource, filters: FilterState) -> dict[str, Any]:
    """Describe the currently selected analytical population.

    Powers the "Analysing N journeys across M regions" line shown under the filters.
    """
    where, params = build_where(filters)
    sql = f"""
    SELECT
        count(*)                        AS journeys,
        count(DISTINCT customer_id)     AS customers,
        count(DISTINCT region)          AS regions,
        count(DISTINCT transport_mode)  AS modes,
        count(DISTINCT service_line)    AS service_lines,
        min(date)                       AS first_date,
        max(date)                       AS last_date
    FROM {ANALYTICS_VIEW}
    {where}
    """
    frame = run_query(source, sql, params)
    if frame.empty or int(frame.loc[0, "journeys"] or 0) == 0:
        return {
            "journeys": 0,
            "customers": 0,
            "regions": 0,
            "modes": 0,
            "service_lines": 0,
            "first_date": None,
            "last_date": None,
        }
    row = frame.iloc[0]
    return {
        "journeys": int(row["journeys"]),
        "customers": int(row["customers"]),
        "regions": int(row["regions"]),
        "modes": int(row["modes"]),
        "service_lines": int(row["service_lines"]),
        "first_date": pd.Timestamp(row["first_date"]).date()
        if pd.notna(row["first_date"])
        else None,
        "last_date": pd.Timestamp(row["last_date"]).date() if pd.notna(row["last_date"]) else None,
    }


def metrics_by_dimension(
    source: DataSource,
    filters: FilterState | None,
    dimension: str,
    keys: Iterable[str] | None = None,
    min_journeys: int = 1,
    limit: int | None = None,
) -> pd.DataFrame:
    """Group the metric set by a dimension column (region, mode, segment, ...).

    ``dimension`` is validated against the metric registry's allow-list of grouping
    columns so it can never carry user input into the SQL text.
    """
    if dimension not in GROUPING_COLUMNS:
        raise ValueError(f"Unsupported grouping column: {dimension}")
    keys = tuple(
        keys
        or (
            "total_journeys",
            "avg_satisfaction",
            "on_time_performance",
            "avg_delay_minutes",
            "complaint_rate",
        )
    )
    where, params = build_where(filters)
    sql = f"""
    SELECT
        {dimension} AS dimension_value,
        {select_clause(keys)}
    FROM {ANALYTICS_VIEW}
    {where}
    GROUP BY 1
    HAVING count(*) >= ?
    ORDER BY total_journeys DESC
    """
    if "total_journeys" not in keys:
        sql = sql.replace("ORDER BY total_journeys DESC", "ORDER BY 1")
    frame = run_query(source, sql, [*params, min_journeys])
    frame = frame.dropna(subset=["dimension_value"])
    if limit is not None:
        frame = frame.head(limit)
    return frame.reset_index(drop=True)


GROUPING_COLUMNS: tuple[str, ...] = (
    "region",
    "transport_mode",
    "customer_segment",
    "service_line",
    "channel",
    "age_group",
    "journey_purpose",
    "time_band",
    "fare_type",
    "origin_region",
    "destination_region",
    "complaint_category",
    "feedback_text_category",
)
