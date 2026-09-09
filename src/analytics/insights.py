"""Deterministic insight generation.

Every statement produced here is derived from a named calculation over the filtered
population, and each carries the numbers it was derived from in its ``evidence``
dictionary. No language model is involved: the findings are identical on every run, and
an analyst can reproduce any sentence from the evidence alone.

The optional AI layer in :mod:`src.ai` may rephrase these findings, but it never
generates them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any
from collections.abc import Sequence

import pandas as pd

from src.analytics.anomalies import Anomaly, detect_anomalies
from src.analytics.data_quality import build_quality_report
from src.analytics.kpis import (
    HEADLINE_KPIS,
    METRICS,
    KpiValue,
    compare_periods,
    format_metric,
    population_summary,
    select_clause,
)
from src.analytics.trends import period_over_period_by_dimension
from src.data.database import ANALYTICS_VIEW, DataSource
from src.data.filters import FilterState, build_where
from src.data.loader import run_query
from src.utils.config import MATERIAL_CHANGE_PCT, MIN_SAMPLE_FOR_COMPARISON
from src.utils.formatting import format_date_range, format_int, format_percent

CATEGORY_PERFORMANCE = "Service performance"
CATEGORY_CUSTOMER = "Customer experience"
CATEGORY_DEMAND = "Demand"
CATEGORY_OPERATIONS = "Operations"
CATEGORY_QUALITY = "Data quality"


@dataclass(frozen=True)
class Insight:
    """A single reproducible finding."""

    key: str
    category: str
    headline: str
    detail: str
    direction: str  # good | bad | neutral
    importance: float
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "category": self.category,
            "headline": self.headline,
            "detail": self.detail,
            "direction": self.direction,
            "importance": round(self.importance, 3),
            "evidence": self.evidence,
        }


# ---------------------------------------------------------------------------
# Individual generators
# ---------------------------------------------------------------------------


def _kpi_movement_insights(kpis: dict[str, KpiValue]) -> list[Insight]:
    """Report headline metrics that moved by more than the material threshold."""
    insights: list[Insight] = []
    for kpi in kpis.values():
        change_pct = kpi.change_pct
        change = kpi.change
        if change is None or change_pct is None:
            continue
        if abs(change_pct) < MATERIAL_CHANGE_PCT:
            continue

        unit = kpi.spec.unit
        movement = "increased" if change > 0 else "decreased"
        if kpi.spec.compare_as_points:
            magnitude = f"{abs(change):.1f} percentage points"
        elif unit == "count":
            magnitude = f"{abs(change_pct):.1f}%"
        else:
            magnitude = f"{abs(change_pct):.1f}%"

        insights.append(
            Insight(
                key=f"kpi_{kpi.key}",
                category=CATEGORY_PERFORMANCE if unit != "count" else CATEGORY_DEMAND,
                headline=f"{kpi.label} {movement} by {magnitude} against the previous period.",
                detail=(
                    f"{format_metric(kpi.current, unit)} this period versus "
                    f"{format_metric(kpi.previous, unit)} previously."
                ),
                direction=kpi.direction,
                importance=min(abs(change_pct) / 10.0, 3.0),
                evidence=kpi.as_evidence(),
            )
        )
    return insights


def _dimension_change_insights(
    source: DataSource,
    filters: FilterState,
    dimension: str,
    metric: str,
    category: str,
    noun: str,
) -> list[Insight]:
    """Best and worst movers for one metric across one dimension."""
    frame = period_over_period_by_dimension(
        source, filters, dimension, metric, min_journeys=MIN_SAMPLE_FOR_COMPARISON
    )
    if frame.empty or frame["previous_value"].isna().all():
        return []

    usable = frame.dropna(subset=["current_value", "previous_value"]).copy()
    if usable.empty:
        return []

    spec = METRICS[metric]
    lower_is_better = spec.lower_is_better
    usable["improvement"] = (
        -usable["absolute_change"] if lower_is_better else usable["absolute_change"]
    )

    insights: list[Insight] = []
    best = usable.loc[usable["improvement"].idxmax()]
    worst = usable.loc[usable["improvement"].idxmin()]

    for row, is_best in ((best, True), (worst, False)):
        change = float(row["absolute_change"])
        if abs(change) < 1e-6:
            continue
        percent = row["percent_change"]
        if percent is None or pd.isna(percent):
            continue
        if abs(float(percent)) < MATERIAL_CHANGE_PCT:
            continue

        value_name = str(row["dimension_value"])
        direction_word = "improvement" if is_best else "deterioration"
        movement = "increased" if change > 0 else "decreased"
        magnitude = (
            f"{abs(change):.1f} percentage points"
            if spec.compare_as_points
            else f"{abs(float(percent)):.1f}%"
        )
        insights.append(
            Insight(
                key=f"{dimension}_{metric}_{'best' if is_best else 'worst'}",
                category=category,
                headline=(
                    f"{value_name} recorded the largest {direction_word} in "
                    f"{spec.label.lower()} among {noun}, which {movement} by {magnitude}."
                ),
                detail=(
                    f"{format_metric(row['current_value'], spec.unit)} this period versus "
                    f"{format_metric(row['previous_value'], spec.unit)} previously, across "
                    f"{format_int(row['current_journeys'])} journeys."
                ),
                direction="good" if is_best else "bad",
                importance=min(abs(float(percent)) / 8.0, 3.0),
                evidence={
                    "dimension": dimension,
                    "dimension_value": value_name,
                    "metric": metric,
                    "current": round(float(row["current_value"]), 3),
                    "previous": round(float(row["previous_value"]), 3),
                    "absolute_change": round(change, 3),
                    "percent_change": round(float(percent), 2),
                    "current_journeys": int(row["current_journeys"]),
                    "previous_journeys": int(row["previous_journeys"]),
                },
            )
        )
    return insights


def _disruption_impact_insight(source: DataSource, filters: FilterState) -> list[Insight]:
    """Quantify how much worse the experience is during recorded disruptions."""
    where, params = build_where(filters)
    sql = f"""
    SELECT
        service_disruption,
        count(*) AS journeys,
        {select_clause(("complaint_rate", "avg_satisfaction", "avg_delay_minutes"))}
    FROM {ANALYTICS_VIEW}
    {where}
    GROUP BY 1
    HAVING service_disruption IS NOT NULL
    """
    frame = run_query(source, sql, params)
    if len(frame) < 2:
        return []

    disrupted = frame[frame["service_disruption"]]
    normal = frame[~frame["service_disruption"]]
    if disrupted.empty or normal.empty:
        return []
    if int(disrupted.iloc[0]["journeys"]) < MIN_SAMPLE_FOR_COMPARISON:
        return []

    disrupted_rate = float(disrupted.iloc[0]["complaint_rate"] or 0)
    normal_rate = float(normal.iloc[0]["complaint_rate"] or 0)
    if normal_rate <= 0:
        return []

    multiplier = disrupted_rate / normal_rate
    satisfaction_gap = float(normal.iloc[0]["avg_satisfaction"] or 0) - float(
        disrupted.iloc[0]["avg_satisfaction"] or 0
    )
    return [
        Insight(
            key="disruption_impact",
            category=CATEGORY_OPERATIONS,
            headline=(
                f"Complaint rates were {multiplier:.1f}x higher on journeys affected by a "
                "service disruption."
            ),
            detail=(
                f"{format_percent(disrupted_rate)} during disruption versus "
                f"{format_percent(normal_rate)} otherwise, with satisfaction "
                f"{satisfaction_gap:.2f} points lower across "
                f"{format_int(disrupted.iloc[0]['journeys'])} disrupted journeys."
            ),
            direction="bad" if multiplier > 1.2 else "neutral",
            importance=min(multiplier, 3.0),
            evidence={
                "disrupted_complaint_rate": round(disrupted_rate, 3),
                "baseline_complaint_rate": round(normal_rate, 3),
                "multiplier": round(multiplier, 2),
                "satisfaction_gap": round(satisfaction_gap, 3),
                "disrupted_journeys": int(disrupted.iloc[0]["journeys"]),
                "baseline_journeys": int(normal.iloc[0]["journeys"]),
            },
        )
    ]


def _peak_pressure_insight(source: DataSource, filters: FilterState) -> list[Insight]:
    """Find the weekday and time band with the worst delay relative to the average."""
    where, params = build_where(filters)
    sql = f"""
    SELECT
        day_of_week,
        time_band,
        count(*) AS journeys,
        {select_clause(("avg_delay_minutes", "on_time_performance"))}
    FROM {ANALYTICS_VIEW}
    {where}
    GROUP BY 1, 2
    HAVING count(*) >= ?
    """
    frame = run_query(source, sql, [*params, MIN_SAMPLE_FOR_COMPARISON])
    if frame.empty:
        return []

    overall = float(
        (frame["avg_delay_minutes"] * frame["journeys"]).sum() / frame["journeys"].sum()
    )
    worst = frame.loc[frame["avg_delay_minutes"].idxmax()]
    worst_delay = float(worst["avg_delay_minutes"])
    if overall <= 0 or worst_delay <= overall * 1.15:
        return []

    names = {
        0: "Sunday",
        1: "Monday",
        2: "Tuesday",
        3: "Wednesday",
        4: "Thursday",
        5: "Friday",
        6: "Saturday",
    }
    day_name = names.get(int(worst["day_of_week"]), "Unknown")
    band = str(worst["time_band"])
    return [
        Insight(
            key="peak_pressure",
            category=CATEGORY_OPERATIONS,
            headline=(f"{day_name} {band.lower()} delay levels exceeded the period baseline."),
            detail=(
                f"Average delay of {worst_delay:.1f} min against a period average of "
                f"{overall:.1f} min, with on-time performance at "
                f"{format_percent(worst['on_time_performance'])} across "
                f"{format_int(worst['journeys'])} journeys."
            ),
            direction="bad",
            importance=min((worst_delay / overall - 1) * 4, 3.0),
            evidence={
                "day": day_name,
                "time_band": band,
                "avg_delay_minutes": round(worst_delay, 2),
                "period_average_delay_minutes": round(overall, 2),
                "ratio": round(worst_delay / overall, 2),
                "journeys": int(worst["journeys"]),
            },
        )
    ]


def _complaint_concentration_insight(source: DataSource, filters: FilterState) -> list[Insight]:
    """Identify whether complaints concentrate in one category."""
    where, params = build_where(filters)
    connector = "AND" if where else "WHERE"
    sql = f"""
    SELECT complaint_category, count(*) AS complaints
    FROM {ANALYTICS_VIEW}
    {where}
    {connector} complaint_flag AND complaint_category IS NOT NULL
    GROUP BY 1
    ORDER BY complaints DESC
    """
    frame = run_query(source, sql, params)
    if frame.empty or frame["complaints"].sum() < MIN_SAMPLE_FOR_COMPARISON:
        return []

    total = int(frame["complaints"].sum())
    top = frame.iloc[0]
    share = float(top["complaints"]) / total * 100.0
    if share < 20:
        return []

    return [
        Insight(
            key="complaint_concentration",
            category=CATEGORY_CUSTOMER,
            headline=(
                f"'{top['complaint_category']}' accounts for {share:.0f}% of all complaints "
                "in the selected population."
            ),
            detail=(
                f"{format_int(top['complaints'])} of {format_int(total)} complaints, ahead of "
                f"'{frame.iloc[1]['complaint_category']}' at "
                f"{float(frame.iloc[1]['complaints']) / total * 100:.0f}%."
                if len(frame) > 1
                else f"{format_int(top['complaints'])} of {format_int(total)} complaints."
            ),
            direction="neutral",
            importance=min(share / 20.0, 2.5),
            evidence={
                "top_category": str(top["complaint_category"]),
                "top_complaints": int(top["complaints"]),
                "total_complaints": total,
                "share_pct": round(share, 1),
                "breakdown": frame.head(5).to_dict(orient="records"),
            },
        )
    ]


def _anomaly_insight(anomalies: Sequence[Anomaly]) -> list[Insight]:
    """Promote the most severe adverse anomaly into the insight feed."""
    adverse = [a for a in anomalies if a.is_adverse]
    if not adverse:
        return []
    worst = max(adverse, key=lambda a: abs(a.z_score))
    return [
        Insight(
            key="top_anomaly",
            category=CATEGORY_PERFORMANCE,
            headline=worst.describe(),
            detail=f"{worst.context()} Severity: {worst.severity} (z = {worst.z_score:.1f}).",
            direction="bad",
            importance=min(abs(worst.z_score) / 2.0, 3.0),
            evidence=worst.to_evidence(),
        )
    ]


def _data_quality_insight(source: DataSource, filters: FilterState) -> list[Insight]:
    """Warn when excluded rows or missing values are large enough to matter."""
    report = build_quality_report(source, filters)
    if report.total_rows == 0:
        return []
    excluded_pct = report.excluded_rows / report.total_rows * 100.0
    if excluded_pct < 0.1 and report.completeness_pct > 99.5:
        return []
    return [
        Insight(
            key="data_quality",
            category=CATEGORY_QUALITY,
            headline=(
                f"{format_int(report.excluded_rows)} of {format_int(report.total_rows)} rows "
                f"({excluded_pct:.2f}%) were excluded from analysis by validity rules."
            ),
            detail=(
                f"Field completeness averages {report.completeness_pct:.1f}% and the overall "
                f"quality score is {report.score}/100 ({report.rating})."
            ),
            direction="bad" if excluded_pct > 1 else "neutral",
            importance=min(excluded_pct, 1.5),
            evidence=report.to_evidence(),
        )
    ]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def generate_insights(
    source: DataSource,
    filters: FilterState,
    limit: int = 8,
    include_quality: bool = True,
    anomalies: Sequence[Anomaly] | None = None,
) -> list[Insight]:
    """Run every generator and return the most important findings, ranked."""
    kpis = compare_periods(source, filters, HEADLINE_KPIS)

    if anomalies is None:
        anomalies = detect_anomalies(
            source, filters, dimension="region", adverse_only=True, limit=20
        )

    collected: list[Insight] = []
    collected += _kpi_movement_insights(kpis)
    collected += _dimension_change_insights(
        source,
        filters,
        "customer_segment",
        "avg_satisfaction",
        CATEGORY_CUSTOMER,
        "customer segments",
    )
    collected += _dimension_change_insights(
        source, filters, "region", "on_time_performance", CATEGORY_PERFORMANCE, "regions"
    )
    collected += _dimension_change_insights(
        source, filters, "transport_mode", "total_journeys", CATEGORY_DEMAND, "transport modes"
    )
    collected += _dimension_change_insights(
        source, filters, "region", "complaint_rate", CATEGORY_CUSTOMER, "regions"
    )
    collected += _disruption_impact_insight(source, filters)
    collected += _peak_pressure_insight(source, filters)
    collected += _complaint_concentration_insight(source, filters)
    collected += _anomaly_insight(anomalies)
    if include_quality:
        collected += _data_quality_insight(source, filters)

    # Stable ordering: importance first, then key, so equal-weight findings never swap.
    collected.sort(key=lambda i: (-i.importance, i.key))
    return collected[:limit]


# ---------------------------------------------------------------------------
# Executive brief
# ---------------------------------------------------------------------------


@dataclass
class ExecutiveBrief:
    """Structured summary assembled entirely from deterministic analytics."""

    period_label: str
    population: dict[str, Any]
    current_performance: list[str]
    key_changes: list[str]
    attention: list[str]
    positives: list[str]
    generated_on: date
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Render the brief as Markdown for download."""
        lines = [
            "# Customer Intelligence Studio - Executive Brief",
            "",
            f"**Period:** {self.period_label}  ",
            f"**Population:** {format_int(self.population.get('journeys'))} journeys, "
            f"{format_int(self.population.get('customers'))} customers, "
            f"{self.population.get('regions', 0)} regions  ",
            f"**Generated:** {self.generated_on:%d %b %Y}",
            "",
            "## Current performance",
            "",
        ]
        lines += [f"- {item}" for item in self.current_performance] or ["- No data available."]
        lines += ["", "## Key changes", ""]
        lines += [f"- {item}" for item in self.key_changes] or ["- No material changes detected."]
        lines += ["", "## Areas requiring attention", ""]
        lines += [f"- {item}" for item in self.attention] or ["- Nothing flagged this period."]
        lines += ["", "## Positive developments", ""]
        lines += [f"- {item}" for item in self.positives] or ["- None identified this period."]
        lines += [
            "",
            "---",
            "",
            "All figures are computed from the selected population by deterministic "
            "analytics functions. Data is synthetic and generated for demonstration.",
        ]
        return "\n".join(lines)

    def to_text(self) -> str:
        """Plain-text rendering for pasting into email or a ticket."""
        markdown = self.to_markdown()
        return markdown.replace("# ", "").replace("**", "").replace("- ", "  - ")


def build_executive_brief(
    source: DataSource,
    filters: FilterState,
    insights: Sequence[Insight] | None = None,
) -> ExecutiveBrief:
    """Assemble the executive brief from KPIs and ranked insights."""
    kpis = compare_periods(source, filters, HEADLINE_KPIS)
    population = population_summary(source, filters)
    findings = (
        list(insights) if insights is not None else generate_insights(source, filters, limit=10)
    )

    current_performance = [
        f"{kpi.label}: {kpi.formatted()} ({kpi.formatted_delta()})" for kpi in kpis.values()
    ]

    key_changes = [
        f"{item.headline} {item.detail}"
        for item in findings
        if item.category in {CATEGORY_PERFORMANCE, CATEGORY_DEMAND, CATEGORY_CUSTOMER}
    ][:5]

    attention = [f"{item.headline} {item.detail}" for item in findings if item.direction == "bad"][
        :5
    ]

    positives = [f"{item.headline} {item.detail}" for item in findings if item.direction == "good"][
        :4
    ]

    return ExecutiveBrief(
        period_label=format_date_range(filters.start_date, filters.end_date),
        population=population,
        current_performance=current_performance,
        key_changes=key_changes,
        attention=attention,
        positives=positives,
        generated_on=date.today(),
        evidence={
            "kpis": [kpi.as_evidence() for kpi in kpis.values()],
            "insights": [item.to_dict() for item in findings],
            "population": {
                key: (value.isoformat() if isinstance(value, date) else value)
                for key, value in population.items()
            },
        },
    )
