"""Analytical tools available to the Ask the Data assistant.

Each tool is an ordinary Python function over the analytics layer. It returns a
:class:`ToolResult` containing

* ``answer``   - a complete, deterministic answer written from the numbers,
* ``evidence`` - the structured figures the answer was derived from,
* ``table``    - an optional DataFrame shown alongside the answer.

The language model, when enabled, receives only ``evidence`` and rewrites ``answer``.
It cannot query the database, invent a metric or choose a different calculation, which
is what keeps the feature grounded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable

import pandas as pd

from src.analytics.anomalies import anomalies_to_frame, detect_anomalies, summarise_anomalies
from src.analytics.data_quality import build_quality_report
from src.analytics.insights import generate_insights
from src.analytics.kpis import (
    HEADLINE_KPIS,
    METRICS,
    compare_periods,
    format_metric,
    metrics_by_dimension,
    population_summary,
)
from src.analytics.segmentation import compare_segments as compare_segment_metrics
from src.analytics.trends import period_over_period_by_dimension
from src.data.database import DataSource
from src.data.filters import FilterState
from src.utils.config import CUSTOMER_SEGMENTS, MIN_SAMPLE_FOR_COMPARISON
from src.utils.formatting import format_date_range, format_int, humanise_list


@dataclass
class ToolResult:
    """Output of one analytical tool."""

    tool: str
    title: str
    answer: str
    evidence: dict[str, Any] = field(default_factory=dict)
    table: pd.DataFrame | None = None
    caveats: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Tool:
    """A named analytical capability the assistant is allowed to call."""

    name: str
    description: str
    keywords: tuple[str, ...]
    runner: Callable[..., ToolResult]
    example: str


def _period_label(filters: FilterState) -> str:
    return format_date_range(filters.start_date, filters.end_date)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def get_kpi_summary(source: DataSource, filters: FilterState, **_: Any) -> ToolResult:
    """Headline KPIs for the selected population, with period-over-period movement."""
    kpis = compare_periods(source, filters, HEADLINE_KPIS)
    population = population_summary(source, filters)

    sentences = [
        f"Across {format_int(population['journeys'])} journeys between "
        f"{_period_label(filters)}, the network recorded "
        f"{kpis['on_time_performance'].formatted()} on-time performance and an average "
        f"satisfaction score of {kpis['avg_satisfaction'].formatted()}."
    ]
    movers = [k for k in kpis.values() if k.change_pct is not None and abs(k.change_pct) >= 2]
    if movers:
        movers.sort(key=lambda k: abs(k.change_pct or 0), reverse=True)
        described = [
            f"{m.label} {m.formatted_delta().replace(' vs previous period', '')}"
            for m in movers[:3]
        ]
        sentences.append("The largest movements were " + humanise_list(described) + ".")
    else:
        sentences.append("No headline metric moved by more than 2% against the previous period.")

    return ToolResult(
        tool="get_kpi_summary",
        title="Headline performance",
        answer=" ".join(sentences),
        evidence={
            "period": _period_label(filters),
            "population": {
                k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in population.items()
            },
            "kpis": [k.as_evidence() for k in kpis.values()],
        },
        table=pd.DataFrame(
            [
                {
                    "Metric": k.label,
                    "Current": k.formatted(),
                    "Previous": format_metric(k.previous, k.spec.unit),
                    "Change": k.formatted_delta().replace(" vs previous period", ""),
                }
                for k in kpis.values()
            ]
        ),
    )


def compare_periods_tool(source: DataSource, filters: FilterState, **_: Any) -> ToolResult:
    """What changed between the selected period and the one before it."""
    kpis = compare_periods(source, filters, tuple(METRICS.keys())[:8])
    previous_start, previous_end = filters.previous_period()

    changed = [k for k in kpis.values() if k.change_pct is not None]
    changed.sort(key=lambda k: abs(k.change_pct or 0), reverse=True)
    material = [k for k in changed if abs(k.change_pct or 0) >= 2][:4]

    if material:
        parts = [
            f"{k.label} moved from {format_metric(k.previous, k.spec.unit)} to "
            f"{k.formatted()} ({k.formatted_delta().replace(' vs previous period', '')})"
            for k in material
        ]
        answer = (
            f"Comparing {_period_label(filters)} with "
            f"{format_date_range(previous_start, previous_end)}: " + humanise_list(parts) + "."
        )
    else:
        answer = (
            f"No metric changed by more than 2% between {_period_label(filters)} and the "
            f"preceding period ({format_date_range(previous_start, previous_end)})."
        )

    return ToolResult(
        tool="compare_periods",
        title="Period-over-period comparison",
        answer=answer,
        evidence={
            "current_period": _period_label(filters),
            "previous_period": format_date_range(previous_start, previous_end),
            "kpis": [k.as_evidence() for k in changed],
        },
        table=pd.DataFrame(
            [
                {
                    "Metric": k.label,
                    "Current": k.formatted(),
                    "Previous": format_metric(k.previous, k.spec.unit),
                    "Change": k.formatted_delta().replace(" vs previous period", ""),
                }
                for k in changed
            ]
        ),
    )


def compare_segments(
    source: DataSource,
    filters: FilterState,
    segment_a: str | None = None,
    segment_b: str | None = None,
    **_: Any,
) -> ToolResult:
    """Compare two customer segments across the standard metric set."""
    segment_a = segment_a or CUSTOMER_SEGMENTS[0]
    segment_b = segment_b or CUSTOMER_SEGMENTS[1]
    frame = compare_segment_metrics(source, filters, segment_a, segment_b)

    if frame.empty:
        return ToolResult(
            tool="compare_segments",
            title="Segment comparison",
            answer=f"No journeys were found for {segment_a} or {segment_b} in this period.",
            evidence={"segment_a": segment_a, "segment_b": segment_b},
        )

    differences = []
    for _, row in frame.iterrows():
        if row["Difference"] is None or pd.isna(row["Difference"]):
            continue
        unit = row["unit"]
        differences.append(
            f"{row['Metric']} is {format_metric(row[segment_a], unit)} for {segment_a} "
            f"versus {format_metric(row[segment_b], unit)} for {segment_b}"
        )

    display = frame[["Metric", segment_a, segment_b]].copy()
    for column in (segment_a, segment_b):
        display[column] = [
            format_metric(value, unit)
            for value, unit in zip(frame[column], frame["unit"], strict=False)
        ]

    return ToolResult(
        tool="compare_segments",
        title=f"{segment_a} versus {segment_b}",
        answer=humanise_list(differences[:4]) + "." if differences else "No comparable metrics.",
        evidence={
            "segment_a": segment_a,
            "segment_b": segment_b,
            "period": _period_label(filters),
            "metrics": frame.drop(columns=["Difference"]).to_dict(orient="records"),
        },
        table=display,
    )


def _analyse_dimension(
    source: DataSource,
    filters: FilterState,
    dimension: str,
    label: str,
    focus_value: str | None,
    metric: str,
) -> ToolResult:
    """Shared implementation for the region and mode analysis tools."""
    breakdown = metrics_by_dimension(source, filters, dimension)
    movement = period_over_period_by_dimension(
        source, filters, dimension, metric, min_journeys=MIN_SAMPLE_FOR_COMPARISON
    )

    if breakdown.empty:
        return ToolResult(
            tool=f"analyse_{label}",
            title=f"{label.title()} analysis",
            answer="No journeys match the current filters.",
            evidence={},
        )

    spec = METRICS[metric]
    sentences: list[str] = []

    if focus_value:
        row = breakdown[breakdown["dimension_value"] == focus_value]
        if not row.empty:
            row = row.iloc[0]
            sentences.append(
                f"{focus_value} handled {format_int(row['total_journeys'])} journeys with "
                f"{format_metric(row['on_time_performance'], 'percent')} on-time performance, "
                f"an average delay of {format_metric(row['avg_delay_minutes'], 'minutes')} and "
                f"a satisfaction score of {format_metric(row['avg_satisfaction'], 'score')}."
            )
            change = movement[movement["dimension_value"] == focus_value]
            if not change.empty and pd.notna(change.iloc[0]["absolute_change"]):
                delta = float(change.iloc[0]["absolute_change"])
                sentences.append(
                    f"{spec.label} changed by {delta:+.2f} against the previous period."
                )

    if not movement.empty:
        usable = movement.dropna(subset=["absolute_change"])
        if not usable.empty:
            worst = usable.iloc[
                usable["absolute_change"].map(lambda v: -v if spec.lower_is_better else v).argmin()
            ]
            best = usable.iloc[
                usable["absolute_change"].map(lambda v: -v if spec.lower_is_better else v).argmax()
            ]
            sentences.append(
                f"The largest deterioration in {spec.label.lower()} was "
                f"{worst['dimension_value']} ({float(worst['absolute_change']):+.2f}), and the "
                f"largest improvement was {best['dimension_value']} "
                f"({float(best['absolute_change']):+.2f})."
            )

    display = breakdown.rename(
        columns={
            "dimension_value": label.title(),
            "total_journeys": "Journeys",
            "avg_satisfaction": "Satisfaction",
            "on_time_performance": "On-Time %",
            "avg_delay_minutes": "Avg Delay",
            "complaint_rate": "Complaint %",
        }
    ).round(2)

    return ToolResult(
        tool=f"analyse_{label}",
        title=f"{label.title()} analysis",
        answer=" ".join(sentences) or "No material differences were found.",
        evidence={
            "dimension": dimension,
            "focus": focus_value,
            "period": _period_label(filters),
            "breakdown": breakdown.round(3).to_dict(orient="records"),
            "movement": movement.round(3).to_dict(orient="records"),
        },
        table=display,
    )


def analyse_region(
    source: DataSource, filters: FilterState, region: str | None = None, **_: Any
) -> ToolResult:
    """Performance by region, optionally focused on one region."""
    return _analyse_dimension(source, filters, "region", "region", region, "on_time_performance")


def analyse_mode(
    source: DataSource, filters: FilterState, transport_mode: str | None = None, **_: Any
) -> ToolResult:
    """Performance by transport mode, optionally focused on one mode."""
    return _analyse_dimension(
        source, filters, "transport_mode", "mode", transport_mode, "on_time_performance"
    )


def find_anomalies(
    source: DataSource, filters: FilterState, dimension: str | None = "region", **_: Any
) -> ToolResult:
    """Statistically unusual days within the selected period."""
    anomalies = detect_anomalies(source, filters, dimension=dimension, adverse_only=True, limit=10)
    if not anomalies:
        return ToolResult(
            tool="find_anomalies",
            title="Anomaly scan",
            answer=(
                "No adverse anomalies were detected in this period: every daily value sat "
                "inside its trailing 28-day expected range."
            ),
            evidence={"total_detected": 0, "period": _period_label(filters)},
        )

    top = anomalies[:3]
    answer = " ".join(f"{a.describe()} {a.context()}" for a in top)
    return ToolResult(
        tool="find_anomalies",
        title="Anomaly scan",
        answer=answer,
        evidence=summarise_anomalies(anomalies),
        table=anomalies_to_frame(anomalies),
    )


def analyse_complaints(source: DataSource, filters: FilterState, **_: Any) -> ToolResult:
    """Where complaints concentrate, and how disruption affects them."""
    by_mode = metrics_by_dimension(
        source, filters, "transport_mode", keys=("total_journeys", "complaint_rate")
    )
    by_category = metrics_by_dimension(
        source, filters, "complaint_category", keys=("total_journeys",)
    )

    if by_mode.empty:
        return ToolResult(
            tool="analyse_complaints",
            title="Complaint analysis",
            answer="No journeys match the current filters.",
            evidence={},
        )

    worst = by_mode.loc[by_mode["complaint_rate"].idxmax()]
    sentences = [
        f"{worst['dimension_value']} has the highest complaint rate at "
        f"{format_metric(worst['complaint_rate'], 'percent')} across "
        f"{format_int(worst['total_journeys'])} journeys."
    ]
    if not by_category.empty:
        top_category = by_category.iloc[0]
        total = by_category["total_journeys"].sum()
        share = float(top_category["total_journeys"]) / total * 100 if total else 0
        sentences.append(
            f"The most common complaint reason is '{top_category['dimension_value']}', "
            f"accounting for {share:.0f}% of complaints."
        )

    return ToolResult(
        tool="analyse_complaints",
        title="Complaint analysis",
        answer=" ".join(sentences),
        evidence={
            "period": _period_label(filters),
            "by_mode": by_mode.round(3).to_dict(orient="records"),
            "by_category": by_category.to_dict(orient="records"),
        },
        table=by_mode.rename(
            columns={
                "dimension_value": "Transport Mode",
                "total_journeys": "Journeys",
                "complaint_rate": "Complaint %",
            }
        ).round(2),
    )


def analyse_satisfaction(source: DataSource, filters: FilterState, **_: Any) -> ToolResult:
    """Which customer segments moved most on satisfaction."""
    movement = period_over_period_by_dimension(
        source,
        filters,
        "customer_segment",
        "avg_satisfaction",
        min_journeys=MIN_SAMPLE_FOR_COMPARISON,
    )
    breakdown = metrics_by_dimension(source, filters, "customer_segment")

    if movement.empty or movement["absolute_change"].isna().all():
        answer = "There is not enough history in this period to compare satisfaction."
    else:
        usable = movement.dropna(subset=["absolute_change"])
        worst = usable.loc[usable["absolute_change"].idxmin()]
        best = usable.loc[usable["absolute_change"].idxmax()]
        answer = (
            f"{worst['dimension_value']} recorded the largest decline in satisfaction, from "
            f"{float(worst['previous_value']):.2f} to {float(worst['current_value']):.2f} "
            f"({float(worst['percent_change']):+.1f}%). "
            f"{best['dimension_value']} improved most, from "
            f"{float(best['previous_value']):.2f} to {float(best['current_value']):.2f} "
            f"({float(best['percent_change']):+.1f}%)."
        )

    return ToolResult(
        tool="analyse_satisfaction",
        title="Satisfaction by segment",
        answer=answer,
        evidence={
            "period": _period_label(filters),
            "movement": movement.round(3).to_dict(orient="records"),
            "current_levels": breakdown.round(3).to_dict(orient="records"),
        },
        table=breakdown.rename(
            columns={
                "dimension_value": "Customer Segment",
                "total_journeys": "Journeys",
                "avg_satisfaction": "Satisfaction",
                "on_time_performance": "On-Time %",
                "avg_delay_minutes": "Avg Delay",
                "complaint_rate": "Complaint %",
            }
        ).round(2),
    )


def top_issues(source: DataSource, filters: FilterState, **_: Any) -> ToolResult:
    """The findings most worth investigating, from the deterministic insight engine."""
    findings = generate_insights(source, filters, limit=10)
    adverse = [f for f in findings if f.direction == "bad"][:3]
    chosen = adverse or findings[:3]

    if not chosen:
        return ToolResult(
            tool="top_issues",
            title="Priority issues",
            answer="No material issues were detected in the selected population.",
            evidence={"insights": []},
        )

    answer = " ".join(f"{i + 1}. {item.headline} {item.detail}" for i, item in enumerate(chosen))
    return ToolResult(
        tool="top_issues",
        title="Priority issues",
        answer=answer,
        evidence={"insights": [item.to_dict() for item in chosen]},
        table=pd.DataFrame(
            [{"Category": i.category, "Finding": i.headline, "Detail": i.detail} for i in chosen]
        ),
    )


def assess_data_quality(source: DataSource, filters: FilterState, **_: Any) -> ToolResult:
    """Quality of the data behind the current numbers."""
    report = build_quality_report(source, filters)
    failing = report.failing_checks()
    answer = (
        f"The data quality score for this population is {report.score}/100 ({report.rating}). "
        f"{format_int(report.analysable_rows)} of {format_int(report.total_rows)} rows are "
        f"usable for analysis, and field completeness averages "
        f"{report.completeness_pct:.1f}%."
    )
    if failing:
        worst = max(failing, key=lambda c: c.failure_rate)
        answer += (
            f" The largest single issue is '{worst.label}', affecting "
            f"{format_int(worst.failed_rows)} rows ({worst.failure_rate:.2f}%)."
        )
    return ToolResult(
        tool="assess_data_quality",
        title="Data quality",
        answer=answer,
        evidence=report.to_evidence(),
        table=report.checks_frame(),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TOOLS: dict[str, Tool] = {
    "get_kpi_summary": Tool(
        name="get_kpi_summary",
        description="Headline KPIs and their movement against the previous period.",
        keywords=(
            "summar",
            "overview",
            "how are we",
            "performance",
            "current",
            "kpi",
            "headline",
            "overall",
        ),
        runner=get_kpi_summary,
        example="Summarise current service performance.",
    ),
    "compare_periods": Tool(
        name="compare_periods",
        description="Everything that changed against the previous equal-length period.",
        keywords=(
            "change",
            "changed",
            "this month",
            "versus last",
            "vs last",
            "compared with",
            "movement",
            "trend",
            "since",
        ),
        runner=compare_periods_tool,
        example="What changed significantly this month?",
    ),
    "compare_segments": Tool(
        name="compare_segments",
        description="Side-by-side comparison of two customer segments.",
        keywords=(
            "segment",
            "commuter",
            "traveller",
            "traveler",
            "digital-first",
            "compare customers",
            "versus",
            "vs",
        ),
        runner=compare_segments,
        example="Compare frequent commuters with occasional travellers.",
    ),
    "analyse_region": Tool(
        name="analyse_region",
        description="Performance by region, including the biggest movers.",
        keywords=(
            "region",
            "where",
            "area",
            "corridor",
            "north",
            "south",
            "east",
            "west",
            "central",
            "northwest",
            "location",
        ),
        runner=analyse_region,
        example="Where have delays increased most?",
    ),
    "analyse_mode": Tool(
        name="analyse_mode",
        description="Performance by transport mode.",
        keywords=("mode", "bus", "train", "metro", "ferry", "light rail", "service type"),
        runner=analyse_mode,
        example="Which transport mode has the highest complaint rate?",
    ),
    "find_anomalies": Tool(
        name="find_anomalies",
        description="Days whose values fall outside their expected statistical range.",
        keywords=("anomal", "unusual", "spike", "outlier", "abnormal", "strange", "unexpected"),
        runner=find_anomalies,
        example="Were there any unusual days this period?",
    ),
    "analyse_complaints": Tool(
        name="analyse_complaints",
        description="Complaint rates by mode and the most common complaint reasons.",
        keywords=("complaint", "complain", "dissatisf", "issue reported", "feedback"),
        runner=analyse_complaints,
        example="Which transport mode has the highest complaint rate?",
    ),
    "analyse_satisfaction": Tool(
        name="analyse_satisfaction",
        description="Satisfaction levels and movement by customer segment.",
        keywords=("satisfaction", "satisfied", "csat", "score", "happy", "experience"),
        runner=analyse_satisfaction,
        example="Which customer segment experienced the largest decline in satisfaction?",
    ),
    "top_issues": Tool(
        name="top_issues",
        description="The highest-priority findings to investigate.",
        keywords=(
            "investigate",
            "issues",
            "problems",
            "worry",
            "attention",
            "three main",
            "what should i",
            "priorit",
            "concern",
        ),
        runner=top_issues,
        example="What are the three main issues I should investigate?",
    ),
    "assess_data_quality": Tool(
        name="assess_data_quality",
        description="Completeness, validity and usability of the underlying data.",
        keywords=("data quality", "missing", "trust", "reliable", "complete", "duplicate", "clean"),
        runner=assess_data_quality,
        example="How complete is the underlying data?",
    ),
}

SUPPORTED_QUESTIONS: tuple[str, ...] = tuple(tool.example for tool in TOOLS.values())
