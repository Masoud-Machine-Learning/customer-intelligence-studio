"""Data quality profiling.

Quality is measured against the *raw* view (``v_journeys``) rather than the analytics
view, because the point of the page is to show what the feed actually delivered. The
analytics view then removes the rows that fail hard validity rules, and the gap between
the two counts is reported explicitly so users can see how much data each KPI is based
on.

Checks are grouped into three families, which are also the weights behind the quality
score:

* **Completeness** - is the value present at all?
* **Validity**     - is the value possible?
* **Uniqueness**   - has the row been delivered more than once?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from src.data.database import ANALYTICS_VIEW, RAW_VIEW, DataSource
from src.data.filters import FilterState, build_where
from src.data.loader import run_query
from src.utils.config import (
    MAX_PLAUSIBLE_DELAY_MINUTES,
    MAX_PLAUSIBLE_DURATION_MINUTES,
    SATISFACTION_MAX,
    SATISFACTION_MIN,
    TRANSPORT_MODES,
)

CATEGORY_WEIGHTS: dict[str, float] = {
    "Completeness": 0.40,
    "Validity": 0.40,
    "Uniqueness": 0.20,
}

# Fields profiled for completeness, in the order shown on the page.
PROFILED_FIELDS: tuple[str, ...] = (
    "journey_id",
    "customer_id",
    "date",
    "region",
    "transport_mode",
    "service_line",
    "customer_segment",
    "channel",
    "journey_duration_minutes",
    "delay_minutes",
    "customer_satisfaction",
    "complaint_flag",
    "complaint_category",
    "resolution_time_hours",
    "fare_type",
    "feedback_text_category",
)

# Columns that are legitimately sparse: they only apply to a subset of journeys.
CONDITIONAL_FIELDS: dict[str, str] = {
    "complaint_category": "Only populated for journeys with a complaint.",
    "resolution_time_hours": "Only populated for journeys with a complaint.",
}


@dataclass(frozen=True)
class QualityCheck:
    """One rule evaluated across the filtered population."""

    key: str
    label: str
    category: str
    failed_rows: int
    total_rows: int
    description: str
    impact: str

    @property
    def failure_rate(self) -> float:
        if self.total_rows == 0:
            return 0.0
        return self.failed_rows / self.total_rows * 100.0

    @property
    def passed(self) -> bool:
        return self.failed_rows == 0

    @property
    def severity(self) -> str:
        rate = self.failure_rate
        if rate == 0:
            return "Pass"
        if rate < 0.5:
            return "Low"
        if rate < 2.0:
            return "Medium"
        return "High"

    def to_evidence(self) -> dict[str, Any]:
        return {
            "check": self.key,
            "label": self.label,
            "category": self.category,
            "failed_rows": self.failed_rows,
            "total_rows": self.total_rows,
            "failure_rate_pct": round(self.failure_rate, 3),
            "severity": self.severity,
        }


@dataclass
class QualityReport:
    """Everything the Data Quality page needs, computed in one pass."""

    total_rows: int
    analysable_rows: int
    checks: list[QualityCheck]
    field_completeness: pd.DataFrame
    latest_date: date | None
    freshness_days: int | None
    outliers: dict[str, int] = field(default_factory=dict)

    @property
    def excluded_rows(self) -> int:
        return max(0, self.total_rows - self.analysable_rows)

    @property
    def completeness_pct(self) -> float:
        """Mean completeness across profiled fields that are not conditionally empty."""
        if self.field_completeness.empty:
            return 100.0
        unconditional = self.field_completeness[~self.field_completeness["conditional"]]
        if unconditional.empty:
            return 100.0
        return float(unconditional["completeness_pct"].mean())

    def category_scores(self) -> dict[str, float]:
        """Score per check family: 100 minus the worst failure rate in that family."""
        scores: dict[str, float] = {}
        for category in CATEGORY_WEIGHTS:
            members = [c for c in self.checks if c.category == category]
            if not members:
                scores[category] = 100.0
                continue
            penalty = sum(c.failure_rate for c in members)
            scores[category] = max(0.0, 100.0 - penalty)
        return scores

    @property
    def score(self) -> float:
        """Weighted quality score on a 0-100 scale."""
        scores = self.category_scores()
        return round(sum(scores[cat] * weight for cat, weight in CATEGORY_WEIGHTS.items()), 1)

    @property
    def rating(self) -> str:
        score = self.score
        if score >= 97:
            return "Good"
        if score >= 92:
            return "Acceptable"
        return "Needs attention"

    def failing_checks(self) -> list[QualityCheck]:
        return [c for c in self.checks if not c.passed]

    def checks_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Check": c.label,
                    "Category": c.category,
                    "Failing Rows": c.failed_rows,
                    "Failure Rate": f"{c.failure_rate:.2f}%",
                    "Severity": c.severity,
                    "Why it matters": c.impact,
                }
                for c in self.checks
            ]
        )

    def to_evidence(self) -> dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "analysable_rows": self.analysable_rows,
            "excluded_rows": self.excluded_rows,
            "quality_score": self.score,
            "rating": self.rating,
            "completeness_pct": round(self.completeness_pct, 2),
            "freshness_days": self.freshness_days,
            "failing_checks": [c.to_evidence() for c in self.failing_checks()],
        }


def _check_definitions() -> list[dict[str, str]]:
    """SQL predicate for each rule, plus why a failure matters analytically."""
    modes = ", ".join(f"'{m}'" for m in TRANSPORT_MODES)
    return [
        {
            "key": "null_customer_id",
            "label": "Null customer ID",
            "category": "Completeness",
            "predicate": "customer_id IS NULL",
            "description": "Journeys delivered without a customer identifier.",
            "impact": "Understates active customers and breaks repeat-usage analysis.",
        },
        {
            "key": "missing_region",
            "label": "Missing region",
            "category": "Completeness",
            "predicate": "region IS NULL OR trim(region) = ''",
            "description": "Journeys with no region recorded.",
            "impact": "Rows vanish from regional breakdowns while still counting in totals.",
        },
        {
            "key": "missing_satisfaction",
            "label": "Missing satisfaction score",
            "category": "Completeness",
            "predicate": "customer_satisfaction IS NULL",
            "description": "Journeys with no satisfaction response.",
            "impact": "Satisfaction is a survey metric; a low response rate widens its error.",
        },
        {
            "key": "invalid_satisfaction",
            "label": "Satisfaction outside 1-5",
            "category": "Validity",
            "predicate": (
                f"customer_satisfaction IS NOT NULL AND "
                f"(customer_satisfaction < {SATISFACTION_MIN} "
                f"OR customer_satisfaction > {SATISFACTION_MAX})"
            ),
            "description": "Scores outside the defined survey scale.",
            "impact": "Would drag the mean score toward an impossible value.",
        },
        {
            "key": "negative_duration",
            "label": "Non-positive journey duration",
            "category": "Validity",
            "predicate": "journey_duration_minutes IS NOT NULL AND journey_duration_minutes <= 0",
            "description": "Journeys recorded as taking zero or negative time.",
            "impact": "Corrupts duration averages and any speed-derived measure.",
        },
        {
            "key": "implausible_duration",
            "label": "Implausible journey duration",
            "category": "Validity",
            "predicate": (f"journey_duration_minutes > {MAX_PLAUSIBLE_DURATION_MINUTES}"),
            "description": f"Durations above {MAX_PLAUSIBLE_DURATION_MINUTES:.0f} minutes.",
            "impact": "Usually an un-closed journey record rather than a real trip.",
        },
        {
            "key": "invalid_delay",
            "label": "Impossible delay value",
            "category": "Validity",
            "predicate": (
                f"delay_minutes IS NULL OR delay_minutes < 0 "
                f"OR delay_minutes > {MAX_PLAUSIBLE_DELAY_MINUTES}"
            ),
            "description": (
                f"Delays that are negative, missing or above "
                f"{MAX_PLAUSIBLE_DELAY_MINUTES:.0f} minutes (often a 999 sentinel)."
            ),
            "impact": "A single sentinel value can move on-time performance by whole points.",
        },
        {
            "key": "invalid_transport_mode",
            "label": "Unrecognised transport mode",
            "category": "Validity",
            "predicate": f"transport_mode IS NULL OR transport_mode NOT IN ({modes})",
            "description": "Mode values outside the reference list.",
            "impact": "Creates phantom categories and splits mode-level aggregates.",
        },
        {
            "key": "duplicate_journey_id",
            "label": "Duplicate journey ID",
            "category": "Uniqueness",
            "predicate": None,  # handled separately - needs a window function
            "description": "The same journey delivered more than once.",
            "impact": "Double counts demand and over-weights the duplicated experience.",
        },
    ]


def build_quality_report(source: DataSource, filters: FilterState | None = None) -> QualityReport:
    """Profile the raw feed for the filtered population."""
    where, params = build_where(filters)

    definitions = _check_definitions()
    row_expressions = [
        f"count(*) FILTER (WHERE {d['predicate']}) AS {d['key']}"
        for d in definitions
        if d["predicate"]
    ]
    completeness_expressions = [
        f"count({column}) AS present_{column}" for column in PROFILED_FIELDS
    ]

    sql = f"""
    SELECT
        count(*) AS total_rows,
        max(date) AS latest_date,
        count(*) - count(DISTINCT journey_id) AS duplicate_journey_id,
        {", ".join(row_expressions)},
        {", ".join(completeness_expressions)}
    FROM {RAW_VIEW}
    {where}
    """
    summary = run_query(source, sql, params)
    row = summary.iloc[0]
    total_rows = int(row["total_rows"] or 0)

    analysable = run_query(
        source,
        f"SELECT count(*) AS n FROM {ANALYTICS_VIEW} {where}",
        params,
    )
    analysable_rows = int(analysable.loc[0, "n"] or 0)

    checks: list[QualityCheck] = []
    for definition in definitions:
        failed = int(row[definition["key"]] or 0)
        checks.append(
            QualityCheck(
                key=definition["key"],
                label=definition["label"],
                category=definition["category"],
                failed_rows=failed,
                total_rows=total_rows,
                description=definition["description"],
                impact=definition["impact"],
            )
        )

    completeness_rows = []
    for column in PROFILED_FIELDS:
        present = int(row[f"present_{column}"] or 0)
        missing = total_rows - present
        completeness_rows.append(
            {
                "field": column,
                "present": present,
                "missing": missing,
                "completeness_pct": 100.0 if total_rows == 0 else present / total_rows * 100.0,
                "conditional": column in CONDITIONAL_FIELDS,
                "note": CONDITIONAL_FIELDS.get(column, ""),
            }
        )
    field_completeness = pd.DataFrame(completeness_rows)

    latest = row["latest_date"]
    latest_date = pd.Timestamp(latest).date() if pd.notna(latest) else None
    freshness_days = (date.today() - latest_date).days if latest_date else None

    outliers = _detect_value_outliers(source, where, params)

    return QualityReport(
        total_rows=total_rows,
        analysable_rows=analysable_rows,
        checks=checks,
        field_completeness=field_completeness,
        latest_date=latest_date,
        freshness_days=freshness_days,
        outliers=outliers,
    )


def _detect_value_outliers(source: DataSource, where: str, params: list[Any]) -> dict[str, int]:
    """Count statistical outliers (beyond 3x IQR) for the main numeric fields.

    These are flagged for review rather than excluded: an extreme but possible delay is
    real operational information, unlike an impossible one.
    """
    sql = f"""
    WITH bounds AS (
        SELECT
            quantile_cont(delay_minutes, 0.25) AS delay_q1,
            quantile_cont(delay_minutes, 0.75) AS delay_q3,
            quantile_cont(journey_duration_minutes, 0.25) AS duration_q1,
            quantile_cont(journey_duration_minutes, 0.75) AS duration_q3
        FROM {RAW_VIEW}
        {where}
    )
    SELECT
        count(*) FILTER (
            WHERE delay_minutes > b.delay_q3 + 3 * (b.delay_q3 - b.delay_q1)
        ) AS delay_outliers,
        count(*) FILTER (
            WHERE journey_duration_minutes > b.duration_q3 + 3 * (b.duration_q3 - b.duration_q1)
               OR journey_duration_minutes < b.duration_q1 - 3 * (b.duration_q3 - b.duration_q1)
        ) AS duration_outliers
    FROM {RAW_VIEW} j, bounds b
    {where}
    """
    frame = run_query(source, sql, [*params, *params])
    if frame.empty:
        return {}
    return {
        "delay_minutes": int(frame.loc[0, "delay_outliers"] or 0),
        "journey_duration_minutes": int(frame.loc[0, "duration_outliers"] or 0),
    }
