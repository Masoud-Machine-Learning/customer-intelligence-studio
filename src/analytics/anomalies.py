"""Statistical anomaly detection over daily metric series.

Method
------
For every day (optionally within a dimension value such as a region) the detector
compares the observed metric against a *trailing* baseline: the mean and standard
deviation of the preceding ``ANOMALY_ROLLING_WINDOW`` days. The current day is excluded
from its own baseline, so a single extreme day cannot hide inside the statistic it is
being tested against.

Two deterministic methods are offered:

``rolling_z``
    Flags a day when ``|value - baseline_mean| / baseline_std`` exceeds the configured
    threshold. Sensitive to sustained shifts as well as single spikes.

``iqr``
    Flags a day outside ``[Q1 - 1.5 x IQR, Q3 + 1.5 x IQR]`` of the trailing window.
    More robust when the baseline itself contains outliers.

Nothing here is learned or probabilistic: the same data always produces the same
anomalies, and every reported figure is recomputable from the evidence dictionary.

Baseline history is fetched from *before* the selected period so that filtering to a
short recent window still has something to compare against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from src.analytics.kpis import GROUPING_COLUMNS, METRICS, format_metric, select_clause
from src.data.database import ANALYTICS_VIEW, DataSource
from src.data.filters import FilterState, build_where
from src.data.loader import run_query
from src.utils.config import (
    ANOMALY_MIN_HISTORY,
    ANOMALY_ROLLING_WINDOW,
    ANOMALY_Z_THRESHOLD,
    LOWER_IS_BETTER,
)

DETECTION_METHODS: tuple[str, ...] = ("rolling_z", "iqr")

# Metrics scanned by default, with the minimum absolute deviation worth reporting.
# Without these floors a very stable series produces large z-scores from trivial moves.
DEFAULT_SCAN_METRICS: dict[str, float] = {
    "avg_delay_minutes": 1.0,
    "total_journeys": 0.0,
    "complaint_rate": 1.5,
    "avg_satisfaction": 0.15,
    "on_time_performance": 3.0,
}

COMPANION_METRICS: tuple[str, ...] = (
    "total_journeys",
    "avg_delay_minutes",
    "on_time_performance",
    "avg_satisfaction",
    "complaint_rate",
    "disruption_rate",
)


@dataclass(frozen=True)
class Anomaly:
    """A single flagged day for one metric within one scope."""

    metric: str
    metric_label: str
    dimension: str
    dimension_value: str
    period: date
    actual: float
    expected: float
    lower_bound: float
    upper_bound: float
    z_score: float
    journeys: int
    method: str
    companions: dict[str, float] = field(default_factory=dict)
    companion_baselines: dict[str, float] = field(default_factory=dict)

    @property
    def scope(self) -> str:
        """Readable scope label, e.g. 'Region: West' or 'Network'."""
        if self.dimension == "network":
            return "Network"
        label = self.dimension.replace("_", " ").title()
        return f"{label}: {self.dimension_value}"

    @property
    def direction(self) -> str:
        return "above" if self.actual > self.expected else "below"

    @property
    def magnitude_pct(self) -> float | None:
        if self.expected == 0:
            return None
        return (self.actual - self.expected) / abs(self.expected) * 100.0

    @property
    def severity(self) -> str:
        magnitude = abs(self.z_score)
        if magnitude >= 4.0:
            return "High"
        if magnitude >= 3.0:
            return "Medium"
        return "Low"

    @property
    def is_adverse(self) -> bool:
        """Whether the deviation is in the unwanted direction for this metric."""
        higher_is_worse = self.metric in LOWER_IS_BETTER
        return (self.actual > self.expected) == higher_is_worse

    def expected_range(self) -> str:
        unit = METRICS[self.metric].unit
        return f"{format_metric(self.lower_bound, unit)} to {format_metric(self.upper_bound, unit)}"

    def describe(self) -> str:
        """Deterministic, evidence-based sentence describing the anomaly."""
        unit = METRICS[self.metric].unit
        scope = "the network" if self.dimension == "network" else self.scope
        verb = "reached" if self.direction == "above" else "fell to"
        return (
            f"{self.metric_label} for {scope} {verb} {format_metric(self.actual, unit)} on "
            f"{self.period:%d %b %Y}, compared with a recent baseline of "
            f"{format_metric(self.expected, unit)}."
        )

    def context(self) -> str:
        """Supporting evidence drawn from companion metrics on the same day."""
        notes: list[str] = []
        for name in ("disruption_rate", "complaint_rate", "total_journeys"):
            if name == self.metric or name not in self.companions:
                continue
            actual = self.companions.get(name)
            baseline = self.companion_baselines.get(name)
            if actual is None or baseline is None or pd.isna(actual) or pd.isna(baseline):
                continue
            if baseline == 0:
                continue
            change = (actual - baseline) / abs(baseline) * 100.0
            if abs(change) < 25:
                continue
            unit = METRICS[name].unit
            notes.append(
                f"{METRICS[name].label} was {format_metric(actual, unit)} against a "
                f"baseline of {format_metric(baseline, unit)} ({change:+.0f}%)"
            )
        if not notes:
            return (
                f"No companion metric moved materially; the deviation is specific to "
                f"{self.metric_label.lower()}."
            )
        return "On the same day, " + "; ".join(notes) + "."

    def to_evidence(self) -> dict[str, Any]:
        """Machine-readable record handed to the AI assistant and the export."""
        return {
            "metric": self.metric,
            "metric_label": self.metric_label,
            "scope": self.scope,
            "date": self.period.isoformat(),
            "actual": round(float(self.actual), 3),
            "expected_baseline": round(float(self.expected), 3),
            "expected_range": [
                round(float(self.lower_bound), 3),
                round(float(self.upper_bound), 3),
            ],
            "z_score": round(float(self.z_score), 2),
            "severity": self.severity,
            "direction": self.direction,
            "adverse": self.is_adverse,
            "journeys": int(self.journeys),
            "method": self.method,
            "statement": self.describe(),
            "context": self.context(),
        }


def _daily_metric_frame(
    source: DataSource,
    filters: FilterState,
    dimension: str | None,
    lookback_days: int,
    metrics: Sequence[str],
) -> pd.DataFrame:
    """Fetch one row per day (and dimension value) with every scanned metric.

    A single query covers the selected window plus the baseline history that precedes it.
    """
    extended = filters.with_dates(
        filters.start_date - timedelta(days=lookback_days), filters.end_date
    )
    where, params = build_where(extended)

    if dimension is None:
        group_expression = "'Network' AS dimension_value"
        group_by = "GROUP BY 1"
    else:
        if dimension not in GROUPING_COLUMNS:
            raise ValueError(f"Unsupported grouping column: {dimension}")
        group_expression = f"{dimension} AS dimension_value"
        group_by = "GROUP BY 1, 2"

    sql = f"""
    SELECT
        CAST(date AS DATE) AS period,
        {group_expression},
        {select_clause(metrics)}
    FROM {ANALYTICS_VIEW}
    {where}
    {group_by}
    ORDER BY 1
    """
    frame = run_query(source, sql, params)
    if frame.empty:
        return frame
    frame["period"] = pd.to_datetime(frame["period"])
    return frame.dropna(subset=["dimension_value"])


def _baseline_bounds(
    series: pd.Series, method: str, window: int, threshold: float
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Trailing baseline, bounds and z-scores for one series.

    ``shift(1)`` excludes the day under test from its own baseline.
    """
    trailing = series.shift(1).rolling(window=window, min_periods=ANOMALY_MIN_HISTORY)

    if method == "iqr":
        q1 = trailing.quantile(0.25)
        q3 = trailing.quantile(0.75)
        iqr = q3 - q1
        centre = trailing.median()
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        spread = (iqr / 1.349).replace(0, np.nan)
    else:
        centre = trailing.mean()
        spread = trailing.std(ddof=0).replace(0, np.nan)
        lower = centre - threshold * spread
        upper = centre + threshold * spread

    z_scores = (series - centre) / spread
    return centre, lower, upper, z_scores


def detect_anomalies(
    source: DataSource,
    filters: FilterState,
    dimension: str | None = None,
    metrics: Iterable[str] | None = None,
    method: str = "rolling_z",
    window: int = ANOMALY_ROLLING_WINDOW,
    threshold: float = ANOMALY_Z_THRESHOLD,
    min_journeys_per_day: int = 25,
    adverse_only: bool = False,
    limit: int | None = None,
) -> list[Anomaly]:
    """Scan daily series for statistically unusual days.

    ``dimension`` of ``None`` scans the whole filtered network; otherwise each value of
    that dimension is scanned independently against its own baseline.
    """
    if method not in DETECTION_METHODS:
        raise ValueError(f"Unsupported detection method: {method}")

    scan = {m: DEFAULT_SCAN_METRICS.get(m, 0.0) for m in (metrics or DEFAULT_SCAN_METRICS)}
    needed = tuple(dict.fromkeys([*scan.keys(), *COMPANION_METRICS]))

    frame = _daily_metric_frame(source, filters, dimension, window + 7, needed)
    if frame.empty:
        return []

    results: list[Anomaly] = []
    for value, group in frame.groupby("dimension_value", sort=True):
        group = group.sort_values("period").reset_index(drop=True)
        if len(group) < ANOMALY_MIN_HISTORY + 1:
            continue

        baselines: dict[str, pd.Series] = {}
        for name in needed:
            baselines[name] = (
                group[name].shift(1).rolling(window=window, min_periods=ANOMALY_MIN_HISTORY).mean()
            )

        for metric_name, floor in scan.items():
            centre, lower, upper, z_scores = _baseline_bounds(
                group[metric_name], method, window, threshold
            )
            for index, row in group.iterrows():
                period = row["period"].date()
                if period < filters.start_date or period > filters.end_date:
                    continue
                actual = row[metric_name]
                expected = centre.iloc[index]
                z_value = z_scores.iloc[index]
                if pd.isna(actual) or pd.isna(expected) or pd.isna(z_value):
                    continue
                if row["total_journeys"] < min_journeys_per_day:
                    continue
                if abs(actual - expected) < floor:
                    continue

                if method == "iqr":
                    flagged = actual < lower.iloc[index] or actual > upper.iloc[index]
                else:
                    flagged = abs(z_value) >= threshold
                if not flagged:
                    continue

                anomaly = Anomaly(
                    metric=metric_name,
                    metric_label=METRICS[metric_name].label,
                    dimension=dimension or "network",
                    dimension_value=str(value),
                    period=period,
                    actual=float(actual),
                    expected=float(expected),
                    lower_bound=float(lower.iloc[index]),
                    upper_bound=float(upper.iloc[index]),
                    z_score=float(z_value),
                    journeys=int(row["total_journeys"]),
                    method=method,
                    companions={
                        name: (None if pd.isna(row[name]) else float(row[name]))
                        for name in COMPANION_METRICS
                        if name in group.columns
                    },
                    companion_baselines={
                        name: (
                            None
                            if pd.isna(baselines[name].iloc[index])
                            else float(baselines[name].iloc[index])
                        )
                        for name in COMPANION_METRICS
                        if name in baselines
                    },
                )
                if adverse_only and not anomaly.is_adverse:
                    continue
                results.append(anomaly)

    results.sort(key=lambda a: (abs(a.z_score), a.period), reverse=True)
    return results[:limit] if limit else results


def anomalies_to_frame(anomalies: Sequence[Anomaly]) -> pd.DataFrame:
    """Tabular presentation of detected anomalies."""
    if not anomalies:
        return pd.DataFrame(
            columns=[
                "Date",
                "Scope",
                "Metric",
                "Actual",
                "Expected Range",
                "Magnitude",
                "Severity",
                "Journeys",
            ]
        )
    rows = []
    for anomaly in anomalies:
        unit = METRICS[anomaly.metric].unit
        magnitude = anomaly.magnitude_pct
        rows.append(
            {
                "Date": anomaly.period,
                "Scope": anomaly.scope,
                "Metric": anomaly.metric_label,
                "Actual": format_metric(anomaly.actual, unit),
                "Expected Range": anomaly.expected_range(),
                "Magnitude": "-" if magnitude is None else f"{magnitude:+.0f}%",
                "Z-Score": round(anomaly.z_score, 2),
                "Severity": anomaly.severity,
                "Journeys": anomaly.journeys,
            }
        )
    return pd.DataFrame(rows)


def summarise_anomalies(anomalies: Sequence[Anomaly], top_n: int = 5) -> dict[str, Any]:
    """Compact summary used as grounding evidence for the AI assistant."""
    adverse = [a for a in anomalies if a.is_adverse]
    return {
        "total_detected": len(anomalies),
        "adverse_detected": len(adverse),
        "most_severe": [a.to_evidence() for a in adverse[:top_n]],
        "scopes_affected": sorted({a.scope for a in adverse}),
        "date_range": (
            [
                min(a.period for a in anomalies).isoformat(),
                max(a.period for a in anomalies).isoformat(),
            ]
            if anomalies
            else []
        ),
    }


def anomaly_series(
    source: DataSource,
    filters: FilterState,
    metric: str,
    dimension: str | None = None,
    dimension_value: str | None = None,
    method: str = "rolling_z",
    window: int = ANOMALY_ROLLING_WINDOW,
    threshold: float = ANOMALY_Z_THRESHOLD,
) -> pd.DataFrame:
    """Daily series with its expected band, ready to plot.

    Returns columns ``period``, ``value``, ``expected``, ``lower``, ``upper`` and
    ``flagged`` for the selected scope.
    """
    needed = tuple(dict.fromkeys([metric, *COMPANION_METRICS]))
    frame = _daily_metric_frame(source, filters, dimension, window + 7, needed)
    if frame.empty:
        return pd.DataFrame(columns=["period", "value", "expected", "lower", "upper", "flagged"])

    if dimension is not None and dimension_value is not None:
        frame = frame[frame["dimension_value"] == dimension_value]
    if frame.empty:
        return pd.DataFrame(columns=["period", "value", "expected", "lower", "upper", "flagged"])

    group = frame.sort_values("period").reset_index(drop=True)
    centre, lower, upper, z_scores = _baseline_bounds(group[metric], method, window, threshold)

    if method == "iqr":
        flagged = (group[metric] < lower) | (group[metric] > upper)
    else:
        flagged = z_scores.abs() >= threshold

    result = pd.DataFrame(
        {
            "period": group["period"],
            "value": group[metric],
            "expected": centre,
            "lower": lower,
            "upper": upper,
            "flagged": flagged.fillna(False),
        }
    )
    # Only show the selected window; the extra history existed to build the baseline.
    mask = (result["period"].dt.date >= filters.start_date) & (
        result["period"].dt.date <= filters.end_date
    )
    return result[mask].reset_index(drop=True)
