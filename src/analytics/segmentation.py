"""Customer segmentation - rule-based and clustered.

Two complementary approaches are offered, because they answer different questions.

**Behavioural rules** produce segments an analyst can defend in a meeting: the
definition is a handful of thresholds, every customer's assignment can be explained in
one sentence, and the segments are stable when the data changes.

**K-means clustering** is offered as a discovery tool. Features are standardised before
clustering (they are on wildly different scales), the number of clusters is chosen by
the user with a silhouette score and an inertia curve to guide the choice, and clusters
are labelled afterwards from their centroids so the output is readable rather than
"Cluster 3".

Both operate on customer-level features aggregated in SQL, not on raw journeys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.analytics.kpis import METRICS, select_clause
from src.data.database import ANALYTICS_VIEW, DataSource
from src.data.filters import FilterState, build_where
from src.data.loader import run_query

CLUSTER_RANDOM_STATE = 42
SILHOUETTE_SAMPLE = 4_000

# Features used for clustering, with the label shown in the UI.
CLUSTER_FEATURES: dict[str, str] = {
    "journeys_per_week": "Journeys per week",
    "peak_share": "Share of travel in peak (%)",
    "digital_share": "Digital interaction rate (%)",
    "avg_satisfaction": "Average satisfaction (1-5)",
    "complaint_rate": "Complaint rate (%)",
    "avg_delay_minutes": "Average delay (min)",
    "modes_used": "Distinct modes used",
}

# Rule thresholds, kept together so the page can display the definition verbatim.
RULE_THRESHOLDS: dict[str, float] = {
    "frequent_journeys_per_week": 3.0,
    "frequent_peak_share": 45.0,
    "high_need_accessibility_share": 20.0,
    "high_need_complaint_rate": 12.0,
    "digital_first_share": 70.0,
    "regular_journeys_per_week": 1.2,
    "regular_peak_share": 30.0,
}

RULE_DEFINITIONS: tuple[tuple[str, str], ...] = (
    (
        "High-Service-Need Customers",
        f"Accessibility services used on at least "
        f"{RULE_THRESHOLDS['high_need_accessibility_share']:.0f}% of journeys, or a "
        f"complaint rate of at least {RULE_THRESHOLDS['high_need_complaint_rate']:.0f}%.",
    ),
    (
        "Frequent Commuters",
        f"At least {RULE_THRESHOLDS['frequent_journeys_per_week']:.0f} journeys per week "
        f"with at least {RULE_THRESHOLDS['frequent_peak_share']:.0f}% of travel in peak "
        "periods.",
    ),
    (
        "Digital-First Customers",
        f"A digital interaction on at least {RULE_THRESHOLDS['digital_first_share']:.0f}% "
        "of journeys.",
    ),
    (
        "Off-Peak Regulars",
        f"At least {RULE_THRESHOLDS['regular_journeys_per_week']:.1f} journeys per week "
        f"with under {RULE_THRESHOLDS['regular_peak_share']:.0f}% of travel in peak "
        "periods.",
    ),
    ("Occasional Travellers", "Everyone who does not meet the rules above."),
)


@dataclass
class SegmentationResult:
    """Customer-level features plus their rule and (optionally) cluster assignment."""

    customers: pd.DataFrame
    feature_columns: tuple[str, ...]
    method: str
    n_clusters: int | None = None
    silhouette: float | None = None
    inertia_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    limitations: list[str] = field(default_factory=list)

    @property
    def label_column(self) -> str:
        return "cluster_label" if self.method == "kmeans" else "rule_segment"

    def profile(self) -> pd.DataFrame:
        """Mean feature values and size for each segment."""
        if self.customers.empty:
            return pd.DataFrame()
        grouped = self.customers.groupby(self.label_column, sort=False)
        profile = grouped[list(self.feature_columns)].mean().round(2)
        profile.insert(0, "customers", grouped.size())
        profile["share_pct"] = (profile["customers"] / profile["customers"].sum() * 100).round(1)
        return profile.reset_index().rename(columns={self.label_column: "segment"})

    def to_evidence(self) -> dict[str, Any]:
        profile = self.profile()
        return {
            "method": self.method,
            "n_customers": int(len(self.customers)),
            "n_segments": int(profile.shape[0]) if not profile.empty else 0,
            "silhouette": None if self.silhouette is None else round(self.silhouette, 3),
            "segments": profile.to_dict(orient="records") if not profile.empty else [],
        }


def customer_features(
    source: DataSource,
    filters: FilterState | None,
    min_journeys: int = 3,
) -> pd.DataFrame:
    """Aggregate journeys into one row per customer.

    The aggregation is pure SQL: it collapses tens of thousands of journeys into a few
    thousand customer rows before any Python touches the data.
    """
    where, params = build_where(filters)
    sql = f"""
    SELECT
        customer_id,
        count(*)                                     AS journeys,
        count(DISTINCT date)                         AS active_days,
        count(DISTINCT transport_mode)               AS modes_used,
        count(DISTINCT region)                       AS regions_used,
        avg(delay_minutes)                           AS avg_delay_minutes,
        avg(customer_satisfaction)                   AS avg_satisfaction,
        100.0 * count(*) FILTER (WHERE complaint_flag)
            / nullif(count(*), 0)                    AS complaint_rate,
        100.0 * count(*) FILTER (WHERE is_peak)
            / nullif(count(*), 0)                    AS peak_share,
        100.0 * count(*) FILTER (WHERE digital_interaction)
            / nullif(count(*), 0)                    AS digital_share,
        100.0 * count(*) FILTER (WHERE accessibility_service_used)
            / nullif(count(*), 0)                    AS accessibility_share,
        100.0 * count(*) FILTER (WHERE on_time)
            / nullif(count(*) FILTER (WHERE on_time IS NOT NULL), 0) AS on_time_performance,
        min(date)                                    AS first_journey,
        max(date)                                    AS last_journey,
        any_value(customer_segment)                  AS assigned_segment,
        any_value(age_group)                         AS age_group,
        mode(channel)                                AS primary_channel,
        mode(transport_mode)                         AS primary_mode
    FROM {ANALYTICS_VIEW}
    {where}
    GROUP BY customer_id
    HAVING count(*) >= ? AND customer_id IS NOT NULL
    """
    frame = run_query(source, sql, [*params, min_journeys])
    if frame.empty:
        return frame

    # Journeys per week normalises for how long each customer was observed - a Pandas
    # step because it depends on two aggregates rather than on the rows themselves.
    span_days = (
        pd.to_datetime(frame["last_journey"]) - pd.to_datetime(frame["first_journey"])
    ).dt.days.clip(lower=6) + 1
    frame["journeys_per_week"] = (frame["journeys"] / span_days * 7).round(2)
    frame["avg_delay_minutes"] = frame["avg_delay_minutes"].round(2)
    frame["avg_satisfaction"] = frame["avg_satisfaction"].round(2)
    for column in (
        "complaint_rate",
        "peak_share",
        "digital_share",
        "accessibility_share",
        "on_time_performance",
    ):
        frame[column] = frame[column].round(1)
    return frame


def apply_rule_segments(features: pd.DataFrame) -> pd.DataFrame:
    """Assign behavioural segments from documented thresholds.

    Rules are evaluated in order and the first match wins, so a customer who is both
    high-need and frequent is reported as high-need: service need is the more
    operationally actionable label.
    """
    if features.empty:
        return features

    frame = features.copy()
    thresholds = RULE_THRESHOLDS

    high_need = (
        frame["accessibility_share"].fillna(0) >= thresholds["high_need_accessibility_share"]
    ) | (frame["complaint_rate"].fillna(0) >= thresholds["high_need_complaint_rate"])

    frequent = (frame["journeys_per_week"] >= thresholds["frequent_journeys_per_week"]) & (
        frame["peak_share"].fillna(0) >= thresholds["frequent_peak_share"]
    )

    digital = frame["digital_share"].fillna(0) >= thresholds["digital_first_share"]

    regular = (frame["journeys_per_week"] >= thresholds["regular_journeys_per_week"]) & (
        frame["peak_share"].fillna(0) < thresholds["regular_peak_share"]
    )

    frame["rule_segment"] = np.select(
        [high_need, frequent, digital, regular],
        [
            "High-Service-Need Customers",
            "Frequent Commuters",
            "Digital-First Customers",
            "Off-Peak Regulars",
        ],
        default="Occasional Travellers",
    )
    return frame


def _label_clusters(centroids: pd.DataFrame, feature_columns: Sequence[str]) -> dict[int, str]:
    """Derive a readable name for each cluster from its most distinctive features.

    The centroid is compared with the overall mean in standardised units; the two
    features furthest from average decide the label. The mapping is fixed, so the same
    centroids always yield the same names.
    """
    phrases = {
        "journeys_per_week": ("High-frequency", "Low-frequency"),
        "peak_share": ("peak-focused", "off-peak"),
        "digital_share": ("digitally engaged", "assisted-channel"),
        "avg_satisfaction": ("satisfied", "dissatisfied"),
        "complaint_rate": ("complaint-prone", "low-complaint"),
        "avg_delay_minutes": ("delay-affected", "reliable-service"),
        "modes_used": ("multi-modal", "single-mode"),
    }
    standardised = (centroids - centroids.mean()) / centroids.std(ddof=0).replace(0, np.nan)
    labels: dict[int, str] = {}
    used: set[str] = set()
    for cluster_id, row in standardised.iterrows():
        ranked = row.abs().sort_values(ascending=False)
        parts: list[str] = []
        for feature in ranked.index[:2]:
            if feature not in phrases or pd.isna(row[feature]):
                continue
            high, low = phrases[feature]
            parts.append(high if row[feature] > 0 else low)
        name = " ".join(parts).strip() or "Mixed profile"
        name = name[0].upper() + name[1:] + " customers"
        # Guarantee unique, stable names even when two clusters share a description.
        if name in used:
            name = f"{name} ({cluster_id + 1})"
        used.add(name)
        labels[int(cluster_id)] = name
    return labels


def cluster_customers(
    features: pd.DataFrame,
    n_clusters: int = 4,
    feature_columns: Sequence[str] | None = None,
) -> SegmentationResult:
    """Cluster customer features with standardised K-means."""
    columns = tuple(feature_columns or CLUSTER_FEATURES.keys())
    limitations = [
        "Clusters describe behaviour in the selected period only; they are not a "
        "permanent customer classification.",
        "K-means assumes roughly spherical, similarly sized groups, which behavioural "
        "data rarely satisfies exactly.",
        "Cluster names are generated from centroid positions, not from qualitative "
        "customer research.",
        "Customers below the minimum journey threshold are excluded, so the clusters "
        "under-represent very occasional travellers.",
    ]

    if features.empty or len(features) < n_clusters:
        return SegmentationResult(
            customers=features,
            feature_columns=columns,
            method="kmeans",
            n_clusters=n_clusters,
            limitations=limitations,
        )

    frame = features.copy()
    matrix = frame[list(columns)].astype(float)
    matrix = matrix.fillna(matrix.median())

    scaled = StandardScaler().fit_transform(matrix)
    model = KMeans(n_clusters=n_clusters, random_state=CLUSTER_RANDOM_STATE, n_init=10)
    frame["cluster"] = model.fit_predict(scaled)

    centroids = frame.groupby("cluster")[list(columns)].mean()
    labels = _label_clusters(centroids, columns)
    frame["cluster_label"] = frame["cluster"].map(labels)

    score: float | None = None
    if 1 < n_clusters < len(frame):
        sample_size = min(SILHOUETTE_SAMPLE, len(frame))
        rng = np.random.default_rng(CLUSTER_RANDOM_STATE)
        idx = rng.choice(len(frame), size=sample_size, replace=False)
        try:
            score = float(silhouette_score(scaled[idx], frame["cluster"].to_numpy()[idx]))
        except ValueError:
            score = None

    return SegmentationResult(
        customers=frame,
        feature_columns=columns,
        method="kmeans",
        n_clusters=n_clusters,
        silhouette=score,
        inertia_curve=inertia_curve(matrix),
        limitations=limitations,
    )


def inertia_curve(
    matrix: pd.DataFrame, k_values: Sequence[int] = (2, 3, 4, 5, 6, 7, 8)
) -> pd.DataFrame:
    """Inertia and silhouette across candidate cluster counts (the elbow chart)."""
    if matrix.empty or len(matrix) < max(k_values) + 1:
        return pd.DataFrame(columns=["k", "inertia", "silhouette"])

    scaled = StandardScaler().fit_transform(matrix.fillna(matrix.median()))
    rng = np.random.default_rng(CLUSTER_RANDOM_STATE)
    sample_size = min(SILHOUETTE_SAMPLE, len(scaled))
    idx = rng.choice(len(scaled), size=sample_size, replace=False)

    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=CLUSTER_RANDOM_STATE, n_init=10)
        assignments = model.fit_predict(scaled)
        try:
            score = float(silhouette_score(scaled[idx], assignments[idx]))
        except ValueError:
            score = float("nan")
        rows.append({"k": k, "inertia": float(model.inertia_), "silhouette": round(score, 3)})
    return pd.DataFrame(rows)


def rule_segmentation(
    source: DataSource, filters: FilterState | None, min_journeys: int = 3
) -> SegmentationResult:
    """Convenience wrapper: features plus rule-based assignment."""
    features = apply_rule_segments(customer_features(source, filters, min_journeys))
    return SegmentationResult(
        customers=features,
        feature_columns=tuple(CLUSTER_FEATURES.keys()),
        method="rules",
        limitations=[
            "Thresholds are analyst-set business rules, not learned from the data.",
            "A customer near a threshold can move segment between periods.",
            "Customers below the minimum journey threshold are excluded.",
        ],
    )


def compare_segments(
    source: DataSource,
    filters: FilterState,
    segment_a: str,
    segment_b: str,
    keys: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Side-by-side metric comparison of two values of ``customer_segment``.

    Returns a tidy frame with one row per metric so the page can render it directly.
    """
    keys = tuple(
        keys
        or (
            "total_journeys",
            "active_customers",
            "avg_satisfaction",
            "on_time_performance",
            "avg_delay_minutes",
            "complaint_rate",
            "digital_rate",
            "repeat_rate",
        )
    )
    where, params = build_where(filters.without("customer_segments"))
    connector = "AND" if where else "WHERE"
    sql = f"""
    SELECT customer_segment, {select_clause(keys)}
    FROM {ANALYTICS_VIEW}
    {where}
    {connector} customer_segment IN (?, ?)
    GROUP BY 1
    """
    frame = run_query(source, sql, [*params, segment_a, segment_b])
    if frame.empty:
        return pd.DataFrame(columns=["Metric", segment_a, segment_b, "Difference"])

    indexed = frame.set_index("customer_segment")
    rows = []
    for key in keys:
        spec = METRICS[key]
        value_a = indexed[key].get(segment_a)
        value_b = indexed[key].get(segment_b)
        difference = (
            None
            if value_a is None or value_b is None or pd.isna(value_a) or pd.isna(value_b)
            else float(value_a) - float(value_b)
        )
        rows.append(
            {
                "metric": key,
                "Metric": spec.label,
                "unit": spec.unit,
                segment_a: None if value_a is None or pd.isna(value_a) else float(value_a),
                segment_b: None if value_b is None or pd.isna(value_b) else float(value_b),
                "Difference": difference,
            }
        )
    return pd.DataFrame(rows)
