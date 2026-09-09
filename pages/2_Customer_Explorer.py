"""Customer Explorer - who travels, how often, and how their experience differs."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analytics.kpis import format_metric, metrics_by_dimension, population_summary
from src.analytics.segmentation import compare_segments, customer_features
from src.ui import charts, components as ui
from src.ui.filters import render_filters
from src.ui.state import active_source
from src.utils.config import CUSTOMER_SEGMENTS

ui.configure_page("Customer Explorer")

source = active_source()
filters = render_filters(source, key_prefix="customer")

ui.page_header(
    "Customer Explorer",
    "Segment composition, engagement and experience for the selected population.",
)

summary = population_summary(source, filters)
ui.population_banner(summary, filters.describe())
if summary["journeys"] == 0:
    ui.footer()
    st.stop()

# ---------------------------------------------------------------------------
# Segment composition and experience
# ---------------------------------------------------------------------------

by_segment = metrics_by_dimension(
    source,
    filters,
    "customer_segment",
    keys=(
        "total_journeys",
        "active_customers",
        "avg_satisfaction",
        "on_time_performance",
        "avg_delay_minutes",
        "complaint_rate",
        "digital_rate",
    ),
)

ui.section("Segment composition")
left, right = st.columns([1, 1.4], gap="large")

with left:
    st.plotly_chart(
        charts.donut_chart(
            by_segment,
            labels="dimension_value",
            values="total_journeys",
            title="Share of journeys by segment",
            height=290,
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )

with right:
    display = by_segment.rename(columns={"dimension_value": "Segment"}).copy()
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Segment": st.column_config.TextColumn(width="medium"),
            "total_journeys": st.column_config.NumberColumn("Journeys", format="%d"),
            "active_customers": st.column_config.NumberColumn("Customers", format="%d"),
            "avg_satisfaction": st.column_config.NumberColumn("Satisfaction", format="%.2f"),
            "on_time_performance": st.column_config.NumberColumn("On-time %", format="%.1f"),
            "avg_delay_minutes": st.column_config.NumberColumn("Avg delay", format="%.2f"),
            "complaint_rate": st.column_config.NumberColumn("Complaint %", format="%.2f"),
            "digital_rate": st.column_config.NumberColumn("Digital %", format="%.1f"),
        },
    )

ui.section("Experience by segment")
chart_left, chart_right = st.columns(2, gap="large")
with chart_left:
    st.plotly_chart(
        charts.bar_chart(
            by_segment.sort_values("avg_satisfaction"),
            x="dimension_value",
            y="avg_satisfaction",
            title="Average satisfaction",
            y_title="Score (1-5)",
            horizontal=True,
            value_format=",.2f",
            height=260,
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )
with chart_right:
    st.plotly_chart(
        charts.bar_chart(
            by_segment.sort_values("complaint_rate", ascending=False),
            x="dimension_value",
            y="complaint_rate",
            title="Complaint rate",
            y_title="% of journeys",
            horizontal=True,
            value_format=",.2f",
            height=260,
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )

# ---------------------------------------------------------------------------
# Channel and mode preference
# ---------------------------------------------------------------------------

ui.section("Channel and mode preference")
pref_left, pref_right = st.columns(2, gap="large")

with pref_left:
    by_channel = metrics_by_dimension(
        source, filters, "channel", keys=("total_journeys", "avg_satisfaction")
    )
    st.plotly_chart(
        charts.bar_chart(
            by_channel.sort_values("total_journeys", ascending=False),
            x="dimension_value",
            y="total_journeys",
            title="Journeys by channel",
            y_title="Journeys",
            value_format=",.0f",
            height=270,
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )

with pref_right:
    by_purpose = metrics_by_dimension(
        source, filters, "journey_purpose", keys=("total_journeys", "avg_satisfaction")
    )
    st.plotly_chart(
        charts.bar_chart(
            by_purpose.sort_values("total_journeys", ascending=False),
            x="dimension_value",
            y="total_journeys",
            title="Journeys by purpose",
            y_title="Journeys",
            value_format=",.0f",
            height=270,
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )

# ---------------------------------------------------------------------------
# Segment comparison
# ---------------------------------------------------------------------------

ui.section("Compare two segments", "Select any two segments to see where they differ.")

available = list(by_segment["dimension_value"]) or list(CUSTOMER_SEGMENTS)
col_a, col_b = st.columns(2)
with col_a:
    segment_a = st.selectbox("Segment A", available, index=0, key="cmp_a")
with col_b:
    default_b = 1 if len(available) > 1 else 0
    segment_b = st.selectbox("Segment B", available, index=default_b, key="cmp_b")

if segment_a == segment_b:
    st.info("Choose two different segments to compare.")
else:
    comparison = compare_segments(source, filters, segment_a, segment_b)
    if comparison.empty:
        ui.empty_state("No overlapping data for these segments.")
    else:
        rendered = pd.DataFrame(
            {
                "Metric": comparison["Metric"],
                segment_a: [
                    format_metric(v, u)
                    for v, u in zip(comparison[segment_a], comparison["unit"], strict=False)
                ],
                segment_b: [
                    format_metric(v, u)
                    for v, u in zip(comparison[segment_b], comparison["unit"], strict=False)
                ],
                "Difference (A - B)": [
                    "-" if pd.isna(d) else f"{d:+,.2f}" for d in comparison["Difference"]
                ],
            }
        )
        st.dataframe(rendered, use_container_width=True, hide_index=True)

        chart_frame = comparison[
            comparison["metric"].isin(
                [
                    "avg_satisfaction",
                    "on_time_performance",
                    "avg_delay_minutes",
                    "complaint_rate",
                    "digital_rate",
                ]
            )
        ]
        st.plotly_chart(
            charts.grouped_bar_chart(
                chart_frame,
                x="Metric",
                series=[segment_a, segment_b],
                names=[segment_a, segment_b],
                title="Metric comparison (raw values)",
                height=300,
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        ui.download_frame(rendered, "segment-comparison.csv", "Download comparison")

# ---------------------------------------------------------------------------
# Customer-level engagement
# ---------------------------------------------------------------------------

ui.section(
    "Customer engagement",
    "Journeys aggregated to one row per customer. Frequency is normalised by how long "
    "each customer was observed.",
)

features = customer_features(source, filters, min_journeys=2)
if features.empty:
    ui.empty_state("Not enough repeat travel in this period to profile customers.")
else:
    metric_left, metric_right = st.columns(2, gap="large")
    with metric_left:
        st.plotly_chart(
            charts.histogram(
                features["journeys_per_week"].clip(
                    upper=features["journeys_per_week"].quantile(0.99)
                ),
                title="Distribution of journeys per week",
                x_title="Journeys per week",
                height=270,
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with metric_right:
        engagement = (
            features.assign(
                band=pd.cut(
                    features["journeys_per_week"],
                    bins=[0, 1, 3, 7, 1000],
                    labels=["Under 1/week", "1-3/week", "3-7/week", "7+/week"],
                )
            )
            .groupby("band", observed=False)
            .agg(customers=("customer_id", "size"), satisfaction=("avg_satisfaction", "mean"))
            .reset_index()
        )
        st.plotly_chart(
            charts.bar_chart(
                engagement,
                x="band",
                y="customers",
                title="Customers by travel frequency band",
                y_title="Customers",
                value_format=",.0f",
                height=270,
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    top = (
        features.sort_values("journeys", ascending=False)
        .head(200)[
            [
                "customer_id",
                "assigned_segment",
                "age_group",
                "journeys",
                "journeys_per_week",
                "avg_satisfaction",
                "complaint_rate",
                "peak_share",
                "digital_share",
                "primary_mode",
                "primary_channel",
            ]
        ]
        .rename(
            columns={
                "customer_id": "Customer",
                "assigned_segment": "Segment",
                "age_group": "Age group",
                "journeys": "Journeys",
                "journeys_per_week": "Journeys/week",
                "avg_satisfaction": "Satisfaction",
                "complaint_rate": "Complaint %",
                "peak_share": "Peak %",
                "digital_share": "Digital %",
                "primary_mode": "Main mode",
                "primary_channel": "Main channel",
            }
        )
    )
    st.dataframe(top, use_container_width=True, hide_index=True, height=320)
    st.caption(f"Showing the 200 most active of {len(features):,} profiled customers.")
    ui.download_frame(top, "customer-engagement.csv", "Download customer table")

ui.footer()
