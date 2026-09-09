"""Segmentation - behavioural rules and optional K-means clustering."""

from __future__ import annotations

import streamlit as st

from src.analytics.kpis import population_summary
from src.analytics.segmentation import (
    CLUSTER_FEATURES,
    RULE_DEFINITIONS,
    apply_rule_segments,
    cluster_customers,
    customer_features,
    rule_segmentation,
)
from src.ui import charts, components as ui
from src.ui.filters import render_filters
from src.ui.state import active_source

ui.configure_page("Segmentation")

source = active_source()
filters = render_filters(source, key_prefix="segmentation")

ui.page_header(
    "Customer Segmentation",
    "Two views of the same customers: transparent business rules, and K-means "
    "clustering used as a discovery tool.",
)

summary = population_summary(source, filters)
ui.population_banner(summary, filters.describe())
if summary["journeys"] == 0:
    ui.footer()
    st.stop()

min_journeys = st.slider(
    "Minimum journeys per customer",
    1,
    20,
    3,
    help="Customers below this threshold are excluded: their behaviour cannot be "
    "characterised reliably from too few journeys.",
    key="seg_min_journeys",
)

approach = st.radio(
    "Approach",
    ["Behavioural rules", "K-means clustering"],
    horizontal=True,
    key="seg_approach",
)

features = customer_features(source, filters, min_journeys=min_journeys)
if features.empty:
    ui.empty_state("No customers meet the minimum journey threshold in this period.")
    ui.footer()
    st.stop()

st.caption(f"{len(features):,} customers profiled from {summary['journeys']:,} journeys.")

# ---------------------------------------------------------------------------
# Rule-based segmentation
# ---------------------------------------------------------------------------

if approach == "Behavioural rules":
    result = rule_segmentation(source, filters, min_journeys=min_journeys)
    profile = result.profile()

    ui.section("Segment definitions", "Rules are evaluated in order; the first match wins.")
    for name, definition in RULE_DEFINITIONS:
        st.markdown(f"**{name}** - {definition}")

    ui.section("Segment profile")
    left, right = st.columns([1, 1.5], gap="large")
    with left:
        st.plotly_chart(
            charts.bar_chart(
                profile.sort_values("customers", ascending=False),
                x="segment",
                y="customers",
                title="Customers per segment",
                y_title="Customers",
                horizontal=True,
                value_format=",.0f",
                height=280,
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with right:
        st.dataframe(
            profile,
            use_container_width=True,
            hide_index=True,
            column_config={
                "segment": st.column_config.TextColumn("Segment", width="medium"),
                "customers": st.column_config.NumberColumn("Customers", format="%d"),
                "share_pct": st.column_config.NumberColumn("Share %", format="%.1f"),
                "journeys_per_week": st.column_config.NumberColumn("Journeys/week", format="%.2f"),
                "peak_share": st.column_config.NumberColumn("Peak %", format="%.1f"),
                "digital_share": st.column_config.NumberColumn("Digital %", format="%.1f"),
                "avg_satisfaction": st.column_config.NumberColumn("Satisfaction", format="%.2f"),
                "complaint_rate": st.column_config.NumberColumn("Complaint %", format="%.2f"),
                "avg_delay_minutes": st.column_config.NumberColumn("Avg delay", format="%.2f"),
                "modes_used": st.column_config.NumberColumn("Modes", format="%.1f"),
            },
        )

    st.plotly_chart(
        charts.scatter_chart(
            result.customers.sample(min(4000, len(result.customers)), random_state=42),
            x="journeys_per_week",
            y="avg_satisfaction",
            colour="rule_segment",
            title="Travel frequency against satisfaction",
            x_title="Journeys per week",
            y_title="Average satisfaction (1-5)",
            height=380,
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    ui.download_frame(profile, "rule-segments.csv", "Download segment profile", key="dl_rules")

# ---------------------------------------------------------------------------
# K-means clustering
# ---------------------------------------------------------------------------

else:
    ui.section(
        "Clustering setup",
        "Features are standardised before clustering because they are on different "
        "scales (journeys per week against percentages against a 1-5 score).",
    )

    setup_left, setup_right = st.columns([1, 1.6])
    with setup_left:
        n_clusters = st.slider("Number of clusters (k)", 2, 8, 4, key="seg_k")
    with setup_right:
        selected_features = st.multiselect(
            "Clustering variables",
            list(CLUSTER_FEATURES.keys()),
            default=[
                "journeys_per_week",
                "peak_share",
                "digital_share",
                "avg_satisfaction",
                "complaint_rate",
            ],
            format_func=lambda key: CLUSTER_FEATURES[key],
            key="seg_features",
        )

    if len(selected_features) < 2:
        st.warning("Select at least two variables to cluster on.")
        ui.footer()
        st.stop()

    with st.spinner("Fitting K-means..."):
        result = cluster_customers(
            apply_rule_segments(features),
            n_clusters=n_clusters,
            feature_columns=tuple(selected_features),
        )

    profile = result.profile()

    metric_left, metric_right = st.columns([1, 1])
    with metric_left:
        st.metric(
            "Silhouette score",
            "-" if result.silhouette is None else f"{result.silhouette:.3f}",
            help="Between -1 and 1. Above roughly 0.25 suggests the clusters are "
            "separated rather than arbitrary slices of a continuum.",
        )
    with metric_right:
        if not result.inertia_curve.empty:
            st.plotly_chart(
                charts.line_chart(
                    result.inertia_curve.rename(columns={"k": "period"}),
                    x="period",
                    y="inertia",
                    title="Inertia across candidate k (elbow)",
                    y_title="Inertia",
                    hover_format=",.0f",
                    height=200,
                ),
                use_container_width=True,
                config={"displayModeBar": False},
            )

    ui.section("Cluster profile", "Names are derived from each cluster's centroid.")
    st.dataframe(profile, use_container_width=True, hide_index=True)

    plot_x = selected_features[0]
    plot_y = selected_features[1]
    st.plotly_chart(
        charts.scatter_chart(
            result.customers.sample(min(4000, len(result.customers)), random_state=42),
            x=plot_x,
            y=plot_y,
            colour="cluster_label",
            title=f"{CLUSTER_FEATURES[plot_x]} against {CLUSTER_FEATURES[plot_y]}",
            x_title=CLUSTER_FEATURES[plot_x],
            y_title=CLUSTER_FEATURES[plot_y],
            height=400,
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    crosstab = (
        result.customers.groupby(["cluster_label", "rule_segment"]).size().unstack(fill_value=0)
    )
    ui.section("Clusters against rule-based segments")
    st.caption(
        "Where the two approaches disagree is usually the interesting part: it shows "
        "behaviour the fixed rules do not capture."
    )
    st.dataframe(crosstab, use_container_width=True)
    ui.download_frame(profile, "cluster-profile.csv", "Download cluster profile", key="dl_clusters")

ui.section("Limitations")
for limitation in result.limitations:
    st.markdown(f"- {limitation}")
st.caption(
    "These segments describe synthetic demonstration data. They do not represent any "
    "real operator's customer classification."
)

ui.footer()
