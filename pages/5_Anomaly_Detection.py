"""Anomaly Detection - days that fall outside their expected statistical range."""

from __future__ import annotations

import streamlit as st

from src.analytics.anomalies import (
    ANOMALY_ROLLING_WINDOW,
    ANOMALY_Z_THRESHOLD,
    DEFAULT_SCAN_METRICS,
    anomalies_to_frame,
    anomaly_series,
    detect_anomalies,
)
from src.analytics.kpis import METRICS, population_summary
from src.ui import charts, components as ui
from src.ui.filters import render_filters
from src.ui.state import active_source

ui.configure_page("Anomaly Detection")

source = active_source()
filters = render_filters(source, key_prefix="anomaly")

ui.page_header(
    "Anomaly Detection",
    "Each day is compared with a trailing baseline built from the days before it. "
    "Nothing is learned or predicted: the same data always produces the same result.",
)

summary = population_summary(source, filters)
ui.population_banner(summary, filters.describe())
if summary["journeys"] == 0:
    ui.footer()
    st.stop()

setup_a, setup_b, setup_c, setup_d = st.columns([1.1, 1.1, 1, 1])
with setup_a:
    scope_label = st.selectbox(
        "Scope", ["By region", "By transport mode", "Whole network"], key="anom_scope"
    )
    dimension = {
        "By region": "region",
        "By transport mode": "transport_mode",
        "Whole network": None,
    }[scope_label]
with setup_b:
    method_label = st.selectbox(
        "Method",
        ["Rolling mean + standard deviation", "Interquartile range (IQR)"],
        key="anom_method",
        help="Rolling z-scores react to sustained shifts; IQR is more robust when the "
        "baseline period itself contains outliers.",
    )
    method = "rolling_z" if method_label.startswith("Rolling") else "iqr"
with setup_c:
    threshold = st.slider(
        "Sensitivity (z)",
        1.5,
        4.0,
        float(ANOMALY_Z_THRESHOLD),
        0.1,
        key="anom_threshold",
        help="Lower values flag more days.",
    )
with setup_d:
    adverse_only = st.toggle(
        "Adverse only",
        value=True,
        key="anom_adverse",
        help="Hide deviations that are improvements.",
    )

anomalies = detect_anomalies(
    source,
    filters,
    dimension=dimension,
    method=method,
    threshold=threshold,
    adverse_only=adverse_only,
    limit=100,
)

count_a, count_b, count_c = st.columns(3)
count_a.metric("Anomalies detected", f"{len(anomalies):,}")
count_b.metric("High severity", f"{sum(1 for a in anomalies if a.severity == 'High'):,}")
count_c.metric("Baseline window", f"{ANOMALY_ROLLING_WINDOW} days")

if not anomalies:
    st.success(
        "No days in this period fell outside their expected range at the current "
        "sensitivity. Lower the threshold or widen the date range to scan more days."
    )
    ui.footer()
    st.stop()

# ---------------------------------------------------------------------------
# Detail
# ---------------------------------------------------------------------------

ui.section("Detected anomalies")
st.dataframe(anomalies_to_frame(anomalies), use_container_width=True, hide_index=True, height=300)
ui.download_frame(
    anomalies_to_frame(anomalies), "anomalies.csv", "Download anomalies", key="dl_anom"
)

ui.section("Explanations", "Each statement is generated from the figures, not written by a model.")
for anomaly in anomalies[:6]:
    with st.container():
        st.markdown(
            f"**{anomaly.describe()}**  \n"
            f"{anomaly.context()}  \n"
            f"Expected range {anomaly.expected_range()} &middot; z = {anomaly.z_score:.2f} "
            f"&middot; severity {anomaly.severity} &middot; {anomaly.journeys:,} journeys"
        )

# ---------------------------------------------------------------------------
# Series view
# ---------------------------------------------------------------------------

ui.section("Series view", "The scanned series with its expected band and flagged days.")

view_a, view_b = st.columns(2)
with view_a:
    metric = st.selectbox(
        "Metric",
        list(DEFAULT_SCAN_METRICS.keys()),
        format_func=lambda key: METRICS[key].label,
        key="anom_metric",
    )
with view_b:
    if dimension is None:
        scope_value = None
        st.caption("Scope: whole network")
    else:
        values = sorted({a.dimension_value for a in anomalies})
        scope_value = st.selectbox("Scope value", values or ["-"], key="anom_scope_value")

series = anomaly_series(
    source,
    filters,
    metric,
    dimension=dimension,
    dimension_value=None if dimension is None else scope_value,
    method=method,
    threshold=threshold,
)

if series.empty:
    ui.empty_state("Not enough history to build a baseline for this scope.")
else:
    st.plotly_chart(
        charts.anomaly_chart(
            series,
            value_column="value",
            title=f"{METRICS[metric].label}"
            + (f" - {scope_value}" if scope_value else " - network"),
            y_title=METRICS[metric].label,
            height=330,
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )

with st.expander("How detection works"):
    st.markdown(
        f"""
1. Aggregate the filtered population to one row per day (per scope) in SQL.
2. For each day, build a baseline from the **previous {ANOMALY_ROLLING_WINDOW} days**.
   The day under test is excluded from its own baseline.
3. Compute the deviation:
   - **Rolling z-score**: `(value - baseline mean) / baseline standard deviation`,
     flagged when the absolute value reaches the sensitivity threshold.
   - **IQR**: flagged outside `Q1 - 1.5 x IQR` to `Q3 + 1.5 x IQR` of the baseline window.
4. Require a minimum number of journeys that day, and a minimum absolute movement, so
   a quiet day with three journeys cannot produce a headline.
5. Attach companion metrics from the same day as evidence for the explanation.
        """
    )

ui.footer()
