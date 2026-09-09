"""Service Performance - operational reliability, peak pressure and disruption impact."""

from __future__ import annotations

import streamlit as st

from src.analytics.kpis import METRICS, compare_periods, metrics_by_dimension, population_summary
from src.analytics.records import (
    DISPLAY_COLUMNS,
    SORT_OPTIONS,
    SUBSET_FILTERS,
    count_records,
    fetch_records,
)
from src.analytics.trends import heatmap_matrix, hourly_profile, weekday_profile
from src.ui import charts, components as ui
from src.ui.filters import render_filters
from src.ui.state import active_source

ui.configure_page("Service Performance")

source = active_source()
filters = render_filters(source, key_prefix="service")

ui.page_header(
    "Service Performance",
    "Reliability and delay behaviour by region, mode, service line and time of day.",
)

summary = population_summary(source, filters)
ui.population_banner(summary, filters.describe())
if summary["journeys"] == 0:
    ui.footer()
    st.stop()

operational = compare_periods(
    source,
    filters,
    (
        "on_time_performance",
        "avg_delay_minutes",
        "p90_delay_minutes",
        "disruption_rate",
        "complaint_rate",
        "avg_resolution_hours",
    ),
)
ui.kpi_row(list(operational.values()), columns=3)

# ---------------------------------------------------------------------------
# Breakdown by dimension
# ---------------------------------------------------------------------------

ui.section("Performance breakdown")

dimension_label = st.radio(
    "Break down by",
    ["Region", "Transport mode", "Service line", "Time band"],
    horizontal=True,
    label_visibility="collapsed",
    key="service_dimension",
)
dimension_column = {
    "Region": "region",
    "Transport mode": "transport_mode",
    "Service line": "service_line",
    "Time band": "time_band",
}[dimension_label]

breakdown = metrics_by_dimension(
    source,
    filters,
    dimension_column,
    keys=(
        "total_journeys",
        "on_time_performance",
        "avg_delay_minutes",
        "p90_delay_minutes",
        "complaint_rate",
        "avg_satisfaction",
    ),
    min_journeys=20,
)

if breakdown.empty:
    ui.empty_state("Not enough journeys to break this selection down.")
else:
    limited = breakdown.head(15)
    chart_left, chart_right = st.columns(2, gap="large")
    with chart_left:
        st.plotly_chart(
            charts.bar_chart(
                limited.sort_values("on_time_performance"),
                x="dimension_value",
                y="on_time_performance",
                title=f"On-time performance by {dimension_label.lower()}",
                y_title="On-time %",
                horizontal=True,
                value_format=",.1f",
                height=320,
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with chart_right:
        st.plotly_chart(
            charts.bar_chart(
                limited.sort_values("avg_delay_minutes", ascending=False),
                x="dimension_value",
                y="avg_delay_minutes",
                title=f"Average delay by {dimension_label.lower()}",
                y_title="Minutes",
                horizontal=True,
                value_format=",.2f",
                height=320,
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.dataframe(
        breakdown.rename(columns={"dimension_value": dimension_label}),
        use_container_width=True,
        hide_index=True,
        height=260,
        column_config={
            "total_journeys": st.column_config.NumberColumn("Journeys", format="%d"),
            "on_time_performance": st.column_config.NumberColumn("On-time %", format="%.1f"),
            "avg_delay_minutes": st.column_config.NumberColumn("Avg delay", format="%.2f"),
            "p90_delay_minutes": st.column_config.NumberColumn("P90 delay", format="%.1f"),
            "complaint_rate": st.column_config.NumberColumn("Complaint %", format="%.2f"),
            "avg_satisfaction": st.column_config.NumberColumn("Satisfaction", format="%.2f"),
        },
    )
    ui.download_frame(
        breakdown,
        f"performance-by-{dimension_column}.csv",
        "Download breakdown",
        key="dl_breakdown",
    )

# ---------------------------------------------------------------------------
# Peak analysis
# ---------------------------------------------------------------------------

ui.section("Peak-hour analysis", "Weekday peaks are 07:00-09:00 and 16:00-18:00.")

hourly = hourly_profile(source, filters)
if hourly.empty:
    ui.empty_state("No hourly data for this selection.")
else:
    peak_left, peak_right = st.columns(2, gap="large")
    with peak_left:
        st.plotly_chart(
            charts.multi_line_chart(
                hourly,
                x="hour",
                y="total_journeys",
                colour="day_type",
                title="Demand by hour of day",
                y_title="Journeys",
                hover_format=",.0f",
                height=290,
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with peak_right:
        st.plotly_chart(
            charts.multi_line_chart(
                hourly,
                x="hour",
                y="avg_delay_minutes",
                colour="day_type",
                title="Average delay by hour of day",
                y_title="Minutes",
                hover_format=",.2f",
                height=290,
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )

heat_metric = st.selectbox(
    "Heatmap metric",
    ["avg_delay_minutes", "on_time_performance", "avg_satisfaction", "total_journeys"],
    format_func=lambda key: METRICS[key].label,
    key="service_heat_metric",
)
matrix = heatmap_matrix(source, filters, heat_metric)
if matrix.empty:
    ui.empty_state("Not enough journeys per hour to build a heatmap.")
else:
    st.plotly_chart(
        charts.heatmap(
            matrix,
            title=f"{METRICS[heat_metric].label} by day and hour",
            colourbar_title="",
            value_format=",.1f",
            height=320,
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )

weekday = weekday_profile(source, filters)
if not weekday.empty:
    st.plotly_chart(
        charts.grouped_bar_chart(
            weekday,
            x="day_name",
            series=["avg_delay_minutes", "complaint_rate"],
            names=["Average delay (min)", "Complaint rate (%)"],
            title="Delay and complaints by day of week",
            height=290,
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )

# ---------------------------------------------------------------------------
# Detailed records
# ---------------------------------------------------------------------------

ui.section(
    "Underlying records",
    "The journeys behind the metrics above. Filtered, sorted and limited in SQL rather "
    "than loaded in full.",
)

control_a, control_b, control_c = st.columns([1.1, 1.1, 1.4])
with control_a:
    subset = st.selectbox("Show", list(SUBSET_FILTERS.keys()), key="records_subset")
with control_b:
    sort = st.selectbox("Sort by", list(SORT_OPTIONS.keys()), key="records_sort")
with control_c:
    search = st.text_input(
        "Search",
        placeholder="Journey ID, service line, region or complaint category",
        key="records_search",
    )

columns = st.multiselect(
    "Columns",
    list(DISPLAY_COLUMNS),
    default=[
        "journey_id",
        "date",
        "region",
        "transport_mode",
        "service_line",
        "customer_segment",
        "delay_minutes",
        "on_time",
        "customer_satisfaction",
        "complaint_flag",
    ],
    key="records_columns",
)
row_limit = st.slider("Rows to display", 50, 2000, 300, step=50, key="records_limit")

matching = count_records(source, filters, subset)
records = fetch_records(
    source,
    filters,
    columns=tuple(columns) if columns else DISPLAY_COLUMNS,
    sort=sort,
    subset=subset,
    search=search,
    limit=row_limit,
)

if records.empty:
    ui.empty_state("No records match this combination of filters and search text.")
else:
    st.caption(
        f"Showing {len(records):,} of {matching:,} matching journeys, sorted by {sort.lower()}."
    )
    st.dataframe(
        records,
        use_container_width=True,
        hide_index=True,
        height=420,
        column_config={
            "delay_minutes": st.column_config.NumberColumn("Delay (min)", format="%.1f"),
            "journey_duration_minutes": st.column_config.NumberColumn(
                "Duration (min)", format="%.1f"
            ),
            "customer_satisfaction": st.column_config.NumberColumn("Satisfaction", format="%.0f"),
            "resolution_time_hours": st.column_config.NumberColumn("Resolution (h)", format="%.1f"),
        },
    )
    ui.download_frame(records, "journey-records.csv", "Download these records", key="dl_records")

ui.metric_help(
    [
        METRICS[k]
        for k in (
            "on_time_performance",
            "avg_delay_minutes",
            "p90_delay_minutes",
            "disruption_rate",
        )
    ]
)
ui.footer()
