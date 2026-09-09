"""Data Quality - completeness, validity, uniqueness and their effect on analytics."""

from __future__ import annotations

import streamlit as st

from src.analytics.data_quality import CATEGORY_WEIGHTS, build_quality_report
from src.ui import charts, components as ui
from src.ui.filters import render_filters
from src.ui.state import active_source

ui.configure_page("Data Quality")

source = active_source()
filters = render_filters(source, key_prefix="quality")

ui.page_header(
    "Data Quality",
    "Profiled against the raw feed, before validity rules are applied. Analytics pages "
    "use only the rows that pass these rules.",
)

report = build_quality_report(source, filters)

if report.total_rows == 0:
    ui.empty_state("No rows match the current filters.")
    ui.footer()
    st.stop()

# ---------------------------------------------------------------------------
# Headline
# ---------------------------------------------------------------------------

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Quality score", f"{report.score}/100", report.rating)
col_b.metric("Total records", f"{report.total_rows:,}")
col_c.metric(
    "Usable for analysis",
    f"{report.analysable_rows:,}",
    f"-{report.excluded_rows:,} excluded" if report.excluded_rows else "none excluded",
    delta_color="inverse" if report.excluded_rows else "off",
)
col_d.metric(
    "Data freshness",
    "-" if report.freshness_days is None else f"{report.freshness_days} day(s)",
    help="Days between today and the most recent journey in the selection.",
)

st.caption(
    "Score = "
    + " + ".join(f"{int(w * 100)}% {cat.lower()}" for cat, w in CATEGORY_WEIGHTS.items())
    + ". Each family scores 100 minus the summed failure rates of its checks."
)

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

ui.section("Rule results")
checks = report.checks_frame()
st.dataframe(
    checks,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Check": st.column_config.TextColumn(width="medium"),
        "Why it matters": st.column_config.TextColumn(width="large"),
        "Failing Rows": st.column_config.NumberColumn(format="%d"),
    },
)
ui.download_frame(checks, "data-quality-checks.csv", "Download check results", key="dl_checks")

failing = report.failing_checks()
if failing:
    worst = max(failing, key=lambda c: c.failure_rate)
    st.warning(
        f"{len(failing)} check(s) are failing. The largest is '{worst.label}' at "
        f"{worst.failure_rate:.2f}% of rows ({worst.failed_rows:,} records). "
        f"{worst.impact}"
    )
else:
    st.success("All quality rules pass for the current selection.")

# ---------------------------------------------------------------------------
# Field-level completeness
# ---------------------------------------------------------------------------

ui.section("Field completeness")
completeness = report.field_completeness.copy()
chart_left, chart_right = st.columns([1.3, 1], gap="large")

with chart_left:
    st.plotly_chart(
        charts.bar_chart(
            completeness.sort_values("completeness_pct"),
            x="field",
            y="completeness_pct",
            title="Completeness by field",
            y_title="% populated",
            horizontal=True,
            value_format=",.2f",
            height=420,
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )

with chart_right:
    st.dataframe(
        completeness.rename(
            columns={
                "field": "Field",
                "present": "Present",
                "missing": "Missing",
                "completeness_pct": "Complete %",
                "conditional": "Conditional",
                "note": "Note",
            }
        ),
        use_container_width=True,
        hide_index=True,
        height=420,
        column_config={
            "Complete %": st.column_config.NumberColumn(format="%.2f"),
            "Present": st.column_config.NumberColumn(format="%d"),
            "Missing": st.column_config.NumberColumn(format="%d"),
        },
    )

st.caption(
    "Fields marked conditional are only populated for a subset of journeys - a complaint "
    "category exists only where a complaint was raised - so their sparsity is expected "
    "and is excluded from the completeness average."
)

# ---------------------------------------------------------------------------
# Outliers and impact
# ---------------------------------------------------------------------------

ui.section("Statistical outliers")
outlier_a, outlier_b = st.columns(2)
outlier_a.metric("Extreme delay values", f"{report.outliers.get('delay_minutes', 0):,}")
outlier_b.metric(
    "Extreme duration values", f"{report.outliers.get('journey_duration_minutes', 0):,}"
)
st.caption(
    "Values beyond three interquartile ranges from the median. These are flagged for "
    "review but kept in the analysis: an extreme yet possible delay is real operational "
    "information, unlike an impossible one."
)

ui.section("How quality affects the numbers")
excluded_pct = 0.0 if report.total_rows == 0 else report.excluded_rows / report.total_rows * 100
st.markdown(
    f"""
- **{report.excluded_rows:,} rows ({excluded_pct:.2f}%)** are excluded from every KPI on
  the other pages, because they fail a hard validity rule or duplicate a journey ID.
- **Missing satisfaction scores reduce the denominator** for the satisfaction metric
  rather than counting as zero, so a falling response rate widens the uncertainty around
  the score but does not artificially depress it.
- **Rows with a missing region** still count in network totals but disappear from
  regional breakdowns, which is why the two can differ slightly.
- **Duplicated journey IDs** are de-duplicated by keeping the first record per ID, so
  demand is not double counted.
    """
)

with st.expander("Where these rules are defined"):
    st.markdown(
        "Validity rules live in `src/analytics/data_quality.py` and the SQL predicate "
        "that filters the analytics view lives in `src/data/database.py`. Both read "
        "their thresholds from `src/utils/config.py`, so the page and the view can "
        "never disagree about what counts as valid."
    )

ui.footer()
