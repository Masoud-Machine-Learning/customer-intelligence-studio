"""KPI arithmetic and period comparison."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from src.analytics.kpis import (
    HEADLINE_KPIS,
    METRICS,
    compare_periods,
    compute_metrics,
    metrics_by_dimension,
    population_summary,
)
from src.data.database import ANALYTICS_VIEW, query
from src.data.filters import FilterState, build_where
from src.utils.formatting import percent_change


def test_headline_metrics_are_within_valid_ranges(source, filters):
    values = compute_metrics(source, filters, HEADLINE_KPIS)

    assert values["total_journeys"] > 0
    assert values["active_customers"] > 0
    assert 1.0 <= values["avg_satisfaction"] <= 5.0
    assert 0.0 <= values["on_time_performance"] <= 100.0
    assert values["avg_delay_minutes"] >= 0.0
    assert 0.0 <= values["complaint_rate"] <= 100.0


def test_total_journeys_matches_a_direct_count(source, filters):
    """The metric registry must agree with a plain SQL count over the same filter."""
    where, params = build_where(filters)
    direct = query(
        source.connection, f"SELECT count(*) AS n FROM {ANALYTICS_VIEW} {where}", params
    ).loc[0, "n"]

    assert compute_metrics(source, filters, ["total_journeys"])["total_journeys"] == direct


def test_on_time_performance_is_computed_over_non_null_rows(source, filters):
    where, params = build_where(filters)
    frame = query(
        source.connection,
        f"""
        SELECT count(*) FILTER (WHERE on_time) AS hits,
               count(*) FILTER (WHERE on_time IS NOT NULL) AS total
        FROM {ANALYTICS_VIEW} {where}
        """,
        params,
    )
    expected = frame.loc[0, "hits"] / frame.loc[0, "total"] * 100
    actual = compute_metrics(source, filters, ["on_time_performance"])["on_time_performance"]

    assert actual == pytest.approx(expected)


def test_previous_period_is_the_preceding_equal_window(filters):
    previous_start, previous_end = filters.previous_period()

    assert previous_end == filters.start_date - timedelta(days=1)
    assert (previous_end - previous_start).days + 1 == filters.period_days


def test_compare_periods_reports_both_sides_and_a_consistent_delta(source, filters):
    kpis = compare_periods(source, filters, HEADLINE_KPIS)

    for kpi in kpis.values():
        assert kpi.current is not None
        assert kpi.previous is not None
        assert kpi.change == pytest.approx(kpi.current - kpi.previous)
        assert kpi.change_pct == pytest.approx(percent_change(kpi.current, kpi.previous))


def test_delta_direction_respects_metric_polarity(source, filters):
    """A rise in delay is bad; a rise in satisfaction is good."""
    kpis = compare_periods(source, filters, ("avg_delay_minutes", "avg_satisfaction"))

    delay = kpis["avg_delay_minutes"]
    if delay.change and delay.change > 0:
        assert delay.direction == "bad"

    satisfaction = kpis["avg_satisfaction"]
    if satisfaction.change and satisfaction.change > 0:
        assert satisfaction.direction == "good"


def test_dimension_breakdown_sums_back_to_the_total(source, filters):
    """Splitting by mode must not create or lose journeys."""
    breakdown = metrics_by_dimension(source, filters, "transport_mode")
    total = compute_metrics(source, filters, ["total_journeys"])["total_journeys"]

    assert breakdown["total_journeys"].sum() == total


def test_population_summary_counts_distinct_dimensions(source, filters):
    summary = population_summary(source, filters)

    assert summary["journeys"] > 0
    assert 0 < summary["regions"] <= 6
    assert 0 < summary["modes"] <= 5
    assert summary["first_date"] >= filters.start_date
    assert summary["last_date"] <= filters.end_date


def test_filtering_to_one_region_reduces_the_population(source, filters):
    unfiltered = compute_metrics(source, filters, ["total_journeys"])["total_journeys"]
    narrowed = FilterState.create(filters.start_date, filters.end_date, regions=["West"])
    filtered = compute_metrics(source, narrowed, ["total_journeys"])["total_journeys"]

    assert 0 < filtered < unfiltered


def test_empty_population_returns_none_rather_than_raising(source):
    """A filter matching nothing must degrade gracefully, not error."""
    long_before_the_data = date(2000, 1, 1)
    impossible = FilterState.create(long_before_the_data, long_before_the_data)
    values = compute_metrics(source, impossible, HEADLINE_KPIS)

    assert values["total_journeys"] == 0
    assert values["avg_satisfaction"] is None


def test_every_registered_metric_executes(source, filters):
    """Guards against a typo in any SQL expression in the registry."""
    values = compute_metrics(source, filters, METRICS.keys())

    assert set(values) == set(METRICS)
    assert not any(isinstance(v, pd.Series) for v in values.values())
