"""Anomaly detection, checked against deliberately injected anomalies."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.analytics.anomalies import (
    anomalies_to_frame,
    anomaly_series,
    detect_anomalies,
    summarise_anomalies,
)
from src.data.filters import FilterState
from src.data.generator import known_anomaly_windows

TEST_END_DATE = date(2026, 6, 30)


@pytest.fixture(scope="module")
def wide_filters() -> FilterState:
    """A window wide enough to contain every injected anomaly."""
    return FilterState.create(TEST_END_DATE - timedelta(days=130), TEST_END_DATE)


def test_detector_finds_the_injected_delay_spike(source, wide_filters):
    """The generator plants a delay spike in the West region; the detector must find it."""
    planted = next(a for a in known_anomaly_windows(TEST_END_DATE) if a["kind"] == "delay")
    found = detect_anomalies(source, wide_filters, dimension="region", method="rolling_z")

    matches = [
        a
        for a in found
        if a.dimension_value == planted["value"]
        and a.metric == "avg_delay_minutes"
        and planted["start"] <= a.period <= planted["end"]
    ]
    assert matches, "the injected West delay spike was not detected"
    assert all(a.actual > a.expected for a in matches)
    assert any(a.severity in {"Medium", "High"} for a in matches)


def test_detected_anomaly_reports_a_usable_expected_range(source, wide_filters):
    found = detect_anomalies(source, wide_filters, dimension="region", limit=5)
    assert found

    for anomaly in found:
        assert anomaly.lower_bound <= anomaly.expected <= anomaly.upper_bound
        assert anomaly.actual < anomaly.lower_bound or anomaly.actual > anomaly.upper_bound
        assert anomaly.journeys > 0


def test_statements_are_deterministic(source, wide_filters):
    """The same inputs must always produce the same sentences."""
    first = detect_anomalies(source, wide_filters, dimension="region", limit=5)
    second = detect_anomalies(source, wide_filters, dimension="region", limit=5)

    assert [a.describe() for a in first] == [a.describe() for a in second]
    assert [a.context() for a in first] == [a.context() for a in second]


def test_higher_threshold_flags_fewer_days(source, wide_filters):
    sensitive = detect_anomalies(source, wide_filters, dimension="region", threshold=2.0)
    strict = detect_anomalies(source, wide_filters, dimension="region", threshold=3.5)

    assert len(strict) <= len(sensitive)


def test_both_methods_run_and_agree_on_the_worst_scope(source, wide_filters):
    rolling = detect_anomalies(source, wide_filters, dimension="region", method="rolling_z")
    iqr = detect_anomalies(source, wide_filters, dimension="region", method="iqr")

    assert rolling and iqr
    assert {a.dimension_value for a in iqr} & {a.dimension_value for a in rolling}


def test_unknown_method_is_rejected(source, wide_filters):
    with pytest.raises(ValueError):
        detect_anomalies(source, wide_filters, method="isolation_forest")


def test_adverse_only_filters_out_improvements(source, wide_filters):
    everything = detect_anomalies(source, wide_filters, dimension="region")
    adverse = detect_anomalies(source, wide_filters, dimension="region", adverse_only=True)

    assert len(adverse) <= len(everything)
    assert all(a.is_adverse for a in adverse)


def test_anomalies_are_confined_to_the_selected_window(source):
    """Baseline history is read from before the window but never reported from it."""
    narrow = FilterState.create(TEST_END_DATE - timedelta(days=20), TEST_END_DATE)
    found = detect_anomalies(source, narrow, dimension="region", threshold=2.0)

    assert all(narrow.start_date <= a.period <= narrow.end_date for a in found)


def test_series_view_returns_a_plottable_band(source, wide_filters):
    series = anomaly_series(
        source, wide_filters, "avg_delay_minutes", dimension="region", dimension_value="West"
    )

    assert not series.empty
    assert {"period", "value", "expected", "lower", "upper", "flagged"} <= set(series.columns)
    assert series["flagged"].any()


def test_frame_and_summary_shapes(source, wide_filters):
    found = detect_anomalies(source, wide_filters, dimension="region", limit=8)
    frame = anomalies_to_frame(found)
    summary = summarise_anomalies(found)

    assert len(frame) == len(found)
    assert summary["total_detected"] == len(found)
    assert isinstance(summary["scopes_affected"], list)


def test_empty_input_produces_an_empty_frame():
    assert anomalies_to_frame([]).empty
