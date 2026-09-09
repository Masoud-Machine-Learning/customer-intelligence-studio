"""Filter state and its translation into parameterised SQL."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.analytics.kpis import compute_metrics
from src.data.filters import FILTER_COLUMNS, FilterState, build_where, where_fragment


def test_selections_are_normalised_and_deduplicated():
    state = FilterState.create(
        date(2026, 1, 1), date(2026, 1, 31), regions=["West", "West", "North", "", None]
    )

    assert state.regions == ("West", "North")


def test_reversed_dates_are_corrected():
    state = FilterState.create(date(2026, 3, 31), date(2026, 3, 1))

    assert state.start_date == date(2026, 3, 1)
    assert state.end_date == date(2026, 3, 31)


def test_period_length_is_inclusive():
    state = FilterState.create(date(2026, 1, 1), date(2026, 1, 31))

    assert state.period_days == 31


def test_previous_period_abuts_the_selection_without_overlap():
    state = FilterState.create(date(2026, 4, 1), date(2026, 4, 30))
    previous = state.for_previous_period()

    assert previous.end_date == date(2026, 3, 31)
    assert previous.start_date == date(2026, 3, 2)
    assert previous.period_days == state.period_days
    assert previous.end_date < state.start_date


def test_empty_state_produces_only_a_date_clause():
    state = FilterState.create(date(2026, 1, 1), date(2026, 1, 31))
    clause, params = build_where(state)

    assert clause == "WHERE date BETWEEN ? AND ?"
    assert params == [date(2026, 1, 1), date(2026, 1, 31)]


def test_values_are_bound_as_parameters_not_interpolated():
    """The SQL text must never contain a user-supplied value."""
    state = FilterState.create(
        date(2026, 1, 1), date(2026, 1, 31), regions=["West"], transport_modes=["Bus", "Metro"]
    )
    clause, params = build_where(state)

    assert "West" not in clause
    assert "Bus" not in clause
    assert clause.count("?") == len(params)
    assert params[2:] == ["West", "Bus", "Metro"]


def test_dates_can_be_excluded_from_the_clause():
    state = FilterState.create(date(2026, 1, 1), date(2026, 1, 31), regions=["North"])
    clause, params = build_where(state, include_dates=False)

    assert clause == "WHERE region IN (?)"
    assert params == ["North"]


def test_alias_is_applied_to_every_column():
    state = FilterState.create(date(2026, 1, 1), date(2026, 1, 31), regions=["North"])
    clause, _ = build_where(state, alias="j")

    assert "j.date BETWEEN" in clause
    assert "j.region IN" in clause


def test_none_state_produces_no_clause():
    assert build_where(None) == ("", [])
    assert where_fragment(None) == ("TRUE", [])


def test_without_clears_named_dimensions_only():
    state = FilterState.create(
        date(2026, 1, 1), date(2026, 1, 31), regions=["North"], transport_modes=["Bus"]
    )
    reduced = state.without("regions")

    assert reduced.regions == ()
    assert reduced.transport_modes == ("Bus",)


def test_every_filter_attribute_has_a_mapped_column():
    state = FilterState.create(date(2026, 1, 1), date(2026, 1, 2))

    for attribute in FILTER_COLUMNS:
        assert hasattr(state, attribute)


def test_filter_state_is_hashable_for_caching():
    """Streamlit caches on the filter object, so it must be hashable."""
    state = FilterState.create(date(2026, 1, 1), date(2026, 1, 31), regions=["North"])

    assert hash(state) == hash(
        FilterState.create(date(2026, 1, 1), date(2026, 1, 31), regions=["North"])
    )


@pytest.mark.parametrize(
    ("dimension", "value"),
    [("regions", "West"), ("transport_modes", "Bus"), ("customer_segments", "Frequent Commuters")],
)
def test_each_filter_actually_restricts_the_query(source, dimension, value):
    end = date(2026, 6, 30)
    base = FilterState.create(end - timedelta(days=59), end)
    narrowed = FilterState.create(end - timedelta(days=59), end, **{dimension: [value]})

    total = compute_metrics(source, base, ["total_journeys"])["total_journeys"]
    subset = compute_metrics(source, narrowed, ["total_journeys"])["total_journeys"]

    assert 0 < subset < total


def test_combined_filters_are_conjunctive(source):
    """Adding a second filter can only narrow the population further."""
    end = date(2026, 6, 30)
    one = FilterState.create(end - timedelta(days=59), end, regions=["West"])
    two = FilterState.create(
        end - timedelta(days=59), end, regions=["West"], transport_modes=["Bus"]
    )

    assert (
        compute_metrics(source, two, ["total_journeys"])["total_journeys"]
        <= compute_metrics(source, one, ["total_journeys"])["total_journeys"]
    )
