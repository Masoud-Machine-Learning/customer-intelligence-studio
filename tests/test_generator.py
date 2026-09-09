"""Synthetic data generation: reproducibility, structure and causal relationships."""

from __future__ import annotations

from datetime import date

import pytest

from src.data.generator import GenerationConfig, generate_dataset
from src.data.schema import JOURNEY_COLUMNS
from src.utils.config import ON_TIME_THRESHOLD_MINUTES


def test_generation_is_reproducible(generation_config):
    """The same configuration must produce byte-identical frames."""
    first = generate_dataset(generation_config)
    second = generate_dataset(generation_config)

    assert first.journeys.equals(second.journeys)
    assert first.customers.equals(second.customers)
    assert first.service_lines.equals(second.service_lines)
    assert first.manifest["fingerprint"] == second.manifest["fingerprint"]


def test_a_different_seed_produces_different_data(generation_config):
    other = GenerationConfig(
        seed=generation_config.seed + 1,
        n_journeys=generation_config.n_journeys,
        n_customers=generation_config.n_customers,
        history_days=generation_config.history_days,
        end_date=generation_config.end_date,
    )

    assert not generate_dataset(generation_config).journeys.equals(generate_dataset(other).journeys)


def test_schema_and_volume(dataset, generation_config):
    journeys = dataset.journeys

    assert list(journeys.columns) == list(JOURNEY_COLUMNS)
    # Duplicates are appended after generation, so the row count exceeds the request.
    assert len(journeys) >= generation_config.n_journeys
    assert dataset.customers["customer_id"].is_unique
    assert dataset.service_lines["service_line"].is_unique


def test_dates_stay_inside_the_configured_history(dataset, generation_config):
    journeys = dataset.journeys

    assert journeys["date"].min() >= generation_config.start_date()
    assert journeys["date"].max() <= generation_config.resolved_end_date()


def test_on_time_flag_matches_the_delay_threshold(dataset):
    valid = dataset.journeys[dataset.journeys["delay_minutes"] < 900]
    expected = valid["delay_minutes"] <= ON_TIME_THRESHOLD_MINUTES

    assert (valid["on_time"] == expected).all()


def test_longer_delays_reduce_satisfaction(dataset):
    """The generator must encode the relationship the application later reports."""
    valid = dataset.journeys[
        (dataset.journeys["delay_minutes"] < 900)
        & dataset.journeys["customer_satisfaction"].between(1, 5)
    ]
    low_delay = valid[valid["delay_minutes"] <= 2]["customer_satisfaction"].mean()
    high_delay = valid[valid["delay_minutes"] >= 15]["customer_satisfaction"].mean()

    assert high_delay < low_delay - 0.3


def test_disruption_increases_complaints(dataset):
    valid = dataset.journeys[dataset.journeys["delay_minutes"] < 900]
    disrupted = valid[valid["service_disruption"]]["complaint_flag"].mean()
    normal = valid[~valid["service_disruption"]]["complaint_flag"].mean()

    assert disrupted > normal * 1.5


def test_peak_periods_carry_more_delay(dataset):
    journeys = dataset.journeys[dataset.journeys["delay_minutes"] < 900].copy()
    hour = journeys["timestamp"].dt.hour
    weekday = journeys["timestamp"].dt.dayofweek < 5
    peak = weekday & (hour.between(7, 9) | hour.between(16, 18))

    assert journeys[peak]["delay_minutes"].mean() > journeys[~peak]["delay_minutes"].mean()


def test_weekends_are_quieter_than_weekdays(dataset):
    journeys = dataset.journeys.copy()
    per_day = journeys.groupby("date").size()
    weekend = [d for d in per_day.index if d.weekday() >= 5]
    weekday = [d for d in per_day.index if d.weekday() < 5]

    assert per_day[weekend].mean() < per_day[weekday].mean()


def test_quality_defects_are_injected(dataset):
    journeys = dataset.journeys

    assert journeys["customer_satisfaction"].isna().sum() > 0
    assert journeys["region"].isna().sum() > 0
    assert journeys["journey_id"].duplicated().sum() > 0
    assert (journeys["transport_mode"] == "UNKNOWN").sum() > 0


def test_quality_degradation_can_be_disabled():
    clean = generate_dataset(
        GenerationConfig(
            seed=3,
            n_journeys=2_000,
            n_customers=400,
            history_days=60,
            end_date=date(2026, 6, 30),
            degrade_quality=False,
        )
    )

    assert clean.journeys["journey_id"].is_unique
    assert clean.journeys["customer_satisfaction"].notna().all()
    assert not (clean.journeys["transport_mode"] == "UNKNOWN").any()


@pytest.mark.parametrize("column", ["region", "transport_mode", "customer_segment"])
def test_key_dimensions_have_multiple_values(dataset, column):
    assert dataset.journeys[column].nunique(dropna=True) >= 3
