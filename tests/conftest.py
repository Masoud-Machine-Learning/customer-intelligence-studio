"""Shared test fixtures.

A small dataset is generated once per session and loaded into an in-memory DuckDB
database using the same code path as the real application, so tests exercise the actual
schema, views and validity rules rather than a stand-in.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.data.database import DataSource, build_database
from src.data.filters import FilterState
from src.data.generator import GenerationConfig, generate_dataset

TEST_END_DATE = date(2026, 6, 30)
TEST_SEED = 7


@pytest.fixture(scope="session")
def generation_config() -> GenerationConfig:
    """Small but structurally complete configuration.

    Volume matters here: the anomaly detector ignores days with too few journeys to be
    meaningful, so a fixture with only a handful of journeys per region per day would
    make the detector look broken. 40,000 journeys over 200 days gives roughly 30 per
    region per day, which clears that floor while still generating in under a second.
    """
    return GenerationConfig(
        seed=TEST_SEED,
        n_journeys=40_000,
        n_customers=3_000,
        history_days=200,
        end_date=TEST_END_DATE,
    )


@pytest.fixture(scope="session")
def dataset(generation_config: GenerationConfig):
    return generate_dataset(generation_config)


@pytest.fixture(scope="session")
def source(dataset) -> DataSource:
    """In-memory DataSource built with the production build function."""
    connection = build_database(dataset, db_path=None)
    return DataSource(connection=connection, manifest=dataset.manifest, origin="test")


@pytest.fixture()
def filters() -> FilterState:
    """A 60-day window with 60 days of history available before it."""
    end = TEST_END_DATE
    return FilterState.create(end - timedelta(days=59), end)
