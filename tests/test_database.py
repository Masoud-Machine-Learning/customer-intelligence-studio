"""DuckDB schema, views and the validity rules that separate them."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.data.database import (
    ANALYTICS_VIEW,
    CUSTOMER_TABLE,
    FACT_TABLE,
    RAW_VIEW,
    SERVICE_LINE_TABLE,
    build_upload_database,
    dataset_bounds,
    distinct_values,
    query,
    row_counts,
    scalar,
)
from src.data.schema import JOURNEY_COLUMNS
from src.utils.config import MAX_PLAUSIBLE_DELAY_MINUTES, TRANSPORT_MODES


def test_star_schema_tables_exist(source):
    tables = query(source.connection, "SHOW TABLES")["name"].tolist()

    assert FACT_TABLE in tables
    assert CUSTOMER_TABLE in tables
    assert SERVICE_LINE_TABLE in tables


def test_fact_table_matches_the_canonical_schema(source):
    columns = query(source.connection, f"DESCRIBE {FACT_TABLE}")["column_name"].tolist()

    assert columns == list(JOURNEY_COLUMNS)


def test_raw_view_adds_derived_time_columns(source):
    columns = query(source.connection, f"DESCRIBE {RAW_VIEW}")["column_name"].tolist()

    for derived in (
        "journey_hour",
        "day_of_week",
        "is_weekend",
        "is_peak",
        "time_band",
        "week_start",
        "month_start",
    ):
        assert derived in columns


def test_raw_view_joins_the_dimension_tables(source):
    """Dimension attributes exist only on the dimensions, so a null column means no join."""
    columns = query(source.connection, f"DESCRIBE {RAW_VIEW}")["column_name"].tolist()
    assert "line_reliability_index" in columns

    populated = scalar(
        source.connection,
        f"SELECT count(*) FROM {RAW_VIEW} WHERE line_reliability_index IS NOT NULL",
    )
    assert populated > 0


def test_analytics_view_excludes_invalid_rows(source):
    counts = row_counts(source.connection)

    assert counts["analysable_rows"] < counts["raw_rows"]

    offenders = scalar(
        source.connection,
        f"""
        SELECT count(*) FROM {ANALYTICS_VIEW}
        WHERE delay_minutes < 0
           OR delay_minutes > {MAX_PLAUSIBLE_DELAY_MINUTES}
           OR journey_duration_minutes <= 0
           OR customer_satisfaction NOT BETWEEN 1 AND 5
        """,
    )
    assert offenders == 0


def test_analytics_view_only_contains_known_transport_modes(source):
    modes = set(distinct_values(source.connection, "transport_mode"))

    assert modes.issubset(set(TRANSPORT_MODES))
    assert "UNKNOWN" not in modes


def test_analytics_view_deduplicates_journey_ids(source):
    duplicates = scalar(
        source.connection,
        f"SELECT count(*) - count(DISTINCT journey_id) FROM {ANALYTICS_VIEW}",
    )
    raw_duplicates = scalar(
        source.connection,
        f"SELECT count(*) - count(DISTINCT journey_id) FROM {RAW_VIEW}",
    )

    assert raw_duplicates > 0, "the generator should inject duplicates to detect"
    assert duplicates == 0


def test_time_band_classification_is_consistent(source):
    """No weekend row may be labelled as a peak, by definition."""
    contradictions = scalar(
        source.connection,
        f"SELECT count(*) FROM {ANALYTICS_VIEW} WHERE is_weekend AND is_peak",
    )
    assert contradictions == 0


def test_parameterised_query_binds_values(source):
    frame = query(
        source.connection,
        f"SELECT count(*) AS n FROM {ANALYTICS_VIEW} WHERE region = ?",
        ["West"],
    )
    assert frame.loc[0, "n"] > 0


def test_dataset_bounds_returns_ordered_dates(source):
    minimum, maximum = dataset_bounds(source.connection)

    assert isinstance(minimum, date)
    assert minimum < maximum


def test_distinct_values_rejects_unknown_columns(source):
    with pytest.raises(ValueError):
        distinct_values(source.connection, "1=1; DROP TABLE journeys")


def test_upload_database_accepts_a_minimal_schema():
    """An uploaded file with only the required columns must still work end to end."""
    frame = pd.DataFrame(
        {
            "journey_id": ["A1", "A2", "A3"],
            "customer_id": ["C1", "C1", "C2"],
            "date": ["2026-01-01", "2026-01-02", "2026-01-02"],
            "region": ["North", "North", "South"],
            "transport_mode": ["Bus", "Metro", "Bus"],
            "customer_segment": ["Frequent Commuters"] * 3,
            "delay_minutes": [1.0, 12.0, 3.0],
            "customer_satisfaction": [5, 2, 4],
        }
    )
    connection = build_upload_database(frame)

    assert row_counts(connection)["analysable_rows"] == 3
    # on_time is derivable and must be filled in when the feed omits it.
    on_time = query(connection, f"SELECT on_time FROM {ANALYTICS_VIEW} ORDER BY journey_id")
    assert on_time["on_time"].tolist() == [True, False, True]
