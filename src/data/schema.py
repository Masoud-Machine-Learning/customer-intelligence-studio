"""Canonical dataset schema.

The flat journey schema defined here is the contract used in three places:

* the synthetic generator writes it,
* the DuckDB view ``v_journeys`` exposes it (dimensions joined back onto the fact),
* uploaded CSV files are validated against it.

Keeping one definition means an uploaded file and the demo dataset are interchangeable
everywhere downstream.
"""

from __future__ import annotations

from typing import Final

# Columns an uploaded file must contain for the application to work at all.
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "journey_id",
    "customer_id",
    "date",
    "region",
    "transport_mode",
    "customer_segment",
    "delay_minutes",
    "customer_satisfaction",
)

# Columns that unlock additional pages when present but are not strictly required.
OPTIONAL_COLUMNS: Final[tuple[str, ...]] = (
    "timestamp",
    "origin_region",
    "destination_region",
    "service_line",
    "journey_purpose",
    "age_group",
    "channel",
    "journey_duration_minutes",
    "on_time",
    "complaint_flag",
    "complaint_category",
    "resolution_time_hours",
    "repeat_customer",
    "service_disruption",
    "accessibility_service_used",
    "fare_type",
    "digital_interaction",
    "feedback_text_category",
)

JOURNEY_COLUMNS: Final[tuple[str, ...]] = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

# DuckDB column types for the fact table, keyed by column name.
DUCKDB_TYPES: Final[dict[str, str]] = {
    "journey_id": "VARCHAR",
    "customer_id": "VARCHAR",
    "date": "DATE",
    "timestamp": "TIMESTAMP",
    "region": "VARCHAR",
    "origin_region": "VARCHAR",
    "destination_region": "VARCHAR",
    "transport_mode": "VARCHAR",
    "service_line": "VARCHAR",
    "journey_purpose": "VARCHAR",
    "customer_segment": "VARCHAR",
    "age_group": "VARCHAR",
    "channel": "VARCHAR",
    "journey_duration_minutes": "DOUBLE",
    "delay_minutes": "DOUBLE",
    "on_time": "BOOLEAN",
    "customer_satisfaction": "DOUBLE",
    "complaint_flag": "BOOLEAN",
    "complaint_category": "VARCHAR",
    "resolution_time_hours": "DOUBLE",
    "repeat_customer": "BOOLEAN",
    "service_disruption": "BOOLEAN",
    "accessibility_service_used": "BOOLEAN",
    "fare_type": "VARCHAR",
    "digital_interaction": "BOOLEAN",
    "feedback_text_category": "VARCHAR",
}

NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    "journey_duration_minutes",
    "delay_minutes",
    "customer_satisfaction",
    "resolution_time_hours",
)

BOOLEAN_COLUMNS: Final[tuple[str, ...]] = (
    "on_time",
    "complaint_flag",
    "repeat_customer",
    "service_disruption",
    "accessibility_service_used",
    "digital_interaction",
)

# Dimension tables kept separate from the fact table (see docs/architecture notes).
CUSTOMER_COLUMNS: Final[tuple[str, ...]] = (
    "customer_id",
    "customer_segment",
    "age_group",
    "home_region",
    "fare_type",
    "repeat_customer",
    "digital_affinity",
    "first_seen_date",
)

SERVICE_LINE_COLUMNS: Final[tuple[str, ...]] = (
    "service_line",
    "transport_mode",
    "region",
    "scheduled_duration_minutes",
    "reliability_index",
    "capacity_index",
)


class SchemaValidationError(ValueError):
    """Raised when an uploaded dataset cannot be used by the application."""


def missing_required_columns(columns: list[str]) -> list[str]:
    """Return required columns absent from ``columns`` (case-insensitive)."""
    present = {str(c).strip().lower() for c in columns}
    return [c for c in REQUIRED_COLUMNS if c not in present]
