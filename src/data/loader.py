"""Dataset loading, Streamlit caching and upload validation.

This module is the only place that knows about Streamlit's cache primitives, so the
analytics and data layers stay importable from plain Python (and from pytest).

Caching strategy
----------------
``@st.cache_resource``
    The DuckDB connection. It is a long-lived handle, not a value, and must not be
    copied per session.

``@st.cache_data``
    Query results. They are plain DataFrames keyed by the SQL text and bound
    parameters, so repeated interactions with unchanged filters cost nothing.
"""

from __future__ import annotations

import io
from typing import Any, Sequence

import duckdb
import pandas as pd

from src.data import database
from src.data.schema import (
    JOURNEY_COLUMNS,
    REQUIRED_COLUMNS,
    SchemaValidationError,
    missing_required_columns,
)

MAX_UPLOAD_ROWS = 400_000


def _streamlit():
    """Return the Streamlit module, or ``None`` when running outside the app."""
    try:
        import streamlit as st

        return st
    except ModuleNotFoundError:  # pragma: no cover - Streamlit is a hard dependency
        return None


# ---------------------------------------------------------------------------
# Demo dataset
# ---------------------------------------------------------------------------


def load_demo_source() -> database.DataSource:
    """Open (building if necessary) the synthetic demo dataset."""
    manifest = database.ensure_demo_database()
    conn = database.open_demo_database()
    return database.DataSource(connection=conn, manifest=manifest, origin="demo")


def get_demo_source() -> database.DataSource:
    """Streamlit-cached demo data source."""
    st = _streamlit()
    if st is None:  # pragma: no cover
        return load_demo_source()

    @st.cache_resource(show_spinner="Preparing the demo dataset...")
    def _cached() -> database.DataSource:
        return load_demo_source()

    return _cached()


# ---------------------------------------------------------------------------
# Uploaded datasets
# ---------------------------------------------------------------------------


class UploadValidationResult:
    """Outcome of validating a user-supplied file."""

    def __init__(
        self,
        frame: pd.DataFrame | None,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        self.frame = frame
        self.errors = errors
        self.warnings = warnings

    @property
    def ok(self) -> bool:
        return not self.errors and self.frame is not None


def validate_upload(raw: bytes | io.BytesIO | pd.DataFrame) -> UploadValidationResult:
    """Validate an uploaded CSV against the canonical journey schema.

    Returns a result object rather than raising, so the page can show every problem at
    once instead of failing on the first one.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if isinstance(raw, pd.DataFrame):
        frame = raw.copy()
    else:
        try:
            frame = pd.read_csv(raw)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
            return UploadValidationResult(None, [f"The file could not be read as CSV: {exc}"], [])

    if frame.empty:
        return UploadValidationResult(None, ["The uploaded file contains no rows."], [])

    if len(frame) > MAX_UPLOAD_ROWS:
        return UploadValidationResult(
            None,
            [f"The file has {len(frame):,} rows; the demo accepts up to {MAX_UPLOAD_ROWS:,}."],
            [],
        )

    frame.columns = [str(c).strip().lower() for c in frame.columns]

    missing = missing_required_columns(list(frame.columns))
    if missing:
        errors.append(
            "Missing required column(s): " + ", ".join(missing) + ". "
            f"Required columns are: {', '.join(REQUIRED_COLUMNS)}."
        )

    if not errors:
        parsed_dates = pd.to_datetime(frame["date"], errors="coerce")
        if parsed_dates.isna().all():
            errors.append("The 'date' column could not be parsed as dates.")
        elif parsed_dates.isna().any():
            warnings.append(
                f"{int(parsed_dates.isna().sum()):,} row(s) have an unparseable date and "
                "will be excluded from analysis."
            )

        numeric_checks = {
            "delay_minutes": "delay_minutes",
            "customer_satisfaction": "customer_satisfaction",
        }
        for label, column in numeric_checks.items():
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().sum() == 0:
                errors.append(f"The '{label}' column contains no numeric values.")

    absent_optional = [c for c in JOURNEY_COLUMNS if c not in frame.columns]
    if absent_optional and not errors:
        warnings.append(
            f"{len(absent_optional)} optional column(s) are absent and the related charts "
            "will be empty: " + ", ".join(absent_optional[:6])
            + ("..." if len(absent_optional) > 6 else "")
        )

    if errors:
        return UploadValidationResult(None, errors, warnings)
    return UploadValidationResult(frame, [], warnings)


def build_upload_source(frame: pd.DataFrame, name: str = "uploaded file") -> database.DataSource:
    """Load a validated frame into an in-session, in-memory database."""
    conn = database.build_upload_database(frame)
    counts = database.row_counts(conn)
    start, end = database.dataset_bounds(conn)
    manifest = {
        "source_name": name,
        "n_journeys": counts["raw_rows"],
        "analysable_rows": counts["analysable_rows"],
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "quality_degraded": False,
    }
    return database.DataSource(connection=conn, manifest=manifest, origin="upload")


# ---------------------------------------------------------------------------
# Cached querying
# ---------------------------------------------------------------------------


def run_query(
    source: database.DataSource,
    sql: str,
    params: Sequence[Any] | None = None,
) -> pd.DataFrame:
    """Execute SQL against a data source, caching the result where possible."""
    st = _streamlit()
    params = list(params or [])

    # Only the demo source is cacheable: its connection can be resolved from a cache key.
    # Uploaded (and test) sources live for one session only, so they query directly.
    if st is None or source.origin != "demo":
        return database.query(source.connection, sql, params)

    @st.cache_data(show_spinner=False, ttl=1800, max_entries=256)
    def _cached(origin: str, statement: str, bound: tuple[Any, ...]) -> pd.DataFrame:
        return database.query(_connection_for(origin), statement, list(bound))

    return _cached(source.origin, sql, tuple(params))


def _connection_for(origin: str) -> duckdb.DuckDBPyConnection:
    """Resolve a cached connection from its origin key."""
    if origin == "demo":
        return get_demo_source().connection
    raise KeyError(f"Unknown data source origin: {origin}")


def dataset_manifest(source: database.DataSource) -> dict:
    """Manifest describing the active dataset, for display in the sidebar/About page."""
    return dict(source.manifest)
