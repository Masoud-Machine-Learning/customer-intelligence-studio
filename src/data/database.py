"""DuckDB analytical layer.

The application does not treat its data as "a CSV in a DataFrame". Data is loaded into
DuckDB as a small star schema and every filter, aggregation and KPI is expressed in SQL.
Pandas is used afterwards, on the already-aggregated result, for reshaping and chart
preparation.

Two views sit on top of the fact table:

``v_journeys``
    Every row, enriched with dimension attributes and derived time columns. This is what
    the Data Quality page profiles, because quality has to be measured on the raw feed.

``v_journeys_analytics``
    The subset that passes hard validity rules, de-duplicated on ``journey_id``. Every
    analytical query reads this view, so impossible values cannot leak into KPIs.

The difference between the two row counts is itself reported on the Data Quality page.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import duckdb
import pandas as pd

from src.data.generator import GenerationConfig, SyntheticDataset, generate_dataset
from src.data.schema import (
    BOOLEAN_COLUMNS,
    DUCKDB_TYPES,
    JOURNEY_COLUMNS,
    NUMERIC_COLUMNS,
)
from src.utils.config import (
    DB_PATH,
    GENERATED_DIR,
    MANIFEST_PATH,
    MAX_PLAUSIBLE_DELAY_MINUTES,
    MAX_PLAUSIBLE_DURATION_MINUTES,
    ON_TIME_THRESHOLD_MINUTES,
    PEAK_HOURS_AM,
    PEAK_HOURS_PM,
    SAMPLE_CSV_PATH,
    SATISFACTION_MAX,
    SATISFACTION_MIN,
    TRANSPORT_MODES,
)

_BUILD_LOCK = threading.Lock()

FACT_TABLE = "journeys"
CUSTOMER_TABLE = "customers"
SERVICE_LINE_TABLE = "service_lines"
RAW_VIEW = "v_journeys"
ANALYTICS_VIEW = "v_journeys_analytics"


@dataclass
class DataSource:
    """A queryable dataset plus the metadata describing where it came from."""

    connection: duckdb.DuckDBPyConnection
    manifest: dict
    origin: str = "demo"  # "demo" or "upload"

    @property
    def label(self) -> str:
        return "Demo dataset (synthetic)" if self.origin == "demo" else "Uploaded dataset"


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------


def _fact_ddl() -> str:
    columns = ",\n    ".join(f'"{name}" {DUCKDB_TYPES[name]}' for name in JOURNEY_COLUMNS)
    return f"CREATE TABLE {FACT_TABLE} (\n    {columns}\n)"


def _valid_row_predicate() -> str:
    """SQL predicate defining an analytically usable journey row."""
    modes = ", ".join(f"'{mode}'" for mode in TRANSPORT_MODES)
    return f"""
        delay_minutes IS NOT NULL
        AND delay_minutes >= 0
        AND delay_minutes <= {MAX_PLAUSIBLE_DELAY_MINUTES}
        AND (journey_duration_minutes IS NULL
             OR (journey_duration_minutes > 0
                 AND journey_duration_minutes <= {MAX_PLAUSIBLE_DURATION_MINUTES}))
        AND (customer_satisfaction IS NULL
             OR customer_satisfaction BETWEEN {SATISFACTION_MIN} AND {SATISFACTION_MAX})
        AND transport_mode IN ({modes})
        AND journey_id IS NOT NULL
    """


def _create_views(conn: duckdb.DuckDBPyConnection, *, with_dimensions: bool) -> None:
    """Create the enriched raw view and the analytics view."""
    if with_dimensions:
        dimension_select = """,
            c.home_region              AS customer_home_region,
            c.digital_affinity         AS digital_affinity,
            c.first_seen_date          AS customer_first_seen_date,
            s.reliability_index        AS line_reliability_index,
            s.capacity_index           AS line_capacity_index,
            s.scheduled_duration_minutes AS line_scheduled_minutes"""
        dimension_join = f"""
        LEFT JOIN {CUSTOMER_TABLE} c ON c.customer_id = j.customer_id
        LEFT JOIN {SERVICE_LINE_TABLE} s ON s.service_line = j.service_line"""
    else:
        # Uploaded datasets have no dimension tables; expose the columns as NULL so
        # every downstream query keeps working unchanged.
        dimension_select = """,
            CAST(NULL AS VARCHAR) AS customer_home_region,
            CAST(NULL AS DOUBLE)  AS digital_affinity,
            CAST(NULL AS DATE)    AS customer_first_seen_date,
            CAST(NULL AS DOUBLE)  AS line_reliability_index,
            CAST(NULL AS DOUBLE)  AS line_capacity_index,
            CAST(NULL AS DOUBLE)  AS line_scheduled_minutes"""
        dimension_join = ""

    am_start, am_end = PEAK_HOURS_AM
    pm_start, pm_end = PEAK_HOURS_PM

    conn.execute(f"DROP VIEW IF EXISTS {ANALYTICS_VIEW}")
    conn.execute(f"DROP VIEW IF EXISTS {RAW_VIEW}")
    conn.execute(
        f"""
        CREATE VIEW {RAW_VIEW} AS
        SELECT
            j.*,
            CAST(EXTRACT(hour FROM j.timestamp) AS INTEGER)  AS journey_hour,
            CAST(EXTRACT(dow  FROM j.date) AS INTEGER)       AS day_of_week,
            CASE WHEN EXTRACT(dow FROM j.date) IN (0, 6) THEN TRUE ELSE FALSE END
                                                            AS is_weekend,
            CASE
                WHEN EXTRACT(dow FROM j.date) IN (0, 6) THEN FALSE
                WHEN EXTRACT(hour FROM j.timestamp) BETWEEN {am_start} AND {am_end} THEN TRUE
                WHEN EXTRACT(hour FROM j.timestamp) BETWEEN {pm_start} AND {pm_end} THEN TRUE
                ELSE FALSE
            END                                             AS is_peak,
            CASE
                WHEN EXTRACT(dow FROM j.date) IN (0, 6) THEN 'Weekend'
                WHEN EXTRACT(hour FROM j.timestamp) BETWEEN {am_start} AND {am_end} THEN 'AM Peak'
                WHEN EXTRACT(hour FROM j.timestamp) BETWEEN {pm_start} AND {pm_end} THEN 'PM Peak'
                ELSE 'Off-Peak'
            END                                             AS time_band,
            date_trunc('week', j.date)                      AS week_start,
            date_trunc('month', j.date)                     AS month_start{dimension_select}
        FROM {FACT_TABLE} j{dimension_join}
        """
    )
    conn.execute(
        f"""
        CREATE VIEW {ANALYTICS_VIEW} AS
        SELECT * FROM {RAW_VIEW}
        WHERE {_valid_row_predicate()}
        QUALIFY row_number() OVER (
            PARTITION BY journey_id ORDER BY timestamp, date
        ) = 1
        """
    )


def _coerce_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Align an arbitrary journey frame to the canonical schema and dtypes."""
    prepared = frame.copy()
    prepared.columns = [str(c).strip() for c in prepared.columns]
    rename = {c: c.lower() for c in prepared.columns if c != c.lower()}
    if rename:
        prepared = prepared.rename(columns=rename)

    for column in JOURNEY_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = None

    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce").dt.date
    if prepared["timestamp"].isna().all():
        prepared["timestamp"] = pd.to_datetime(prepared["date"], errors="coerce")
    else:
        prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="coerce")

    for column in NUMERIC_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    for column in BOOLEAN_COLUMNS:
        values = prepared[column]
        if values.dtype == object:
            values = values.replace(
                {
                    "true": True, "True": True, "TRUE": True, "1": True, 1: True, "yes": True,
                    "false": False, "False": False, "FALSE": False, "0": False, 0: False,
                    "no": False,
                }
            )
        prepared[column] = values.astype("object").where(values.notna(), None)

    # on_time is derivable and is recomputed when a feed omits it.
    missing_on_time = prepared["on_time"].isna()
    if missing_on_time.any():
        derived = prepared["delay_minutes"] <= ON_TIME_THRESHOLD_MINUTES
        prepared.loc[missing_on_time, "on_time"] = derived[missing_on_time]

    return prepared[list(JOURNEY_COLUMNS)]


def build_database(
    dataset: SyntheticDataset,
    db_path: Path | str | None = DB_PATH,
) -> duckdb.DuckDBPyConnection:
    """Create (or replace) the DuckDB database from a generated dataset."""
    target = ":memory:" if db_path is None else str(db_path)
    if db_path is not None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()

    conn = duckdb.connect(target)
    journeys = _coerce_frame(dataset.journeys)
    customers = dataset.customers
    service_lines = dataset.service_lines

    conn.execute(_fact_ddl())
    conn.register("_journeys_src", journeys)
    conn.execute(f"INSERT INTO {FACT_TABLE} SELECT * FROM _journeys_src")
    conn.unregister("_journeys_src")

    conn.register("_customers_src", customers)
    conn.execute(f"CREATE TABLE {CUSTOMER_TABLE} AS SELECT * FROM _customers_src")
    conn.unregister("_customers_src")

    conn.register("_lines_src", service_lines)
    conn.execute(f"CREATE TABLE {SERVICE_LINE_TABLE} AS SELECT * FROM _lines_src")
    conn.unregister("_lines_src")

    _create_views(conn, with_dimensions=True)
    return conn


def build_upload_database(journeys: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    """Load a user-supplied journey frame into an in-memory database.

    Uploaded data is never written to disk: the connection lives only for the Streamlit
    session, which keeps user data inside the session as promised on the upload screen.
    """
    conn = duckdb.connect(":memory:")
    prepared = _coerce_frame(journeys)
    conn.execute(_fact_ddl())
    conn.register("_journeys_src", prepared)
    conn.execute(f"INSERT INTO {FACT_TABLE} SELECT * FROM _journeys_src")
    conn.unregister("_journeys_src")
    _create_views(conn, with_dimensions=False)
    return conn


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def query(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    params: Sequence[Any] | None = None,
) -> pd.DataFrame:
    """Run a parameterised SQL statement and return a DataFrame.

    A cursor is used per call so concurrent Streamlit reruns cannot interleave on the
    same connection object.
    """
    cursor = conn.cursor()
    try:
        result = cursor.execute(sql, list(params) if params else [])
        return result.fetch_df()
    finally:
        cursor.close()


def scalar(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    params: Sequence[Any] | None = None,
) -> Any:
    """Run a query expected to return a single value."""
    frame = query(conn, sql, params)
    if frame.empty:
        return None
    return frame.iloc[0, 0]


def dataset_bounds(conn: duckdb.DuckDBPyConnection) -> tuple[date, date]:
    """Earliest and latest journey date available for analysis."""
    frame = query(
        conn,
        f"SELECT min(date) AS min_date, max(date) AS max_date FROM {ANALYTICS_VIEW}",
    )
    min_date = frame.loc[0, "min_date"]
    max_date = frame.loc[0, "max_date"]
    if pd.isna(min_date) or pd.isna(max_date):
        today = date.today()
        return today, today
    return pd.Timestamp(min_date).date(), pd.Timestamp(max_date).date()


def distinct_values(conn: duckdb.DuckDBPyConnection, column: str) -> list[str]:
    """Sorted distinct non-null values for a dimension column (filter options)."""
    if column not in set(JOURNEY_COLUMNS) | {"time_band"}:
        raise ValueError(f"Unsupported filter column: {column}")
    frame = query(
        conn,
        f"""
        SELECT DISTINCT {column} AS value
        FROM {ANALYTICS_VIEW}
        WHERE {column} IS NOT NULL
        ORDER BY 1
        """,
    )
    return [str(v) for v in frame["value"].tolist()]


def row_counts(conn: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Raw versus analysable row counts - the headline of the Data Quality page."""
    raw = scalar(conn, f"SELECT count(*) FROM {RAW_VIEW}")
    usable = scalar(conn, f"SELECT count(*) FROM {ANALYTICS_VIEW}")
    return {"raw_rows": int(raw or 0), "analysable_rows": int(usable or 0)}


# ---------------------------------------------------------------------------
# Demo dataset lifecycle
# ---------------------------------------------------------------------------


def _write_manifest(manifest: dict) -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def read_manifest() -> dict | None:
    """Manifest of the database currently on disk, if any."""
    if not MANIFEST_PATH.exists():
        return None
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def ensure_demo_database(
    config: GenerationConfig | None = None, force: bool = False
) -> dict:
    """Generate and load the demo dataset if it is missing or out of date.

    The manifest fingerprint covers the seed, size and date range, so changing any of
    them triggers a rebuild while an unchanged configuration reuses the existing file.
    """
    config = config or GenerationConfig()
    with _BUILD_LOCK:
        existing = read_manifest()
        if (
            not force
            and existing
            and existing.get("fingerprint") == config.fingerprint()
            and DB_PATH.exists()
        ):
            return existing

        dataset = generate_dataset(config)
        conn = build_database(dataset, DB_PATH)
        try:
            # A small CSV template so users can see the expected upload shape.
            sample = dataset.journeys.head(300)
            SAMPLE_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
            sample.to_csv(SAMPLE_CSV_PATH, index=False)
        finally:
            conn.close()

        _write_manifest(dataset.manifest)
        return dataset.manifest


def open_demo_database() -> duckdb.DuckDBPyConnection:
    """Open the on-disk demo database read-only, generating it first if needed."""
    ensure_demo_database()
    return duckdb.connect(str(DB_PATH), read_only=True)
