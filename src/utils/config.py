"""Central configuration for Customer Intelligence Studio.

Every tunable value (paths, domain vocabulary, thresholds, palette) lives here so that
no other module has to hard-code magic strings or numbers. Import from this module
rather than redefining constants locally.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Paths - all relative to the project folder so the app deploys standalone.
# ---------------------------------------------------------------------------

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
GENERATED_DIR: Final[Path] = DATA_DIR / "generated"
DB_PATH: Final[Path] = GENERATED_DIR / "customer_intelligence.duckdb"
MANIFEST_PATH: Final[Path] = GENERATED_DIR / "manifest.json"
SAMPLE_CSV_PATH: Final[Path] = GENERATED_DIR / "sample_upload_template.csv"

# Load a local .env if one exists, for development convenience. Real environment
# variables always win, which is what deployment platforms rely on.
try:  # pragma: no cover - depends on an optional file being present
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ModuleNotFoundError:  # python-dotenv is optional
    pass

# ---------------------------------------------------------------------------
# Application identity
# ---------------------------------------------------------------------------

APP_NAME: Final[str] = "Customer Intelligence Studio"
APP_SUBTITLE: Final[str] = (
    "Interactive Customer & Mobility Analytics with AI-Assisted Decision Support"
)
APP_TAGLINE: Final[str] = (
    "Explore customer behaviour, service performance and emerging issues through "
    "interactive analytics and AI-assisted insights."
)
DISCLAIMER: Final[str] = (
    "Portfolio demonstration using synthetic data. Not affiliated with Transport for NSW."
)

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

RANDOM_SEED: Final[int] = 4257
DEFAULT_N_JOURNEYS: Final[int] = 60_000
DEFAULT_N_CUSTOMERS: Final[int] = 9_000
DEFAULT_HISTORY_DAYS: Final[int] = 540  # ~18 months of daily history

# Share of rows deliberately degraded so the Data Quality page has real work to do.
MISSING_SATISFACTION_RATE: Final[float] = 0.045
MISSING_REGION_RATE: Final[float] = 0.004
MISSING_CUSTOMER_ID_RATE: Final[float] = 0.002
DUPLICATE_ROW_RATE: Final[float] = 0.0015
INVALID_ROW_RATE: Final[float] = 0.0018

# ---------------------------------------------------------------------------
# Domain vocabulary - synthetic and deliberately generic (no operator branding).
# ---------------------------------------------------------------------------

REGIONS: Final[tuple[str, ...]] = (
    "Central",
    "North",
    "South",
    "East",
    "West",
    "Northwest",
)

TRANSPORT_MODES: Final[tuple[str, ...]] = (
    "Metro",
    "Train",
    "Bus",
    "Light Rail",
    "Ferry",
)

CUSTOMER_SEGMENTS: Final[tuple[str, ...]] = (
    "Frequent Commuters",
    "Occasional Travellers",
    "Off-Peak Regulars",
    "Digital-First Customers",
    "High-Service-Need Customers",
)

AGE_GROUPS: Final[tuple[str, ...]] = ("16-24", "25-34", "35-49", "50-64", "65+")

CHANNELS: Final[tuple[str, ...]] = (
    "Mobile App",
    "Website",
    "Contact Centre",
    "Station Kiosk",
    "Third-Party App",
)

JOURNEY_PURPOSES: Final[tuple[str, ...]] = (
    "Commute",
    "Education",
    "Business",
    "Leisure",
    "Healthcare",
    "Shopping",
    "Other",
)

FARE_TYPES: Final[tuple[str, ...]] = (
    "Adult",
    "Concession",
    "Senior",
    "Student",
    "Visitor",
)

COMPLAINT_CATEGORIES: Final[tuple[str, ...]] = (
    "Delay",
    "Crowding",
    "Cleanliness",
    "Staff Conduct",
    "Accessibility",
    "Information & Wayfinding",
    "Ticketing & Fares",
)

FEEDBACK_CATEGORIES: Final[tuple[str, ...]] = (
    "Positive - Reliability",
    "Positive - Staff",
    "Neutral",
    "Negative - Delay",
    "Negative - Crowding",
    "Negative - Information",
    "Negative - Cleanliness",
)

# ---------------------------------------------------------------------------
# Analytical thresholds
# ---------------------------------------------------------------------------

ON_TIME_THRESHOLD_MINUTES: Final[float] = 5.0
PEAK_HOURS_AM: Final[tuple[int, int]] = (7, 9)
PEAK_HOURS_PM: Final[tuple[int, int]] = (16, 18)
SATISFACTION_MIN: Final[int] = 1
SATISFACTION_MAX: Final[int] = 5
MAX_PLAUSIBLE_DURATION_MINUTES: Final[float] = 240.0
MAX_PLAUSIBLE_DELAY_MINUTES: Final[float] = 180.0

ANOMALY_ROLLING_WINDOW: Final[int] = 28
ANOMALY_Z_THRESHOLD: Final[float] = 2.5
ANOMALY_MIN_HISTORY: Final[int] = 14

# Minimum rows before a comparison is reported as meaningful rather than noise.
MIN_SAMPLE_FOR_COMPARISON: Final[int] = 100
MATERIAL_CHANGE_PCT: Final[float] = 3.0

# ---------------------------------------------------------------------------
# Visual identity - restrained enterprise analytics palette (no gradients/neon).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Palette:
    """Colour tokens shared by the Streamlit CSS and every Plotly chart."""

    ink: str = "#16212b"
    body: str = "#33424f"
    muted: str = "#6a7986"
    line: str = "#dfe4e9"
    surface: str = "#ffffff"
    canvas: str = "#f5f7f9"
    accent: str = "#1f6f8b"
    accent_soft: str = "#e3eef2"
    positive: str = "#2f7d55"
    negative: str = "#a63a35"
    caution: str = "#9a6b1e"
    categorical: tuple[str, ...] = field(
        default_factory=lambda: (
            "#1f6f8b",
            "#7a8f9c",
            "#2f7d55",
            "#9a6b1e",
            "#6b5b95",
            "#a63a35",
            "#4c8ca3",
            "#57697a",
        )
    )
    sequential: tuple[str, ...] = field(
        default_factory=lambda: (
            "#eef3f6",
            "#cfdde4",
            "#a9c6d2",
            "#7eaabc",
            "#4f8ba3",
            "#1f6f8b",
        )
    )


PALETTE: Final[Palette] = Palette()

# Metrics where a decrease is the good outcome (drives delta colour coding).
LOWER_IS_BETTER: Final[frozenset[str]] = frozenset(
    {
        "avg_delay_minutes",
        "complaint_rate",
        "avg_resolution_hours",
        "disruption_rate",
        "p90_delay_minutes",
    }
)


def openai_api_key() -> str | None:
    """Return the configured OpenAI key, or ``None`` when AI features are disabled.

    Resolution order: Streamlit secrets (used on Streamlit Community Cloud), then the
    ``OPENAI_API_KEY`` environment variable. Never raises when secrets are unavailable.
    """
    try:  # pragma: no cover - depends on the Streamlit runtime being present
        import streamlit as st

        secret = st.secrets.get("OPENAI_API_KEY")
        if secret:
            return str(secret).strip() or None
    except Exception:
        pass

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    return key or None


def openai_model() -> str:
    """Model used for the optional natural-language wording layer."""
    return os.environ.get("CIS_OPENAI_MODEL", "gpt-4o-mini")
