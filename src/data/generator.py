"""Deterministic synthetic customer and mobility dataset generator.

The generator builds a small star schema:

* ``customers``      - one row per synthetic customer (behavioural attributes)
* ``service_lines``  - one row per synthetic service line (operational attributes)
* ``journeys``       - the fact table, one row per journey

Relationships are causal rather than random. Delay is driven by line reliability, peak
hour, disruption and injected anomaly windows; satisfaction is driven by delay,
disruption and complaint handling; complaints are driven by delay, disruption and
segment. Noise, missing values, duplicates and invalid rows are added last so that the
Data Quality page has genuine defects to find.

Everything is seeded: the same configuration always produces byte-identical frames.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Final

import numpy as np
import pandas as pd

from src.utils.config import (
    AGE_GROUPS,
    CHANNELS,
    COMPLAINT_CATEGORIES,
    CUSTOMER_SEGMENTS,
    DEFAULT_HISTORY_DAYS,
    DEFAULT_N_CUSTOMERS,
    DEFAULT_N_JOURNEYS,
    DUPLICATE_ROW_RATE,
    FARE_TYPES,
    INVALID_ROW_RATE,
    JOURNEY_PURPOSES,
    MAX_PLAUSIBLE_DELAY_MINUTES,
    MISSING_CUSTOMER_ID_RATE,
    MISSING_REGION_RATE,
    MISSING_SATISFACTION_RATE,
    ON_TIME_THRESHOLD_MINUTES,
    RANDOM_SEED,
    REGIONS,
    TRANSPORT_MODES,
)
from src.data.schema import JOURNEY_COLUMNS

# Lines are laid out mode-major then region-major so a journey's line index can be
# derived arithmetically instead of with a per-row lookup.
LINES_PER_MODE: Final[dict[str, int]] = {
    "Metro": 1,
    "Train": 2,
    "Bus": 3,
    "Light Rail": 1,
    "Ferry": 1,
}
LINE_PREFIX: Final[dict[str, str]] = {
    "Metro": "MET",
    "Train": "TRN",
    "Bus": "BUS",
    "Light Rail": "LRT",
    "Ferry": "FRY",
}
MODE_SCHEDULED_MINUTES: Final[dict[str, float]] = {
    "Metro": 24.0,
    "Train": 38.0,
    "Bus": 31.0,
    "Light Rail": 21.0,
    "Ferry": 29.0,
}
MODE_DELAY_PRESSURE: Final[dict[str, float]] = {
    "Metro": 1.15,
    "Train": 1.95,
    "Bus": 2.35,
    "Light Rail": 1.55,
    "Ferry": 1.70,
}
# Global scale factor tuned so network on-time performance sits near 90%, which is a
# realistic headline figure for a mixed urban network.
DELAY_CALIBRATION: Final[float] = 0.80

SEGMENT_SHARE: Final[dict[str, float]] = {
    "Frequent Commuters": 0.22,
    "Occasional Travellers": 0.34,
    "Off-Peak Regulars": 0.18,
    "Digital-First Customers": 0.16,
    "High-Service-Need Customers": 0.10,
}
# Relative likelihood of a journey belonging to each segment, by day type.
SEGMENT_WEEKDAY_ACTIVITY: Final[dict[str, float]] = {
    "Frequent Commuters": 3.60,
    "Occasional Travellers": 0.70,
    "Off-Peak Regulars": 1.40,
    "Digital-First Customers": 2.10,
    "High-Service-Need Customers": 0.95,
}
SEGMENT_WEEKEND_ACTIVITY: Final[dict[str, float]] = {
    "Frequent Commuters": 0.85,
    "Occasional Travellers": 2.20,
    "Off-Peak Regulars": 1.55,
    "Digital-First Customers": 1.30,
    "High-Service-Need Customers": 1.05,
}

REGION_POPULATION_WEIGHT: Final[dict[str, float]] = {
    "Central": 0.26,
    "North": 0.16,
    "South": 0.17,
    "East": 0.14,
    "West": 0.18,
    "Northwest": 0.09,
}
REGION_DELAY_MULTIPLIER: Final[dict[str, float]] = {
    "Central": 1.12,
    "North": 0.94,
    "South": 1.02,
    "East": 0.97,
    "West": 1.18,
    "Northwest": 0.90,
}

# Region x mode affinity (rows are normalised at sampling time).
REGION_MODE_AFFINITY: Final[dict[str, dict[str, float]]] = {
    "Central": {"Metro": 0.28, "Train": 0.30, "Bus": 0.22, "Light Rail": 0.14, "Ferry": 0.06},
    "North": {"Metro": 0.18, "Train": 0.30, "Bus": 0.34, "Light Rail": 0.06, "Ferry": 0.12},
    "South": {"Metro": 0.10, "Train": 0.34, "Bus": 0.46, "Light Rail": 0.06, "Ferry": 0.04},
    "East": {"Metro": 0.12, "Train": 0.22, "Bus": 0.44, "Light Rail": 0.10, "Ferry": 0.12},
    "West": {"Metro": 0.22, "Train": 0.34, "Bus": 0.40, "Light Rail": 0.03, "Ferry": 0.01},
    "Northwest": {"Metro": 0.34, "Train": 0.26, "Bus": 0.36, "Light Rail": 0.03, "Ferry": 0.01},
}
SEGMENT_MODE_BIAS: Final[dict[str, dict[str, float]]] = {
    "Frequent Commuters": {"Metro": 1.25, "Train": 1.20, "Bus": 0.85, "Light Rail": 1.0, "Ferry": 0.8},
    "Occasional Travellers": {"Metro": 0.9, "Train": 0.95, "Bus": 1.1, "Light Rail": 1.15, "Ferry": 1.4},
    "Off-Peak Regulars": {"Metro": 0.95, "Train": 0.9, "Bus": 1.25, "Light Rail": 1.05, "Ferry": 1.0},
    "Digital-First Customers": {"Metro": 1.45, "Train": 1.05, "Bus": 0.8, "Light Rail": 1.1, "Ferry": 0.95},
    "High-Service-Need Customers": {"Metro": 0.7, "Train": 0.85, "Bus": 1.5, "Light Rail": 0.9, "Ferry": 0.7},
}

SEGMENT_CHANNEL_PROFILE: Final[dict[str, dict[str, float]]] = {
    "Frequent Commuters": {"Mobile App": 0.52, "Website": 0.16, "Contact Centre": 0.06, "Station Kiosk": 0.14, "Third-Party App": 0.12},
    "Occasional Travellers": {"Mobile App": 0.28, "Website": 0.22, "Contact Centre": 0.12, "Station Kiosk": 0.28, "Third-Party App": 0.10},
    "Off-Peak Regulars": {"Mobile App": 0.34, "Website": 0.20, "Contact Centre": 0.11, "Station Kiosk": 0.26, "Third-Party App": 0.09},
    "Digital-First Customers": {"Mobile App": 0.68, "Website": 0.18, "Contact Centre": 0.02, "Station Kiosk": 0.03, "Third-Party App": 0.09},
    "High-Service-Need Customers": {"Mobile App": 0.18, "Website": 0.14, "Contact Centre": 0.38, "Station Kiosk": 0.26, "Third-Party App": 0.04},
}

SEGMENT_AGE_PROFILE: Final[dict[str, dict[str, float]]] = {
    "Frequent Commuters": {"16-24": 0.14, "25-34": 0.34, "35-49": 0.32, "50-64": 0.17, "65+": 0.03},
    "Occasional Travellers": {"16-24": 0.18, "25-34": 0.24, "35-49": 0.25, "50-64": 0.20, "65+": 0.13},
    "Off-Peak Regulars": {"16-24": 0.10, "25-34": 0.15, "35-49": 0.22, "50-64": 0.28, "65+": 0.25},
    "Digital-First Customers": {"16-24": 0.30, "25-34": 0.40, "35-49": 0.22, "50-64": 0.07, "65+": 0.01},
    "High-Service-Need Customers": {"16-24": 0.08, "25-34": 0.10, "35-49": 0.16, "50-64": 0.26, "65+": 0.40},
}

AGE_FARE_PROFILE: Final[dict[str, dict[str, float]]] = {
    "16-24": {"Adult": 0.25, "Concession": 0.15, "Senior": 0.0, "Student": 0.55, "Visitor": 0.05},
    "25-34": {"Adult": 0.80, "Concession": 0.08, "Senior": 0.0, "Student": 0.06, "Visitor": 0.06},
    "35-49": {"Adult": 0.86, "Concession": 0.07, "Senior": 0.0, "Student": 0.01, "Visitor": 0.06},
    "50-64": {"Adult": 0.74, "Concession": 0.18, "Senior": 0.02, "Student": 0.0, "Visitor": 0.06},
    "65+": {"Adult": 0.10, "Concession": 0.16, "Senior": 0.70, "Student": 0.0, "Visitor": 0.04},
}

# Satisfaction offsets applied per segment (expectations differ by customer type).
SEGMENT_SATISFACTION_OFFSET: Final[dict[str, float]] = {
    "Frequent Commuters": -0.18,
    "Occasional Travellers": 0.12,
    "Off-Peak Regulars": 0.20,
    "Digital-First Customers": -0.05,
    "High-Service-Need Customers": -0.30,
}
SEGMENT_COMPLAINT_OFFSET: Final[dict[str, float]] = {
    "Frequent Commuters": 0.25,
    "Occasional Travellers": -0.20,
    "Off-Peak Regulars": -0.30,
    "Digital-First Customers": 0.10,
    "High-Service-Need Customers": 0.55,
}


@dataclass(frozen=True)
class AnomalyWindow:
    """A deliberately injected, detectable deviation from baseline behaviour."""

    name: str
    kind: str  # delay | demand | complaints | improvement
    dimension: str  # region | transport_mode
    value: str
    start_offset: int  # days before the final date (inclusive, larger = older)
    end_offset: int
    magnitude: float
    note: str

    def window(self, end_date: date) -> tuple[date, date]:
        """Resolve the window to absolute dates for a dataset ending on ``end_date``."""
        return (
            end_date - timedelta(days=self.start_offset),
            end_date - timedelta(days=self.end_offset),
        )


KNOWN_ANOMALIES: Final[tuple[AnomalyWindow, ...]] = (
    AnomalyWindow(
        name="West corridor delay spike",
        kind="delay",
        dimension="region",
        value="West",
        start_offset=52,
        end_offset=41,
        magnitude=2.30,
        note="Sustained delay elevation on West region services.",
    ),
    AnomalyWindow(
        name="Central major event demand surge",
        kind="demand",
        dimension="region",
        value="Central",
        start_offset=118,
        end_offset=116,
        magnitude=3.40,
        note="Three-day demand spike concentrated in the Central region.",
    ),
    AnomalyWindow(
        name="Bus complaint surge",
        kind="complaints",
        dimension="transport_mode",
        value="Bus",
        start_offset=31,
        end_offset=18,
        magnitude=1.15,
        note="Elevated complaint rate on bus services with lower satisfaction.",
    ),
    AnomalyWindow(
        name="North reliability improvement",
        kind="improvement",
        dimension="region",
        value="North",
        start_offset=70,
        end_offset=1,
        magnitude=0.68,
        note="Timetable change reduced delays on North region services.",
    ),
)


@dataclass(frozen=True)
class GenerationConfig:
    """Inputs that fully determine the generated dataset."""

    seed: int = RANDOM_SEED
    n_journeys: int = DEFAULT_N_JOURNEYS
    n_customers: int = DEFAULT_N_CUSTOMERS
    history_days: int = DEFAULT_HISTORY_DAYS
    end_date: date | None = None
    degrade_quality: bool = True

    def resolved_end_date(self) -> date:
        """End date of the generated history (defaults to today)."""
        return self.end_date or date.today()

    def start_date(self) -> date:
        return self.resolved_end_date() - timedelta(days=self.history_days - 1)

    def fingerprint(self) -> str:
        """Stable hash of the configuration, used to decide whether to regenerate."""
        payload = asdict(self)
        payload["end_date"] = self.resolved_end_date().isoformat()
        raw = "|".join(f"{k}={payload[k]}" for k in sorted(payload))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class SyntheticDataset:
    """Generated star schema plus a manifest describing how it was produced."""

    journeys: pd.DataFrame
    customers: pd.DataFrame
    service_lines: pd.DataFrame
    manifest: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Vectorised sampling helpers
# ---------------------------------------------------------------------------


def _sample_categorical(rng: np.random.Generator, probabilities: np.ndarray) -> np.ndarray:
    """Row-wise categorical sampling from an ``(n, k)`` probability matrix.

    Uses inverse-CDF sampling so 60k draws stay a single vectorised operation.
    """
    normalised = probabilities / probabilities.sum(axis=1, keepdims=True)
    cumulative = normalised.cumsum(axis=1)
    draws = rng.random((normalised.shape[0], 1))
    return (draws > cumulative).sum(axis=1).clip(0, normalised.shape[1] - 1)


def _profile_matrix(profile: dict[str, dict[str, float]], keys: tuple[str, ...],
                    columns: tuple[str, ...]) -> np.ndarray:
    """Turn a nested ``{key: {column: weight}}`` mapping into a dense matrix."""
    return np.array([[profile[key][col] for col in columns] for key in keys], dtype=float)


def _hour_profiles() -> tuple[np.ndarray, np.ndarray]:
    """Hour-of-day probability profiles per segment for weekdays and weekends."""
    hours = np.arange(24)

    def bell(centre: float, width: float, height: float) -> np.ndarray:
        return height * np.exp(-0.5 * ((hours - centre) / width) ** 2)

    weekday = {
        "Frequent Commuters": bell(8.0, 1.05, 1.0) + bell(17.3, 1.15, 0.95) + 0.03,
        "Occasional Travellers": bell(11.5, 3.3, 0.6) + bell(17.5, 2.2, 0.45) + 0.05,
        "Off-Peak Regulars": bell(11.0, 2.1, 0.9) + bell(14.5, 2.0, 0.7) + 0.03,
        "Digital-First Customers": bell(8.5, 1.6, 0.85) + bell(17.8, 1.8, 0.9) + 0.06,
        "High-Service-Need Customers": bell(10.5, 2.2, 0.95) + bell(14.0, 2.0, 0.6) + 0.03,
    }
    weekend = {
        "Frequent Commuters": bell(11.0, 3.0, 0.7) + bell(18.0, 2.4, 0.5) + 0.05,
        "Occasional Travellers": bell(12.5, 3.4, 0.9) + bell(19.0, 2.6, 0.55) + 0.06,
        "Off-Peak Regulars": bell(11.5, 3.0, 0.9) + 0.05,
        "Digital-First Customers": bell(13.0, 3.6, 0.8) + bell(20.0, 2.4, 0.5) + 0.06,
        "High-Service-Need Customers": bell(11.0, 2.8, 0.85) + 0.04,
    }
    # Services do not run overnight in this synthetic network.
    mask = np.where((hours >= 5) & (hours <= 23), 1.0, 0.02)
    weekday_matrix = np.array([weekday[s] * mask for s in CUSTOMER_SEGMENTS])
    weekend_matrix = np.array([weekend[s] * mask for s in CUSTOMER_SEGMENTS])
    return weekday_matrix, weekend_matrix


def _build_service_lines(rng: np.random.Generator) -> pd.DataFrame:
    """Create the service line dimension, ordered mode-major then region-major."""
    records: list[dict[str, object]] = []
    for mode in TRANSPORT_MODES:
        per_region = LINES_PER_MODE[mode]
        for region_index, region in enumerate(REGIONS):
            for slot in range(per_region):
                sequence = region_index * per_region + slot + 1
                records.append(
                    {
                        "service_line": f"{LINE_PREFIX[mode]}-{sequence:02d}",
                        "transport_mode": mode,
                        "region": region,
                        "scheduled_duration_minutes": round(
                            float(MODE_SCHEDULED_MINUTES[mode] * rng.uniform(0.78, 1.28)), 1
                        ),
                        "reliability_index": round(float(rng.uniform(0.72, 1.24)), 3),
                        "capacity_index": round(float(rng.uniform(0.65, 1.35)), 3),
                    }
                )
    return pd.DataFrame.from_records(records)


def _build_customers(rng: np.random.Generator, n_customers: int, start: date) -> pd.DataFrame:
    """Create the customer dimension with behavioural attributes."""
    segment_probs = np.array([SEGMENT_SHARE[s] for s in CUSTOMER_SEGMENTS])
    segment_idx = rng.choice(len(CUSTOMER_SEGMENTS), size=n_customers, p=segment_probs / segment_probs.sum())

    age_matrix = _profile_matrix(SEGMENT_AGE_PROFILE, CUSTOMER_SEGMENTS, AGE_GROUPS)
    age_idx = _sample_categorical(rng, age_matrix[segment_idx])

    fare_matrix = _profile_matrix(AGE_FARE_PROFILE, AGE_GROUPS, FARE_TYPES)
    fare_idx = _sample_categorical(rng, fare_matrix[age_idx] + 1e-9)

    region_probs = np.array([REGION_POPULATION_WEIGHT[r] for r in REGIONS])
    home_idx = rng.choice(len(REGIONS), size=n_customers, p=region_probs / region_probs.sum())

    # Journey volume per customer is heavy-tailed: a minority travel constantly.
    base_activity = np.array([3.5, 0.8, 1.6, 2.2, 1.1])[segment_idx]
    activity_weight = base_activity * rng.lognormal(mean=0.0, sigma=0.55, size=n_customers)

    digital_affinity = np.clip(
        np.array([0.62, 0.45, 0.40, 0.90, 0.28])[segment_idx] + rng.normal(0, 0.11, n_customers),
        0.02,
        0.99,
    )

    repeat_base = np.array([0.88, 0.24, 0.72, 0.80, 0.55])[segment_idx]
    repeat_customer = rng.random(n_customers) < repeat_base

    first_seen = pd.to_datetime(start) + pd.to_timedelta(
        rng.integers(0, 400, size=n_customers), unit="D"
    )

    return pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(1, n_customers + 1)],
            "customer_segment": np.array(CUSTOMER_SEGMENTS)[segment_idx],
            "age_group": np.array(AGE_GROUPS)[age_idx],
            "home_region": np.array(REGIONS)[home_idx],
            "fare_type": np.array(FARE_TYPES)[fare_idx],
            "repeat_customer": repeat_customer,
            "digital_affinity": np.round(digital_affinity, 3),
            "first_seen_date": first_seen.date,
            "_segment_idx": segment_idx,
            "_home_idx": home_idx,
            "_activity": activity_weight,
        }
    )


def _daily_demand_weights(dates: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Daily journey volume weights combining weekly, seasonal and event effects."""
    day_of_week = dates.dayofweek.to_numpy()
    weekday_factor = np.select(
        [day_of_week <= 3, day_of_week == 4, day_of_week == 5, day_of_week == 6],
        [1.00, 1.06, 0.63, 0.49],
        default=1.0,
    )

    day_of_year = dates.dayofyear.to_numpy()
    seasonal = 1.0 + 0.11 * np.sin(2 * np.pi * (day_of_year - 40) / 365.25)

    # Late December / early January holiday trough.
    month = dates.month.to_numpy()
    day = dates.day.to_numpy()
    holiday = np.where(((month == 12) & (day >= 20)) | ((month == 1) & (day <= 12)), 0.63, 1.0)

    # Gentle patronage growth across the observed history.
    trend = np.linspace(0.94, 1.08, len(dates))

    noise = rng.normal(1.0, 0.045, len(dates))

    weights = weekday_factor * seasonal * holiday * trend * noise

    # Demand anomalies operate at the day level.
    end_date = dates[-1].date()
    for anomaly in KNOWN_ANOMALIES:
        if anomaly.kind != "demand":
            continue
        start, stop = anomaly.window(end_date)
        mask = (dates.date >= start) & (dates.date <= stop)
        weights = np.where(mask, weights * 1.55, weights)

    return np.clip(weights, 0.05, None)


def _anomaly_masks(
    journey_dates: np.ndarray,
    regions: np.ndarray,
    modes: np.ndarray,
    end_date: date,
) -> dict[str, np.ndarray]:
    """Boolean masks marking journeys inside each injected anomaly window."""
    masks: dict[str, np.ndarray] = {}
    for anomaly in KNOWN_ANOMALIES:
        start, stop = anomaly.window(end_date)
        in_window = (journey_dates >= np.datetime64(start)) & (journey_dates <= np.datetime64(stop))
        dimension = regions if anomaly.dimension == "region" else modes
        masks[anomaly.name] = in_window & (dimension == anomaly.value)
    return masks


def _apply_quality_degradation(
    journeys: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Introduce realistic defects: nulls, duplicates and impossible values."""
    n = len(journeys)

    missing_sat = rng.random(n) < MISSING_SATISFACTION_RATE
    journeys.loc[missing_sat, "customer_satisfaction"] = np.nan

    missing_region = rng.random(n) < MISSING_REGION_RATE
    journeys.loc[missing_region, "region"] = None

    missing_customer = rng.random(n) < MISSING_CUSTOMER_ID_RATE
    journeys.loc[missing_customer, "customer_id"] = None

    # Impossible values that the data quality rules must catch.
    invalid = rng.random(n) < INVALID_ROW_RATE
    invalid_idx = np.flatnonzero(invalid)
    if len(invalid_idx):
        thirds = np.array_split(invalid_idx, 4)
        journeys.loc[journeys.index[thirds[0]], "journey_duration_minutes"] = -np.abs(
            rng.normal(12, 4, len(thirds[0]))
        ).round(1)
        journeys.loc[journeys.index[thirds[1]], "customer_satisfaction"] = rng.choice(
            [0.0, 7.0], size=len(thirds[1])
        )
        journeys.loc[journeys.index[thirds[2]], "delay_minutes"] = 999.0
        journeys.loc[journeys.index[thirds[3]], "transport_mode"] = "UNKNOWN"

    # Duplicated rows (same journey_id emitted twice by an upstream feed).
    n_duplicates = int(round(n * DUPLICATE_ROW_RATE))
    if n_duplicates > 0:
        dup_idx = rng.choice(n, size=n_duplicates, replace=False)
        duplicates = journeys.iloc[dup_idx].copy()
        journeys = pd.concat([journeys, duplicates], ignore_index=True)

    return journeys


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_dataset(config: GenerationConfig | None = None) -> SyntheticDataset:
    """Generate the full synthetic star schema for ``config``.

    The function is pure: identical configuration in, identical frames out.
    """
    config = config or GenerationConfig()
    rng = np.random.default_rng(config.seed)

    end_date = config.resolved_end_date()
    start_date = config.start_date()
    dates = pd.date_range(start_date, end_date, freq="D")

    service_lines = _build_service_lines(rng)
    customers = _build_customers(rng, config.n_customers, start_date)

    # --- 1. How many journeys happen on each day -------------------------------------
    day_weights = _daily_demand_weights(dates, rng)
    per_day = rng.multinomial(config.n_journeys, day_weights / day_weights.sum())
    day_index = np.repeat(np.arange(len(dates)), per_day)
    journey_dates = dates.to_numpy()[day_index]
    n = len(day_index)

    day_of_week = dates.dayofweek.to_numpy()[day_index]
    is_weekend = day_of_week >= 5

    # --- 2. Who travels ---------------------------------------------------------------
    weekday_seg = np.array([SEGMENT_WEEKDAY_ACTIVITY[s] * SEGMENT_SHARE[s] for s in CUSTOMER_SEGMENTS])
    weekend_seg = np.array([SEGMENT_WEEKEND_ACTIVITY[s] * SEGMENT_SHARE[s] for s in CUSTOMER_SEGMENTS])
    segment_probs = np.where(is_weekend[:, None], weekend_seg, weekday_seg)
    segment_idx = _sample_categorical(rng, segment_probs)

    customer_row = np.empty(n, dtype=np.int64)
    cust_segment_idx = customers["_segment_idx"].to_numpy()
    activity = customers["_activity"].to_numpy()
    for seg in range(len(CUSTOMER_SEGMENTS)):
        journey_positions = np.flatnonzero(segment_idx == seg)
        if journey_positions.size == 0:
            continue
        candidates = np.flatnonzero(cust_segment_idx == seg)
        weights = activity[candidates]
        customer_row[journey_positions] = rng.choice(
            candidates, size=journey_positions.size, p=weights / weights.sum()
        )

    home_idx = customers["_home_idx"].to_numpy()[customer_row]
    age_idx = np.searchsorted(np.array(AGE_GROUPS), customers["age_group"].to_numpy()[customer_row])
    digital_affinity = customers["digital_affinity"].to_numpy()[customer_row]
    repeat_customer = customers["repeat_customer"].to_numpy()[customer_row]

    # --- 3. Where and how they travel -------------------------------------------------
    # Most journeys start in the customer's home region; the rest are spread elsewhere.
    stays_home = rng.random(n) < 0.76
    random_region = rng.integers(0, len(REGIONS), size=n)
    region_idx = np.where(stays_home, home_idx, random_region)

    # A Central demand surge pulls journeys into Central on the event days.
    for anomaly in KNOWN_ANOMALIES:
        if anomaly.kind != "demand":
            continue
        start, stop = anomaly.window(end_date)
        in_window = (journey_dates >= np.datetime64(start)) & (journey_dates <= np.datetime64(stop))
        pull = in_window & (rng.random(n) < 0.45)
        region_idx = np.where(pull, REGIONS.index(anomaly.value), region_idx)

    mode_matrix = _profile_matrix(REGION_MODE_AFFINITY, REGIONS, TRANSPORT_MODES)
    mode_bias = _profile_matrix(SEGMENT_MODE_BIAS, CUSTOMER_SEGMENTS, TRANSPORT_MODES)
    mode_idx = _sample_categorical(rng, mode_matrix[region_idx] * mode_bias[segment_idx])

    modes = np.array(TRANSPORT_MODES)[mode_idx]
    regions = np.array(REGIONS)[region_idx]

    # Service line index: modes are laid out contiguously in the dimension table.
    mode_offsets = np.cumsum([0] + [LINES_PER_MODE[m] * len(REGIONS) for m in TRANSPORT_MODES[:-1]])
    per_region_counts = np.array([LINES_PER_MODE[m] for m in TRANSPORT_MODES])
    slot = (rng.random(n) * per_region_counts[mode_idx]).astype(int)
    line_idx = mode_offsets[mode_idx] + region_idx * per_region_counts[mode_idx] + slot

    line_reliability = service_lines["reliability_index"].to_numpy()[line_idx]
    line_capacity = service_lines["capacity_index"].to_numpy()[line_idx]
    line_scheduled = service_lines["scheduled_duration_minutes"].to_numpy()[line_idx]

    # --- 4. When they travel ----------------------------------------------------------
    weekday_hours, weekend_hours = _hour_profiles()
    hour_probs = np.where(
        is_weekend[:, None], weekend_hours[segment_idx], weekday_hours[segment_idx]
    )
    hour = _sample_categorical(rng, hour_probs)
    minute = rng.integers(0, 60, size=n)
    timestamps = (
        pd.to_datetime(journey_dates)
        + pd.to_timedelta(hour, unit="h")
        + pd.to_timedelta(minute, unit="m")
    )

    is_peak = (~is_weekend) & (((hour >= 7) & (hour <= 9)) | ((hour >= 16) & (hour <= 18)))

    # --- 5. Operational outcomes ------------------------------------------------------
    masks = _anomaly_masks(journey_dates, regions, modes, end_date)

    delay_pressure = np.array([MODE_DELAY_PRESSURE[m] for m in TRANSPORT_MODES])[mode_idx]
    delay_pressure = delay_pressure / line_reliability
    delay_pressure *= np.array([REGION_DELAY_MULTIPLIER[r] for r in REGIONS])[region_idx]
    delay_pressure *= 1.0 + 0.62 * is_peak
    delay_pressure *= 1.0 + 0.18 * (day_of_week == 4)  # Friday effect
    delay_pressure *= np.clip(line_capacity, 0.7, 1.4) ** 0.5

    for anomaly in KNOWN_ANOMALIES:
        if anomaly.kind in {"delay", "improvement"}:
            delay_pressure = np.where(masks[anomaly.name], delay_pressure * anomaly.magnitude, delay_pressure)

    disruption_prob = 0.030 * (1.6 - np.clip(line_reliability, 0.5, 1.3)) * (1.0 + 0.55 * is_peak)
    for anomaly in KNOWN_ANOMALIES:
        if anomaly.kind == "delay":
            disruption_prob = np.where(masks[anomaly.name], disruption_prob * 2.4, disruption_prob)
    service_disruption = rng.random(n) < np.clip(disruption_prob, 0, 0.5)

    delay_pressure *= 1.0 + 2.4 * service_disruption
    delay_pressure *= DELAY_CALIBRATION
    delay_minutes = rng.gamma(shape=1.35, scale=delay_pressure / 1.35)
    delay_minutes = np.clip(np.round(delay_minutes, 1), 0.0, MAX_PLAUSIBLE_DELAY_MINUTES)
    on_time = delay_minutes <= ON_TIME_THRESHOLD_MINUTES

    duration = line_scheduled * rng.lognormal(0.0, 0.17, n) + 0.85 * delay_minutes
    duration = np.clip(np.round(duration, 1), 3.0, 235.0)

    # --- 6. Complaints ----------------------------------------------------------------
    complaint_logit = (
        -3.55
        + 0.072 * np.minimum(delay_minutes, 60.0)
        + 1.05 * service_disruption
        + np.array([SEGMENT_COMPLAINT_OFFSET[s] for s in CUSTOMER_SEGMENTS])[segment_idx]
        - 0.14 * repeat_customer
        + 0.22 * is_peak
    )
    for anomaly in KNOWN_ANOMALIES:
        if anomaly.kind == "complaints":
            complaint_logit = np.where(
                masks[anomaly.name], complaint_logit + anomaly.magnitude, complaint_logit
            )
    complaint_flag = rng.random(n) < 1.0 / (1.0 + np.exp(-complaint_logit))

    resolution_hours = np.where(
        complaint_flag,
        np.round(rng.lognormal(3.18, 0.62, n) * (1.0 + 0.45 * service_disruption), 1),
        np.nan,
    )
    resolution_hours = np.where(np.isnan(resolution_hours), np.nan, np.clip(resolution_hours, 0.5, 400.0))

    # Complaint reason follows the dominant cause of the journey's experience.
    complaint_weights = np.zeros((n, len(COMPLAINT_CATEGORIES)))
    complaint_weights[:, 0] = 0.15 + 0.09 * np.minimum(delay_minutes, 40)  # Delay
    complaint_weights[:, 1] = 0.12 + 0.55 * is_peak + 0.3 * (line_capacity < 0.85)  # Crowding
    complaint_weights[:, 2] = 0.18  # Cleanliness
    complaint_weights[:, 3] = 0.14  # Staff Conduct
    complaint_weights[:, 4] = 0.06 + 0.85 * (segment_idx == CUSTOMER_SEGMENTS.index("High-Service-Need Customers"))
    complaint_weights[:, 5] = 0.14 + 0.75 * service_disruption  # Information & Wayfinding
    complaint_weights[:, 6] = 0.13  # Ticketing & Fares
    complaint_idx = _sample_categorical(rng, complaint_weights)
    complaint_category = np.where(
        complaint_flag, np.array(COMPLAINT_CATEGORIES)[complaint_idx], None
    )

    # --- 7. Satisfaction --------------------------------------------------------------
    satisfaction = (
        4.42
        + np.array([SEGMENT_SATISFACTION_OFFSET[s] for s in CUSTOMER_SEGMENTS])[segment_idx]
        - 0.082 * np.minimum(delay_minutes, 45.0)
        - 0.52 * service_disruption
        - 0.20 * is_peak
        + 0.16 * repeat_customer
        - 0.42 * complaint_flag
        - 0.0075 * np.nan_to_num(resolution_hours, nan=0.0)
        + rng.normal(0.0, 0.62, n)
    )
    for anomaly in KNOWN_ANOMALIES:
        if anomaly.kind == "complaints":
            satisfaction = np.where(masks[anomaly.name], satisfaction - 0.42, satisfaction)
    satisfaction = np.clip(np.rint(satisfaction), 1, 5).astype(float)

    # --- 8. Customer-facing attributes ------------------------------------------------
    channel_matrix = _profile_matrix(SEGMENT_CHANNEL_PROFILE, CUSTOMER_SEGMENTS, CHANNELS)
    channel_idx = _sample_categorical(rng, channel_matrix[segment_idx])
    channels = np.array(CHANNELS)[channel_idx]

    digital_channel = np.isin(channels, ["Mobile App", "Website", "Third-Party App"])
    digital_interaction = rng.random(n) < np.clip(
        digital_affinity * np.where(digital_channel, 1.0, 0.35), 0.01, 0.99
    )

    accessibility_prob = (
        0.015
        + 0.24 * (segment_idx == CUSTOMER_SEGMENTS.index("High-Service-Need Customers"))
        + 0.06 * (age_idx == len(AGE_GROUPS) - 1)
    )
    accessibility_service_used = rng.random(n) < accessibility_prob

    purpose_weights = np.zeros((n, len(JOURNEY_PURPOSES)))
    purpose_weights[:, 0] = 0.10 + 1.6 * is_peak + 0.8 * (segment_idx == 0)  # Commute
    purpose_weights[:, 1] = 0.18 + 0.9 * (age_idx == 0)  # Education
    purpose_weights[:, 2] = 0.16 + 0.4 * (~is_weekend)  # Business
    purpose_weights[:, 3] = 0.30 + 0.9 * is_weekend  # Leisure
    purpose_weights[:, 4] = 0.10 + 0.5 * (segment_idx == CUSTOMER_SEGMENTS.index("High-Service-Need Customers"))
    purpose_weights[:, 5] = 0.22 + 0.4 * is_weekend  # Shopping
    purpose_weights[:, 6] = 0.07  # Other
    purpose_idx = _sample_categorical(rng, purpose_weights)
    purposes = np.array(JOURNEY_PURPOSES)[purpose_idx]

    origin_idx = np.where(rng.random(n) < 0.82, region_idx, rng.integers(0, len(REGIONS), n))
    commute_mask = purposes == "Commute"
    destination_idx = np.where(
        commute_mask & (rng.random(n) < 0.48),
        REGIONS.index("Central"),
        rng.integers(0, len(REGIONS), n),
    )

    # Feedback theme mirrors the satisfaction score and its dominant driver.
    feedback = np.full(n, "Neutral", dtype=object)
    positive = satisfaction >= 5
    good = satisfaction == 4
    poor = satisfaction <= 2
    feedback[good] = "Positive - Reliability"
    feedback[positive] = np.where(
        rng.random(int(positive.sum())) < 0.45, "Positive - Staff", "Positive - Reliability"
    )
    poor_reason = np.select(
        [
            delay_minutes[poor] > 8,
            service_disruption[poor],
            is_peak[poor],
        ],
        ["Negative - Delay", "Negative - Information", "Negative - Crowding"],
        default="Negative - Cleanliness",
    )
    feedback[poor] = poor_reason

    # --- 9. Assemble the fact table ---------------------------------------------------
    journeys = pd.DataFrame(
        {
            "journey_id": [f"J{i:07d}" for i in range(1, n + 1)],
            "customer_id": customers["customer_id"].to_numpy()[customer_row],
            "date": pd.to_datetime(journey_dates).date,
            "timestamp": timestamps,
            "region": regions,
            "origin_region": np.array(REGIONS)[origin_idx],
            "destination_region": np.array(REGIONS)[destination_idx],
            "transport_mode": modes,
            "service_line": service_lines["service_line"].to_numpy()[line_idx],
            "journey_purpose": purposes,
            "customer_segment": np.array(CUSTOMER_SEGMENTS)[segment_idx],
            "age_group": np.array(AGE_GROUPS)[age_idx],
            "channel": channels,
            "journey_duration_minutes": duration,
            "delay_minutes": delay_minutes,
            "on_time": on_time,
            "customer_satisfaction": satisfaction,
            "complaint_flag": complaint_flag,
            "complaint_category": complaint_category,
            "resolution_time_hours": resolution_hours,
            "repeat_customer": repeat_customer,
            "service_disruption": service_disruption,
            "accessibility_service_used": accessibility_service_used,
            "fare_type": customers["fare_type"].to_numpy()[customer_row],
            "digital_interaction": digital_interaction,
            "feedback_text_category": feedback,
        }
    )
    journeys = journeys.sort_values("timestamp", kind="stable").reset_index(drop=True)

    if config.degrade_quality:
        journeys = _apply_quality_degradation(journeys, rng)

    journeys = journeys[list(JOURNEY_COLUMNS)]

    manifest = {
        "seed": config.seed,
        "fingerprint": config.fingerprint(),
        "n_journeys": int(len(journeys)),
        "n_customers": int(config.n_customers),
        "n_service_lines": int(len(service_lines)),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "history_days": config.history_days,
        "quality_degraded": config.degrade_quality,
    }

    customers = customers.drop(columns=["_segment_idx", "_home_idx", "_activity"])
    return SyntheticDataset(
        journeys=journeys,
        customers=customers,
        service_lines=service_lines,
        manifest=manifest,
    )


def known_anomaly_windows(end_date: date) -> list[dict[str, object]]:
    """Ground truth for the injected anomalies.

    Used by the test suite to confirm the detector finds what was planted. The
    application itself never reads this - anomalies must be discovered from the data.
    """
    resolved = []
    for anomaly in KNOWN_ANOMALIES:
        start, stop = anomaly.window(end_date)
        resolved.append(
            {
                "name": anomaly.name,
                "kind": anomaly.kind,
                "dimension": anomaly.dimension,
                "value": anomaly.value,
                "start": start,
                "end": stop,
                "note": anomaly.note,
            }
        )
    return resolved
