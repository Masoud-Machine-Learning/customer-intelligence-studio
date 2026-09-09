"""Sidebar filter widgets.

These functions only collect user choices and hand back a :class:`FilterState`. All
filtering logic - including how "previous period" is derived and how the SQL is built -
lives in :mod:`src.data.filters` so it can be tested without Streamlit.

The selection is stored in ``st.session_state`` so it persists as the user moves between
pages: a filter set on the Executive Overview still applies on Service Performance.
"""

from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from src.data.database import DataSource, dataset_bounds, distinct_values
from src.data.filters import FilterState

SESSION_KEY = "cis_filters"

PRESETS: dict[str, int | None] = {
    "Last 30 days": 30,
    "Last 90 days": 90,
    "Last 6 months": 182,
    "Last 12 months": 365,
    "All available data": None,
}
DEFAULT_PRESET = "Last 90 days"


@st.cache_data(show_spinner=False, ttl=1800)
def _filter_options(origin: str, column: str) -> list[str]:
    """Distinct values for one filter, cached because they change only on reload."""
    from src.data.loader import get_demo_source

    source = get_demo_source() if origin == "demo" else None
    if source is None:
        return []
    return distinct_values(source.connection, column)


def _options(source: DataSource, column: str) -> list[str]:
    if source.origin == "demo":
        return _filter_options(source.origin, column)
    return distinct_values(source.connection, column)


def _resolve_preset(preset: str, bounds: tuple[date, date]) -> tuple[date, date]:
    minimum, maximum = bounds
    days = PRESETS.get(preset)
    if days is None:
        return minimum, maximum
    start = max(minimum, maximum - timedelta(days=days - 1))
    return start, maximum


def render_filters(source: DataSource, *, key_prefix: str = "main") -> FilterState:
    """Draw the sidebar filters and return the resulting state.

    A preset picker covers the common cases in one click; the explicit date inputs stay
    available for anything else.
    """
    bounds = dataset_bounds(source.connection)
    minimum, maximum = bounds

    with st.sidebar:
        st.markdown("#### Filters")

        preset = st.selectbox(
            "Period",
            list(PRESETS.keys()),
            index=list(PRESETS.keys()).index(DEFAULT_PRESET),
            key=f"{key_prefix}_preset",
            help=(
                "The comparison period is always the equal-length window immediately "
                "before the one selected."
            ),
        )
        default_start, default_end = _resolve_preset(preset, bounds)

        date_range = st.date_input(
            "Date range",
            value=(default_start, default_end),
            min_value=minimum,
            max_value=maximum,
            key=f"{key_prefix}_dates_{preset}",
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = default_start, default_end

        regions = st.multiselect("Region", _options(source, "region"), key=f"{key_prefix}_regions")
        modes = st.multiselect(
            "Transport mode", _options(source, "transport_mode"), key=f"{key_prefix}_modes"
        )
        segments = st.multiselect(
            "Customer segment",
            _options(source, "customer_segment"),
            key=f"{key_prefix}_segments",
        )

        with st.expander("More filters"):
            service_lines = st.multiselect(
                "Service line", _options(source, "service_line"), key=f"{key_prefix}_lines"
            )
            channels = st.multiselect(
                "Channel", _options(source, "channel"), key=f"{key_prefix}_channels"
            )
            time_bands = st.multiselect(
                "Time band", _options(source, "time_band"), key=f"{key_prefix}_bands"
            )
            purposes = st.multiselect(
                "Journey purpose",
                _options(source, "journey_purpose"),
                key=f"{key_prefix}_purposes",
            )

        if st.button("Reset filters", use_container_width=True, key=f"{key_prefix}_reset"):
            for suffix in (
                "regions",
                "modes",
                "segments",
                "lines",
                "channels",
                "bands",
                "purposes",
            ):
                st.session_state.pop(f"{key_prefix}_{suffix}", None)
            st.rerun()

    filters = FilterState.create(
        start_date=start_date,
        end_date=end_date,
        regions=regions,
        transport_modes=modes,
        customer_segments=segments,
        service_lines=service_lines,
        channels=channels,
        time_bands=time_bands,
        journey_purposes=purposes,
    )
    st.session_state[SESSION_KEY] = filters
    return filters


def current_filters(source: DataSource) -> FilterState:
    """Filters from session state, falling back to a sensible default."""
    existing = st.session_state.get(SESSION_KEY)
    if isinstance(existing, FilterState):
        return existing
    minimum, maximum = dataset_bounds(source.connection)
    start, _ = _resolve_preset(DEFAULT_PRESET, (minimum, maximum))
    return FilterState.create(start, maximum)
