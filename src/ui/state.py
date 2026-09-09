"""Session state: which dataset the user is analysing.

The default is always the built-in synthetic demo, so the application is fully
explorable with no setup. An uploaded dataset replaces it for the session only and is
never written to disk.
"""

from __future__ import annotations

import streamlit as st

from src.data.database import DataSource
from src.data.loader import get_demo_source

SOURCE_KEY = "cis_data_source"


def set_source(source: DataSource) -> None:
    """Make ``source`` the active dataset for this session."""
    st.session_state[SOURCE_KEY] = source


def clear_source() -> None:
    """Return to the built-in demo dataset."""
    st.session_state.pop(SOURCE_KEY, None)


def active_source() -> DataSource:
    """The dataset every page should analyse."""
    existing = st.session_state.get(SOURCE_KEY)
    if isinstance(existing, DataSource):
        return existing
    return get_demo_source()


def using_upload() -> bool:
    return active_source().origin == "upload"
