"""Reusable Streamlit components.

Every page is assembled from these, so spacing, typography and the KPI card treatment
are defined once. Pages contain layout and narrative; calculations live in
``src.analytics``.
"""

from __future__ import annotations

import html
from typing import Any
from collections.abc import Iterable, Sequence

import pandas as pd
import streamlit as st

from src.analytics.insights import Insight
from src.analytics.kpis import KpiValue
from src.utils.config import APP_NAME, DISCLAIMER, PALETTE

_CSS = f"""
<style>
  .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }}
  header[data-testid="stHeader"] {{ background: transparent; }}

  .cis-title {{
    font-size: 1.55rem; font-weight: 640; letter-spacing: -0.015em;
    color: {PALETTE.ink}; margin: 0 0 0.15rem 0; line-height: 1.25;
  }}
  .cis-subtitle {{
    font-size: 0.92rem; color: {PALETTE.muted}; margin: 0 0 0.2rem 0; line-height: 1.5;
  }}
  .cis-rule {{ border: 0; border-top: 1px solid {PALETTE.line}; margin: 1.1rem 0 1.3rem 0; }}

  .cis-section {{
    font-size: 0.78rem; font-weight: 660; text-transform: uppercase;
    letter-spacing: 0.09em; color: {PALETTE.muted};
    margin: 1.6rem 0 0.1rem 0;
  }}
  .cis-section-note {{ font-size: 0.85rem; color: {PALETTE.muted}; margin: 0 0 0.6rem 0; }}

  .cis-card {{
    background: {PALETTE.surface}; border: 1px solid {PALETTE.line};
    border-radius: 7px; padding: 0.85rem 0.95rem; height: 100%;
  }}
  .cis-card-label {{
    font-size: 0.74rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.06em; color: {PALETTE.muted}; margin-bottom: 0.35rem;
  }}
  .cis-card-value {{
    font-size: 1.62rem; font-weight: 620; color: {PALETTE.ink};
    line-height: 1.1; letter-spacing: -0.02em; font-variant-numeric: tabular-nums;
  }}
  .cis-card-delta {{ font-size: 0.78rem; margin-top: 0.3rem; font-variant-numeric: tabular-nums; }}
  .cis-good {{ color: {PALETTE.positive}; }}
  .cis-bad {{ color: {PALETTE.negative}; }}
  .cis-flat {{ color: {PALETTE.muted}; }}

  .cis-population {{
    background: {PALETTE.accent_soft}; border: 1px solid #cfe0e7;
    border-radius: 6px; padding: 0.55rem 0.8rem; font-size: 0.86rem;
    color: #1b4f63; margin-bottom: 0.4rem;
  }}

  .cis-insight {{
    border-left: 3px solid {PALETTE.line}; padding: 0.15rem 0 0.15rem 0.8rem;
    margin-bottom: 0.85rem;
  }}
  .cis-insight-bad {{ border-left-color: {PALETTE.negative}; }}
  .cis-insight-good {{ border-left-color: {PALETTE.positive}; }}
  .cis-insight-neutral {{ border-left-color: {PALETTE.categorical[1]}; }}
  .cis-insight-headline {{ font-size: 0.93rem; color: {PALETTE.ink}; font-weight: 560; }}
  .cis-insight-detail {{ font-size: 0.84rem; color: {PALETTE.muted}; margin-top: 0.15rem; }}
  .cis-insight-tag {{
    font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.07em;
    color: {PALETTE.muted}; font-weight: 620;
  }}

  .cis-footer {{
    font-size: 0.76rem; color: {PALETTE.muted}; border-top: 1px solid {PALETTE.line};
    margin-top: 2.4rem; padding-top: 0.7rem; line-height: 1.6;
  }}
  .cis-pill {{
    display: inline-block; font-size: 0.72rem; padding: 0.12rem 0.5rem;
    border-radius: 10px; border: 1px solid {PALETTE.line}; color: {PALETTE.muted};
    margin-right: 0.3rem;
  }}
  [data-testid="stMetricValue"] {{ font-size: 1.35rem; }}
</style>
"""


def configure_page(page_title: str, icon: str = "*") -> None:
    """Standard page configuration and stylesheet injection."""
    st.set_page_config(
        page_title=f"{page_title} - {APP_NAME}",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(title: str, description: str, *, rule: bool = True) -> None:
    """Consistent page title block."""
    st.markdown(
        f'<div class="cis-title">{html.escape(title)}</div>'
        f'<p class="cis-subtitle">{html.escape(description)}</p>',
        unsafe_allow_html=True,
    )
    if rule:
        st.markdown('<hr class="cis-rule"/>', unsafe_allow_html=True)


def section(title: str, note: str = "") -> None:
    """Small uppercase section divider."""
    block = f'<div class="cis-section">{html.escape(title)}</div>'
    if note:
        block += f'<p class="cis-section-note">{html.escape(note)}</p>'
    st.markdown(block, unsafe_allow_html=True)


def kpi_card(kpi: KpiValue) -> str:
    """HTML for one KPI card, including its period-over-period delta."""
    css_class = {"good": "cis-good", "bad": "cis-bad"}.get(kpi.direction, "cis-flat")
    return (
        f'<div class="cis-card">'
        f'<div class="cis-card-label">{html.escape(kpi.label)}</div>'
        f'<div class="cis-card-value">{html.escape(kpi.formatted())}</div>'
        f'<div class="cis-card-delta {css_class}">{html.escape(kpi.formatted_delta())}</div>'
        f"</div>"
    )


def kpi_row(kpis: Sequence[KpiValue], columns: int = 3) -> None:
    """Render KPI cards in a responsive grid."""
    kpis = list(kpis)
    for start in range(0, len(kpis), columns):
        chunk = kpis[start : start + columns]
        cols = st.columns(len(chunk), gap="small")
        for column, kpi in zip(cols, chunk, strict=False):
            with column:
                st.markdown(kpi_card(kpi), unsafe_allow_html=True)
        if start + columns < len(kpis):
            st.write("")


def population_banner(summary: dict[str, Any], filter_description: str = "") -> None:
    """The 'Analysing N journeys across M regions' line."""
    journeys = summary.get("journeys", 0)
    if not journeys:
        st.warning("No journeys match the current filters. Widen the date range or clear a filter.")
        return

    text = (
        f"Analysing <strong>{journeys:,}</strong> journeys from "
        f"<strong>{summary.get('customers', 0):,}</strong> customers across "
        f"<strong>{summary.get('regions', 0)}</strong> region(s), "
        f"<strong>{summary.get('modes', 0)}</strong> mode(s)"
    )
    first, last = summary.get("first_date"), summary.get("last_date")
    if first and last:
        text += f", {first:%d %b %Y} to {last:%d %b %Y}"
    if filter_description:
        text += f" &mdash; {html.escape(filter_description)}"
    st.markdown(f'<div class="cis-population">{text}.</div>', unsafe_allow_html=True)


def insight_list(insights: Iterable[Insight], empty_message: str = "") -> None:
    """Render ranked insights with a direction-coded left border."""
    items = list(insights)
    if not items:
        st.caption(empty_message or "No material findings for the current selection.")
        return

    for item in items:
        css = {
            "bad": "cis-insight-bad",
            "good": "cis-insight-good",
        }.get(item.direction, "cis-insight-neutral")
        st.markdown(
            f'<div class="cis-insight {css}">'
            f'<div class="cis-insight-tag">{html.escape(item.category)}</div>'
            f'<div class="cis-insight-headline">{html.escape(item.headline)}</div>'
            f'<div class="cis-insight-detail">{html.escape(item.detail)}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )


def metric_help(specs: Iterable[Any]) -> None:
    """Expandable definitions so every number on the page is explained."""
    with st.expander("How these metrics are calculated"):
        for spec in specs:
            st.markdown(f"**{spec.label}** - {spec.description}")
            st.caption(f"`{spec.sql}`")


def download_frame(
    frame: pd.DataFrame,
    filename: str,
    label: str = "Download CSV",
    key: str | None = None,
) -> None:
    """CSV download button for any table shown on a page."""
    if frame is None or frame.empty:
        st.button(label, disabled=True, help="Nothing to export for this selection.", key=key)
        return
    st.download_button(
        label=label,
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def empty_state(message: str) -> None:
    """Consistent treatment when a chart or table has no data."""
    st.info(message, icon=None)


def evidence_panel(evidence: dict[str, Any], title: str = "Evidence used") -> None:
    """Expandable panel showing the exact figures behind a generated statement."""
    with st.expander(title):
        st.caption(
            "These figures were computed by the application's analytics layer. "
            "Any generated wording is based only on this evidence."
        )
        st.json(evidence, expanded=False)


def footer(extra: str = "") -> None:
    """Disclaimer and provenance shown at the bottom of every page."""
    note = f"{DISCLAIMER}"
    if extra:
        note += f" {extra}"
    st.markdown(f'<div class="cis-footer">{html.escape(note)}</div>', unsafe_allow_html=True)


def dataset_badge(manifest: dict[str, Any], origin: str) -> None:
    """Compact provenance badge for the sidebar."""
    label = "Demo dataset" if origin == "demo" else "Uploaded dataset"
    rows = manifest.get("n_journeys", 0)
    start = manifest.get("start_date", "?")
    end = manifest.get("end_date", "?")
    st.markdown(
        f'<span class="cis-pill">{html.escape(label)}</span>'
        f'<span class="cis-pill">{rows:,} rows</span><br>'
        f'<span style="font-size:0.74rem;color:{PALETTE.muted}">{start} to {end}</span>',
        unsafe_allow_html=True,
    )
