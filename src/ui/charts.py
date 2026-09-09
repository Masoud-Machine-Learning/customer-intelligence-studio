"""Plotly chart builders sharing one visual language.

Pages never construct a figure directly. Every chart comes from here so that axes,
fonts, gridlines, hover formatting and colours stay identical across the application.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import plotly.graph_objects as go

from src.utils.config import PALETTE

BASE_FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


def _base_layout(title: str | None, height: int) -> dict:
    return {
        "title": (
            {
                "text": title,
                "font": {"size": 14, "color": PALETTE.ink, "family": BASE_FONT},
                "x": 0,
                "xanchor": "left",
                "y": 0.97,
            }
            if title
            else None
        ),
        "height": height,
        "margin": {"l": 8, "r": 8, "t": 42 if title else 16, "b": 8},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": BASE_FONT, "size": 12, "color": PALETTE.body},
        "hoverlabel": {
            "bgcolor": PALETTE.surface,
            "bordercolor": PALETTE.line,
            "font": {"family": BASE_FONT, "size": 12, "color": PALETTE.ink},
        },
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.0,
            "x": 0,
            "font": {"size": 11},
            "title": {"text": ""},
        },
        "xaxis": {
            "showgrid": False,
            "linecolor": PALETTE.line,
            "ticks": "outside",
            "tickcolor": PALETTE.line,
            "tickfont": {"size": 11, "color": PALETTE.muted},
            "title": {"font": {"size": 11, "color": PALETTE.muted}},
        },
        "yaxis": {
            "gridcolor": PALETTE.line,
            "zeroline": False,
            "linecolor": "rgba(0,0,0,0)",
            "tickfont": {"size": 11, "color": PALETTE.muted},
            "title": {"font": {"size": 11, "color": PALETTE.muted}},
        },
    }


def _apply(figure: go.Figure, title: str | None, height: int, y_title: str = "") -> go.Figure:
    figure.update_layout(**_base_layout(title, height))
    if y_title:
        figure.update_yaxes(title_text=y_title)
    return figure


def line_chart(
    frame: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: str | None = None,
    y_title: str = "",
    overlay: str | None = None,
    overlay_name: str = "7-day average",
    height: int = 280,
    hover_format: str = ",.2f",
) -> go.Figure:
    """Single-series trend with an optional smoothed overlay."""
    figure = go.Figure()
    if frame.empty:
        return _apply(figure, title, height, y_title)

    figure.add_trace(
        go.Scatter(
            x=frame[x],
            y=frame[y],
            mode="lines",
            name="Daily",
            line={"color": PALETTE.categorical[1], "width": 1.2},
            hovertemplate=f"%{{x|%d %b %Y}}<br>%{{y:{hover_format}}}<extra></extra>",
        )
    )
    if overlay and overlay in frame.columns:
        figure.add_trace(
            go.Scatter(
                x=frame[x],
                y=frame[overlay],
                mode="lines",
                name=overlay_name,
                line={"color": PALETTE.accent, "width": 2.4},
                hovertemplate=f"%{{x|%d %b %Y}}<br>%{{y:{hover_format}}}<extra></extra>",
            )
        )
    return _apply(figure, title, height, y_title)


def multi_line_chart(
    frame: pd.DataFrame,
    x: str,
    y: str,
    colour: str,
    *,
    title: str | None = None,
    y_title: str = "",
    height: int = 300,
    hover_format: str = ",.2f",
) -> go.Figure:
    """One line per category, using the shared categorical palette."""
    figure = go.Figure()
    if frame.empty:
        return _apply(figure, title, height, y_title)

    for index, (name, group) in enumerate(frame.groupby(colour, sort=True)):
        figure.add_trace(
            go.Scatter(
                x=group[x],
                y=group[y],
                mode="lines",
                name=str(name),
                line={"color": PALETTE.categorical[index % len(PALETTE.categorical)], "width": 1.9},
                hovertemplate=f"<b>{name}</b><br>%{{x|%d %b %Y}}<br>%{{y:{hover_format}}}<extra></extra>",
            )
        )
    return _apply(figure, title, height, y_title)


def bar_chart(
    frame: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: str | None = None,
    y_title: str = "",
    horizontal: bool = False,
    height: int = 280,
    highlight: str | None = None,
    value_format: str = ",.1f",
) -> go.Figure:
    """Categorical comparison. One accent colour, optionally highlighting a category."""
    figure = go.Figure()
    if frame.empty:
        return _apply(figure, title, height, y_title)

    categories = frame[x].astype(str)
    colours = [
        PALETTE.accent if (highlight is None or value == highlight) else PALETTE.categorical[1]
        for value in categories
    ]
    if horizontal:
        figure.add_trace(
            go.Bar(
                x=frame[y],
                y=categories,
                orientation="h",
                marker_color=colours,
                hovertemplate=f"%{{y}}<br>%{{x:{value_format}}}<extra></extra>",
            )
        )
        figure.update_layout(yaxis={"autorange": "reversed"})
    else:
        figure.add_trace(
            go.Bar(
                x=categories,
                y=frame[y],
                marker_color=colours,
                hovertemplate=f"%{{x}}<br>%{{y:{value_format}}}<extra></extra>",
            )
        )
    figure = _apply(figure, title, height, y_title)
    if horizontal:
        figure.update_layout(
            xaxis={"gridcolor": PALETTE.line, "showgrid": True},
            yaxis={"showgrid": False, "autorange": "reversed"},
        )
    return figure


def grouped_bar_chart(
    frame: pd.DataFrame,
    x: str,
    series: Sequence[str],
    *,
    title: str | None = None,
    y_title: str = "",
    height: int = 300,
    names: Sequence[str] | None = None,
) -> go.Figure:
    """Two or more measures side by side for each category."""
    figure = go.Figure()
    if frame.empty:
        return _apply(figure, title, height, y_title)

    labels = list(names) if names else list(series)
    for index, (column, label) in enumerate(zip(series, labels, strict=False)):
        figure.add_trace(
            go.Bar(
                x=frame[x].astype(str),
                y=frame[column],
                name=str(label),
                marker_color=PALETTE.categorical[index % len(PALETTE.categorical)],
                hovertemplate=f"<b>{label}</b><br>%{{x}}<br>%{{y:,.2f}}<extra></extra>",
            )
        )
    figure.update_layout(barmode="group")
    return _apply(figure, title, height, y_title)


def histogram(
    values: pd.Series,
    *,
    title: str | None = None,
    x_title: str = "",
    height: int = 260,
    nbins: int = 40,
) -> go.Figure:
    """Distribution of a numeric measure."""
    figure = go.Figure()
    if values.empty:
        return _apply(figure, title, height)
    figure.add_trace(
        go.Histogram(
            x=values,
            nbinsx=nbins,
            marker_color=PALETTE.accent,
            hovertemplate="%{x}<br>%{y:,} journeys<extra></extra>",
        )
    )
    figure = _apply(figure, title, height, "Journeys")
    figure.update_xaxes(title_text=x_title)
    return figure


def heatmap(
    matrix: pd.DataFrame,
    *,
    title: str | None = None,
    height: int = 300,
    colourbar_title: str = "",
    value_format: str = ".1f",
) -> go.Figure:
    """Day-by-hour matrix using the sequential palette."""
    figure = go.Figure()
    if matrix.empty:
        return _apply(figure, title, height)

    figure.add_trace(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=[str(c) for c in matrix.columns],
            y=list(matrix.index),
            colorscale=[
                [i / (len(PALETTE.sequential) - 1), colour]
                for i, colour in enumerate(PALETTE.sequential)
            ],
            colorbar={
                "title": {"text": colourbar_title, "font": {"size": 11}},
                "thickness": 10,
                "outlinewidth": 0,
            },
            hovertemplate=f"%{{y}} at %{{x}}:00<br>%{{z:{value_format}}}<extra></extra>",
        )
    )
    figure = _apply(figure, title, height)
    figure.update_xaxes(title_text="Hour of day", showgrid=False)
    figure.update_yaxes(showgrid=False, autorange="reversed")
    return figure


def scatter_chart(
    frame: pd.DataFrame,
    x: str,
    y: str,
    colour: str,
    *,
    title: str | None = None,
    x_title: str = "",
    y_title: str = "",
    height: int = 360,
    size: str | None = None,
) -> go.Figure:
    """Customer-level scatter used on the segmentation page."""
    figure = go.Figure()
    if frame.empty:
        return _apply(figure, title, height, y_title)

    for index, (name, group) in enumerate(frame.groupby(colour, sort=True)):
        figure.add_trace(
            go.Scattergl(
                x=group[x],
                y=group[y],
                mode="markers",
                name=str(name),
                marker={
                    "color": PALETTE.categorical[index % len(PALETTE.categorical)],
                    "size": 6 if size is None else None,
                    "opacity": 0.55,
                    "line": {"width": 0},
                },
                hovertemplate=f"<b>{name}</b><br>{x_title or x}: %{{x:,.2f}}<br>"
                f"{y_title or y}: %{{y:,.2f}}<extra></extra>",
            )
        )
    figure = _apply(figure, title, height, y_title)
    figure.update_xaxes(title_text=x_title or x)
    return figure


def anomaly_chart(
    frame: pd.DataFrame,
    *,
    value_column: str,
    title: str | None = None,
    y_title: str = "",
    height: int = 300,
) -> go.Figure:
    """Series with its expected band and flagged points marked.

    Expects columns ``period``, ``value_column``, ``expected``, ``lower``, ``upper`` and
    a boolean ``flagged``.
    """
    figure = go.Figure()
    if frame.empty:
        return _apply(figure, title, height, y_title)

    figure.add_trace(
        go.Scatter(
            x=frame["period"],
            y=frame["upper"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["period"],
            y=frame["lower"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(31, 111, 139, 0.10)",
            line={"width": 0},
            name="Expected range",
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["period"],
            y=frame[value_column],
            mode="lines",
            name="Actual",
            line={"color": PALETTE.body, "width": 1.6},
            hovertemplate="%{x|%d %b %Y}<br>%{y:,.2f}<extra></extra>",
        )
    )
    flagged = frame[frame["flagged"].fillna(False)]
    if not flagged.empty:
        figure.add_trace(
            go.Scatter(
                x=flagged["period"],
                y=flagged[value_column],
                mode="markers",
                name="Anomaly",
                marker={
                    "color": PALETTE.negative,
                    "size": 9,
                    "symbol": "circle-open",
                    "line": {"width": 2},
                },
                hovertemplate="<b>Anomaly</b><br>%{x|%d %b %Y}<br>%{y:,.2f}<extra></extra>",
            )
        )
    return _apply(figure, title, height, y_title)


def donut_chart(
    frame: pd.DataFrame,
    labels: str,
    values: str,
    *,
    title: str | None = None,
    height: int = 280,
) -> go.Figure:
    """Composition of a whole; used sparingly and only for small category counts."""
    figure = go.Figure()
    if frame.empty:
        return _apply(figure, title, height)
    figure.add_trace(
        go.Pie(
            labels=frame[labels].astype(str),
            values=frame[values],
            hole=0.62,
            marker={"colors": list(PALETTE.categorical)},
            sort=False,
            textinfo="none",
            hovertemplate="%{label}<br>%{value:,} journeys (%{percent})<extra></extra>",
        )
    )
    return _apply(figure, title, height)
