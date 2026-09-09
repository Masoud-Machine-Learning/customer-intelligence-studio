"""About - what this project is, how it is built and what it does not claim."""

from __future__ import annotations

import streamlit as st

from src.ai.assistant import ai_available
from src.ui import components as ui
from src.ui.state import active_source
from src.utils.config import APP_NAME, DISCLAIMER

ui.configure_page("About")

source = active_source()

ui.page_header(
    f"About {APP_NAME}",
    "An independent portfolio demonstration of an end-to-end customer analytics application.",
)

st.markdown(
    """
Customer Intelligence Studio is a portfolio demonstration of an end-to-end customer
analytics application. It combines Python, Streamlit, Pandas, SQL, interactive
visualisation, anomaly detection and optional AI-assisted analysis to turn complex
customer and operational data into practical insights.

**All data used in the demonstration is synthetic**, generated programmatically from a
fixed random seed. No real customer, health, employer or operator data appears anywhere
in this application.
    """
)

st.info(
    "This is an independent portfolio demonstration and is not affiliated with Transport for NSW.",
    icon=None,
)

manifest = source.manifest
info_a, info_b, info_c = st.columns(3)
info_a.metric("Journeys in dataset", f"{manifest.get('n_journeys', 0):,}")
info_b.metric("Customers", f"{manifest.get('n_customers', 0):,}")
info_c.metric(
    "History",
    f"{manifest.get('start_date', '?')} to {manifest.get('end_date', '?')}",
)

# ---------------------------------------------------------------------------

ui.section("Technology stack")
st.markdown(
    """
| Technology | How it is used here |
| --- | --- |
| **Python 3.10+** | All application and analytics code |
| **Streamlit** | Multi-page interactive application, caching, session state |
| **Pandas** | Rolling statistics, reshaping, feature engineering, chart preparation |
| **DuckDB** | Analytical database: star schema, views, all aggregation in SQL |
| **SQL** | Filtering, aggregation, KPIs, period comparison, window functions |
| **Plotly** | All charts, built from one shared visual specification |
| **scikit-learn** | Standardisation and K-means for the optional clustering view |
| **pytest** | Unit tests for the analytics, filters, SQL layer and data rules |
| **Ruff** | Linting in CI |
| **GitHub Actions** | Lint and test on every push and pull request |
| **OpenAI API** | Optional wording layer only; the application is complete without it |

Technologies deliberately **not** claimed: Snowflake, Databricks, Azure Data Platform,
Tableau, Power BI and Dash are not used anywhere in this project.
    """
)

ui.section("Architecture")
st.markdown(
    """
```
Synthetic data generator  (numpy, seeded, causal relationships)
            |
            v
      DuckDB database      (journeys fact + customers/service_lines dimensions)
            |
            v
      SQL query layer      (v_journeys, v_journeys_analytics, parameterised filters)
            |
            v
    Pandas analytics layer (rolling windows, reshaping, feature engineering)
            |
   +--------+--------+--------------+----------------+
   |        |        |              |                |
  KPIs   Trends   Anomalies    Segmentation    Data quality
            |
            v
     Deterministic insights engine
            |
            v
         Streamlit UI
            |
            v
   Optional AI wording layer (evidence in, prose out)
```

**Why this shape.** The database does the heavy work: aggregating 60,000 journeys into
a handful of rows is what SQL is for, and it keeps the app responsive without loading
the dataset into memory on every interaction. Pandas takes over where SQL is awkward -
rolling baselines, pivoting, and building the customer feature matrix. Analytics
functions never touch Streamlit, so they are testable in isolation; pages never contain
calculations, so a metric can only be defined once.
    """
)

ui.section("Data flow")
st.markdown(
    """
1. On first run the generator creates the synthetic dataset and loads it into DuckDB.
   A manifest fingerprint (seed, size, date range) decides whether to rebuild.
2. Sidebar filters build a `FilterState`, which is translated into a parameterised
   `WHERE` clause. Values are always bound, never interpolated.
3. Analytics functions run one query per question and return small aggregates.
4. Results are cached by Streamlit against the SQL text and its bound parameters.
5. The insights engine and the AI tools consume those same functions, so every page,
   every finding and every answer agree by construction.
    """
)

ui.section("Analytics methods")
st.markdown(
    """
- **Period comparison** - the previous period is always the equal-length window
  immediately before the selection, so a 30-day view compares with the prior 30 days.
- **Rate metrics** use `FILTER (WHERE ...)` with a null-aware denominator, so missing
  values reduce the sample rather than counting as zeroes.
- **Anomaly detection** - trailing 28-day rolling mean and standard deviation, with the
  tested day excluded from its own baseline. IQR is offered as a robust alternative.
  Minimum volume and minimum movement floors prevent trivial findings.
- **Segmentation** - documented behavioural rules, plus standardised K-means with a
  silhouette score and elbow curve to inform the choice of k.
- **Insights** - a fixed set of generators, each producing a statement plus the evidence
  it was derived from. Findings are ranked by effect size, not by a model.
    """
)

ui.section("AI-assisted analytics")
st.markdown(
    """
The assistant classifies a question, calls **one** of ten named analytical functions,
and receives structured evidence back. Only then, and only if a key is configured, does
a language model see anything - and it is given the evidence plus the already-computed
answer, with instructions to reword rather than calculate. Every answer shows its
evidence in an expander, so a wrong rewording is visible immediately.

Without a key, the deterministic answer is shown directly. No page, chart or metric is
disabled.
    """
)
st.caption(
    f"AI wording is currently **{'available' if ai_available() else 'not configured'}** "
    "in this deployment."
)

ui.section("Testing")
st.markdown(
    """
`pytest` covers the parts where a silent error would be most damaging: KPI arithmetic
and period comparison, filter-to-SQL translation, the DuckDB schema and validity view,
anomaly detection against deliberately injected anomalies, data-quality rules against
deliberately injected defects, segmentation, and the reproducibility of the generator.

Tests run against a small, fast, in-memory database built with the same code path as the
real one.
    """
)

ui.section("Deployment")
st.markdown(
    """
Deployed on **Streamlit Community Cloud** from this folder alone. The dataset is
generated at first run inside the container, so there is no external database, no
credentials and no data to upload. `OPENAI_API_KEY` is optional and, when used, is read
from Streamlit secrets.
    """
)

ui.section("Security and privacy")
st.markdown(
    """
- The demo dataset is synthetic; there is no real personal information to protect.
- Uploaded files are parsed in memory for the session only and are never written to disk.
- Uploaded data is not sent anywhere unless you explicitly enable AI wording, and even
  then only aggregated figures leave the application - never individual records.
- No API key is committed. `.env.example` documents the variable; `.env` is ignored.
    """
)

ui.section("Limitations")
st.markdown(
    """
- The data is synthetic, so findings describe the generator, not the real world.
- Single-node DuckDB: the design suits tens of millions of rows on one machine, not a
  distributed warehouse.
- Segments and clusters are demonstrations of technique, not validated customer research.
- There is no authentication; it is a public demonstration by design.
    """
)

ui.footer(f"{DISCLAIMER}")
