# Skills Matrix - Evidence from Customer Intelligence Studio

An honest mapping of this project against the advertised skill list. Classifications are
deliberately conservative: a skill is only **directly demonstrated** if a reviewer could
open the repository and point at working code that exercises it in a non-trivial way.

Where a skill is not demonstrated, that is stated plainly rather than stretched. It is
better to show six technologies genuinely than to list fifteen superficially.

| Skill | Classification | Evidence |
| --- | --- | --- |
| **Python Development** | **Directly demonstrated** | ~4,500 lines of application code across a layered package (`src/data`, `src/analytics`, `src/ai`, `src/ui`, `src/utils`). Dataclasses, type hints, docstrings, a metric registry, dependency-light design, no calculations in page scripts. |
| **Pandas / Data Processing** | **Directly demonstrated** | Rolling baselines with `shift(1).rolling(...)` in `src/analytics/anomalies.py`; gap-free calendar reindexing and pivoting in `src/analytics/trends.py`; customer feature engineering in `src/analytics/segmentation.py`; vectorised synthetic data generation in `src/data/generator.py`. |
| **Streamlit** | **Directly demonstrated** | A nine-view multi-page application: shared sidebar filters persisted in session state, `@st.cache_resource` and `@st.cache_data` used for their correct purposes, `st.column_config` formatting, file upload with validation, download buttons, custom CSS, and a custom theme. |
| **SQL** | **Directly demonstrated** | All aggregation is SQL: `FILTER (WHERE ...)` null-aware rates, `QUALIFY row_number() OVER (...)` de-duplication, CTEs, `FULL OUTER JOIN` period comparison, `quantile_cont`, `date_trunc`, parameterised `IN` lists. See `src/analytics/kpis.py` and `src/data/database.py`. |
| **Data Modelling** | **Directly demonstrated** | A star schema (`journeys` fact, `customers` and `service_lines` dimensions), a canonical schema contract in `src/data/schema.py` shared by the generator, the database and upload validation, and a two-view design separating raw profiling from analysable data. |
| **Data Visualisation Design** | **Directly demonstrated** | Every chart is built through `src/ui/charts.py` from one specification: a restrained palette defined as tokens, consistent axes and hover formats, accessible contrast, deliberate chart-type choice, and no decorative gradients or 3D. |
| **Performance Optimisation** | **Directly demonstrated** | Aggregation pushed to the database rather than the client; result caching keyed on SQL text plus bound parameters; database-side sorting and limiting for the record table; one grouped query serving a multi-metric, multi-scope anomaly scan; a manifest fingerprint that avoids regenerating data. Measured timings are documented in the README. |
| **Technical Documentation** | **Directly demonstrated** | A README with architecture rationale, a Mermaid diagram, a SQL-versus-Pandas table and measured performance; an in-app About page; module-level docstrings explaining *why*, not only *what*; this document and the interview guide. |
| **Git / Version Control** | **Directly demonstrated** | The project is developed in a Git repository as a self-contained folder with its own `.gitignore`, ignoring generated data and secrets. |
| **CI/CD & Deployment** | **Directly demonstrated** | `.github/workflows/ci.yml` runs Ruff lint, Ruff format check, the pytest suite and a database smoke test on two Python versions, path-filtered to this folder. Deployment to Streamlit Community Cloud is documented step by step and the project is deployable from this folder alone. |
| **Cloud Application Development** | **Partially demonstrated** | The app is built for and deployed to a managed cloud platform (Streamlit Community Cloud): relative POSIX paths, secrets read from the platform's store, no local-only assumptions, ephemeral-filesystem-safe data generation. It does **not** demonstrate IaC, container orchestration, VPC networking or managed cloud data services. |
| **REST API Integration** | **Partially demonstrated** | The OpenAI API is integrated through its official client in `src/ai/assistant.py`, with optional configuration, graceful degradation on failure, and a prompt contract. It is a single third-party integration, not a broad API-integration portfolio, and this project exposes no API of its own. |
| **Stakeholder Engagement** | **Partially demonstrated** | The artefacts are stakeholder-facing by design: an executive brief with four decision-oriented sections and Markdown/text export, plain-language explanations of how data quality affects reported numbers, and stated limitations on every analytical method. Actual engagement with stakeholders cannot be shown by a portfolio project. |
| **User Training & Support** | **Partially demonstrated** | The application is self-documenting: an in-app About page covering architecture and methods, a "How detection works" explainer, a "How these metrics are calculated" panel exposing each metric's SQL, tool descriptions on the Ask the Data page, and error messages that say what to do next. No training delivery or support process is evidenced. |
| **JavaScript** | **Not demonstrated** | No JavaScript is written in this project. Plotly's runtime is JavaScript, but that is a library dependency, not authored code. |
| **Dash** | **Not demonstrated** | Streamlit is used. Dash does not appear in this project. |
| **Tableau** | **Not demonstrated** | Not used. |
| **Power BI** | **Not demonstrated** | Not used. |
| **Snowflake** | **Not demonstrated** | DuckDB is used. The SQL is standard enough to port, and the `DataSource` abstraction is where a warehouse connector would go, but no Snowflake code exists here. |
| **Databricks** | **Not demonstrated** | Not used. No Spark, no notebooks, no Delta Lake. |
| **Azure Data Platform** | **Not demonstrated** | Not used. No ADF, Synapse, Fabric or Azure SQL. |
| **AWS / GCP** | **Not demonstrated** | Not used. Deployment is to Streamlit Community Cloud. |

## Summary

| Classification | Count | Skills |
| --- | --- | --- |
| Directly demonstrated | 10 | Python, Pandas, Streamlit, SQL, Data Modelling, Data Visualisation Design, Performance Optimisation, Technical Documentation, Git, CI/CD & Deployment |
| Partially demonstrated | 4 | Cloud Application Development, REST API Integration, Stakeholder Engagement, User Training & Support |
| Not demonstrated | 8 | JavaScript, Dash, Tableau, Power BI, Snowflake, Databricks, Azure Data Platform, AWS/GCP |

## How to talk about the gaps

The gaps are real, and the credible position is to name them and show the nearest
transferable evidence:

- **Snowflake / Databricks / Azure.** "I have not used them in production. What I can
  show is that the analytical work here is standard SQL over a columnar engine, and the
  data access sits behind one abstraction, so moving to a warehouse changes the connector
  and the dialect, not the analytics or the application."
- **Tableau / Power BI.** "I have built the equivalent from first principles: a semantic
  layer where each metric is defined once, consistent cross-page filtering, drill-down to
  underlying records, and export. The concepts transfer; the tool is new."
- **Dash.** "I chose Streamlit deliberately and can explain the trade-off. Dash's callback
  model is closer to a conventional reactive UI, which I would expect to pick up quickly
  from this base."
- **JavaScript.** "Not used here. I would not claim front-end JavaScript."
