# Interview Guide - Customer Intelligence Studio

Answers are based **only** on what is actually built in this repository. Where something
is not implemented, the answer says so and describes what would be done instead.

---

## Design and technology choices

### Why Streamlit?

The audience is analysts and operational managers, and the work is analytical rather than
transactional. Streamlit lets the same Python that computes a metric also render it, so
there is no serialisation boundary between the analysis and the interface and no separate
front-end to keep in sync.

The trade-offs are real and worth naming: Streamlit re-runs the whole script on every
interaction, so performance depends on disciplined caching, and the layout system is
constrained compared with a real component framework. Both are managed here - aggregation
is pushed into SQL so a re-run is cheap, and the visual language is centralised in
`src/ui/components.py` and `src/ui/charts.py` rather than fought with per page.

### Why DuckDB rather than Pandas alone, or Postgres?

DuckDB is an embedded columnar OLAP engine: no server, no credentials, no network hop,
and it deploys as a pip install, which matters for a demo that must run anywhere.

Against Pandas alone: SQL expresses the actual analytical questions better - grouped
aggregation, window functions, `FILTER (WHERE ...)`, full outer joins for period
comparison - and it does not require the whole dataset in memory. Against Postgres: there
is no operational benefit here, and Postgres is row-oriented, so scans over 60,000 rows
with wide aggregations are slower for no gain.

### Why both SQL and Pandas? Isn't that two ways to do the same thing?

They do different jobs, and the split is deliberate and documented.

SQL does set-based work: filtering, grouping, rates, percentiles, de-duplication,
joining. It reduces 60,000 journeys to tens of rows before Python is involved.

Pandas does order-dependent and reshaping work: trailing rolling means and standard
deviations for anomaly baselines, reindexing onto a gap-free calendar, pivoting into a
day-by-hour matrix, and building the customer feature matrix for clustering. Those are
possible in SQL but far less readable, and they operate on small results anyway.

The rule I applied: **if it reduces rows, it belongs in SQL; if it reshapes or sequences
an already-small result, it belongs in Pandas.**

### How did you structure the application?

Four layers, with a strict dependency direction:

```
src/data       generation, DuckDB schema and views, filter-to-SQL, loading and caching
src/analytics  KPIs, trends, anomalies, segmentation, data quality, insights, records
src/ai         analytical tools, prompts, the optional assistant
src/ui         components, charts, filter widgets, session state
pages/         layout and narrative only
```

Analytics never imports Streamlit, which is why the entire analytical surface is testable
without a browser. Pages contain no calculations. Metrics are defined once in a registry
(`src/analytics/kpis.py::METRICS`) as SQL expressions with a unit and a description, so
the Executive Overview, the insights engine and the AI assistant cannot disagree about
what "on-time performance" means.

---

## Application behaviour

### How do filters work?

The sidebar widgets build an immutable `FilterState` (`src/data/filters.py`). Nothing
else in the app touches Streamlit widget values. `build_where()` turns that state into a
`WHERE` clause plus a list of bound parameters, driven by a fixed `FILTER_COLUMNS`
mapping from attribute name to column name.

Three consequences matter:

1. Values are always bound as parameters; only column names are interpolated, and those
   come from an allow-list, so user input cannot reach the SQL text.
2. The state is a frozen dataclass of tuples, so it is hashable and can key a cache.
3. "Previous period" is derived from the state itself - the equal-length window ending
   the day before the selection - so every page compares like with like automatically.

An empty multi-select means "no restriction on this dimension", which is what users
expect from a filter they have not touched.

### How did you improve performance?

Measured, then targeted:

- **`@st.cache_resource`** for the DuckDB connection - a handle created once, never
  copied per session. **`@st.cache_data`** for query results, keyed on the SQL text and
  the bound parameters with a 30-minute TTL.
- **Aggregation pushed into the database.** A KPI row is one scan returning one row. The
  browser never receives 60,000 records.
- **One query serving many metrics.** The anomaly scan needs five metrics across six
  regions over a window plus its baseline history; that is a single grouped query, then
  windowing in Pandas.
- **Server-side paging for the record table** - `ORDER BY ... LIMIT` with the column list
  chosen before the query runs.
- **Filter option lists cached separately** from the data, because they change only when
  the dataset does.
- **A manifest fingerprint** over seed, size and date range, so the dataset is generated
  once rather than on every cold start.

On the 60,000-row demo dataset: generation 0.4 s, database build 1.1 s, KPI comparison
0.9 s, six-region anomaly scan 0.4 s, full insight generation 0.8 s. Warm interactions
are effectively instant.

The honest remaining bottleneck is the anomaly scan's Python loop over days and scopes.
It is fine at this scale; at ten times the scope count I would vectorise it with a
grouped Pandas operation rather than iterating rows.

### How does anomaly detection work?

For each day, and optionally within each dimension value, a baseline is built from the
**preceding 28 days**, with the day under test excluded from its own baseline via
`shift(1)`. That exclusion matters: without it, an extreme day inflates the standard
deviation it is being tested against and hides itself.

Two methods, both deterministic:

- **Rolling z-score** - `|value - baseline mean| / baseline standard deviation` against a
  configurable threshold, default 2.5.
- **IQR** - outside `Q1 - 1.5 x IQR` to `Q3 + 1.5 x IQR` of the trailing window; more
  robust when the baseline itself contains outliers.

Two guards stop noise being reported: a minimum number of journeys that day, and a
minimum absolute movement per metric. Without them a very stable series produces large
z-scores from trivial changes.

Baseline history is fetched from *before* the selected window, so filtering to the last
30 days still yields a real comparison rather than an empty one.

I did not use Isolation Forest. It would add a dependency, non-obvious tuning and an
unexplainable score, in exchange for nothing this problem needs. The test suite verifies
detection against anomalies the generator deliberately plants.

### How is the LLM grounded? How do you prevent hallucinations?

The model is never in the analytical path. The pipeline is:

1. **Classify** the question by keyword score against ten named tools, and extract entities
   from a fixed vocabulary of regions, modes and segments.
2. **Run exactly one tool**, which uses the same analytics functions as the rest of the
   app and returns both a structured `evidence` dictionary and a complete written answer.
3. **Optionally** send the evidence and that answer to the model, with a system prompt
   that forbids computing, estimating or introducing any figure and caps the length.
4. **Show the evidence** in an expander under every answer.

So the model has no database access, cannot choose a calculation, never sees row-level
data, and its output is checkable against the evidence displayed beside it. If the API
call fails, the deterministic answer is shown with a note - the page does not error. With
no key configured the model step is skipped entirely and nothing is disabled.

The deeper point: **the best insights do not need an LLM at all.** The insights engine is
ten deterministic generators, each producing a headline plus the evidence it was computed
from. The model is a wording layer, not an analyst.

### How is data quality measured?

Nine rules in three families, evaluated against the *raw* view rather than the analytics
view, because quality must be measured on what the feed actually delivered:

- **Completeness** - null customer ID, missing region, missing satisfaction.
- **Validity** - satisfaction outside 1-5, non-positive or implausible duration,
  impossible delay (the classic `999` sentinel), unrecognised transport mode.
- **Uniqueness** - duplicate journey ID.

The score is weighted 40/40/20 across those families. Beyond the score, the page reports
field-level completeness (flagging conditionally-populated fields so their expected
sparsity does not distort the average), outliers beyond 3x IQR, freshness, and - the part
that matters most - an explicit account of how each issue changes the numbers on the
other pages.

The architectural decision I would highlight: the validity rules exist as a **SQL view**
(`v_journeys_analytics`), not as a filter each query remembers to apply. No analytical
query can bypass them. The difference between the two views' row counts is reported to
the user rather than hidden.

---

## Scaling, integration and operations

### How would you scale this to millions of records?

DuckDB handles tens of millions of rows on one machine, so the first answer is that much
of it already scales: aggregation is in the database, the client only ever receives small
results, and nothing loads the full dataset into memory.

Beyond that, in order:

1. **Store Parquet, not a single table.** Partition by month so scans prune by date, which
   is the filter that always applies.
2. **Pre-aggregate the common grains** - daily by region, mode and segment - into a
   summary table. Most pages never need journey grain.
3. **Vectorise the anomaly scan** and move rolling statistics into SQL window functions.
4. **Sample for the interactive scatter plots**, which is already done at 4,000 points.
5. **Move the store behind the `DataSource` abstraction** to a warehouse if the data
   outgrows one machine.

The limit I would actually hit first is not the database; it is Streamlit's full-script
re-run model with many concurrent users.

### How would you connect this to Snowflake or Databricks?

I have not used either in production, and I would not claim otherwise.

Structurally, the change is contained. `DataSource` wraps a connection, and every query
goes through `run_query`. Swapping DuckDB for a Snowflake connector means implementing
that interface and adjusting dialect specifics - `QUALIFY` exists in Snowflake,
`quantile_cont` becomes `percentile_cont`, `FILTER (WHERE ...)` becomes
`COUNT_IF`/`CASE`. The analytics functions, the metric registry, the insights engine and
every page stay as they are.

I would also change *where* the work happens: with a warehouse I would push the
pre-aggregation into scheduled models (dbt) and have the app query small marts rather than
re-aggregating the fact table on every interaction.

### How would you deploy it internally?

Streamlit Community Cloud is right for a public demo. Internally I would containerise it
(`python:3.12-slim`, `pip install -r requirements.txt`, `streamlit run app.py`) and run it
behind the organisation's reverse proxy and identity provider, with the dataset replaced
by a warehouse connection and secrets injected from the platform's secret store rather
than a file. The app already reads its key from Streamlit secrets or the environment, and
uses only relative paths, so nothing in the code blocks that.

### How would you handle authentication?

There is none today - it is a public demonstration by design, and it holds only synthetic
data.

For an internal deployment I would not build auth into the app. I would put it behind the
existing SSO at the proxy or platform layer (OIDC through an identity-aware proxy), which
keeps credential handling out of application code. If per-user data restrictions were
needed, I would pass the authenticated identity into the filter layer as a mandatory
predicate applied in `build_where`, so it cannot be forgotten by an individual query - the
same reasoning as putting validity rules in a view.

### How would you secure uploaded customer data?

What the app does now: uploaded files are parsed **in memory only** for the session,
never written to disk, never persisted between sessions, and never sent anywhere unless
the user explicitly enables AI wording - and the page warns them before that is possible.
Even then, only aggregated evidence leaves the application; row-level records never do.

For real customer data I would add: transport-level protection and SSO before the app is
reachable; a documented retention position (currently "nothing is retained"); PII
minimisation at ingest so identifiers never enter the analytical store in the first place;
and an explicit, off-by-default setting for any external AI call, with an audit record of
what was sent.

### How would you monitor the application?

Not implemented, so I would describe rather than claim. Streamlit Community Cloud gives
logs and basic health. For anything real I would add structured logging around
`run_query` with the tool name, filter hash and duration; export those as metrics
(p95 query latency, cache hit ratio, error rate per page); alert on error rate and on the
data-quality score crossing a threshold, since a silent upstream feed change is the most
likely failure; and add uptime checks against the health endpoint.

The data-quality page is itself a monitoring surface - it is the check that would catch a
feed problem before anyone acted on a wrong number.

---

## Working with stakeholders

### How would you gather requirements for something like this?

I would start from decisions, not dashboards: who acts on this, what do they decide, how
often, and what would change their mind. That is what produced the four sections of the
executive brief - current performance, key changes, areas requiring attention, positive
developments - rather than a wall of charts.

Practically: interview the two or three people who would use it weekly; find the report
they build by hand today, because that is the real requirement; agree metric definitions
in writing *before* building, since most analytics disputes are definitional; and get
agreement on the comparison basis, which is why "previous period" here is explicitly the
equal-length preceding window rather than something ambiguous.

### How would you conduct UAT?

Scenario-based rather than feature-based: give each user three real questions they had
last month and ask them to answer them with the app unaided. Watch where they hesitate.

Alongside that: a reconciliation pass against an existing trusted report, with any
difference explained rather than adjusted away - usually it is a definitional or
data-quality difference, and the Data Quality page exists partly to make that
conversation possible; explicit sign-off on metric definitions; and edge cases like an
empty filter combination, which the app handles with a clear message rather than a stack
trace.

### How would you train and support business users?

The strongest form of training is an application that explains itself, which is what the
in-app documentation is for: the About page covering methods, the "How these metrics are
calculated" panel that shows each metric's actual SQL, the "How detection works"
explainer on the anomaly page, and stated limitations on every analytical method.

On top of that I would run a short scenario-led session rather than a feature tour, write
a one-page "how to answer the five most common questions" guide, and keep a feedback route
open. For support, the useful signal is which questions people ask that the app cannot
answer - that is the backlog.

---

## Questions about honesty and limitations

### What are the weaknesses of this project?

- The data is synthetic. Findings describe the generator, not the real world, and I would
  never present them as transport research.
- Question classification is keyword-based, not a learned intent model. It is transparent
  and testable, but it will not understand an arbitrary question - the supported set is
  listed in the UI rather than hidden.
- K-means on behavioural data is a discovery tool, not a validated segmentation. The page
  says so, and shows the silhouette score and limitations rather than only the clusters.
- No authentication, no monitoring, no warehouse integration. Those are deliberate scope
  decisions for a portfolio demonstration, not oversights.

### What would you do differently next time?

I would build the metric registry first. It emerged partway through, and everything got
simpler once every metric had exactly one definition. I would also write the anomaly scan
as a vectorised grouped operation from the start rather than a row loop - it works fine at
this scale, but it is the one place where the code shape would not survive a large
increase in scope.
