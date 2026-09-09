"""Ask the Data - natural-language questions answered from computed evidence."""

from __future__ import annotations

import streamlit as st

from src.ai.assistant import ai_available, answer_question, classify_question
from src.ai.tools import TOOLS
from src.analytics.kpis import population_summary
from src.ui import components as ui
from src.ui.filters import render_filters
from src.ui.state import active_source, using_upload

ui.configure_page("Ask the Data")

source = active_source()
filters = render_filters(source, key_prefix="ask")

ui.page_header(
    "Ask the Data",
    "Questions are routed to an analytical function, answered from the database, and "
    "shown with the evidence used.",
)

summary = population_summary(source, filters)
ui.population_banner(summary, filters.describe())
if summary["journeys"] == 0:
    ui.footer()
    st.stop()

has_key = ai_available()

# ---------------------------------------------------------------------------
# How it works
# ---------------------------------------------------------------------------

with st.expander("How this works", expanded=not has_key):
    st.markdown(
        """
```
Your question
      v
Question classification  ->  selects one analytical tool
      v
Deterministic analytics  ->  SQL + Pandas over the filtered data
      v
Structured evidence      ->  the figures, shown below every answer
      v
Optional language model  ->  rewords the answer, adds no numbers
      v
Answer + evidence
```

The language model is never given database access and never computes anything. It
receives the evidence and the already-written answer, and may only rephrase them. With
no API key configured the application skips that step and shows the computed answer
directly, which is why every feature works without one.
        """
    )
    if has_key:
        st.success("An API key is configured. Natural-language rewording is available.")
    else:
        st.info(
            "No `OPENAI_API_KEY` is configured, so answers are shown exactly as the "
            "analytics layer computes them. Nothing is disabled."
        )

# ---------------------------------------------------------------------------
# Question input
# ---------------------------------------------------------------------------

st.write("")
examples = [tool.example for tool in TOOLS.values()]
chosen_example = st.selectbox(
    "Supported questions", ["Write my own question"] + examples, key="ask_example"
)

default_text = "" if chosen_example == "Write my own question" else chosen_example
question = st.text_input(
    "Your question",
    value=default_text,
    placeholder="Which customer segment experienced the largest decline in satisfaction?",
    key=f"ask_question_{chosen_example}",
)

control_a, control_b = st.columns([1, 2])
with control_a:
    use_llm = st.toggle(
        "Use AI wording",
        value=has_key,
        disabled=not has_key,
        help="Optional. Rephrases the computed answer; it cannot change the figures.",
        key="ask_use_llm",
    )
with control_b:
    if using_upload() and use_llm:
        st.warning(
            "You are analysing an uploaded dataset. With AI wording enabled, aggregated "
            "figures from your data (never individual records) are sent to OpenAI. Turn "
            "the toggle off to keep everything local.",
            icon=None,
        )

submitted = st.button("Ask", type="primary", disabled=not question.strip())

# ---------------------------------------------------------------------------
# Answer
# ---------------------------------------------------------------------------

if submitted and question.strip():
    tool_name, params, matched = classify_question(question)
    with st.spinner("Running the analysis..."):
        response = answer_question(source, filters, question, use_llm=use_llm)

    st.markdown(f"### {response.tool_title}")
    st.markdown(response.answer)

    if response.llm_note:
        st.caption(response.llm_note)

    routing = (
        f"Routed to `{response.tool_used}` - {TOOLS[response.tool_used].description}"
        if response.tool_used in TOOLS
        else f"Routed to `{response.tool_used}`"
    )
    if response.matched_terms:
        routing += f"  \nMatched on: {', '.join(response.matched_terms[:5])}"
    if params:
        routing += f"  \nDetected: {', '.join(f'{k}={v}' for k, v in params.items())}"
    st.caption(routing)

    if response.table is not None and not response.table.empty:
        st.dataframe(response.table, use_container_width=True, hide_index=True)
        ui.download_frame(response.table, "answer-data.csv", "Download this table", key="dl_answer")

    if response.llm_used and response.deterministic_answer != response.answer:
        with st.expander("Computed answer before AI rewording"):
            st.markdown(response.deterministic_answer)

    ui.evidence_panel(response.evidence)

    st.caption(
        "Answers describe synthetic demonstration data for the currently filtered population only."
    )

# ---------------------------------------------------------------------------
# Tool catalogue
# ---------------------------------------------------------------------------

ui.section("Available analytical tools")
st.caption(
    "The assistant may only call one of these. It cannot write its own query or invent a metric."
)
for tool in TOOLS.values():
    st.markdown(f"**`{tool.name}`** - {tool.description}  \n*Example:* {tool.example}")

ui.footer()
