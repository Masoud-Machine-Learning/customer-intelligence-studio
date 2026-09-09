"""Ask the Data assistant: question routing, grounding and optional LLM wording.

Pipeline
--------
1. **Classify** the question against the tool registry using keyword scoring, and
   extract any named entities (region, mode, segment) from the configured vocabulary.
2. **Run the tool**, which computes the answer with the same analytics functions the
   rest of the application uses.
3. **Return structured evidence** plus a complete deterministic answer.
4. **Optionally** hand the evidence to a language model to reword it.

Step 4 is the only part that needs an API key, and the application is fully usable
without it: the deterministic answer from step 3 is always shown or used as the final
answer. A model failure is caught and downgraded to the deterministic answer rather
than surfacing as an error.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.ai import prompts
from src.ai.tools import TOOLS, ToolResult
from src.data.database import DataSource
from src.data.filters import FilterState
from src.utils.config import (
    CUSTOMER_SEGMENTS,
    REGIONS,
    TRANSPORT_MODES,
    openai_api_key,
    openai_model,
)

DEFAULT_TOOL = "get_kpi_summary"


@dataclass
class AssistantResponse:
    """Everything the page needs to render one answer."""

    question: str
    answer: str
    tool_used: str
    tool_title: str
    evidence: dict[str, Any] = field(default_factory=dict)
    table: pd.DataFrame | None = None
    llm_used: bool = False
    llm_note: str = ""
    deterministic_answer: str = ""
    matched_terms: list[str] = field(default_factory=list)


def ai_available() -> bool:
    """True when an API key is configured and the client library is importable."""
    if not openai_api_key():
        return False
    try:
        import openai  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def _extract_entities(question: str) -> dict[str, str]:
    """Pull known dimension values out of the question text."""
    lowered = question.lower()
    found: dict[str, str] = {}

    for region in REGIONS:
        if re.search(rf"\b{region.lower()}\b", lowered):
            found["region"] = region
            break

    for mode in TRANSPORT_MODES:
        if re.search(rf"\b{mode.lower()}\b", lowered):
            found["transport_mode"] = mode
            break

    matched_segments = [s for s in CUSTOMER_SEGMENTS if _segment_mentioned(s, lowered)]
    if matched_segments:
        found["segment_a"] = matched_segments[0]
        if len(matched_segments) > 1:
            found["segment_b"] = matched_segments[1]
    return found


def _segment_mentioned(segment: str, lowered_question: str) -> bool:
    """Match a segment by its distinctive word rather than its full name."""
    tokens = {
        "Frequent Commuters": ("commuter", "frequent"),
        "Occasional Travellers": ("occasional", "traveller", "traveler"),
        "Off-Peak Regulars": ("off-peak regular", "off peak regular"),
        "Digital-First Customers": ("digital-first", "digital first"),
        "High-Service-Need Customers": ("high-service", "high service", "assisted"),
    }
    return any(token in lowered_question for token in tokens.get(segment, ()))


def classify_question(question: str) -> tuple[str, dict[str, Any], list[str]]:
    """Choose the tool best matching the question.

    Returns the tool name, its keyword arguments and the terms that matched, so the UI
    can show why a particular tool was selected.
    """
    lowered = (question or "").lower().strip()
    if not lowered:
        return DEFAULT_TOOL, {}, []

    scores: dict[str, tuple[int, list[str]]] = {}
    for name, tool in TOOLS.items():
        matched = [kw for kw in tool.keywords if kw in lowered]
        if matched:
            # Longer keyword matches are more specific, so weight by length.
            scores[name] = (sum(len(kw) for kw in matched), matched)

    entities = _extract_entities(lowered)

    if not scores:
        return DEFAULT_TOOL, entities, []

    best = max(scores.items(), key=lambda item: item[1][0])
    tool_name, (_, matched_terms) = best

    # A question naming two segments is a comparison regardless of other keywords.
    if "segment_b" in entities:
        tool_name = "compare_segments"

    return tool_name, entities, matched_terms


def _run_llm(question: str, result: ToolResult) -> tuple[str, bool, str]:
    """Reword the deterministic answer with the configured model.

    Any failure returns the deterministic answer with an explanatory note, so the page
    never breaks because of the optional dependency.
    """
    try:
        from openai import OpenAI

        client = OpenAI(api_key=openai_api_key())
        response = client.chat.completions.create(
            model=openai_model(),
            messages=prompts.build_messages(question, result.answer, result.evidence),
            temperature=0.2,
            max_tokens=400,
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            return result.answer, False, "The model returned an empty response."
        return text, True, ""
    except Exception as exc:  # noqa: BLE001 - degrade instead of failing the page
        return (
            result.answer,
            False,
            f"AI wording unavailable ({type(exc).__name__}); showing the computed answer.",
        )


def answer_question(
    source: DataSource,
    filters: FilterState,
    question: str,
    use_llm: bool = False,
) -> AssistantResponse:
    """Answer a question about the currently filtered data."""
    tool_name, params, matched = classify_question(question)
    tool = TOOLS.get(tool_name, TOOLS[DEFAULT_TOOL])
    result = tool.runner(source, filters, **params)

    answer = result.answer
    llm_used = False
    note = ""
    if use_llm and ai_available():
        answer, llm_used, note = _run_llm(question, result)
    elif use_llm:
        note = (
            "No OPENAI_API_KEY is configured, so the computed answer is shown directly. "
            "Every figure below comes from the analytics layer."
        )

    return AssistantResponse(
        question=question,
        answer=answer,
        tool_used=result.tool,
        tool_title=result.title,
        evidence=result.evidence,
        table=result.table,
        llm_used=llm_used,
        llm_note=note,
        deterministic_answer=result.answer,
        matched_terms=matched,
    )


def reword_brief(sections: dict[str, Any]) -> tuple[str, bool, str]:
    """Optionally improve the executive brief's wording; never required."""
    if not ai_available():
        return "", False, "No API key configured - the deterministic brief is used as written."
    try:
        from openai import OpenAI

        client = OpenAI(api_key=openai_api_key())
        response = client.chat.completions.create(
            model=openai_model(),
            messages=prompts.build_brief_messages(sections),
            temperature=0.3,
            max_tokens=700,
        )
        return (response.choices[0].message.content or "").strip(), True, ""
    except Exception as exc:  # noqa: BLE001
        return "", False, f"AI rewording unavailable ({type(exc).__name__})."
