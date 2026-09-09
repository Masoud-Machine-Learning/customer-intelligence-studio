"""Prompts for the optional language-model wording layer.

The model is given one job: turn a structured evidence dictionary into plain English.
It is explicitly forbidden from computing, estimating or introducing figures, and the
application shows the same evidence to the user underneath the answer so any deviation
is visible.
"""

from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = """You are the writing layer of an analytics application.

You receive a user question and a JSON object of figures that the application has
already computed from its database. Your only job is to express those figures as a
clear, plain-English answer for a business analyst.

Rules you must follow:
1. Use only numbers that appear in the supplied evidence. Never calculate, estimate,
   extrapolate or infer a figure that is not there.
2. If the evidence does not answer the question, say so plainly and describe what the
   evidence does show. Do not speculate about causes.
3. Do not invent context about real-world organisations, incidents or policies. The
   data is synthetic.
4. Keep to at most 120 words. Lead with the direct answer, then the supporting figures.
5. Write in British English, in a neutral analytical register. No marketing language,
   no exclamation marks, no emoji.
6. Round exactly as the evidence is rounded. Keep units (%, minutes, points) attached.
"""

USER_TEMPLATE = """Question: {question}

Deterministic answer already produced by the application:
{baseline_answer}

Evidence (JSON):
{evidence}

Rewrite the deterministic answer so it reads naturally and directly answers the
question. Keep every figure identical."""


def build_messages(
    question: str, baseline_answer: str, evidence: dict[str, Any]
) -> list[dict[str, str]]:
    """Assemble the chat messages for the wording call."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                question=question.strip(),
                baseline_answer=baseline_answer.strip(),
                evidence=summarise_evidence(evidence),
            ),
        },
    ]


def summarise_evidence(evidence: dict[str, Any], max_chars: int = 6000) -> str:
    """Serialise evidence compactly, truncating long tables rather than the summary.

    Only aggregated figures are ever serialised here - no row-level records leave the
    application.
    """
    trimmed: dict[str, Any] = {}
    for key, value in evidence.items():
        if isinstance(value, list) and len(value) > 12:
            trimmed[key] = value[:12] + [{"note": f"{len(value) - 12} further rows omitted"}]
        else:
            trimmed[key] = value
    text = json.dumps(trimmed, indent=1, default=str)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... (truncated)"
    return text


BRIEF_SYSTEM_PROMPT = """You are drafting an executive brief for a transport analytics
team. You will be given findings that have already been computed. Rewrite them as a
concise brief with the four supplied section headings.

Use only the supplied figures. Do not add findings, causes or recommendations that are
not implied directly by the numbers. Keep the whole brief under 300 words, in British
English, in a neutral register."""


def build_brief_messages(sections: dict[str, Any]) -> list[dict[str, str]]:
    """Messages for the optional executive-brief rewording."""
    return [
        {"role": "system", "content": BRIEF_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(sections, indent=1, default=str)[:8000]},
    ]
