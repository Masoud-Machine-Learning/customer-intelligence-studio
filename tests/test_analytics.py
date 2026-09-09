"""Segmentation, insights and the grounded AI routing layer."""

from __future__ import annotations

import pytest

from src.ai.assistant import answer_question, classify_question
from src.ai.tools import TOOLS
from src.analytics.insights import build_executive_brief, generate_insights
from src.analytics.records import count_records, fetch_records
from src.analytics.segmentation import (
    CLUSTER_FEATURES,
    RULE_THRESHOLDS,
    apply_rule_segments,
    cluster_customers,
    compare_segments,
    customer_features,
    rule_segmentation,
)
from src.analytics.trends import period_over_period_by_dimension, time_series
from src.utils.config import CUSTOMER_SEGMENTS

# --------------------------------------------------------------------------- trends


def test_time_series_returns_one_row_per_day(source, filters):
    series = time_series(source, filters, grain="day")

    assert not series.empty
    assert series["period"].is_unique
    assert len(series) <= filters.period_days


def test_coarser_grain_produces_fewer_rows(source, filters):
    daily = time_series(source, filters, grain="day")
    monthly = time_series(source, filters, grain="month")

    assert len(monthly) < len(daily)


def test_unknown_grain_is_rejected(source, filters):
    with pytest.raises(ValueError):
        time_series(source, filters, grain="fortnight")


def test_period_comparison_by_dimension_is_internally_consistent(source, filters):
    frame = period_over_period_by_dimension(
        source, filters, "region", "avg_delay_minutes", min_journeys=10
    )
    usable = frame.dropna(subset=["current_value", "previous_value"])

    assert not usable.empty
    for _, row in usable.iterrows():
        assert row["absolute_change"] == pytest.approx(row["current_value"] - row["previous_value"])


# ---------------------------------------------------------------- segmentation


def test_customer_features_respect_the_minimum_journey_threshold(source, filters):
    features = customer_features(source, filters, min_journeys=4)

    assert not features.empty
    assert features["journeys"].min() >= 4
    assert features["customer_id"].is_unique
    assert features["customer_id"].notna().all()


def test_rule_segments_are_exhaustive_and_documented(source, filters):
    result = rule_segmentation(source, filters, min_journeys=2)

    assert result.customers["rule_segment"].notna().all()
    assert set(result.customers["rule_segment"]).issubset(set(CUSTOMER_SEGMENTS))
    assert result.limitations


def test_rule_thresholds_actually_separate_customers(source, filters):
    features = apply_rule_segments(customer_features(source, filters, min_journeys=2))
    frequent = features[features["rule_segment"] == "Frequent Commuters"]

    if not frequent.empty:
        assert frequent["journeys_per_week"].min() >= RULE_THRESHOLDS["frequent_journeys_per_week"]


def test_clustering_is_deterministic_and_labelled(source, filters):
    features = customer_features(source, filters, min_journeys=3)
    first = cluster_customers(features, n_clusters=4)
    second = cluster_customers(features, n_clusters=4)

    assert first.customers["cluster"].tolist() == second.customers["cluster"].tolist()
    assert first.customers["cluster_label"].notna().all()
    assert first.customers["cluster_label"].nunique() == 4
    assert first.silhouette is not None


def test_cluster_profile_accounts_for_every_customer(source, filters):
    features = customer_features(source, filters, min_journeys=3)
    result = cluster_customers(features, n_clusters=3)
    profile = result.profile()

    assert profile["customers"].sum() == len(result.customers)
    assert profile["share_pct"].sum() == pytest.approx(100.0, abs=0.5)


def test_clustering_respects_selected_features(source, filters):
    features = customer_features(source, filters, min_journeys=3)
    chosen = ("journeys_per_week", "peak_share")
    result = cluster_customers(features, n_clusters=3, feature_columns=chosen)

    assert result.feature_columns == chosen
    assert set(chosen).issubset(set(CLUSTER_FEATURES))


def test_segment_comparison_returns_both_sides(source, filters):
    comparison = compare_segments(source, filters, "Frequent Commuters", "Occasional Travellers")

    assert not comparison.empty
    assert "Frequent Commuters" in comparison.columns
    assert "Occasional Travellers" in comparison.columns


# -------------------------------------------------------------------- insights


def test_insights_are_generated_with_evidence(source, filters):
    insights = generate_insights(source, filters, limit=8)

    assert insights
    assert len(insights) <= 8
    for insight in insights:
        assert insight.headline.endswith(".")
        assert insight.evidence
        assert insight.direction in {"good", "bad", "neutral"}


def test_insight_ordering_is_stable(source, filters):
    first = generate_insights(source, filters, limit=8)
    second = generate_insights(source, filters, limit=8)

    assert [i.key for i in first] == [i.key for i in second]
    assert [i.headline for i in first] == [i.headline for i in second]


def test_executive_brief_has_every_section(source, filters):
    brief = build_executive_brief(source, filters)
    markdown = brief.to_markdown()

    assert "Current performance" in markdown
    assert "Key changes" in markdown
    assert "Areas requiring attention" in markdown
    assert "Positive developments" in markdown
    assert brief.current_performance
    assert brief.evidence["kpis"]


# ----------------------------------------------------------------- record table


def test_records_are_limited_and_filtered_in_sql(source, filters):
    records = fetch_records(source, filters, limit=25)

    assert len(records) <= 25
    assert not records.empty


def test_record_subsets_narrow_the_result(source, filters):
    everything = count_records(source, filters, "All journeys")
    complaints = count_records(source, filters, "Complaints only")

    assert 0 < complaints < everything

    frame = fetch_records(source, filters, subset="Complaints only", limit=50)
    assert frame["complaint_flag"].all()


def test_record_search_is_parameterised(source, filters):
    """A search string containing SQL must be treated as text, not code."""
    frame = fetch_records(source, filters, search="'; DROP TABLE journeys; --", limit=10)

    assert frame.empty
    assert count_records(source, filters) > 0


# ----------------------------------------------------------------- ai routing


@pytest.mark.parametrize(
    ("question", "expected_tool"),
    [
        ("Summarise current service performance.", "get_kpi_summary"),
        (
            "Which customer segment experienced the largest decline in satisfaction?",
            "analyse_satisfaction",
        ),
        ("Which transport mode has the highest complaint rate?", "analyse_complaints"),
        ("Were there any unusual days this period?", "find_anomalies"),
        ("What are the three main issues I should investigate?", "top_issues"),
        ("How complete is the underlying data?", "assess_data_quality"),
    ],
)
def test_questions_route_to_the_expected_tool(question, expected_tool):
    tool_name, _, _ = classify_question(question)

    assert tool_name == expected_tool


def test_named_entities_are_extracted():
    _, params, _ = classify_question("How are delays in the West region?")

    assert params.get("region") == "West"


def test_two_named_segments_force_a_comparison():
    tool_name, params, _ = classify_question(
        "Compare frequent commuters with occasional travellers"
    )

    assert tool_name == "compare_segments"
    assert params["segment_a"] == "Frequent Commuters"
    assert params["segment_b"] == "Occasional Travellers"


def test_unrecognised_question_falls_back_to_the_summary():
    tool_name, _, _ = classify_question("xyzzy")

    assert tool_name == "get_kpi_summary"


def test_answers_work_without_an_api_key(source, filters, monkeypatch):
    """The whole feature must function with AI disabled."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = answer_question(
        source, filters, "Summarise current service performance.", use_llm=True
    )

    assert response.answer
    assert response.answer == response.deterministic_answer
    assert response.llm_used is False
    assert response.evidence


def test_every_registered_tool_runs(source, filters):
    """Guards against a tool that raises only when a user happens to ask for it."""
    for name, tool in TOOLS.items():
        result = tool.runner(source, filters)

        assert result.tool
        assert result.answer, f"{name} produced no answer"
        assert isinstance(result.evidence, dict)
