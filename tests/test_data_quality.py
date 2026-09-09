"""Data-quality rules, checked against deliberately injected defects."""

from __future__ import annotations

import pandas as pd

from src.analytics.data_quality import (
    CATEGORY_WEIGHTS,
    PROFILED_FIELDS,
    build_quality_report,
)
from src.data.database import DataSource, build_upload_database


def test_report_covers_every_check_family(source, filters):
    report = build_quality_report(source, filters)
    categories = {check.category for check in report.checks}

    assert categories == set(CATEGORY_WEIGHTS)


def test_injected_defects_are_detected(source, filters):
    """The generator plants nulls, impossible values and duplicates; all must be found."""
    report = build_quality_report(source, filters)
    by_key = {check.key: check for check in report.checks}

    assert by_key["missing_satisfaction"].failed_rows > 0
    assert by_key["invalid_delay"].failed_rows > 0
    assert by_key["invalid_transport_mode"].failed_rows > 0
    assert by_key["duplicate_journey_id"].failed_rows > 0


def test_excluded_rows_equal_the_gap_between_the_views(source, filters):
    report = build_quality_report(source, filters)

    assert report.total_rows > report.analysable_rows
    assert report.excluded_rows == report.total_rows - report.analysable_rows


def test_score_is_bounded_and_reflects_failures(source, filters):
    report = build_quality_report(source, filters)

    assert 0.0 <= report.score <= 100.0
    assert report.score < 100.0, "planted defects must cost score"
    assert report.rating in {"Good", "Acceptable", "Needs attention"}


def test_failure_rate_is_a_percentage_of_total_rows(source, filters):
    report = build_quality_report(source, filters)

    for check in report.checks:
        assert 0.0 <= check.failure_rate <= 100.0
        if check.failed_rows == 0:
            assert check.passed
            assert check.severity == "Pass"


def test_field_completeness_covers_the_profiled_fields(source, filters):
    report = build_quality_report(source, filters)

    assert set(report.field_completeness["field"]) == set(PROFILED_FIELDS)
    assert (report.field_completeness["completeness_pct"] <= 100.0).all()
    assert (report.field_completeness["completeness_pct"] >= 0.0).all()


def test_conditional_fields_are_excluded_from_the_average(source, filters):
    """Complaint category is empty for most rows by design and must not drag the score."""
    report = build_quality_report(source, filters)
    conditional = report.field_completeness[report.field_completeness["conditional"]]

    assert not conditional.empty
    assert report.completeness_pct > conditional["completeness_pct"].max()


def test_outliers_are_counted_but_not_excluded(source, filters):
    report = build_quality_report(source, filters)

    assert "delay_minutes" in report.outliers
    assert report.outliers["delay_minutes"] >= 0


def test_a_clean_dataset_scores_full_marks():
    """A feed with no defects must reach 100, otherwise the score is not meaningful."""
    frame = pd.DataFrame(
        {
            "journey_id": [f"J{i}" for i in range(50)],
            "customer_id": [f"C{i % 10}" for i in range(50)],
            "date": ["2026-01-01"] * 50,
            "region": ["North"] * 50,
            "transport_mode": ["Bus"] * 50,
            "customer_segment": ["Frequent Commuters"] * 50,
            "delay_minutes": [2.0] * 50,
            "customer_satisfaction": [4] * 50,
            "journey_duration_minutes": [20.0] * 50,
            "complaint_flag": [False] * 50,
        }
    )
    connection = build_upload_database(frame)
    clean = DataSource(connection=connection, manifest={}, origin="test-clean")
    report = build_quality_report(clean, None)

    assert report.excluded_rows == 0
    assert report.score == 100.0


def test_evidence_is_json_serialisable(source, filters):
    report = build_quality_report(source, filters)
    evidence = report.to_evidence()

    assert evidence["total_rows"] == report.total_rows
    assert isinstance(evidence["failing_checks"], list)
