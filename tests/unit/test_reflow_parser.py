"""Unit tests for reflow.py's result.md gap-analysis parser."""

import pytest

import reflow


def test_parses_full_job_match_report(tmp_path):
    report = tmp_path / "result.md"
    report.write_text(
        "# Job Match Evaluation: Jane\n"
        "**Target Role:** Backend Engineer\n\n"
        "**Overall Match:** 62.0/100\n"
        "**Weight profile:** engineering\n\n"
        "**Required skills MISSING:**\n"
        "kubernetes, terraform\n\n"
        "**Preferred skills missing:**\n"
        "None\n\n"
        "## Areas for Improvement\n"
        "- Quantify impact\n"
        "- Mention CI/CD\n",
        encoding="utf-8",
    )

    gap = reflow.parse_result_markdown(str(report))

    assert gap.job_title == "Backend Engineer"
    assert gap.weight_profile == "engineering"
    assert gap.missing_required_skills == ["kubernetes", "terraform"]
    assert gap.missing_preferred_skills == []
    assert gap.improvement_areas == ["Quantify impact", "Mention CI/CD"]


def test_parses_flagged_requirement_gate_report(tmp_path):
    report = tmp_path / "result.md"
    report.write_text(
        "# Requirement Gate: Jane\n"
        "**Target Role:** Backend Engineer\n\n"
        "**Status:** FLAGGED\n\n"
        "## Features Kept\n"
        "- python\n\n"
        "## Features to Add\n"
        "- kubernetes\n"
        "- security clearance\n",
        encoding="utf-8",
    )

    gap = reflow.parse_result_markdown(str(report))

    assert gap.job_title == "Backend Engineer"
    assert gap.weight_profile == "engineering"
    assert gap.missing_required_skills == ["kubernetes", "security clearance"]


def test_exits_when_result_file_missing(tmp_path):
    missing_path = tmp_path / "does_not_exist.md"

    with pytest.raises(SystemExit):
        reflow.parse_result_markdown(str(missing_path))


def test_exits_when_result_file_has_unrecognized_shape(tmp_path):
    report = tmp_path / "result.md"
    report.write_text(
        "# Some Other Report\nNot a job-match report.\n", encoding="utf-8"
    )

    with pytest.raises(SystemExit):
        reflow.parse_result_markdown(str(report))
