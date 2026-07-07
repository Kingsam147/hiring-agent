"""Additional unit tests for keyword_matching.py's evidence-building and
industry-mention helpers."""

from keyword_matching import build_skills_evidence, compute_industry_mentions
from models import JSONResume, Work, KeywordMatchResult


def test_build_skills_evidence_reports_matched_and_missing():
    result = KeywordMatchResult(
        matched_required=["Python"],
        missing_required=["Kubernetes"],
        matched_preferred=["Docker"],
        missing_preferred=[],
        coverage_score=75.0,
    )

    evidence = build_skills_evidence(result)

    assert "Matched 1/2 required skills (Python)" in evidence
    assert "Missing required: Kubernetes" in evidence
    assert "Matched 1/1 preferred skills (Docker)" in evidence


def test_build_skills_evidence_notes_gate_cap():
    result = KeywordMatchResult(
        matched_required=[],
        missing_required=["Python"],
        coverage_score=30.0,
        gated=True,
    )

    evidence = build_skills_evidence(result)

    assert "capped at 60" in evidence.lower()


def test_compute_industry_mentions_finds_matching_work_entry():
    resume_data = JSONResume(
        work=[
            Work(
                name="FinBank", position="Engineer", highlights=["Built FinTech tools"]
            )
        ]
    )

    match = compute_industry_mentions("FinTech", resume_data)

    assert match.mention_count == 1
    assert "FinBank" in match.matched_entries[0]


def test_compute_industry_mentions_returns_zero_for_no_match():
    resume_data = JSONResume(
        work=[Work(name="Bakery Co", position="Baker", highlights=["Baked bread"])]
    )

    match = compute_industry_mentions("FinTech", resume_data)

    assert match.mention_count == 0


def test_compute_industry_mentions_returns_none_when_no_industry_given():
    assert compute_industry_mentions(None, None) is None
