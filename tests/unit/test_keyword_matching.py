"""Unit tests for deterministic keyword matching against job requirements."""

from keyword_matching import apply_knockout_resolutions, compute_keyword_match
from models import JobDescriptionData


def _job_data(**overrides) -> JobDescriptionData:
    defaults = dict(
        job_title="Backend Engineer",
        required_skills=["Python", "SQL", "Kubernetes"],
        preferred_skills=["React", "Docker"],
        must_have_qualifications=[],
    )
    defaults.update(overrides)
    return JobDescriptionData(**defaults)


def test_keyword_match_finds_required_and_preferred_skills():
    resume_text = "Backend engineer experienced in Python, SQL, and Docker."

    result = compute_keyword_match(_job_data(), resume_text)

    assert set(result.matched_required) == {"Python", "SQL"}
    assert result.missing_required == ["Kubernetes"]
    assert result.matched_preferred == ["Docker"]
    assert result.missing_preferred == ["React"]
    assert 0 <= result.coverage_score <= 100


def test_keyword_match_gates_coverage_when_must_have_missing():
    job_data = _job_data(must_have_qualifications=["US citizenship required"])
    resume_text = (
        "Backend engineer experienced in Python, SQL, Kubernetes, React, and Docker."
    )

    result = compute_keyword_match(job_data, resume_text)

    assert result.gated is True
    assert result.coverage_score <= 60.0
    assert result.must_have_status[0].status == "not_found"


def test_apply_knockout_resolutions_resolves_unverifiable_must_have():
    job_data = _job_data(
        must_have_qualifications=[
            "Willingness to work in a fast paced, high pressure environment"
        ]
    )
    resume_text = (
        "Backend engineer experienced in Python, SQL, Kubernetes, React, and Docker."
    )
    result = compute_keyword_match(job_data, resume_text)
    assert result.must_have_status[0].status == "unverifiable"

    resolved = apply_knockout_resolutions(result, resolver=lambda qualification: True)

    assert resolved.must_have_status[0].resolved is True
    assert resolved.knockout_failed is False
    assert resolved.gated is False


def test_apply_knockout_resolutions_marks_knockout_failed_on_rejection():
    result = compute_keyword_match(
        _job_data(must_have_qualifications=["Security clearance"]),
        "Backend engineer experienced in Python, SQL, Kubernetes.",
    )

    resolved = apply_knockout_resolutions(result, resolver=lambda qualification: False)

    assert resolved.knockout_failed is True
    assert resolved.gated is True
    assert resolved.coverage_score <= 60.0
