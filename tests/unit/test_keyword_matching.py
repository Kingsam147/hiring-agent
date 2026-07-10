"""Unit tests for deterministic keyword matching against job requirements."""

from keyword_matching import (
    compute_keyword_match,
    requirement_satisfied,
    normalize_text,
)
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


def test_keyword_match_leaves_long_qualifications_unverifiable_and_uncapped():
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
    assert result.gated is False


def test_requirement_satisfied_matches_any_parenthetical_alternative():
    corpus = normalize_text("Built UIs with React and Next.js.")

    assert requirement_satisfied("SPA frameworks (Angular, React, or Vue)", corpus)


def test_requirement_satisfied_fails_when_no_alternative_present():
    corpus = normalize_text("Built UIs with jQuery and vanilla JS.")

    assert not requirement_satisfied("SPA frameworks (Angular, React, or Vue)", corpus)


def test_requirement_satisfied_matches_bare_or_alternative_via_db_engine_synonym():
    postgres_corpus = normalize_text("Data layer built on PostgreSQL.")
    mongo_corpus = normalize_text("Data layer built on MongoDB.")

    assert requirement_satisfied("Relational or NoSQL databases", postgres_corpus)
    assert requirement_satisfied("Relational or NoSQL databases", mongo_corpus)


def test_requirement_satisfied_matches_common_abbreviation():
    corpus = normalize_text("Skills: OOP, Data Structures & Algorithms.")

    assert requirement_satisfied("Object-oriented programming", corpus)


def test_requirement_satisfied_still_requires_plain_skill_to_be_present():
    corpus = normalize_text("Backend engineer experienced in Python and SQL.")

    assert not requirement_satisfied("Kubernetes", corpus)
    assert requirement_satisfied("Python", corpus)


def test_requirement_satisfied_matches_dotnet_platform_via_asp_net_core():
    corpus = normalize_text("Full-stack experience with C# and ASP.NET Core.")

    assert requirement_satisfied(".NET platform", corpus)


def test_requirement_satisfied_dotnet_platform_still_fails_without_any_alias():
    corpus = normalize_text("Backend engineer experienced in Python and SQL.")

    assert not requirement_satisfied(".NET platform", corpus)


def test_requirement_satisfied_matches_actual_preferred_skill_phrases_via_short_forms():
    # These five are the real preferred-skill phrases the JD extraction
    # produced for the Crestron XiO Cloud posting; each alias is the short
    # form that actually shows up in the resume text.
    assert requirement_satisfied(
        "Asynchronous programming", normalize_text("Built async REST APIs.")
    )
    assert requirement_satisfied(
        "RESTful API development", normalize_text("Designed 32 REST APIs.")
    )
    assert requirement_satisfied(
        "Client-server application architecture",
        normalize_text("Built a client-server platform."),
    )
    assert requirement_satisfied(
        "Cloud-based applications",
        normalize_text("Deployed cloud-native services."),
    )
    assert requirement_satisfied(
        "Full-stack web development",
        normalize_text("Full-stack engineer."),
    )


def test_requirement_satisfied_matches_short_forms_across_categories():
    # Languages
    assert requirement_satisfied("JavaScript", normalize_text("Skilled in JS."))
    assert requirement_satisfied("TypeScript", normalize_text("Skilled in TS."))
    # Backend
    assert requirement_satisfied("Node.js", normalize_text("Built APIs with Node."))
    # Cloud & DevOps
    assert requirement_satisfied(
        "Amazon Web Services", normalize_text("Deployed on AWS.")
    )
    assert requirement_satisfied(
        "Continuous Integration/Continuous Deployment", normalize_text("Uses CI/CD.")
    )
    # Architecture & Security
    assert requirement_satisfied(
        "Domain-Driven Design", normalize_text("Applied DDD patterns.")
    )
    assert requirement_satisfied(
        "Row-Level Security", normalize_text("Enforced RLS policies.")
    )
    # Frontend
    assert requirement_satisfied(
        "Single Page Application", normalize_text("Built an SPA.")
    )
    # Database & Querying
    assert requirement_satisfied(
        "Structured Query Language", normalize_text("Proficient in SQL.")
    )
    # Development Practice
    assert requirement_satisfied(
        "Test-Driven Development", normalize_text("Practices TDD.")
    )


def test_keyword_match_uses_requirement_satisfied_for_or_bundled_skills():
    job_data = _job_data(
        required_skills=["SPA frameworks (Angular, React, or Vue)"],
        preferred_skills=[],
    )
    resume_text = "Full-stack engineer building React interfaces."

    result = compute_keyword_match(job_data, resume_text)

    assert result.matched_required == ["SPA frameworks (Angular, React, or Vue)"]
    assert result.missing_required == []


def test_keyword_match_reports_matched_and_missing_soft_skills():
    job_data = _job_data(
        soft_skills=["Communication", "Leadership", "Teamwork"],
    )
    resume_text = (
        "Demonstrated strong communication and leadership across engineering teams."
    )

    result = compute_keyword_match(job_data, resume_text)

    assert set(result.matched_soft_skills) == {"Communication", "Leadership"}
    assert result.missing_soft_skills == ["Teamwork"]


def test_keyword_match_soft_skills_do_not_affect_coverage_score():
    job_data = _job_data(
        required_skills=["Python"],
        preferred_skills=[],
        soft_skills=["Leadership"],
    )
    resume_text_with_soft_skill = "Backend engineer skilled in Python. Led the team."
    resume_text_without_soft_skill = "Backend engineer skilled in Python."

    with_soft_skill = compute_keyword_match(job_data, resume_text_with_soft_skill)
    without_soft_skill = compute_keyword_match(job_data, resume_text_without_soft_skill)

    assert with_soft_skill.coverage_score == without_soft_skill.coverage_score
