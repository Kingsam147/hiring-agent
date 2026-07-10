"""Unit tests for JobDescriptionEvaluator's grading orchestration.

The LLM boundary (extract_job_requirements, _score_resume,
generate_score_summary) and the embedding model are mocked so these tests
run fast and deterministically, focused on verifying that the evaluator
correctly assembles a final grade from its inputs.
"""

from evaluator import JobDescriptionEvaluator
from models import (
    JobCategoryScore,
    JobDescriptionData,
    JobScores,
    LLMJobEvaluationResponse,
    LLMJobScores,
)
from weight_profiles import get_profile


def _make_evaluator() -> JobDescriptionEvaluator:
    return JobDescriptionEvaluator(
        job_description="We need a backend engineer skilled in Python and SQL.",
        model_name="gemini-2.5-flash",
        weight_profile="engineering",
    )


def test_compute_weighted_total_applies_engineering_weights():
    evaluator = _make_evaluator()
    weights = get_profile("engineering")
    scores = JobScores(
        skills_match=JobCategoryScore(score=80, evidence="matched most skills"),
        experience_match=JobCategoryScore(score=90, evidence="strong experience"),
        job_title_alignment=JobCategoryScore(score=70, evidence="close title match"),
        education=JobCategoryScore(score=100, evidence="meets requirement"),
        resume_quality=JobCategoryScore(score=85, evidence="well written"),
        missing_critical_requirements=JobCategoryScore(
            score=100, evidence="none missing"
        ),
    )
    semantic_score = 60.0

    weighted_total = evaluator._compute_weighted_total(scores, semantic_score)

    expected = round(
        80 * weights["skills_match"]
        + 90 * weights["experience_match"]
        + 60.0 * weights["semantic_match"]
        + 70 * weights["job_title_alignment"]
        + 100 * weights["education"]
        + 85 * weights["resume_quality"]
        + 100 * weights["missing_critical_requirements"],
        1,
    )
    assert weighted_total == expected


def test_compute_weighted_total_caps_at_100():
    evaluator = _make_evaluator()
    scores = JobScores(
        skills_match=JobCategoryScore(score=100, evidence="x"),
        experience_match=JobCategoryScore(score=100, evidence="x"),
        job_title_alignment=JobCategoryScore(score=100, evidence="x"),
        education=JobCategoryScore(score=100, evidence="x"),
        resume_quality=JobCategoryScore(score=100, evidence="x"),
        missing_critical_requirements=JobCategoryScore(score=100, evidence="x"),
    )

    weighted_total = evaluator._compute_weighted_total(scores, 100.0)

    assert weighted_total == 100.0


def test_evaluate_orchestration_with_mocked_llm_and_embeddings(monkeypatch):
    evaluator = _make_evaluator()

    job_data = JobDescriptionData(
        job_title="Backend Engineer",
        required_skills=["Python", "SQL"],
        preferred_skills=["Docker"],
        must_have_qualifications=[],
    )
    monkeypatch.setattr(evaluator, "extract_job_requirements", lambda: job_data)

    llm_result = LLMJobEvaluationResponse(
        scores=LLMJobScores(
            experience_match=JobCategoryScore(
                score=90, evidence="strong backend experience"
            ),
            job_title_alignment=JobCategoryScore(
                score=80, evidence="close title match"
            ),
            education=JobCategoryScore(score=100, evidence="meets requirement"),
            resume_quality=JobCategoryScore(score=85, evidence="well written"),
            missing_critical_requirements=JobCategoryScore(
                score=100, evidence="none missing"
            ),
        ),
        key_strengths=["Strong Python and SQL skills"],
        areas_for_improvement=["Could add Docker experience"],
    )
    monkeypatch.setattr(evaluator, "_score_resume", lambda *args, **kwargs: llm_result)
    monkeypatch.setattr(
        evaluator, "generate_score_summary", lambda evaluation: "canned summary"
    )

    def fake_load_embedding_model():
        evaluator.embedding_model = object()

    monkeypatch.setattr(evaluator, "_load_embedding_model", fake_load_embedding_model)
    monkeypatch.setattr(evaluator, "compute_semantic_score", lambda resume_text: 55.0)

    resume_text = "Backend engineer with 3 years of Python and SQL experience."
    result = evaluator.evaluate(resume_text, resume_data=None)

    assert result.job_title == "Backend Engineer"
    # Weighted coverage: both required skills matched (2/2) but the one
    # preferred skill (Docker) is not, so 100*(0.8*1.0 + 0.2*0.0) == 80.0.
    assert result.scores.skills_match.score == 80.0
    assert result.semantic_match_score == 55.0
    assert result.score_summary == "canned summary"
    assert result.key_strengths == ["Strong Python and SQL skills"]
    assert 0 <= result.weighted_total <= 100
