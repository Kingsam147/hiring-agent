"""Unit tests for score.py's build_job_evaluation_markdown report builder."""

from models import (
    JobCategoryScore,
    JobEvaluationData,
    JobScores,
    KeywordMatchResult,
)
from score import build_job_evaluation_markdown


def _category(score=80.0, evidence="Solid evidence.") -> JobCategoryScore:
    return JobCategoryScore(score=score, evidence=evidence)


def _evaluation(keyword_match=None) -> JobEvaluationData:
    return JobEvaluationData(
        scores=JobScores(
            skills_match=_category(),
            experience_match=_category(),
            job_title_alignment=_category(),
            education=_category(),
            resume_quality=_category(),
            missing_critical_requirements=_category(),
        ),
        semantic_match_score=75.0,
        weighted_total=80.0,
        key_strengths=["Strong backend experience."],
        areas_for_improvement=["Add more leadership examples."],
        job_title="Backend Engineer",
        keyword_match=keyword_match,
    )


def test_markdown_lists_matched_and_missing_soft_skills():
    keyword_match = KeywordMatchResult(
        matched_soft_skills=["Communication"],
        missing_soft_skills=["Leadership"],
        coverage_score=70.0,
    )

    markdown = build_job_evaluation_markdown(_evaluation(keyword_match))

    assert "**Soft skills present:**" in markdown
    assert "Communication" in markdown
    assert "**Soft skills missing:**" in markdown
    assert "Leadership" in markdown


def test_markdown_omits_soft_skills_section_when_none_extracted():
    keyword_match = KeywordMatchResult(coverage_score=70.0)

    markdown = build_job_evaluation_markdown(_evaluation(keyword_match))

    assert "Soft skills" not in markdown
