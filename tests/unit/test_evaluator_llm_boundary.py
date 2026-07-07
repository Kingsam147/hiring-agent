"""Unit tests that exercise the real LLM-call boundary in evaluator.py
(extract_job_requirements, check_requirements, _score_resume, generate
score summary, and the legacy ResumeEvaluator.evaluate_resume) via a mocked
provider.chat(), rather than mocking the methods themselves. This gives
much more real coverage of how a resume is actually graded.
"""

import json

from evaluator import JobDescriptionEvaluator, ResumeEvaluator


def _content(payload: dict) -> dict:
    return {"message": {"role": "assistant", "content": json.dumps(payload)}}


class _DispatchingProvider:
    """Returns a different canned response depending on which schema the
    caller asked for (mirrors how each evaluator call site passes a
    different `format=<Model>.model_json_schema()`), or a plain-text
    response when no format is requested (generate_score_summary)."""

    def __init__(
        self,
        responses_by_schema_title: dict,
        plain_text_response: str = "canned summary",
    ):
        self._responses = responses_by_schema_title
        self._plain_text_response = plain_text_response

    def chat(self, model, messages, options=None, **kwargs):
        schema = kwargs.get("format")
        if schema is None:
            return _content_text(self._plain_text_response)
        title = schema.get("title")
        return _content(self._responses[title])


def _content_text(text: str) -> dict:
    return {"message": {"role": "assistant", "content": text}}


def test_extract_job_requirements_parses_llm_response():
    job_data_payload = {
        "job_title": "Backend Engineer",
        "required_skills": ["Python", "SQL"],
        "preferred_skills": ["Docker"],
        "must_have_qualifications": [],
    }
    evaluator = JobDescriptionEvaluator(
        job_description="We need a backend engineer.",
        model_name="gemini-2.5-flash",
        weight_profile="engineering",
    )
    evaluator.provider = _DispatchingProvider({"JobDescriptionData": job_data_payload})

    job_data = evaluator.extract_job_requirements()

    assert job_data.job_title == "Backend Engineer"
    assert job_data.required_skills == ["Python", "SQL"]


def test_check_requirements_gates_on_missing_required_skill():
    job_data_payload = {
        "job_title": "Backend Engineer",
        "required_skills": ["Python", "Kubernetes"],
        "preferred_skills": [],
        "must_have_qualifications": [],
    }
    evaluator = JobDescriptionEvaluator(
        job_description="We need a backend engineer skilled in Python and Kubernetes.",
        model_name="gemini-2.5-flash",
        weight_profile="engineering",
    )
    recheck_payload = {
        "verdicts": [
            {
                "requirement": "Kubernetes",
                "status": "not_met",
                "reasoning": "no mention",
            }
        ]
    }
    evaluator.provider = _DispatchingProvider(
        {
            "JobDescriptionData": job_data_payload,
            "RequirementRecheckResponse": recheck_payload,
        }
    )

    gate_result = evaluator.check_requirements(
        "Backend engineer experienced in Python.",
        resume_data=None,
        knockout_resolver=None,
    )

    assert gate_result.passed is False
    assert gate_result.missing_required_skills == ["Kubernetes"]
    assert gate_result.kept_required_skills == ["Python"]


def test_evaluate_full_flow_with_dispatched_llm_responses(monkeypatch):
    job_data_payload = {
        "job_title": "Backend Engineer",
        "required_skills": ["Python", "SQL"],
        "preferred_skills": [],
        "must_have_qualifications": [],
    }
    llm_score_payload = {
        "scores": {
            "experience_match": {"score": 90, "evidence": "strong experience"},
            "job_title_alignment": {"score": 80, "evidence": "close match"},
            "education": {"score": 100, "evidence": "meets requirement"},
            "resume_quality": {"score": 85, "evidence": "well written"},
            "missing_critical_requirements": {"score": 100, "evidence": "none missing"},
        },
        "key_strengths": ["Strong Python and SQL skills"],
        "areas_for_improvement": ["Could add more metrics"],
    }
    evaluator = JobDescriptionEvaluator(
        job_description="We need a backend engineer skilled in Python and SQL.",
        model_name="gemini-2.5-flash",
        weight_profile="engineering",
    )
    evaluator.provider = _DispatchingProvider(
        {
            "JobDescriptionData": job_data_payload,
            "LLMJobEvaluationResponse": llm_score_payload,
        }
    )

    def fake_load_embedding_model():
        evaluator.embedding_model = object()

    monkeypatch.setattr(evaluator, "_load_embedding_model", fake_load_embedding_model)
    monkeypatch.setattr(evaluator, "compute_semantic_score", lambda resume_text: 50.0)

    result = evaluator.evaluate(
        "Backend engineer with 3 years of Python and SQL experience.",
        resume_data=None,
        knockout_resolver=None,
    )

    assert result.job_title == "Backend Engineer"
    assert result.scores.skills_match.score == 100.0
    assert result.score_summary == "canned summary"
    assert 0 <= result.weighted_total <= 100


def test_resume_evaluator_evaluate_resume_with_mocked_llm():
    evaluation_payload = {
        "scores": {
            "open_source": {
                "score": 20,
                "max": 35,
                "evidence": "some open source work",
            },
            "self_projects": {
                "score": 25,
                "max": 30,
                "evidence": "solid self projects",
            },
            "production": {
                "score": 15,
                "max": 25,
                "evidence": "some production experience",
            },
            "technical_skills": {
                "score": 8,
                "max": 10,
                "evidence": "good technical skills",
            },
        },
        "bonus_points": {"total": 5, "breakdown": "bonus for extra activity"},
        "deductions": {"total": 0, "reasons": "none"},
        "key_strengths": ["Strong self projects"],
        "areas_for_improvement": ["More open source contributions"],
    }
    resume_evaluator = ResumeEvaluator(model_name="gemini-2.5-flash")
    resume_evaluator.provider = _DispatchingProvider(
        {"EvaluationData": evaluation_payload}
    )

    result = resume_evaluator.evaluate_resume("Backend engineer resume text.")

    assert result.scores.open_source.score == 20
    assert result.bonus_points.total == 5
    assert result.key_strengths == ["Strong self projects"]
