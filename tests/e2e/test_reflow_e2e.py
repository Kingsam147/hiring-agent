"""End-to-end test of reflow.py's tailor-and-regrade loop.

Runs the real run_reflow_loop() control flow (tailor -> validate -> remap ->
regrade -> track best -> stop on plateau) against the fake_reflow_module
fixture. The Claude tailoring call and the regrade scoring call are both
mocked/stubbed so this test makes zero real LLM API calls and costs nothing
to run, while still exercising the actual loop logic end to end.
"""

import ast
import importlib.util
import json

import reflow
from models import (
    Basics,
    JobCategoryScore,
    JobDescriptionData,
    JobEvaluationData,
    JobScores,
    JSONResume,
)
from reflow import GapAnalysis


def _fake_evaluation(score: float) -> JobEvaluationData:
    category = JobCategoryScore(score=score, evidence="Stubbed evidence.")
    return JobEvaluationData(
        scores=JobScores(
            skills_match=category,
            experience_match=category,
            job_title_alignment=category,
            education=category,
            resume_quality=category,
            missing_critical_requirements=category,
        ),
        semantic_match_score=score,
        weighted_total=score,
        key_strengths=[],
        areas_for_improvement=[],
        job_title="Backend Engineer",
    )


class _StubJobDescriptionEvaluator:
    """Never constructs a real provider or calls an LLM -- returns a fixed
    JobDescriptionData so run_reflow_loop's job_data fetch costs nothing."""

    def __init__(self, *args, **kwargs):
        pass

    def extract_job_requirements(self) -> JobDescriptionData:
        return JobDescriptionData(
            job_title="Backend Engineer",
            required_skills=["Kubernetes"],
            preferred_skills=[],
        )


class _StubTailorProvider:
    """Always returns the fixture's own (guardrail-clean) content unchanged."""

    def __init__(self, module):
        self._payload = json.dumps(
            {
                "summary": module.SUMMARY,
                "skills": [
                    {"label": label, "rest": rest} for label, rest in module.SKILLS
                ],
                "experience": [
                    {"title": entry["title"], "bullets": list(entry["bullets"])}
                    for entry in module.EXPERIENCE
                ],
                "projects": [
                    {"title": entry["title"], "bullets": list(entry["bullets"])}
                    for entry in module.PROJECTS
                ],
            }
        )
        self.call_count = 0

    def chat(self, model, messages, options=None, **kwargs):
        self.call_count += 1
        return {"message": {"role": "assistant", "content": self._payload}}


def _gap() -> GapAnalysis:
    return GapAnalysis(
        job_title="Backend Engineer",
        weight_profile="engineering",
        missing_required_skills=["Kubernetes"],
        missing_preferred_skills=[],
        improvement_areas=["Quantify impact more"],
    )


def _candidate_payload(module, summary):
    return json.dumps(
        {
            "summary": summary,
            "skills": [{"label": label, "rest": rest} for label, rest in module.SKILLS],
            "experience": [
                {"title": entry["title"], "bullets": list(entry["bullets"])}
                for entry in module.EXPERIENCE
            ],
            "projects": [
                {"title": entry["title"], "bullets": list(entry["bullets"])}
                for entry in module.PROJECTS
            ],
        }
    )


class _SequentialStubTailorProvider:
    """Returns a different scripted payload on each call and records the
    exact messages it received, so tests can inspect what content each
    iteration was actually tailoring from."""

    def __init__(self, payloads):
        self._responses = iter(payloads)
        self.received_messages = []

    def chat(self, model, messages, options=None, **kwargs):
        self.received_messages.append(messages)
        return {"message": {"role": "assistant", "content": next(self._responses)}}


def test_reflow_loop_tracks_best_score_and_stops_on_plateau(
    fake_reflow_module, monkeypatch
):
    scores = iter([70.0, 75.0, 78.0, 77.0, 76.0, 999.0])  # 999 must never be reached
    monkeypatch.setattr(
        reflow, "regrade_candidate", lambda *a, **kw: _fake_evaluation(next(scores))
    )
    monkeypatch.setattr(reflow, "JobDescriptionEvaluator", _StubJobDescriptionEvaluator)

    tailor_provider = _StubTailorProvider(fake_reflow_module)
    template_manager = reflow.TemplateManager()
    original_resume = reflow.apply_candidate_to_resume(
        JSONResume(basics=Basics(name="Test Candidate")),
        reflow.build_candidate_from_module(fake_reflow_module),
    )

    best_candidate, best_score, score_history, best_evaluation = reflow.run_reflow_loop(
        tailor_provider,
        template_manager,
        _gap(),
        original_resume,
        fake_reflow_module,
        skills_bank="",
        job_description="Backend engineer role requiring Kubernetes.",
    )

    assert score_history == [70.0, 75.0, 78.0, 77.0, 76.0]
    assert best_score == 78.0
    assert best_candidate is not None
    assert reflow.resolve_band(best_score) == "70% - 80%"
    assert tailor_provider.call_count == 5  # stopped early, iteration 6 never ran
    assert best_evaluation is not None
    assert best_evaluation.weighted_total == 78.0


def test_reflow_loop_reports_not_compatible_when_score_plateaus_below_seventy(
    fake_reflow_module, monkeypatch
):
    monkeypatch.setattr(
        reflow, "regrade_candidate", lambda *a, **kw: _fake_evaluation(40.0)
    )
    monkeypatch.setattr(reflow, "JobDescriptionEvaluator", _StubJobDescriptionEvaluator)

    tailor_provider = _StubTailorProvider(fake_reflow_module)
    template_manager = reflow.TemplateManager()
    original_resume = reflow.apply_candidate_to_resume(
        JSONResume(basics=Basics(name="Test Candidate")),
        reflow.build_candidate_from_module(fake_reflow_module),
    )

    best_candidate, best_score, score_history, best_evaluation = reflow.run_reflow_loop(
        tailor_provider,
        template_manager,
        _gap(),
        original_resume,
        fake_reflow_module,
        skills_bank="",
        job_description="Backend engineer role requiring Kubernetes.",
    )

    assert best_score == 40.0
    assert (
        reflow.resolve_band(best_score) is None
    )  # main() would print "not compatible"


def test_run_reflow_loop_refreshes_gap_from_job_data_before_first_iteration(
    fake_reflow_module, monkeypatch
):
    """The gap passed in may be stale (parsed from an old result.md). The
    loop must recompute missing required/preferred skills fresh from
    job_data against the original resume before tailoring starts, so the
    model is never working from an out-of-date target list."""
    stale_gap = _gap()
    stale_gap.missing_required_skills = []
    stale_gap.missing_preferred_skills = []

    class _StubJobDescriptionEvaluatorWithSkills:
        def __init__(self, *args, **kwargs):
            pass

        def extract_job_requirements(self) -> JobDescriptionData:
            return JobDescriptionData(
                job_title="Backend Engineer",
                required_skills=["CSS"],
                preferred_skills=["Async programming"],
            )

    monkeypatch.setattr(
        reflow, "JobDescriptionEvaluator", _StubJobDescriptionEvaluatorWithSkills
    )
    monkeypatch.setattr(
        reflow, "regrade_candidate", lambda *a, **kw: _fake_evaluation(80.0)
    )

    same_payload = _candidate_payload(fake_reflow_module, fake_reflow_module.SUMMARY)
    tailor_provider = _SequentialStubTailorProvider(
        [same_payload, same_payload, same_payload]
    )
    template_manager = reflow.TemplateManager()
    original_resume = reflow.apply_candidate_to_resume(
        JSONResume(basics=Basics(name="Test Candidate")),
        reflow.build_candidate_from_module(fake_reflow_module),
    )

    reflow.run_reflow_loop(
        tailor_provider,
        template_manager,
        stale_gap,
        original_resume,
        fake_reflow_module,
        skills_bank="",
        job_description="Backend engineer role requiring CSS and async programming.",
    )

    first_call_user_message = tailor_provider.received_messages[0][1]["content"]
    assert "CSS" in first_call_user_message
    assert "Async programming" in first_call_user_message


def test_reflow_loop_advances_current_content_even_without_improving_score(
    fake_reflow_module, monkeypatch
):
    monkeypatch.setattr(reflow, "MAX_ITERATIONS", 4)
    tailor_provider = _SequentialStubTailorProvider(
        [
            _candidate_payload(fake_reflow_module, "Candidate A summary."),
            _candidate_payload(fake_reflow_module, "Candidate B summary."),
            _candidate_payload(fake_reflow_module, "Candidate C summary."),
            _candidate_payload(fake_reflow_module, "Candidate D summary."),
        ]
    )
    scores = iter([80.0, 70.0, 75.0, 76.0])
    monkeypatch.setattr(
        reflow, "regrade_candidate", lambda *a, **kw: _fake_evaluation(next(scores))
    )
    monkeypatch.setattr(reflow, "JobDescriptionEvaluator", _StubJobDescriptionEvaluator)

    template_manager = reflow.TemplateManager()
    original_resume = reflow.apply_candidate_to_resume(
        JSONResume(basics=Basics(name="Test Candidate")),
        reflow.build_candidate_from_module(fake_reflow_module),
    )

    best_candidate, best_score, score_history, best_evaluation = reflow.run_reflow_loop(
        tailor_provider,
        template_manager,
        _gap(),
        original_resume,
        fake_reflow_module,
        skills_bank="",
        job_description="Backend engineer role requiring Kubernetes.",
    )

    assert score_history == [80.0, 70.0, 75.0, 76.0]
    assert best_score == 80.0
    assert best_candidate.summary == "Candidate A summary."
    # All 4 iterations ran: under the old best-ever comparison this chain
    # would have stopped after iteration 3 (neither 70 nor 75 beats 80).
    assert len(tailor_provider.received_messages) == 4

    iteration_three_user_message = tailor_provider.received_messages[2][1]["content"]
    assert "Candidate B summary." in iteration_three_user_message
    assert "Candidate A summary." not in iteration_three_user_message

    iteration_four_user_message = tailor_provider.received_messages[3][1]["content"]
    assert "Candidate C summary." in iteration_four_user_message


def test_reflow_loop_stops_after_two_consecutive_local_regressions(
    fake_reflow_module, monkeypatch
):
    tailor_provider = _SequentialStubTailorProvider(
        [
            _candidate_payload(fake_reflow_module, "Candidate A summary."),
            _candidate_payload(fake_reflow_module, "Candidate B summary."),
            _candidate_payload(fake_reflow_module, "Candidate C summary."),
            _candidate_payload(fake_reflow_module, "Should never be reached."),
        ]
    )
    scores = iter([80.0, 70.0, 65.0, 999.0])  # 999 must never be reached
    monkeypatch.setattr(
        reflow, "regrade_candidate", lambda *a, **kw: _fake_evaluation(next(scores))
    )
    monkeypatch.setattr(reflow, "JobDescriptionEvaluator", _StubJobDescriptionEvaluator)

    template_manager = reflow.TemplateManager()
    original_resume = reflow.apply_candidate_to_resume(
        JSONResume(basics=Basics(name="Test Candidate")),
        reflow.build_candidate_from_module(fake_reflow_module),
    )

    best_candidate, best_score, score_history, best_evaluation = reflow.run_reflow_loop(
        tailor_provider,
        template_manager,
        _gap(),
        original_resume,
        fake_reflow_module,
        skills_bank="",
        job_description="Backend engineer role requiring Kubernetes.",
    )

    assert score_history == [80.0, 70.0, 65.0]
    assert best_score == 80.0
    assert best_candidate.summary == "Candidate A summary."
    assert len(tailor_provider.received_messages) == 3


def test_reflow_writes_correctly_spliced_tailored_generator(
    fake_reflow_module, monkeypatch, tmp_path
):
    tailored_output_path = tmp_path / "fake_reflow_resume_tailored.py"
    monkeypatch.setattr(reflow, "REFLOW_RESUME_PATH", _fixture_path())
    monkeypatch.setattr(reflow, "TAILORED_RESUME_PATH", tailored_output_path)

    candidate = reflow.build_candidate_from_module(fake_reflow_module)
    candidate.summary = "A freshly tailored fictional summary for testing."

    reflow.write_tailored_generator(candidate, fake_reflow_module)

    source = tailored_output_path.read_text(encoding="utf-8")
    ast.parse(source)  # must be valid Python
    assert "A freshly tailored fictional summary for testing." in source
    assert fake_reflow_module.NAME in source  # untouched
    assert fake_reflow_module.EDUCATION_LINE in source  # untouched

    spec = importlib.util.spec_from_file_location(
        "tailored_fixture", tailored_output_path
    )
    tailored_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tailored_module)
    assert tailored_module.SUMMARY == candidate.summary
    assert tailored_module.NAME == fake_reflow_module.NAME
    out_pdf = tailored_module.build(str(tmp_path / "fixture_out.pdf"))
    assert out_pdf


def test_render_tailored_resume_writes_pdf_next_to_generator(
    fake_reflow_module, monkeypatch, tmp_path
):
    tailored_output_path = tmp_path / "fake_reflow_resume_tailored.py"
    monkeypatch.setattr(reflow, "REFLOW_RESUME_PATH", _fixture_path())
    monkeypatch.setattr(reflow, "TAILORED_RESUME_PATH", tailored_output_path)

    candidate = reflow.build_candidate_from_module(fake_reflow_module)
    reflow.write_tailored_generator(candidate, fake_reflow_module)

    output_pdf_path = reflow.render_tailored_resume()

    assert output_pdf_path == tailored_output_path.with_suffix(".pdf")
    assert output_pdf_path.exists()


def _fixture_path():
    from pathlib import Path

    return Path(__file__).parent.parent / "fixtures" / "fake_reflow_resume.py"
