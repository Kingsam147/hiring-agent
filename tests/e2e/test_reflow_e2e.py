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
from models import Basics, JSONResume
from reflow import GapAnalysis


class _StubTailorProvider:
    """Always returns the fixture's own (guardrail-clean) content unchanged."""

    def __init__(self, module):
        self._payload = json.dumps(
            {
                "summary": module.SUMMARY,
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


def test_reflow_loop_tracks_best_score_and_stops_on_plateau(fake_reflow_module, monkeypatch):
    scores = iter([70.0, 75.0, 78.0, 77.0, 76.0, 999.0])  # 999 must never be reached
    monkeypatch.setattr(reflow, "regrade_candidate", lambda *a, **kw: next(scores))

    tailor_provider = _StubTailorProvider(fake_reflow_module)
    template_manager = reflow.TemplateManager()
    original_resume = reflow.apply_candidate_to_resume(
        JSONResume(basics=Basics(name="Test Candidate")),
        reflow.build_candidate_from_module(fake_reflow_module),
    )

    best_candidate, best_score, score_history = reflow.run_reflow_loop(
        tailor_provider,
        template_manager,
        _gap(),
        original_resume,
        fake_reflow_module,
        skills_bank="",
        job_description="Backend engineer role requiring Kubernetes.",
        frozen_resolver=lambda qualification: None,
    )

    assert score_history == [70.0, 75.0, 78.0, 77.0, 76.0]
    assert best_score == 78.0
    assert best_candidate is not None
    assert reflow.resolve_band(best_score) == "70% - 80%"
    assert tailor_provider.call_count == 5  # stopped early, iteration 6 never ran


def test_reflow_loop_reports_not_compatible_when_score_plateaus_below_seventy(
    fake_reflow_module, monkeypatch
):
    monkeypatch.setattr(reflow, "regrade_candidate", lambda *a, **kw: 40.0)

    tailor_provider = _StubTailorProvider(fake_reflow_module)
    template_manager = reflow.TemplateManager()
    original_resume = reflow.apply_candidate_to_resume(
        JSONResume(basics=Basics(name="Test Candidate")),
        reflow.build_candidate_from_module(fake_reflow_module),
    )

    best_candidate, best_score, score_history = reflow.run_reflow_loop(
        tailor_provider,
        template_manager,
        _gap(),
        original_resume,
        fake_reflow_module,
        skills_bank="",
        job_description="Backend engineer role requiring Kubernetes.",
        frozen_resolver=lambda qualification: None,
    )

    assert best_score == 40.0
    assert reflow.resolve_band(best_score) is None  # main() would print "not compatible"


def test_reflow_writes_correctly_spliced_tailored_generator(fake_reflow_module, monkeypatch, tmp_path):
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

    spec = importlib.util.spec_from_file_location("tailored_fixture", tailored_output_path)
    tailored_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tailored_module)
    assert tailored_module.SUMMARY == candidate.summary
    assert tailored_module.NAME == fake_reflow_module.NAME
    out_pdf = tailored_module.build(str(tmp_path / "fixture_out.pdf"))
    assert out_pdf


def _fixture_path():
    from pathlib import Path

    return Path(__file__).parent.parent / "fixtures" / "fake_reflow_resume.py"
