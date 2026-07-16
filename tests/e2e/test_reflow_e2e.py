"""End-to-end tests of reflow.py's agent-driven candidate flow.

Runs the real evaluate_candidate() control flow (load context -> validate ->
remap -> regrade -> compare against session state -> write outputs) against
the fake_reflow_module fixture. The job-requirement extraction and the
regrade scoring call are both stubbed so these tests make zero real LLM API
calls, while still exercising the actual CLI logic end to end. Iteration and
retry ordering now live in the calling agent (the resume-reflow-pipeline
skill), so there is no loop to test here anymore.
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




def _candidate_dict(module, summary=None):
    return {
        "summary": summary if summary is not None else module.SUMMARY,
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


def _context(module, missing_soft_skills=None):
    return reflow.ReflowContext(
        gap=_gap(),
        job_description="Backend engineer role requiring Kubernetes.",
        original_resume=reflow.apply_candidate_to_resume(
            JSONResume(basics=Basics(name="Test Candidate")),
            reflow.build_candidate_from_module(module),
        ),
        skills_bank="",
        job_data=JobDescriptionData(
            job_title="Backend Engineer",
            required_skills=["Kubernetes"],
            preferred_skills=[],
        ),
        current_content=reflow.build_candidate_from_module(module),
        missing_soft_skills=missing_soft_skills or [],
    )


def _gap() -> GapAnalysis:
    return GapAnalysis(
        job_title="Backend Engineer",
        weight_profile="engineering",
        missing_required_skills=["Kubernetes"],
        missing_preferred_skills=[],
        improvement_areas=["Quantify impact more"],
    )


def _wire_evaluate_candidate(monkeypatch, tmp_path, module, score, context=None):
    """Point every filesystem/LLM edge of evaluate_candidate at stubs."""
    monkeypatch.setattr(
        reflow, "gather_context", lambda: (context or _context(module), module)
    )
    monkeypatch.setattr(
        reflow, "regrade_candidate", lambda *a, **kw: _fake_evaluation(score)
    )
    monkeypatch.setattr(reflow, "REFLOW_RESUME_PATH", _fixture_path())
    monkeypatch.setattr(
        reflow, "TAILORED_RESUME_PATH", tmp_path / "fake_reflow_resume_tailored.py"
    )
    monkeypatch.setattr(reflow, "REFLOW_STATE_PATH", tmp_path / "reflow_state.json")
    monkeypatch.setattr(
        reflow, "load_job_description", lambda: "Backend engineer role."
    )
    monkeypatch.setattr(reflow, "write_result_markdown", lambda markdown: None)
    monkeypatch.setattr(reflow, "load_potential_skills", lambda: [])


def _write_candidate(tmp_path, module, summary=None):
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(_candidate_dict(module, summary)), encoding="utf-8")
    return str(path)


def test_evaluate_candidate_writes_new_best_and_saves_state(
    fake_reflow_module, monkeypatch, tmp_path, capsys
):
    _wire_evaluate_candidate(monkeypatch, tmp_path, fake_reflow_module, score=78.0)

    reflow.evaluate_candidate(_write_candidate(tmp_path, fake_reflow_module))

    out = capsys.readouterr().out
    assert "SCORE: 78.0/100 (previous best: none/100)" in out
    assert "NEW BEST" in out
    assert (tmp_path / "fake_reflow_resume_tailored.py").exists()
    assert (tmp_path / "fake_reflow_resume_tailored.pdf").exists()
    state = json.loads((tmp_path / "reflow_state.json").read_text(encoding="utf-8"))
    assert state["best_score"] == 78.0


def test_evaluate_candidate_below_seventy_writes_nothing(
    fake_reflow_module, monkeypatch, tmp_path, capsys
):
    _wire_evaluate_candidate(monkeypatch, tmp_path, fake_reflow_module, score=40.0)

    reflow.evaluate_candidate(_write_candidate(tmp_path, fake_reflow_module))

    out = capsys.readouterr().out
    assert "SCORE: 40.0/100" in out
    assert "below 70" in out
    assert not (tmp_path / "fake_reflow_resume_tailored.py").exists()
    assert not (tmp_path / "reflow_state.json").exists()


def test_evaluate_candidate_must_beat_saved_best(
    fake_reflow_module, monkeypatch, tmp_path, capsys
):
    _wire_evaluate_candidate(monkeypatch, tmp_path, fake_reflow_module, score=80.0)
    reflow.evaluate_candidate(_write_candidate(tmp_path, fake_reflow_module))
    capsys.readouterr()

    _wire_evaluate_candidate(monkeypatch, tmp_path, fake_reflow_module, score=75.0)
    reflow.evaluate_candidate(
        _write_candidate(tmp_path, fake_reflow_module, summary="A different summary.")
    )

    out = capsys.readouterr().out
    assert "SCORE: 75.0/100 (previous best: 80.0/100)" in out
    assert "did not beat the saved best" in out
    state = json.loads((tmp_path / "reflow_state.json").read_text(encoding="utf-8"))
    assert state["best_score"] == 80.0


def test_state_resets_when_job_description_changes(
    fake_reflow_module, monkeypatch, tmp_path, capsys
):
    _wire_evaluate_candidate(monkeypatch, tmp_path, fake_reflow_module, score=80.0)
    reflow.evaluate_candidate(_write_candidate(tmp_path, fake_reflow_module))
    capsys.readouterr()

    _wire_evaluate_candidate(monkeypatch, tmp_path, fake_reflow_module, score=72.0)
    monkeypatch.setattr(
        reflow, "load_job_description", lambda: "A completely different role."
    )
    reflow.evaluate_candidate(_write_candidate(tmp_path, fake_reflow_module))

    out = capsys.readouterr().out
    assert "previous best score reset" in out
    assert "SCORE: 72.0/100 (previous best: none/100)" in out
    assert "NEW BEST" in out


def test_evaluate_candidate_reports_validation_problems(
    fake_reflow_module, monkeypatch, tmp_path, capsys
):
    _wire_evaluate_candidate(monkeypatch, tmp_path, fake_reflow_module, score=90.0)
    bad = _candidate_dict(fake_reflow_module)
    bad["summary"] = "A summary with an em dash \u2014 which is forbidden."
    path = tmp_path / "bad_candidate.json"
    path.write_text(json.dumps(bad), encoding="utf-8")

    reflow.evaluate_candidate(str(path))

    out = capsys.readouterr().out
    assert "VALIDATION FAILED:" in out
    assert "em dash" in out.lower()
    assert not (tmp_path / "fake_reflow_resume_tailored.py").exists()


def test_evaluate_candidate_requires_missing_soft_skills_verbatim(
    fake_reflow_module, monkeypatch, tmp_path, capsys
):
    context = _context(fake_reflow_module, missing_soft_skills=["Prioritization"])
    _wire_evaluate_candidate(
        monkeypatch, tmp_path, fake_reflow_module, score=90.0, context=context
    )

    reflow.evaluate_candidate(_write_candidate(tmp_path, fake_reflow_module))

    out = capsys.readouterr().out
    assert "VALIDATION FAILED:" in out
    assert 'missing soft skill "Prioritization"' in out


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
