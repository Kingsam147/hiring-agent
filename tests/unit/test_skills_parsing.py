"""Unit tests for skills parsing: both the resume's LLM-extracted skills
section and reflow.py's skills-bank / SKILLS handling.
"""

from pathlib import Path

import reflow
from models import JSONResume, Basics, Work, Project, Skill
from reflow import TailoredEntry, TailoredResume, TailoredSkill
from transform import transform_parsed_data


def test_resume_skills_extraction_parses_categories():
    raw_llm_response = {
        "skills": [
            {"name": "Languages", "level": None, "keywords": ["Python", "Java"]},
            {"name": "Backend", "level": None, "keywords": ["Node.js", "PostgreSQL"]},
        ]
    }

    transformed = transform_parsed_data(raw_llm_response)

    assert len(transformed["skills"]) == 2
    assert transformed["skills"][0]["name"] == "Languages"
    assert transformed["skills"][0]["keywords"] == ["Python", "Java"]
    assert transformed["skills"][1]["keywords"] == ["Node.js", "PostgreSQL"]


def test_resume_skills_extraction_handles_flat_string_list():
    raw_llm_response = {"skills": ["Python", "Java", "SQL"]}

    transformed = transform_parsed_data(raw_llm_response)

    assert len(transformed["skills"]) == 1
    assert transformed["skills"][0]["name"] == "Programming Languages"
    assert transformed["skills"][0]["keywords"] == ["Python", "Java", "SQL"]


def test_load_skills_bank_returns_empty_for_untouched_skeleton(monkeypatch, tmp_path):
    skeleton = tmp_path / "skills_bank.txt"
    skeleton.write_text(
        "# comment\n[Languages]\n\n[Backend]\n\n[Projects]\n### Example Project\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(reflow, "SKILLS_BANK_PATH", skeleton)

    assert reflow.load_skills_bank() == ""


def test_load_skills_bank_returns_content_once_filled_in(monkeypatch, tmp_path):
    filled = tmp_path / "skills_bank.txt"
    filled.write_text(
        "# comment\n[Languages]\nGo\n\n[Projects]\n### Example Project\nUsed Redis for caching\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(reflow, "SKILLS_BANK_PATH", filled)

    content = reflow.load_skills_bank()

    assert "Go" in content
    assert "Used Redis for caching" in content


def test_apply_candidate_rebuilds_skills_from_tailored_pairs():
    original_resume = JSONResume(
        basics=Basics(name="Jane Doe", summary="Old summary."),
        skills=[Skill(name="Languages", keywords=["Python"])],
        work=[Work(name="Acme", position="Engineer", highlights=["Old bullet."])],
        projects=[Project(name="Cool Project", highlights=["Old project bullet."])],
    )
    candidate = TailoredResume(
        summary="New summary.",
        skills=[
            TailoredSkill(label="Languages: ", rest="Python, Go"),
            TailoredSkill(label="Backend: ", rest="Node.js, Redis"),
        ],
        experience=[TailoredEntry(title="Acme | Engineer", bullets=["New bullet."])],
        projects=[TailoredEntry(title="Cool Project", bullets=["New project bullet."])],
    )

    tailored = reflow.apply_candidate_to_resume(original_resume, candidate)

    assert tailored.basics.summary == "New summary."
    assert len(tailored.skills) == 2
    assert tailored.skills[0].name == "Languages"
    assert tailored.skills[0].keywords == ["Python", "Go"]
    assert tailored.skills[1].name == "Backend"
    assert tailored.skills[1].keywords == ["Node.js", "Redis"]
    assert tailored.work[0].highlights == ["New bullet."]
    assert tailored.projects[0].highlights == ["New project bullet."]
    # Original object must not be mutated.
    assert original_resume.skills[0].keywords == ["Python"]
    assert original_resume.work[0].highlights == ["Old bullet."]
