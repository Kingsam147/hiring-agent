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


def test_load_potential_skills_returns_empty_when_section_missing(
    monkeypatch, tmp_path
):
    skeleton = tmp_path / "skills_bank.txt"
    skeleton.write_text("[Languages]\nGo\n", encoding="utf-8")
    monkeypatch.setattr(reflow, "SKILLS_BANK_PATH", skeleton)

    assert reflow.load_potential_skills() == []


def test_load_potential_skills_returns_lines_from_section(monkeypatch, tmp_path):
    filled = tmp_path / "skills_bank.txt"
    filled.write_text(
        "[Languages]\nGo\n\n[Potential Skills]\n"
        "# comment should be skipped\n"
        "Kubernetes (used briefly, needs a refresher)\n"
        "GraphQL\n\n[Projects]\n### Example\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(reflow, "SKILLS_BANK_PATH", filled)

    assert reflow.load_potential_skills() == [
        "Kubernetes (used briefly, needs a refresher)",
        "GraphQL",
    ]


def _candidate_with_text(summary: str, bullet: str) -> TailoredResume:
    return TailoredResume(
        summary=summary,
        skills=[TailoredSkill(label="Languages: ", rest="Python, Go")],
        experience=[TailoredEntry(title="Acme | Engineer", bullets=[bullet])],
        projects=[],
    )


def test_find_used_potential_skills_matches_core_name_case_insensitively():
    potential_skills = ["Kubernetes (used briefly, needs a refresher)"]
    candidate = _candidate_with_text(
        "Backend engineer.", "Deployed services with kubernetes at scale."
    )

    used = reflow.find_used_potential_skills(potential_skills, candidate)

    assert used == ["Kubernetes (used briefly, needs a refresher)"]


def test_find_used_potential_skills_ignores_skills_not_present():
    potential_skills = ["Kubernetes (used briefly, needs a refresher)", "GraphQL"]
    candidate = _candidate_with_text(
        "Backend engineer.", "Built REST APIs with Node.js."
    )

    used = reflow.find_used_potential_skills(potential_skills, candidate)

    assert used == []


def test_load_potential_skills_parses_markdown_header_bullet_list(
    monkeypatch, tmp_path
):
    filled = tmp_path / "skills_bank.txt"
    filled.write_text(
        "## Verified Skills\n\n### Languages\nPython\n\n---\n\n"
        "## Potential Skills\n\n"
        "- C# (Crestron XiO Cloud co-op)\n"
        "- ASP.NET Core (Crestron XiO Cloud co-op)\n\n"
        "---\n\n# Job Search Context\nSome notes.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(reflow, "SKILLS_BANK_PATH", filled)

    assert reflow.load_potential_skills() == [
        "C# (Crestron XiO Cloud co-op)",
        "ASP.NET Core (Crestron XiO Cloud co-op)",
    ]


def test_load_potential_skills_markdown_section_stops_at_next_header_without_rule(
    monkeypatch, tmp_path
):
    filled = tmp_path / "skills_bank.txt"
    filled.write_text(
        "## Potential Skills\n\n- Kubernetes\n\n## Projects\n- Should not appear\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(reflow, "SKILLS_BANK_PATH", filled)

    assert reflow.load_potential_skills() == ["Kubernetes"]
