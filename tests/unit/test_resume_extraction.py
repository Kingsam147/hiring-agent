"""Unit tests for the LLM-driven resume section extraction pipeline.

These tests mock the LLM provider's chat() response and verify that
pdf.py + transform.py correctly turn that response into structured resume
data. No real LLM/network calls are made.
"""

import json

from pdf import PDFHandler


def _canned_response(payload: dict) -> dict:
    return {"message": {"role": "assistant", "content": json.dumps(payload)}}


def test_extract_basics_section_transforms_llm_response(monkeypatch):
    handler = PDFHandler()
    canned = {
        "basics": {
            "name": "Jane Doe",
            "email": "jane@example.com",
            "profiles": [{"url": "https://github.com/janedoe"}],
        }
    }
    monkeypatch.setattr(handler.provider, "chat", lambda **kwargs: _canned_response(canned))

    result = handler.extract_basics_section("some resume markdown text")

    assert result is not None
    assert result["basics"]["name"] == "Jane Doe"
    assert result["basics"]["email"] == "jane@example.com"
    assert result["basics"]["profiles"][0]["network"] == "GitHub"
    assert result["basics"]["profiles"][0]["username"] == "janedoe"


def test_extract_skills_section_transforms_llm_response(monkeypatch):
    handler = PDFHandler()
    canned = {
        "skills": [
            {"name": "Languages", "level": None, "keywords": ["Python", "Java"]},
            {"name": "Backend", "level": None, "keywords": ["Node.js", "PostgreSQL"]},
        ]
    }
    monkeypatch.setattr(handler.provider, "chat", lambda **kwargs: _canned_response(canned))

    result = handler.extract_skills_section("some resume markdown text")

    assert result is not None
    assert len(result["skills"]) == 2
    assert result["skills"][0]["name"] == "Languages"
    assert result["skills"][0]["keywords"] == ["Python", "Java"]


def test_extract_section_returns_none_on_malformed_json(monkeypatch):
    handler = PDFHandler()
    monkeypatch.setattr(
        handler.provider,
        "chat",
        lambda **kwargs: {"message": {"role": "assistant", "content": "not valid json"}},
    )

    result = handler.extract_basics_section("some resume markdown text")

    assert result is None
