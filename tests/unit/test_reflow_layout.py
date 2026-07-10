"""Unit tests for reflow.py's layout-fit validator.

Uses the fake_reflow_module fixture (tests/fixtures/fake_reflow_resume.py)
instead of the real, gitignored reflow_resume.py.
"""

import copy

import reflow


def test_current_fixture_content_passes_layout_fit(fake_reflow_module):
    candidate = reflow.build_candidate_from_module(fake_reflow_module)

    assert reflow.check_layout_fit(fake_reflow_module, candidate) == []


def test_overlong_bullet_fails_two_line_check(fake_reflow_module):
    candidate = reflow.build_candidate_from_module(fake_reflow_module)
    candidate.experience[0].bullets[0] = "built and shipped things " * 30

    problems = reflow.check_layout_fit(fake_reflow_module, candidate)

    assert any("wraps to" in problem for problem in problems)


def test_overlong_summary_causes_page_overflow(fake_reflow_module):
    candidate = reflow.build_candidate_from_module(fake_reflow_module)
    candidate.summary = "This is a very long summary sentence. " * 200

    problems = reflow.check_layout_fit(fake_reflow_module, candidate)

    assert any("overflows" in problem for problem in problems)


def test_overwide_skills_line_is_flagged(fake_reflow_module):
    candidate = reflow.build_candidate_from_module(fake_reflow_module)
    candidate.skills[0].rest = "a very long list of skills, " * 10

    problems = reflow.check_layout_fit(fake_reflow_module, candidate)

    assert any("too wide" in problem for problem in problems)
