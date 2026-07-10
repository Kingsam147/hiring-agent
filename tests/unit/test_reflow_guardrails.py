"""Unit tests for reflow.py's structure/em-dash/fixed-metric guardrails.

Uses the fake_reflow_module fixture instead of the real, gitignored
reflow_resume.py.
"""

import copy

import reflow
from models import JobDescriptionData


def test_current_fixture_content_passes_all_guardrails(fake_reflow_module):
    candidate = reflow.build_candidate_from_module(fake_reflow_module)

    assert reflow.validate_candidate(fake_reflow_module, candidate) == []


def test_metric_homes_locates_all_eight_groups(fake_reflow_module):
    homes = reflow._metric_home_entries(fake_reflow_module)

    assert len(homes) == 8


def test_check_structure_flags_wrong_bullet_count(fake_reflow_module):
    candidate = reflow.build_candidate_from_module(fake_reflow_module)
    candidate.experience[0].bullets.append("An extra bullet that should not exist.")

    problems = reflow.check_structure(fake_reflow_module, candidate)

    assert any("must keep" in problem for problem in problems)


def test_check_structure_flags_retitled_entry(fake_reflow_module):
    candidate = reflow.build_candidate_from_module(fake_reflow_module)
    candidate.projects[0].title = "A Completely Different Title"

    problems = reflow.check_structure(fake_reflow_module, candidate)

    assert any("must stay verbatim" in problem for problem in problems)


def test_check_em_dashes_detects_em_dash_in_summary(fake_reflow_module):
    candidate = reflow.build_candidate_from_module(fake_reflow_module)
    candidate.summary = "A summary with an em dash — right here."

    problems = reflow.check_em_dashes(candidate)

    assert any("em dash" in problem for problem in problems)


def test_check_fixed_metrics_detects_altered_metric(fake_reflow_module):
    candidate = reflow.build_candidate_from_module(fake_reflow_module)
    candidate.experience[0].bullets[0] = (
        candidate.experience[0].bullets[0].replace("1,384ms", "1384 ms")
    )

    problems = reflow.check_fixed_metrics(fake_reflow_module, candidate)

    assert any("fixed metric" in problem for problem in problems)


def test_validate_candidate_short_circuits_on_structure_failure(fake_reflow_module):
    candidate = reflow.build_candidate_from_module(fake_reflow_module)
    candidate.skills.pop()

    problems = reflow.validate_candidate(fake_reflow_module, candidate)

    assert any("skills lines" in problem for problem in problems)


def _job_data(**overrides):
    defaults = dict(
        job_title="Backend Engineer",
        required_skills=["Kubernetes", "React"],
        preferred_skills=[],
    )
    defaults.update(overrides)
    return JobDescriptionData(**defaults)


def test_check_no_matched_keyword_regression_flags_dropped_match(fake_reflow_module):
    previous_content = reflow.build_candidate_from_module(fake_reflow_module)
    previous_content.skills[0] = reflow.TailoredSkill(
        label="Languages: ", rest="Python, Kubernetes"
    )
    candidate = copy.deepcopy(previous_content)
    candidate.skills[0] = reflow.TailoredSkill(label="Languages: ", rest="Python, Go")

    problems = reflow.check_no_matched_keyword_regression(
        _job_data(), previous_content, candidate
    )

    assert any("Kubernetes" in problem for problem in problems)


def test_check_no_matched_keyword_regression_allows_unrelated_changes(
    fake_reflow_module,
):
    previous_content = reflow.build_candidate_from_module(fake_reflow_module)
    previous_content.skills[0] = reflow.TailoredSkill(
        label="Languages: ", rest="Python, Kubernetes"
    )
    candidate = copy.deepcopy(previous_content)
    candidate.summary = "A rewritten but still Kubernetes-mentioning summary."

    problems = reflow.check_no_matched_keyword_regression(
        _job_data(), previous_content, candidate
    )

    assert problems == []


def test_check_no_matched_keyword_regression_ignores_never_matched_skills(
    fake_reflow_module,
):
    previous_content = reflow.build_candidate_from_module(fake_reflow_module)
    candidate = copy.deepcopy(previous_content)

    problems = reflow.check_no_matched_keyword_regression(
        _job_data(), previous_content, candidate
    )

    assert problems == []


def test_validate_candidate_skips_regression_check_without_job_data(
    fake_reflow_module,
):
    candidate = reflow.build_candidate_from_module(fake_reflow_module)

    assert reflow.validate_candidate(fake_reflow_module, candidate) == []
