"""Unit tests for the pure resume-transformation helpers in transform.py.

These turn raw (LLM-extracted) section dicts into JSON-Resume-shaped dicts.
All pure functions, no mocking needed.
"""

from transform import (
    parse_date_range,
    transform_achievements,
    transform_education,
    transform_projects,
    transform_work_experience,
)


def test_parse_date_range_handles_month_range():
    start, end = parse_date_range("Jan-Mar 2021")
    assert start == "Jan 2021"
    assert end == "Mar 2021"


def test_parse_date_range_handles_single_month():
    start, end = parse_date_range("Jan 2021")
    assert start == "Jan 2021"
    assert end is None


def test_parse_date_range_handles_year_range():
    start, end = parse_date_range("2020-2021")
    assert start == "2020-01"
    assert end == "2021-12"


def test_parse_date_range_handles_onwards():
    start, end = parse_date_range("Jun 2023 onwards")
    assert start == "Jun 2023"
    assert end == "Present"


def test_parse_date_range_handles_empty_input():
    assert parse_date_range("") == (None, None)


def test_transform_work_experience_uses_explicit_dates():
    raw = [
        {
            "name": "Acme Corp",
            "position": "Software Engineer",
            "startDate": "2022-01",
            "endDate": "2023-06",
            "description": "Built things.",
            "highlights": ["Shipped feature X"],
        }
    ]

    transformed = transform_work_experience(raw)

    assert transformed[0]["name"] == "Acme Corp"
    assert transformed[0]["position"] == "Software Engineer"
    assert transformed[0]["startDate"] == "2022-01"
    assert transformed[0]["endDate"] == "2023-06"
    assert transformed[0]["highlights"] == ["Shipped feature X"]


def test_transform_work_experience_parses_month_range_start_date():
    raw = [{"name": "Acme Corp", "startDate": "Jan-Mar 2021", "highlights": []}]

    transformed = transform_work_experience(raw)

    assert transformed[0]["startDate"] == "Jan 2021"
    assert transformed[0]["endDate"] == "Mar 2021"


def test_transform_work_experience_joins_list_description_into_summary():
    raw = [{"name": "Acme Corp", "description": ["Line one.", "Line two."]}]

    transformed = transform_work_experience(raw)

    assert transformed[0]["summary"] == "Line one. Line two."


def test_transform_education_splits_degree_and_area():
    raw = [
        {
            "institution": "State University",
            "degree": "B.S., Computer Science",
            "years": "2018-2022",
            "gpa": 3.8,
        }
    ]

    transformed = transform_education(raw)

    assert transformed[0]["institution"] == "State University"
    assert transformed[0]["studyType"] == "B.S."
    assert transformed[0]["area"] == "Computer Science"
    assert transformed[0]["startDate"] == "2018-01"
    assert transformed[0]["endDate"] == "2022-12"
    assert transformed[0]["score"] == "3.8"


def test_transform_achievements_maps_alternate_field_names():
    raw = [
        {"name": "Best Hackathon Project", "organization": "HackXYZ", "year": "2023"}
    ]

    transformed = transform_achievements(raw)

    assert transformed[0]["title"] == "Best Hackathon Project"
    assert transformed[0]["awarder"] == "HackXYZ"
    assert transformed[0]["date"] == "2023-01"


def test_transform_projects_extracts_skills_from_pipe_in_name():
    raw = [
        {
            "name": "Cool Project | Python, Redis",
            "description": "A cool project.",
        }
    ]

    transformed = transform_projects(raw)

    assert transformed[0]["name"] == "Cool Project"
    assert transformed[0]["skills"] == ["Python", "Redis"]


def test_transform_projects_falls_back_to_technologies_for_skills():
    raw = [{"name": "Cool Project", "technologies": "Python, Redis"}]

    transformed = transform_projects(raw)

    assert transformed[0]["technologies"] == ["Python", "Redis"]
    assert transformed[0]["skills"] == ["Python", "Redis"]
