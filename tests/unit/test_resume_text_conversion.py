"""Unit tests for turning structured resume/GitHub data into the plain text
that gets fed into keyword matching and LLM grading."""

from models import (
    Basics,
    Education,
    JSONResume,
    Location,
    Profile,
    Project,
    Skill,
    Work,
)
from transform import convert_github_data_to_text, convert_json_resume_to_text


def test_convert_json_resume_to_text_includes_all_major_sections():
    resume_data = JSONResume(
        basics=Basics(
            name="Jane Doe",
            email="jane@example.com",
            summary="Backend engineer.",
            location=Location(city="Boston", region="MA"),
            profiles=[Profile(network="GitHub", username="janedoe", url="https://github.com/janedoe")],
        ),
        work=[
            Work(
                name="Acme Corp",
                position="Software Engineer",
                startDate="2022-01",
                endDate="Present",
                highlights=["Shipped feature X", "Reduced latency by 50%"],
            )
        ],
        education=[
            Education(
                institution="State University",
                studyType="B.S.",
                area="Computer Science",
                startDate="2018-01",
                endDate="2022-05",
            )
        ],
        skills=[Skill(name="Languages", keywords=["Python", "SQL"])],
        projects=[
            Project(
                name="Cool Project",
                description="A cool project.",
                highlights=["Built a REST API"],
            )
        ],
    )

    text = convert_json_resume_to_text(resume_data)

    assert "Jane Doe" in text
    assert "Acme Corp" in text
    assert "Shipped feature X" in text
    assert "State University" in text
    assert "Python, SQL" in text
    assert "Cool Project" in text
    assert "Built a REST API" in text


def test_convert_json_resume_to_text_handles_empty_resume():
    text = convert_json_resume_to_text(JSONResume())

    assert text == ""


def test_convert_github_data_to_text_includes_profile_and_projects():
    github_data = {
        "profile": {
            "username": "janedoe",
            "name": "Jane Doe",
            "public_repos": 12,
            "followers": 5,
        },
        "projects": [
            {
                "name": "cool-repo",
                "description": "A cool repo.",
                "github_url": "https://github.com/janedoe/cool-repo",
                "github_details": {"stars": 42, "forks": 3, "language": "Python"},
            }
        ],
    }

    text = convert_github_data_to_text(github_data)

    assert "janedoe" in text
    assert "cool-repo" in text
    assert "Stars: 42" in text


def test_convert_github_data_to_text_handles_missing_keys():
    text = convert_github_data_to_text({})

    assert "GITHUB DATA" in text
