"""Integration test for the real sentence-transformers embedding model.

This is the one component intentionally allowed to hit a real (local, no
API key) model in CI: all-MiniLM-L6-v2. No LLM/GitHub API calls happen here.
"""

from evaluator import JobDescriptionEvaluator


def _make_evaluator(job_description: str) -> JobDescriptionEvaluator:
    return JobDescriptionEvaluator(
        job_description=job_description,
        model_name="gemini-2.5-flash",
        weight_profile="engineering",
    )


def test_semantic_score_ranks_relevant_resume_higher_than_unrelated():
    job_description = (
        "We are hiring a backend engineer with strong experience in Python, "
        "PostgreSQL, and building REST APIs at scale."
    )
    evaluator = _make_evaluator(job_description)
    evaluator._load_embedding_model()

    relevant_resume_text = (
        "Backend engineer with 4 years of experience building REST APIs in "
        "Python, using PostgreSQL for data storage and Redis for caching."
    )
    unrelated_resume_text = (
        "Pastry chef with 10 years of experience designing dessert menus "
        "and managing a bakery kitchen for a five-star hotel."
    )

    relevant_score = evaluator.compute_semantic_score(relevant_resume_text)
    unrelated_score = evaluator.compute_semantic_score(unrelated_resume_text)

    assert 0 <= relevant_score <= 100
    assert 0 <= unrelated_score <= 100
    assert relevant_score > unrelated_score


def test_semantic_score_is_deterministic_for_same_input():
    evaluator = _make_evaluator("Backend engineer role requiring Python and SQL.")
    evaluator._load_embedding_model()

    resume_text = "Experienced backend engineer skilled in Python and SQL."

    first_score = evaluator.compute_semantic_score(resume_text)
    second_score = evaluator.compute_semantic_score(resume_text)

    assert first_score == second_score
