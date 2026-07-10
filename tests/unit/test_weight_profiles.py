"""Unit tests for weight-profile selection."""

from weight_profiles import suggest_profile


def test_suggest_profile_detects_sales_title():
    assert suggest_profile("Account Executive") == "sales"


def test_suggest_profile_detects_design_title():
    assert suggest_profile("Senior Product Designer") == "design"


def test_suggest_profile_falls_back_to_engineering_for_unrelated_title():
    assert suggest_profile("Backend Engineer") == "engineering"


def test_suggest_profile_uses_industry_when_title_is_ambiguous():
    assert suggest_profile("Software Engineer", industry="Retail") == "sales"
