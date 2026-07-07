"""Resume reflow orchestrator.

Reads the latest job-match gap analysis (result.md), tailors the CONTENT
block of resume/resume_reflow/reflow_resume.py with Claude Sonnet 5, regrades
each candidate with the existing JobDescriptionEvaluator pipeline, and writes
resume/resume_reflow/reflow_resume_tailored.py when the best score is >= 70.

Run: python reflow.py   (after a successful `python score.py` run)
"""

import copy
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel

from evaluator import JobDescriptionEvaluator
from llm_utils import initialize_llm_provider, extract_json_from_response
from models import JSONResume, Skill
from prompt import DEFAULT_MODEL, MODEL_PARAMETERS, CLAUDE_API_KEY
from prompts.template_manager import TemplateManager
from score import (
    RESULT_FILE_PATH,
    find_resume_file,
    load_job_description,
    _knockout_resolver,
)
from transform import convert_json_resume_to_text

logger = logging.getLogger(__name__)

TAILOR_MODEL = "claude-sonnet-5"
REFLOW_RESUME_PATH = Path("resume/resume_reflow/reflow_resume.py")
TAILORED_RESUME_PATH = Path("resume/resume_reflow/reflow_resume_tailored.py")
SKILLS_BANK_PATH = Path("resume/resume_reflow/skills_bank.txt")

MAX_ITERATIONS = 6
MAX_STAGNANT_ITERATIONS = 2
MAX_VALIDATION_RETRIES = 3

FULL_REPORT_HEADER = "# Job Match Evaluation:"
FLAGGED_REPORT_HEADER = "# Requirement Gate:"
TARGET_ROLE_PREFIX = "**Target Role:** "
WEIGHT_PROFILE_PREFIX = "**Weight profile:** "
REQUIRED_MISSING_HEADER = "**Required skills MISSING:**"
PREFERRED_MISSING_HEADER = "**Preferred skills missing:**"
IMPROVEMENT_SECTION_HEADER = "## Areas for Improvement"
FEATURES_TO_ADD_HEADER = "## Features to Add"


class GapAnalysis(BaseModel):
    job_title: str
    weight_profile: str = "engineering"
    missing_required_skills: List[str] = []
    missing_preferred_skills: List[str] = []
    improvement_areas: List[str] = []


def _line_value_with_prefix(lines: List[str], prefix: str) -> Optional[str]:
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped[len(prefix):].strip()
    return None


def _comma_list_after_header(lines: List[str], header: str) -> List[str]:
    for index, line in enumerate(lines):
        if line.strip() == header and index + 1 < len(lines):
            value = lines[index + 1].strip()
            if not value or value == "None":
                return []
            return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _bullets_under_section(lines: List[str], section_header: str) -> List[str]:
    items = []
    in_section = False
    for line in lines:
        stripped = line.strip()
        if stripped == section_header:
            in_section = True
            continue
        if in_section:
            if stripped.startswith("## "):
                break
            if stripped.startswith("- "):
                items.append(stripped[2:].strip())
    return items


def parse_result_markdown(result_path: str = RESULT_FILE_PATH) -> GapAnalysis:
    if not os.path.exists(result_path):
        sys.exit(
            f"Error: '{result_path}' not found. Run 'python score.py' first to "
            "produce a job-match report."
        )

    lines = Path(result_path).read_text(encoding="utf-8").splitlines()
    first_line = lines[0].strip() if lines else ""

    if first_line.startswith(FULL_REPORT_HEADER):
        job_title = _line_value_with_prefix(lines, TARGET_ROLE_PREFIX)
        weight_profile = _line_value_with_prefix(lines, WEIGHT_PROFILE_PREFIX)
        return GapAnalysis(
            job_title=job_title or "Unknown role",
            weight_profile=weight_profile or "engineering",
            missing_required_skills=_comma_list_after_header(lines, REQUIRED_MISSING_HEADER),
            missing_preferred_skills=_comma_list_after_header(lines, PREFERRED_MISSING_HEADER),
            improvement_areas=_bullets_under_section(lines, IMPROVEMENT_SECTION_HEADER),
        )

    if first_line.startswith(FLAGGED_REPORT_HEADER):
        job_title = _line_value_with_prefix(lines, TARGET_ROLE_PREFIX)
        return GapAnalysis(
            job_title=job_title or "Unknown role",
            missing_required_skills=_bullets_under_section(lines, FEATURES_TO_ADD_HEADER),
        )

    sys.exit(
        f"Error: '{result_path}' does not look like a job-match report "
        f"(expected it to start with '{FULL_REPORT_HEADER}' or "
        f"'{FLAGGED_REPORT_HEADER}'). Re-run 'python score.py' to regenerate it."
    )
