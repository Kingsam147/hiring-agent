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


def load_cached_resume() -> JSONResume:
    pdf_path = find_resume_file()
    resume_file_stem = os.path.splitext(os.path.basename(pdf_path))[0]
    cache_filename = f"cache/resumecache_{resume_file_stem}.json"
    if not os.path.exists(cache_filename):
        sys.exit(
            f"Error: '{cache_filename}' not found. Run 'python score.py' first "
            "so the parsed resume is cached."
        )
    cached_data = json.loads(Path(cache_filename).read_text(encoding="utf-8"))
    return JSONResume(**cached_data)


def load_reflow_resume_module():
    if not REFLOW_RESUME_PATH.exists():
        sys.exit(f"Error: '{REFLOW_RESUME_PATH}' not found.")
    module_spec = importlib.util.spec_from_file_location(
        "reflow_resume", REFLOW_RESUME_PATH
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def load_skills_bank() -> str:
    if not SKILLS_BANK_PATH.exists():
        return ""
    content = SKILLS_BANK_PATH.read_text(encoding="utf-8").strip()
    meaningful_lines = [
        line
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith(("#", "["))
    ]
    return content if meaningful_lines else ""


class TailoredSkill(BaseModel):
    label: str
    rest: str


class TailoredEntry(BaseModel):
    title: str
    bullets: List[str]


class TailoredResume(BaseModel):
    summary: str
    skills: List[TailoredSkill]
    experience: List[TailoredEntry]
    projects: List[TailoredEntry]


def build_candidate_from_module(module) -> TailoredResume:
    return TailoredResume(
        summary=module.SUMMARY,
        skills=[TailoredSkill(label=label, rest=rest) for label, rest in module.SKILLS],
        experience=[
            TailoredEntry(title=entry["title"], bullets=list(entry["bullets"]))
            for entry in module.EXPERIENCE
        ],
        projects=[
            TailoredEntry(title=entry["title"], bullets=list(entry["bullets"]))
            for entry in module.PROJECTS
        ],
    )


MAX_BULLET_LINES = 2


def _replay_vertical_layout(module, candidate: TailoredResume) -> float:
    """Mirror build()'s vertical accumulation for a candidate's content.

    Returns the top-of-line coordinate (from the page top) of the final
    ACTIVITIES line. Must stay in lockstep with reflow_resume.build().
    """
    wrap = module._wrap
    leading = module.LEADING

    top = 75.0

    summary_lines = wrap(
        candidate.summary, module.F_REG, module.SZ_BODY, module.TEXT_RIGHT - module.BODY_X
    )
    y = top + module.GAP_HEADER_TO_BODY
    last = y + (len(summary_lines) - 1) * leading
    top = last + module.GAP_SECTION

    y = top + module.GAP_HEADER_TO_SKILLS
    last = y + (len(candidate.skills) - 1) * leading
    top = last + module.GAP_SECTION

    for entries, gap_header_to_entry in (
        (candidate.experience, module.GAP_HEADER_TO_ENTRY_EXP),
        (candidate.projects, module.GAP_HEADER_TO_ENTRY_PROJ),
    ):
        y = top + gap_header_to_entry
        last = y
        for entry in entries:
            bullet_top = y + module.GAP_ENTRY_TITLE_TO_BULLET
            for bullet_text in entry.bullets:
                line_count = len(
                    wrap(
                        bullet_text,
                        module.F_REG,
                        module.SZ_BODY,
                        module.TEXT_RIGHT - module.BULLET_TEXT_X,
                    )
                )
                bullet_top += line_count * leading
            last = bullet_top - leading
            y = last + module.GAP_ENTRY_TO_ENTRY
        top = last + module.GAP_SECTION

    y = top + module.GAP_HEADER_TO_BODY
    top = y + 16.2  # inline literal in build()'s EDUCATION section — keep in sync

    y = top + module.GAP_HEADER_TO_SKILLS
    return y


def check_layout_fit(module, candidate: TailoredResume) -> List[str]:
    from reportlab.pdfbase.pdfmetrics import stringWidth

    problems = []

    skills_line_width = module.TEXT_RIGHT - module.BODY_X
    for skill in candidate.skills:
        label_width = stringWidth(skill.label, module.F_BOLD, module.SZ_BODY)
        rest_width = stringWidth(skill.rest, module.F_REG, module.SZ_BODY)
        if label_width + rest_width > skills_line_width:
            problems.append(
                f'skills line "{skill.label.strip()}" is too wide for one printed '
                "line; shorten its content"
            )

    bullet_width = module.TEXT_RIGHT - module.BULLET_TEXT_X
    for section_name, entries in (
        ("experience", candidate.experience),
        ("projects", candidate.projects),
    ):
        for entry in entries:
            for bullet_index, bullet_text in enumerate(entry.bullets, 1):
                line_count = len(
                    module._wrap(bullet_text, module.F_REG, module.SZ_BODY, bullet_width)
                )
                if line_count > MAX_BULLET_LINES:
                    problems.append(
                        f'{section_name} entry "{entry.title}" bullet {bullet_index} '
                        f"wraps to {line_count} lines (max {MAX_BULLET_LINES}); shorten it"
                    )

    final_line_top = _replay_vertical_layout(module, candidate)
    if final_line_top + module.LEADING > module.PAGE_H:
        problems.append(
            f"content runs to {final_line_top:.1f}pt from the page top and overflows "
            f"the {module.PAGE_H:.0f}pt page; shorten the summary or bullets"
        )
    return problems
