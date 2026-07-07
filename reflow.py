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


EM_DASH = "—"

FIXED_METRIC_GROUPS = [
    ("1,384ms", "196ms"),
    ("~$60", "~$0.38"),
    ("5-10x",),
    ("32 REST endpoints",),
    ("500+ stars",),
    ("1,200+", "900+", "300+"),
    ("PR #283",),
    ("PR #822",),
]


def check_structure(module, candidate: TailoredResume) -> List[str]:
    problems = []
    if len(candidate.skills) != len(module.SKILLS):
        problems.append(
            f"expected {len(module.SKILLS)} skills lines, got {len(candidate.skills)}"
        )
    for original_entries, tailored_entries, section_name in (
        (module.EXPERIENCE, candidate.experience, "experience"),
        (module.PROJECTS, candidate.projects, "projects"),
    ):
        if len(tailored_entries) != len(original_entries):
            problems.append(
                f"expected {len(original_entries)} {section_name} entries, "
                f"got {len(tailored_entries)}"
            )
            continue
        for original_entry, tailored_entry in zip(original_entries, tailored_entries):
            if tailored_entry.title != original_entry["title"]:
                problems.append(
                    f'{section_name} title must stay verbatim: expected '
                    f'"{original_entry["title"]}", got "{tailored_entry.title}"'
                )
            if len(tailored_entry.bullets) != len(original_entry["bullets"]):
                problems.append(
                    f'{section_name} entry "{original_entry["title"]}" must keep '
                    f'{len(original_entry["bullets"])} bullets, got '
                    f"{len(tailored_entry.bullets)}"
                )
    return problems


def check_em_dashes(candidate: TailoredResume) -> List[str]:
    problems = []
    if EM_DASH in candidate.summary:
        problems.append("summary contains an em dash; replace it")
    for skill in candidate.skills:
        if EM_DASH in skill.label + skill.rest:
            problems.append(f'skills line "{skill.label.strip()}" contains an em dash')
    for entry in candidate.experience + candidate.projects:
        if EM_DASH in entry.title or any(EM_DASH in bullet for bullet in entry.bullets):
            problems.append(f'entry "{entry.title}" contains an em dash')
    return problems


def _metric_home_entries(module) -> Dict[Tuple[str, ...], str]:
    """Map each fixed-metric group to the title of the entry whose bullets
    contain it in the original, unmodified reflow_resume content."""
    homes = {}
    original_entries = list(module.EXPERIENCE) + list(module.PROJECTS)
    for metric_group in FIXED_METRIC_GROUPS:
        for entry in original_entries:
            joined_bullets = " ".join(entry["bullets"])
            if all(substring in joined_bullets for substring in metric_group):
                homes[metric_group] = entry["title"]
                break
    return homes


def check_fixed_metrics(module, candidate: TailoredResume) -> List[str]:
    problems = []
    candidate_bullets_by_title = {
        entry.title: " ".join(entry.bullets)
        for entry in candidate.experience + candidate.projects
    }
    for metric_group, home_title in _metric_home_entries(module).items():
        entry_text = candidate_bullets_by_title.get(home_title, "")
        missing_substrings = [s for s in metric_group if s not in entry_text]
        if missing_substrings:
            problems.append(
                f'fixed metric(s) {missing_substrings} must appear verbatim in the '
                f'bullets of entry "{home_title}"'
            )
    return problems


def validate_candidate(module, candidate: TailoredResume) -> List[str]:
    problems = check_structure(module, candidate)
    if problems:
        return problems  # counts/titles wrong — the scans below assume structure holds
    problems += check_em_dashes(candidate)
    problems += check_fixed_metrics(module, candidate)
    problems += check_layout_fit(module, candidate)
    return problems


def generate_tailored_candidate(
    tailor_provider,
    template_manager: TemplateManager,
    gap: GapAnalysis,
    current_content: TailoredResume,
    skills_bank: str,
    reflow_module,
) -> Optional[TailoredResume]:
    retry_feedback = None
    for attempt in range(1, MAX_VALIDATION_RETRIES + 2):  # 1 try + 3 retries
        system_message = template_manager.render_template("resume_reflow_system_message")
        user_message = template_manager.render_template(
            "resume_reflow_user_message",
            job_title=gap.job_title,
            missing_required_skills=gap.missing_required_skills,
            missing_preferred_skills=gap.missing_preferred_skills,
            improvement_areas=gap.improvement_areas,
            summary=current_content.summary,
            skills=current_content.skills,
            experience=current_content.experience,
            projects=current_content.projects,
            skills_bank=skills_bank,
            retry_feedback=retry_feedback,
        )
        if system_message is None or user_message is None:
            sys.exit("Error: failed to render the resume reflow templates.")

        response = tailor_provider.chat(
            model=TAILOR_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
        )

        try:
            response_text = extract_json_from_response(response["message"]["content"])
            candidate = TailoredResume(**json.loads(response_text))
        except Exception as parse_error:
            logger.warning(f"Tailor attempt {attempt}: unparseable response ({parse_error})")
            retry_feedback = (
                "Your previous reply was not valid JSON matching the required "
                f"structure: {parse_error}. Return ONLY the JSON object."
            )
            continue

        problems = validate_candidate(reflow_module, candidate)
        if not problems:
            return candidate

        logger.warning(f"Tailor attempt {attempt}: {len(problems)} guardrail problem(s)")
        retry_feedback = "; ".join(problems)

    logger.warning("Discarding this iteration's candidate: guardrails still failing after retries.")
    return None
