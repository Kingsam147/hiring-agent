"""Resume reflow validator/regrader.

Tailored content is written by the calling agent (e.g. a Claude chat running
the resume-reflow-pipeline skill), not by an LLM API call from this script.
This script only supplies the tailoring context, validates candidates against
the guardrails, regrades them with the existing JobDescriptionEvaluator
pipeline, and writes resume/resume_reflow/reflow_resume_tailored.py when a
candidate sets a new best score >= 70.

Run (after a successful `python score.py` run):
    python reflow.py --show-context
    python reflow.py --candidate <path-to-candidate.json>
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel

from evaluator import JobDescriptionEvaluator
from keyword_matching import (
    compute_keyword_match,
    normalize_text,
    requirement_satisfied,
)
from models import JobDescriptionData, JobEvaluationData, JSONResume, Skill
from prompt import DEFAULT_MODEL, MODEL_PARAMETERS
from score import (
    RESULT_FILE_PATH,
    build_job_evaluation_markdown,
    find_resume_file,
    load_job_description,
    write_result_markdown,
)
from transform import convert_json_resume_to_text

logger = logging.getLogger(__name__)

REFLOW_RESUME_PATH = Path("resume/resume_reflow/reflow_resume.py")
TAILORED_RESUME_PATH = Path("resume/resume_reflow/reflow_resume_tailored.py")
SKILLS_BANK_PATH = Path("resume/resume_reflow/skills_bank.txt")
REFLOW_STATE_PATH = Path("resume/resume_reflow/reflow_state.json")

FULL_REPORT_HEADER = "# Job Match Evaluation:"
TARGET_ROLE_PREFIX = "**Target Role:** "
WEIGHT_PROFILE_PREFIX = "**Weight profile:** "
REQUIRED_MISSING_HEADER = "**Required skills MISSING:**"
PREFERRED_MISSING_HEADER = "**Preferred skills missing:**"
IMPROVEMENT_SECTION_HEADER = "## Areas for Improvement"


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
            return stripped[len(prefix) :].strip()
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
            missing_required_skills=_comma_list_after_header(
                lines, REQUIRED_MISSING_HEADER
            ),
            missing_preferred_skills=_comma_list_after_header(
                lines, PREFERRED_MISSING_HEADER
            ),
            improvement_areas=_bullets_under_section(lines, IMPROVEMENT_SECTION_HEADER),
        )

    sys.exit(
        f"Error: '{result_path}' does not look like a job-match report "
        f"(expected it to start with '{FULL_REPORT_HEADER}'). "
        "Re-run 'python score.py' to regenerate it."
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


def load_reflow_resume_module(path: Optional[Path] = None):
    path = path if path is not None else REFLOW_RESUME_PATH
    if not path.exists():
        sys.exit(f"Error: '{path}' not found.")
    module_spec = importlib.util.spec_from_file_location("reflow_resume", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def render_tailored_resume() -> Path:
    tailored_module = load_reflow_resume_module(TAILORED_RESUME_PATH)
    output_path = TAILORED_RESUME_PATH.with_suffix(".pdf")
    tailored_module.build(str(output_path))
    return output_path


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


_MARKDOWN_HEADER_PATTERN = re.compile(r"^#{1,6}\s+(.*)$")
_HORIZONTAL_RULE_PATTERN = re.compile(r"^-{3,}$")


def _skills_bank_section(content: str, header: str) -> List[str]:
    """Extract one section's entries from skills_bank.txt.

    Supports both the bracket style ("[Header]", entries stop at the next
    "[...]") and Markdown style ("## Header", "- entry", stops at the next
    "#"-header of any level or a "---" rule) so the file can be freely
    reorganized without breaking Potential Skills detection.
    """
    items = []
    in_section = False
    section_style = None

    for line in content.splitlines():
        stripped = line.strip()

        if not in_section:
            if stripped == f"[{header}]":
                in_section = True
                section_style = "bracket"
            else:
                markdown_match = _MARKDOWN_HEADER_PATTERN.match(stripped)
                if markdown_match and markdown_match.group(1).strip() == header:
                    in_section = True
                    section_style = "markdown"
            continue

        if section_style == "bracket":
            if stripped.startswith("[") and stripped.endswith("]"):
                break
            if not stripped or stripped.startswith("#"):
                continue
            items.append(stripped)
        else:
            if _MARKDOWN_HEADER_PATTERN.match(
                stripped
            ) or _HORIZONTAL_RULE_PATTERN.match(stripped):
                break
            if not stripped:
                continue
            if stripped.startswith("- "):
                stripped = stripped[2:].strip()
            items.append(stripped)

    return items


def load_potential_skills() -> List[str]:
    if not SKILLS_BANK_PATH.exists():
        return []
    content = SKILLS_BANK_PATH.read_text(encoding="utf-8")
    return _skills_bank_section(content, "Potential Skills")


_PARENTHETICAL_PATTERN = re.compile(r"\s*\([^)]*\)")


def _skill_core_name(skill_line: str) -> str:
    return _PARENTHETICAL_PATTERN.sub("", skill_line).strip()


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


def find_used_potential_skills(
    potential_skills: List[str], candidate: TailoredResume
) -> List[str]:
    combined_text = " ".join(
        [candidate.summary]
        + [skill.label + skill.rest for skill in candidate.skills]
        + [
            bullet
            for entry in candidate.experience + candidate.projects
            for bullet in entry.bullets
        ]
    ).lower()
    return [
        skill_line
        for skill_line in potential_skills
        if _skill_core_name(skill_line).lower() in combined_text
    ]


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
        candidate.summary,
        module.F_REG,
        module.SZ_BODY,
        module.TEXT_RIGHT - module.BODY_X,
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
                    module._wrap(
                        bullet_text, module.F_REG, module.SZ_BODY, bullet_width
                    )
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
                    f"{section_name} title must stay verbatim: expected "
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
                f"fixed metric(s) {missing_substrings} must appear verbatim in the "
                f'bullets of entry "{home_title}"'
            )
    return problems


def _candidate_match_corpus(candidate: TailoredResume) -> str:
    parts = [candidate.summary]
    parts.extend(skill.label + skill.rest for skill in candidate.skills)
    for entry in candidate.experience + candidate.projects:
        parts.append(entry.title)
        parts.extend(entry.bullets)
    return normalize_text("\n".join(parts))


def check_no_matched_keyword_regression(
    job_data: JobDescriptionData,
    previous_content: TailoredResume,
    candidate: TailoredResume,
) -> List[str]:
    previous_corpus = _candidate_match_corpus(previous_content)
    candidate_corpus = _candidate_match_corpus(candidate)

    problems = []
    for skill in (job_data.required_skills or []) + (job_data.preferred_skills or []):
        was_matched = requirement_satisfied(skill, previous_corpus)
        still_matched = requirement_satisfied(skill, candidate_corpus)
        if was_matched and not still_matched:
            problems.append(
                f'lost previously-matched skill "{skill}"; keep whatever wording '
                "satisfied this requirement instead of removing or replacing it"
            )
    return problems


def validate_candidate(
    module,
    candidate: TailoredResume,
    job_data: Optional[JobDescriptionData] = None,
    previous_content: Optional[TailoredResume] = None,
) -> List[str]:
    problems = check_structure(module, candidate)
    if problems:
        return problems  # counts/titles wrong — the scans below assume structure holds
    problems += check_em_dashes(candidate)
    problems += check_fixed_metrics(module, candidate)
    problems += check_layout_fit(module, candidate)
    if job_data is not None and previous_content is not None:
        problems += check_no_matched_keyword_regression(
            job_data, previous_content, candidate
        )
    return problems


def _find_work_entry(work_items, title: str):
    if not work_items:
        return None
    parts = title.split(" | ")
    company_name = parts[0].strip()
    position = parts[1].strip() if len(parts) > 1 else None
    for work_item in work_items:
        if work_item.name and work_item.name.strip().lower() == company_name.lower():
            return work_item
    if position:
        for work_item in work_items:
            if work_item.position and position.lower() in work_item.position.lower():
                return work_item
    return None


def _find_project_entry(project_items, title: str):
    if not project_items:
        return None
    project_name = title.split(" | ")[0].strip().lower()
    for project_item in project_items:
        if not project_item.name:
            continue
        cached_name = project_item.name.strip().lower()
        if (
            cached_name == project_name
            or project_name in cached_name
            or cached_name in project_name
        ):
            return project_item
    return None


def apply_candidate_to_resume(
    original_resume: JSONResume, candidate: TailoredResume
) -> JSONResume:
    tailored_resume = copy.deepcopy(original_resume)

    if tailored_resume.basics is not None:
        tailored_resume.basics.summary = candidate.summary

    tailored_resume.skills = [
        Skill(
            name=skill.label.rstrip(": ").strip(),
            keywords=[
                keyword.strip() for keyword in skill.rest.split(",") if keyword.strip()
            ],
        )
        for skill in candidate.skills
    ]

    for entry in candidate.experience:
        work_item = _find_work_entry(tailored_resume.work, entry.title)
        if work_item is not None:
            work_item.highlights = list(entry.bullets)
        else:
            logger.warning(
                f'No cached work entry matches title "{entry.title}"; its tailored '
                "bullets will not affect the regrade."
            )

    for entry in candidate.projects:
        project_item = _find_project_entry(tailored_resume.projects, entry.title)
        if project_item is not None:
            project_item.highlights = list(entry.bullets)
        else:
            logger.warning(
                f'No cached project matches title "{entry.title}"; its tailored '
                "bullets will not affect the regrade."
            )

    return tailored_resume


def regrade_candidate(
    job_description: str,
    weight_profile: str,
    tailored_resume: JSONResume,
) -> JobEvaluationData:
    resume_text = convert_json_resume_to_text(tailored_resume)
    fresh_evaluator = JobDescriptionEvaluator(
        job_description=job_description,
        model_name=DEFAULT_MODEL,
        model_params=MODEL_PARAMETERS.get(DEFAULT_MODEL),
        weight_profile=weight_profile,
    )
    return fresh_evaluator.evaluate(resume_text, resume_data=tailored_resume)


def resolve_band(score: float) -> Optional[str]:
    if score < 70:
        return None
    if score < 80:
        return "70% - 80%"
    if score < 90:
        return "80% - 90%"
    return "90%+"


def _serialize_summary(summary: str) -> str:
    return f"SUMMARY = (\n    {summary!r}\n)\n\n"


def _serialize_skills(skills: List[TailoredSkill]) -> str:
    lines = ["SKILLS = ["]
    for skill in skills:
        lines.append(f"    ({skill.label!r},")
        lines.append(f"     {skill.rest!r}),")
    lines.append("]")
    return "\n".join(lines) + "\n\n"


def _serialize_entries(
    variable_name: str, entries: List[TailoredEntry], original_entries
) -> str:
    lines = [f"{variable_name} = ["]
    for tailored_entry, original_entry in zip(entries, original_entries):
        lines.append(f'    {{"title": {tailored_entry.title!r},')
        lines.append(f'     "meta": {original_entry["meta"]!r},')
        lines.append('     "bullets": [')
        for bullet_text in tailored_entry.bullets:
            lines.append(f"        {bullet_text!r},")
        lines.append("     ]},")
    lines.append("]")
    return "\n".join(lines) + "\n\n"


def _replace_source_block(
    source: str, start_anchor: str, end_anchor: str, replacement: str
) -> str:
    start_index = source.index("\n" + start_anchor) + 1
    end_index = source.index("\n" + end_anchor) + 1
    return source[:start_index] + replacement + source[end_index:]


def write_tailored_generator(candidate: TailoredResume, reflow_module) -> None:
    source = REFLOW_RESUME_PATH.read_text(encoding="utf-8")
    source = _replace_source_block(
        source,
        "SUMMARY = (",
        "# (bold label, regular remainder)",
        _serialize_summary(candidate.summary),
    )
    source = _replace_source_block(
        source, "SKILLS = [", "EXPERIENCE = [", _serialize_skills(candidate.skills)
    )
    source = _replace_source_block(
        source,
        "EXPERIENCE = [",
        "PROJECTS = [",
        _serialize_entries(
            "EXPERIENCE", candidate.experience, reflow_module.EXPERIENCE
        ),
    )
    source = _replace_source_block(
        source,
        "PROJECTS = [",
        "EDUCATION_LINE = ",
        _serialize_entries("PROJECTS", candidate.projects, reflow_module.PROJECTS),
    )
    TAILORED_RESUME_PATH.write_text(source, encoding="utf-8")
    print(f"Wrote {TAILORED_RESUME_PATH}")


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _current_input_hashes() -> Dict[str, str]:
    return {
        "resume_hash": _hash_text(REFLOW_RESUME_PATH.read_text(encoding="utf-8")),
        "jd_hash": _hash_text(load_job_description()),
    }


def load_reflow_state() -> Optional[float]:
    """Return the saved best score for the current inputs, or None.

    The state is hash-gated against the base resume's content and the job
    description's content: if either changed since the state was written,
    the saved best no longer applies and the session starts over.
    """
    if not REFLOW_STATE_PATH.exists():
        return None
    try:
        state = json.loads(REFLOW_STATE_PATH.read_text(encoding="utf-8"))
        saved_best = float(state["best_score"])
        saved_hashes = {
            "resume_hash": state["resume_hash"],
            "jd_hash": state["jd_hash"],
        }
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        sys.exit(
            f"Error: malformed session cache '{REFLOW_STATE_PATH}' ({exc}). "
            "Delete the file to start a fresh tailoring session."
        )
    if saved_hashes != _current_input_hashes():
        print(
            "Note: base resume or job description changed since the last "
            "session; previous best score reset."
        )
        return None
    return saved_best


def save_reflow_state(best_score: float) -> None:
    state = _current_input_hashes()
    state["best_score"] = best_score
    REFLOW_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REFLOW_STATE_PATH.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )


class ReflowContext(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    gap: GapAnalysis
    job_description: str
    original_resume: JSONResume
    skills_bank: str
    job_data: JobDescriptionData
    current_content: TailoredResume
    missing_soft_skills: List[str]


def gather_context() -> Tuple[ReflowContext, object]:
    """Load everything both CLI modes need. Returns (context, reflow_module).

    Requires a prior successful `python score.py` run (result.md and the
    resume cache must exist). The job-requirement extraction goes through the
    configured LLM provider but is cached by content hash, so repeat calls
    within a session are free.
    """
    gap = parse_result_markdown()
    job_description = load_job_description()
    original_resume = load_cached_resume()
    reflow_module = load_reflow_resume_module()
    skills_bank = load_skills_bank()
    if not skills_bank:
        print(
            f"Note: '{SKILLS_BANK_PATH}' is empty - tailoring will only rework "
            "content already in the resume."
        )

    job_data = JobDescriptionEvaluator(
        job_description=job_description,
        model_name=DEFAULT_MODEL,
        model_params=MODEL_PARAMETERS.get(DEFAULT_MODEL),
        weight_profile=gap.weight_profile,
    ).extract_job_requirements()

    # gap.missing_required_skills/missing_preferred_skills may be stale --
    # parsed from a result.md written before this run. Recompute the real gap
    # fresh from job_data against the original resume so tailoring always
    # targets an accurate list.
    original_resume_text = convert_json_resume_to_text(original_resume)
    original_keyword_match = compute_keyword_match(
        job_data, original_resume_text, original_resume
    )
    gap = gap.model_copy(
        update={
            "missing_required_skills": original_keyword_match.missing_required,
            "missing_preferred_skills": original_keyword_match.missing_preferred,
        }
    )

    context = ReflowContext(
        gap=gap,
        job_description=job_description,
        original_resume=original_resume,
        skills_bank=skills_bank,
        job_data=job_data,
        current_content=build_candidate_from_module(reflow_module),
        missing_soft_skills=list(original_keyword_match.missing_soft_skills or []),
    )
    return context, reflow_module


def _print_list(header: str, items: List[str]) -> None:
    print(f"{header}:")
    if not items:
        print("  (none)")
        return
    for item in items:
        print(f"  - {item}")


def show_context() -> None:
    context, _ = gather_context()
    gap = context.gap
    content = context.current_content

    print(f"Target role: {gap.job_title} | weight profile: {gap.weight_profile}")
    print()
    _print_list("Missing REQUIRED skills", gap.missing_required_skills)
    _print_list("Missing preferred skills", gap.missing_preferred_skills)
    _print_list(
        "Missing SOFT skills (each must appear verbatim in your candidate)",
        context.missing_soft_skills,
    )
    _print_list("Areas for improvement (from score.py)", gap.improvement_areas)

    print()
    print("=" * 60)
    print("CURRENT RESUME CONTENT (your candidate must keep this structure)")
    print("=" * 60)
    print(f"\nSUMMARY:\n{content.summary}")
    print(f"\nSKILLS ({len(content.skills)} lines):")
    for skill in content.skills:
        print(f"  label={skill.label!r} rest={skill.rest!r}")
    for section_name, entries in (
        ("EXPERIENCE", content.experience),
        ("PROJECTS", content.projects),
    ):
        print(f"\n{section_name} ({len(entries)} entries):")
        for entry in entries:
            print(f"  title={entry.title!r} ({len(entry.bullets)} bullets)")
            for bullet in entry.bullets:
                print(f"    - {bullet}")

    print()
    print("=" * 60)
    print("SKILLS BANK (verified skills you may draw on)")
    print("=" * 60)
    print(context.skills_bank or "(empty)")

    saved_best = load_reflow_state()
    print()
    if saved_best is None:
        print("Session state: fresh session, no previous best score.")
    else:
        print(f"Session state: resuming, previous best score {saved_best}/100.")


def evaluate_candidate(candidate_path: str) -> None:
    context, reflow_module = gather_context()

    candidate_file = Path(candidate_path)
    if not candidate_file.exists():
        sys.exit(f"Error: candidate file '{candidate_path}' not found.")
    try:
        candidate = TailoredResume(
            **json.loads(candidate_file.read_text(encoding="utf-8"))
        )
    except Exception as exc:
        sys.exit(
            f"Error: '{candidate_path}' is not valid JSON matching the required "
            f"candidate structure: {exc}"
        )

    problems = validate_candidate(
        reflow_module, candidate, context.job_data, context.current_content
    )
    problems += _check_missing_soft_skills(candidate, context.missing_soft_skills)
    if problems:
        print("VALIDATION FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return

    tailored_resume = apply_candidate_to_resume(context.original_resume, candidate)
    evaluation = regrade_candidate(
        context.job_description, context.gap.weight_profile, tailored_resume
    )
    score = evaluation.weighted_total

    previous_best = load_reflow_state()
    previous_display = "none" if previous_best is None else f"{previous_best}"
    print(f"SCORE: {score}/100 (previous best: {previous_display}/100)")

    band = resolve_band(score)
    if band is None:
        print(
            "Not written: score is below 70. This candidate is not a "
            "compatible enough match to render."
        )
        return
    if previous_best is not None and score <= previous_best:
        print(
            f"Not written: score did not beat the saved best of "
            f"{previous_best}/100. Incorporate the remaining gaps and try again."
        )
        return

    print(f"NEW BEST - band {band}")
    write_tailored_generator(candidate, reflow_module)
    output_pdf_path = render_tailored_resume()
    print(f"Rendered {output_pdf_path}")
    save_reflow_state(score)

    candidate_name = "Candidate"
    if context.original_resume.basics is not None and context.original_resume.basics.name:
        candidate_name = context.original_resume.basics.name
    write_result_markdown(build_job_evaluation_markdown(evaluation, candidate_name))
    print(
        f"Wrote final tailoring recommendations (what's boosting your score and "
        f"what's keeping it from 100%) to {RESULT_FILE_PATH}"
    )

    used_potential_skills = find_used_potential_skills(
        load_potential_skills(), candidate
    )
    if used_potential_skills:
        print(
            "\nPolish up before the interview - these rusty skills made it into "
            "this tailored resume:"
        )
        for skill in used_potential_skills:
            print(f"  - {skill}")


def _check_missing_soft_skills(
    candidate: TailoredResume, missing_soft_skills: List[str]
) -> List[str]:
    # Rule 8: soft skills must be demonstrated in EXPERIENCE/PROJECTS bullets;
    # the summary and skills lines do not count.
    bullet_corpus = normalize_text(
        "\n".join(
            bullet
            for entry in candidate.experience + candidate.projects
            for bullet in entry.bullets
        )
    )
    corpus = bullet_corpus
    problems = []
    for skill in missing_soft_skills:
        if normalize_text(skill) not in corpus:
            problems.append(
                f'missing soft skill "{skill}" must appear verbatim in a bullet '
                "(EXPERIENCE or PROJECTS), demonstrated in action"
            )
    return problems


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate and regrade agent-written tailored resume candidates. "
            "Run 'python score.py' first."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--show-context",
        action="store_true",
        help="Print the tailoring context: skill gaps, current resume "
        "content, and the skills bank.",
    )
    mode.add_argument(
        "--candidate",
        metavar="PATH",
        help="Path to a candidate JSON file to validate and regrade.",
    )
    args = parser.parse_args()

    if args.show_context:
        show_context()
    else:
        evaluate_candidate(args.candidate)


if __name__ == "__main__":
    main()
