import os
import sys
import json
import logging
import csv
from pdf import PDFHandler
from github import fetch_and_display_github_info
from models import (
    JSONResume,
    JobEvaluationData,
    ModelProvider,
    get_gemini_daily_spend_line,
)
from evaluator import JobDescriptionEvaluator
from pathlib import Path
from prompt import DEFAULT_MODEL, MODEL_PARAMETERS, MODEL_PROVIDER_MAPPING
from weight_profiles import WEIGHT_PROFILES, DEFAULT_PROFILE, suggest_profile
from transform import (
    transform_job_evaluation_response,
    convert_json_resume_to_text,
    convert_github_data_to_text,
)
from config import DEVELOPMENT_MODE

RESULT_FILE_PATH = "result.md"

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)5s - %(lineno)5d - %(funcName)33s - %(levelname)5s - %(message)s",
)

RESUME_FOLDER = "resume"
JOB_DESCRIPTION_PATH = "job_description.txt"


def find_resume_file(folder: str = RESUME_FOLDER) -> str:
    if not os.path.isdir(folder):
        print(
            f"Error: '{folder}/' folder not found. Create it and place your resume file inside."
        )
        sys.exit(1)

    files = [
        entry
        for entry in sorted(os.listdir(folder))
        if os.path.isfile(os.path.join(folder, entry)) and not entry.startswith(".")
    ]

    if not files:
        print(
            f"Error: no resume file found in '{folder}/'. Place exactly one resume file there."
        )
        sys.exit(1)

    if len(files) > 1:
        print(
            f"Error: multiple files found in '{folder}/': {', '.join(files)}. "
            "Please leave only one resume file in that folder."
        )
        sys.exit(1)

    return os.path.join(folder, files[0])


def load_job_description() -> str:
    if not os.path.exists(JOB_DESCRIPTION_PATH):
        print(f"Error: '{JOB_DESCRIPTION_PATH}' not found in the project root.")
        sys.exit(1)
    content = Path(JOB_DESCRIPTION_PATH).read_text(encoding="utf-8").strip()
    if not content:
        print(
            f"Error: '{JOB_DESCRIPTION_PATH}' is empty. "
            "Paste a job description into it before running in Custom Job Description mode."
        )
        sys.exit(1)
    return content


def build_job_evaluation_markdown(
    evaluation: JobEvaluationData, candidate_name: str = "Candidate"
) -> str:
    lines = [f"# Job Match Evaluation: {candidate_name}"]
    lines.append(f"**Target Role:** {evaluation.job_title}")

    lines.append(f"\n**Overall Match:** {evaluation.weighted_total}/100")
    lines.append(f"**Weight profile:** {evaluation.weight_profile}")

    if evaluation.score_summary:
        lines.append("\n## Why This Score")
        lines.append(evaluation.score_summary)

    weights = WEIGHT_PROFILES.get(
        evaluation.weight_profile, WEIGHT_PROFILES[DEFAULT_PROFILE]
    )

    lines.append("\n## Category Breakdown")

    categories = [
        (
            f"Skills Match ({weights['skills_match']:.0%})",
            evaluation.scores.skills_match,
        ),
        (
            f"Experience Match ({weights['experience_match']:.0%})",
            evaluation.scores.experience_match,
        ),
        (
            f"Title Alignment ({weights['job_title_alignment']:.0%})",
            evaluation.scores.job_title_alignment,
        ),
        (f"Education ({weights['education']:.0%})", evaluation.scores.education),
        (
            f"Resume Quality ({weights['resume_quality']:.0%})",
            evaluation.scores.resume_quality,
        ),
        (
            f"Missing Critical ({weights['missing_critical_requirements']:.0%})",
            evaluation.scores.missing_critical_requirements,
        ),
    ]

    for label, category in categories:
        lines.append(f"\n**{label}:** {category.score:.0f}/100")
        lines.append(f"- Evidence: {category.evidence}")
        if category is evaluation.scores.job_title_alignment and evaluation.seniority:
            seniority = evaluation.seniority
            lines.append(
                f"- Seniority: target={seniority.target_label}, candidate={seniority.candidate_label} "
                f"(gap {seniority.gap:+d})"
            )

    lines.append(
        f"\n**Semantic Match ({weights['semantic_match']:.0%}):** {evaluation.semantic_match_score:.1f}/100"
    )
    lines.append(
        "- Whole-document embedding similarity (all-MiniLM-L6-v2) — supplementary signal."
    )

    if evaluation.keyword_match:
        keyword_match = evaluation.keyword_match
        lines.append("\n## Keyword Match")
        coverage_line = f"Keyword coverage: {keyword_match.coverage_score:.1f}/100"
        if keyword_match.gated:
            coverage_line += " [CAPPED — a must-have qualification was not found]"
        lines.append(coverage_line)

        total_required = len(keyword_match.matched_required) + len(
            keyword_match.missing_required
        )
        lines.append(
            f"\n**Required skills matched ({len(keyword_match.matched_required)}/{total_required}):**"
        )
        lines.append(
            ", ".join(keyword_match.matched_required)
            if keyword_match.matched_required
            else "None"
        )
        lines.append("\n**Required skills MISSING:**")
        lines.append(
            ", ".join(keyword_match.missing_required)
            if keyword_match.missing_required
            else "None"
        )

        total_preferred = len(keyword_match.matched_preferred) + len(
            keyword_match.missing_preferred
        )
        if total_preferred:
            lines.append(
                f"\n**Preferred skills matched ({len(keyword_match.matched_preferred)}/{total_preferred}):**"
            )
            lines.append(
                ", ".join(keyword_match.matched_preferred)
                if keyword_match.matched_preferred
                else "None"
            )
            lines.append("\n**Preferred skills missing:**")
            lines.append(
                ", ".join(keyword_match.missing_preferred)
                if keyword_match.missing_preferred
                else "None"
            )

        if keyword_match.matched_soft_skills or keyword_match.missing_soft_skills:
            lines.append("\n**Soft skills present:**")
            lines.append(
                ", ".join(keyword_match.matched_soft_skills)
                if keyword_match.matched_soft_skills
                else "None"
            )
            lines.append("\n**Soft skills missing:**")
            lines.append(
                ", ".join(keyword_match.missing_soft_skills)
                if keyword_match.missing_soft_skills
                else "None"
            )

        if keyword_match.must_have_status:
            lines.append("\n**Must-have qualifications:**")
            status_labels = {
                "found": "found",
                "not_found": "NOT FOUND",
                "unverifiable": "could not be verified by keyword matching",
            }
            for status in keyword_match.must_have_status:
                lines.append(
                    f"- {status.qualification}: {status_labels[status.status]}"
                )

        if keyword_match.skill_experience:
            lines.append("\n**Skill tenure (deterministic, from work-history dates):**")
            for skill_exp in keyword_match.skill_experience:
                if skill_exp.years > 0:
                    lines.append(f"- {skill_exp.skill}: {skill_exp.years} yrs")
                else:
                    lines.append(f"- {skill_exp.skill}: no dated evidence")
            if (
                evaluation.jd_years_of_experience is not None
                and keyword_match.estimated_total_years is not None
            ):
                lines.append(
                    f"- JD asks for {evaluation.jd_years_of_experience} yrs; candidate total "
                    f"~{keyword_match.estimated_total_years} yrs (from parseable work dates)"
                )

        if evaluation.industry_match:
            industry_match = evaluation.industry_match
            if industry_match.mention_count:
                lines.append(
                    f"\n**Industry ({industry_match.industry}):** mentioned in {industry_match.mention_count} work entr"
                    + ("y" if industry_match.mention_count == 1 else "ies")
                )
            else:
                lines.append(
                    f"\n**Industry ({industry_match.industry}):** no literal mentions "
                    "(LLM judges domain fit within Experience Match)"
                )

        suggested_profile = suggest_profile(
            evaluation.job_title,
            evaluation.industry_match.industry if evaluation.industry_match else None,
        )
        if suggested_profile != evaluation.weight_profile:
            lines.append(
                f"\n> Note: this JD looks like a '{suggested_profile}' role; consider rerunning with that "
                "weight profile (no extra LLM cost)."
            )

    if evaluation.key_strengths:
        lines.append("\n## Key Strengths")
        for strength in evaluation.key_strengths:
            lines.append(f"- {strength}")

    if evaluation.areas_for_improvement:
        lines.append("\n## Areas for Improvement")
        for area in evaluation.areas_for_improvement:
            lines.append(f"- {area}")

    return "\n".join(lines)


def write_result_markdown(markdown: str) -> None:
    Path(RESULT_FILE_PATH).write_text(markdown, encoding="utf-8")
    print(f"Report written to {RESULT_FILE_PATH}")


def is_valid_resume_data(resume_data: JSONResume) -> bool:
    if not resume_data:
        return False
    core_sections = [
        resume_data.basics,
        resume_data.work,
        resume_data.education,
        resume_data.skills,
        resume_data.projects,
    ]
    return any(section is not None for section in core_sections)


def find_profile(profiles, network):
    if not profiles:
        return None
    return next(
        (p for p in profiles if p.network and p.network.lower() == network.lower()),
        None,
    )


def main():
    pdf_path = find_resume_file()

    if MODEL_PROVIDER_MAPPING.get(DEFAULT_MODEL) == ModelProvider.GEMINI:
        print(
            f"Gemini spend so far today: {get_gemini_daily_spend_line(DEFAULT_MODEL)}"
        )

    job_description = load_job_description()
    weight_profile = "engineering"

    resume_file_stem = os.path.splitext(os.path.basename(pdf_path))[0]
    cache_filename = f"cache/resumecache_{resume_file_stem}.json"
    github_cache_filename = f"cache/githubcache_{resume_file_stem}.json"

    resume_data = None
    cache_loaded = False

    if (
        DEVELOPMENT_MODE
        and os.path.exists(cache_filename)
        and os.path.getmtime(cache_filename) >= os.path.getmtime(pdf_path)
    ):
        print(f"Loading cached data from {cache_filename}")
        try:
            cached_data = json.loads(Path(cache_filename).read_text(encoding="utf-8"))
            loaded_resume = JSONResume(**cached_data)
            if not is_valid_resume_data(loaded_resume):
                raise ValueError("Cached resume data contains no core content")
            resume_data = loaded_resume
            cache_loaded = True
        except Exception as e:
            print(f"⚠️ Warning: Invalid cache file {cache_filename}: {e}")
            print("Ignoring cache and reprocessing PDF...")
            try:
                os.remove(cache_filename)
            except Exception as delete_err:
                print(
                    f"Failed to delete invalid cache file {cache_filename}: {delete_err}"
                )

    if not cache_loaded:
        logger.debug(
            f"Extracting data from PDF"
            + (" and caching to " + cache_filename if DEVELOPMENT_MODE else "")
        )
        pdf_handler = PDFHandler()
        resume_data = pdf_handler.extract_json_from_pdf(pdf_path)

        if resume_data is None:
            return None

        if DEVELOPMENT_MODE:
            if is_valid_resume_data(resume_data):
                os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
                Path(cache_filename).write_text(
                    json.dumps(resume_data.model_dump(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            else:
                logger.warning(
                    "Newly extracted resume data is empty/invalid. Skipping cache write."
                )

    candidate_name = os.path.splitext(os.path.basename(pdf_path))[0]
    if (
        resume_data
        and hasattr(resume_data, "basics")
        and resume_data.basics
        and resume_data.basics.name
    ):
        candidate_name = resume_data.basics.name

    model_params = MODEL_PARAMETERS.get(DEFAULT_MODEL)
    job_evaluator = JobDescriptionEvaluator(
        job_description=job_description,
        model_name=DEFAULT_MODEL,
        model_params=model_params,
        weight_profile=weight_profile,
    )

    github_data = {}
    github_cache_loaded = False
    if DEVELOPMENT_MODE and os.path.exists(github_cache_filename):
        print(f"Loading cached data from {github_cache_filename}")
        try:
            loaded_github = json.loads(
                Path(github_cache_filename).read_text(encoding="utf-8")
            )
            if (
                not isinstance(loaded_github, dict)
                or not loaded_github
                or "profile" not in loaded_github
            ):
                raise ValueError("Cached GitHub data is invalid or empty")
            github_data = loaded_github
            github_cache_loaded = True
        except Exception as e:
            print(f"⚠️ Warning: Invalid GitHub cache file {github_cache_filename}: {e}")
            print("Ignoring GitHub cache and refetching...")
            try:
                os.remove(github_cache_filename)
            except Exception as delete_err:
                print(
                    f"Failed to delete invalid GitHub cache file {github_cache_filename}: {delete_err}"
                )

    if not github_cache_loaded:
        profiles = []
        if resume_data and hasattr(resume_data, "basics") and resume_data.basics:
            profiles = resume_data.basics.profiles or []
        github_profile = find_profile(profiles, "Github")

        if github_profile:
            print(
                f"Fetching GitHub data"
                + (
                    " and caching to " + github_cache_filename
                    if DEVELOPMENT_MODE
                    else ""
                )
            )
            github_data = fetch_and_display_github_info(github_profile.url)

            if (
                DEVELOPMENT_MODE
                and github_data
                and isinstance(github_data, dict)
                and "profile" in github_data
            ):
                os.makedirs(os.path.dirname(github_cache_filename), exist_ok=True)
                Path(github_cache_filename).write_text(
                    json.dumps(github_data, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

    resume_text = convert_json_resume_to_text(resume_data)
    if github_data:
        resume_text += convert_github_data_to_text(github_data)

    job_evaluation = job_evaluator.evaluate(resume_text, resume_data=resume_data)
    write_result_markdown(build_job_evaluation_markdown(job_evaluation, candidate_name))

    if DEVELOPMENT_MODE:
        csv_row = transform_job_evaluation_response(
            file_name=os.path.basename(pdf_path),
            evaluation=job_evaluation,
            resume_data=resume_data,
        )
        csv_path = "job_evaluations.csv"
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
            fieldnames = list(csv_row.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(csv_row)

    return job_evaluation


if __name__ == "__main__":
    main()
