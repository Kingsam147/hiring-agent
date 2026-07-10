"""
Deterministic keyword matching between a job description's extracted
requirements and a candidate's resume. No LLM calls, no external
dependencies beyond the standard library.
"""

import re
import unicodedata
from typing import List, Optional

from models import (
    JSONResume,
    JobDescriptionData,
    KeywordMatchResult,
    MustHaveStatus,
    IndustryMatch,
)

MUST_HAVE_VERIFIABLE_MAX_WORDS = 5
GATE_CAP = 60.0
REQUIRED_WEIGHT = 0.8
PREFERRED_WEIGHT = 0.2

_INDUSTRY_SPLIT = re.compile(r"[,/&]|\band\b", re.IGNORECASE)

_PUNCTUATION_TO_STRIP = re.compile(r"[,;:()\[\]{}'\"!?]")
_SEPARATORS_TO_SPACE = re.compile(r"[-_/]")
_WHITESPACE_RUN = re.compile(r"\s+")

_PARENTHETICAL_ALTERNATIVES = re.compile(r"^(.*?)\(([^)]+)\)\s*$")
_BARE_OR_SPLIT = re.compile(r"\s+or\s+", re.IGNORECASE)

# Bounded, hand-maintained aliases for abbreviations we've actually seen a
# grading model fail to connect to their spelled-out requirement phrase.
# Kept short and specific per skill-bank category to avoid over-matching --
# each alias should be an unambiguous short form of its key, not a loose
# synonym that could be true of a different, unrelated skill.
_REQUIREMENT_ALIASES = {
    "object oriented programming": ["oop"],
    "data structures and algorithms": ["data structures & algorithms", "dsa", "ds&a"],
    ".net platform": [
        "asp.net core",
        "asp.net",
        ".net core",
        ".net framework",
        "dotnet",
        "c#",
    ],
    # Languages
    "javascript": ["js"],
    "typescript": ["ts"],
    # Backend
    "restful api development": ["rest api", "rest apis", "restful api", "rest"],
    "asynchronous programming": ["async", "async/await"],
    "client server application architecture": ["client-server", "client server"],
    "node.js": ["node", "nodejs"],
    # Cloud & DevOps
    "amazon web services": ["aws"],
    "google cloud platform": ["gcp"],
    "continuous integration continuous deployment": ["ci/cd", "cicd"],
    "cloud based applications": ["cloud-based", "cloud native"],
    # Architecture & Security
    "domain driven design": ["ddd"],
    "role based access control": ["rbac"],
    "row level security": ["rls"],
    # Frontend
    "single page application": ["spa"],
    # Database & Querying
    "structured query language": ["sql"],
    "relational database management system": ["rdbms"],
    # Development Practice
    "test driven development": ["tdd"],
    "full stack web development": ["full-stack", "full stack", "fullstack"],
}

_RELATIONAL_DB_ENGINES = [
    "postgresql",
    "postgres",
    "mysql",
    "sql server",
    "oracle",
    "sqlite",
    "mariadb",
]
_NOSQL_DB_ENGINES = [
    "mongodb",
    "dynamodb",
    "cassandra",
    "couchdb",
    "firestore",
    "firebase",
]

# Bounded synonym table so a named database engine satisfies a generic
# "relational"/"NoSQL database" alternative, without matching arbitrary text.
_CATEGORY_ENGINE_SYNONYMS = {
    "relational database": _RELATIONAL_DB_ENGINES,
    "relational databases": _RELATIONAL_DB_ENGINES,
    "nosql database": _NOSQL_DB_ENGINES,
    "nosql databases": _NOSQL_DB_ENGINES,
}


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = _SEPARATORS_TO_SPACE.sub(" ", normalized)
    normalized = _PUNCTUATION_TO_STRIP.sub(" ", normalized)
    normalized = _WHITESPACE_RUN.sub(" ", normalized).strip()
    return normalized


def keyword_found(keyword: str, normalized_corpus: str) -> bool:
    normalized_keyword = normalize_text(keyword)
    if not normalized_keyword:
        return False
    pattern = rf"(?<![a-z0-9+#]){re.escape(normalized_keyword)}(?![a-z0-9+#])"
    return re.search(pattern, normalized_corpus) is not None


def build_match_corpus(
    resume_text: str, resume_data: Optional[JSONResume] = None
) -> str:
    parts = [resume_text]

    if resume_data is not None:
        if resume_data.skills:
            for skill in resume_data.skills:
                if skill.name:
                    parts.append(skill.name)
                if skill.keywords:
                    parts.extend(skill.keywords)

        if resume_data.projects:
            for project in resume_data.projects:
                if project.technologies:
                    parts.extend(project.technologies)
                if project.skills:
                    parts.extend(project.skills)

        if resume_data.work:
            for work_item in resume_data.work:
                if work_item.position:
                    parts.append(work_item.position)

        if resume_data.certificates:
            for certificate in resume_data.certificates:
                if certificate.name:
                    parts.append(certificate.name)

    return normalize_text("\n".join(parts))


def _requirement_alternatives(requirement: str) -> List[str]:
    """Split a bundled requirement into individually-satisfying alternatives.

    "SPA frameworks (Angular, React, or Vue)" -> ["SPA frameworks", "Angular",
    "React", "Vue"]. "Relational or NoSQL databases" -> ["Relational
    databases", "NoSQL databases"] (the trailing noun is shared across the
    "or"). A requirement with no alternatives returns [requirement] as-is.
    """
    parens_match = _PARENTHETICAL_ALTERNATIVES.match(requirement)
    if parens_match:
        alternatives = []
        outer = parens_match.group(1).strip()
        if outer:
            alternatives.append(outer)
        for part in re.split(r",|\bor\b", parens_match.group(2), flags=re.IGNORECASE):
            part = part.strip()
            if part:
                alternatives.append(part)
        return alternatives

    or_parts = [part.strip() for part in _BARE_OR_SPLIT.split(requirement)]
    if len(or_parts) > 1:
        last_words = or_parts[-1].split()
        shared_suffix = " ".join(last_words[1:]) if len(last_words) > 1 else ""
        alternatives = []
        for part in or_parts:
            if part == or_parts[-1] or not shared_suffix or " " in part:
                alternatives.append(part)
            else:
                alternatives.append(f"{part} {shared_suffix}")
        return alternatives

    return [requirement]


def requirement_satisfied(requirement: str, normalized_corpus: str) -> bool:
    """Whether a required/preferred skill is satisfied, honoring OR-bundled
    alternatives (parenthetical or bare "A or B" phrasing), common
    abbreviations, and named database engines standing in for their category.
    """
    for alternative in _requirement_alternatives(requirement):
        if keyword_found(alternative, normalized_corpus):
            return True
        normalized_alternative = normalize_text(alternative)
        for alias in _REQUIREMENT_ALIASES.get(normalized_alternative, []):
            if keyword_found(alias, normalized_corpus):
                return True
        for engine in _CATEGORY_ENGINE_SYNONYMS.get(normalized_alternative, []):
            if keyword_found(engine, normalized_corpus):
                return True
    return False


def _dedupe_keywords(keywords: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for keyword in keywords:
        normalized = normalize_text(keyword)
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(keyword)
    return deduped


def _must_have_status(qualification: str, normalized_corpus: str) -> MustHaveStatus:
    word_count = len(normalize_text(qualification).split())
    if word_count > MUST_HAVE_VERIFIABLE_MAX_WORDS:
        return MustHaveStatus(qualification=qualification, status="unverifiable")
    status = "found" if keyword_found(qualification, normalized_corpus) else "not_found"
    return MustHaveStatus(qualification=qualification, status=status)


def _weighted_coverage(
    matched_required_count: int,
    required_total: int,
    matched_preferred_count: int,
    preferred_total: int,
) -> float:
    if required_total and preferred_total:
        return 100 * (
            REQUIRED_WEIGHT * (matched_required_count / required_total)
            + PREFERRED_WEIGHT * (matched_preferred_count / preferred_total)
        )
    elif required_total:
        return 100 * (matched_required_count / required_total)
    elif preferred_total:
        return 100 * (matched_preferred_count / preferred_total)
    else:
        return 50.0


def compute_keyword_match(
    job_data: JobDescriptionData,
    resume_text: str,
    resume_data: Optional[JSONResume] = None,
) -> KeywordMatchResult:
    # Deferred import: skill_experience.py imports keyword_found/normalize_text
    # from this module, so importing it at module scope would be circular.
    from skill_experience import (
        compute_skill_experience,
        compute_total_experience_years,
    )

    normalized_corpus = build_match_corpus(resume_text, resume_data)

    required_skills = _dedupe_keywords(job_data.required_skills or [])
    preferred_skills = _dedupe_keywords(job_data.preferred_skills or [])
    soft_skills = _dedupe_keywords(job_data.soft_skills or [])

    matched_required = [
        skill
        for skill in required_skills
        if requirement_satisfied(skill, normalized_corpus)
    ]
    missing_required = [
        skill for skill in required_skills if skill not in matched_required
    ]
    matched_preferred = [
        skill
        for skill in preferred_skills
        if requirement_satisfied(skill, normalized_corpus)
    ]
    missing_preferred = [
        skill for skill in preferred_skills if skill not in matched_preferred
    ]
    matched_soft_skills = [
        skill
        for skill in soft_skills
        if requirement_satisfied(skill, normalized_corpus)
    ]
    missing_soft_skills = [
        skill for skill in soft_skills if skill not in matched_soft_skills
    ]

    must_have_status = [
        _must_have_status(qualification, normalized_corpus)
        for qualification in (job_data.must_have_qualifications or [])
    ]

    coverage = _weighted_coverage(
        len(matched_required),
        len(required_skills),
        len(matched_preferred),
        len(preferred_skills),
    )

    gated = any(status.status == "not_found" for status in must_have_status)
    if gated:
        coverage = min(coverage, GATE_CAP)

    work = resume_data.work if resume_data is not None else None

    return KeywordMatchResult(
        matched_required=matched_required,
        missing_required=missing_required,
        matched_preferred=matched_preferred,
        missing_preferred=missing_preferred,
        matched_soft_skills=matched_soft_skills,
        missing_soft_skills=missing_soft_skills,
        must_have_status=must_have_status,
        coverage_score=round(coverage, 1),
        gated=gated,
        skill_experience=compute_skill_experience(required_skills, work),
        estimated_total_years=compute_total_experience_years(work),
    )


def build_skills_evidence(result: KeywordMatchResult) -> str:
    total_required = len(result.matched_required) + len(result.missing_required)
    total_preferred = len(result.matched_preferred) + len(result.missing_preferred)

    parts = [
        f"Matched {len(result.matched_required)}/{total_required} required skills"
        + (
            f" ({', '.join(result.matched_required)})"
            if result.matched_required
            else ""
        )
        + "."
    ]
    if result.missing_required:
        parts.append(f"Missing required: {', '.join(result.missing_required)}.")
    if total_preferred:
        parts.append(
            f"Matched {len(result.matched_preferred)}/{total_preferred} preferred skills"
            + (
                f" ({', '.join(result.matched_preferred)})"
                if result.matched_preferred
                else ""
            )
            + "."
        )
    if result.gated:
        parts.append(
            f"Score capped at {GATE_CAP:.0f} because a must-have qualification was not found in the resume."
        )

    return " ".join(parts)


def _work_entry_industry_corpus(work_item) -> str:
    parts = [work_item.name, work_item.summary]
    if work_item.highlights:
        parts.extend(work_item.highlights)
    return normalize_text("\n".join(part for part in parts if part))


def compute_industry_mentions(
    industry: Optional[str], resume_data: Optional[JSONResume]
) -> Optional[IndustryMatch]:
    if not industry or not industry.strip():
        return None

    tokens = [
        token.strip() for token in _INDUSTRY_SPLIT.split(industry) if token.strip()
    ]
    if not tokens:
        tokens = [industry.strip()]

    if resume_data is None or not resume_data.work:
        return IndustryMatch(industry=industry, mention_count=0, matched_entries=[])

    matched_entries = []
    for work_item in resume_data.work:
        corpus = _work_entry_industry_corpus(work_item)
        if any(keyword_found(token, corpus) for token in tokens):
            label = f"{work_item.name or 'Unknown company'} — {work_item.position or 'Unknown role'}"
            matched_entries.append(label)

    return IndustryMatch(
        industry=industry,
        mention_count=len(matched_entries),
        matched_entries=matched_entries,
    )
