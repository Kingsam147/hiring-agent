import threading
import time
import os
import re
import json
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Tuple, Any, Literal, Protocol, runtime_checkable
from pydantic import BaseModel, Field, field_validator
from enum import Enum

logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Enum for supported model providers."""

    OLLAMA = "ollama"
    GEMINI = "gemini"


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Dict[str, Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Send a chat request to the LLM provider."""
        ...


class Location(BaseModel):
    """Location information for JSON Resume format."""

    address: Optional[str] = None
    postalCode: Optional[str] = None
    city: Optional[str] = None
    countryCode: Optional[str] = None
    region: Optional[str] = None


class Profile(BaseModel):
    """Social profile information for JSON Resume format."""

    network: Optional[str] = None
    username: Optional[str] = None
    url: str


class Basics(BaseModel):
    """Basic information for JSON Resume format."""

    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    url: Optional[str] = None
    summary: Optional[str] = None
    location: Optional[Location] = None
    profiles: Optional[List[Profile]] = None


class Work(BaseModel):
    """Work experience for JSON Resume format."""

    name: Optional[str] = None
    position: Optional[str] = None
    url: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    summary: Optional[str] = None
    highlights: Optional[List[str]] = None


class Volunteer(BaseModel):
    """Volunteer experience for JSON Resume format."""

    organization: Optional[str] = None
    position: Optional[str] = None
    url: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    summary: Optional[str] = None
    highlights: Optional[List[str]] = None


class Education(BaseModel):
    """Education information for JSON Resume format."""

    institution: Optional[str] = None
    url: Optional[str] = None
    area: Optional[str] = None
    studyType: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    score: Optional[str] = None
    courses: Optional[List[str]] = None


class Award(BaseModel):
    """Award information for JSON Resume format."""

    title: Optional[str] = None
    date: Optional[str] = None
    awarder: Optional[str] = None
    summary: Optional[str] = None


class Certificate(BaseModel):
    """Certificate information for JSON Resume format."""

    name: Optional[str] = None
    date: Optional[str] = None
    issuer: Optional[str] = None
    url: Optional[str] = None


class Publication(BaseModel):
    """Publication information for JSON Resume format."""

    name: Optional[str] = None
    publisher: Optional[str] = None
    releaseDate: Optional[str] = None
    url: Optional[str] = None
    summary: Optional[str] = None


class Skill(BaseModel):
    """Skill information for JSON Resume format."""

    name: Optional[str] = None
    level: Optional[str] = None
    keywords: Optional[List[str]] = None


class Language(BaseModel):
    """Language information for JSON Resume format."""

    language: Optional[str] = None
    fluency: Optional[str] = None


class Interest(BaseModel):
    """Interest information for JSON Resume format."""

    name: Optional[str] = None
    keywords: Optional[List[str]] = None


class Reference(BaseModel):
    """Reference information for JSON Resume format."""

    name: Optional[str] = None
    reference: Optional[str] = None


class Project(BaseModel):
    """Project information for JSON Resume format."""

    name: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    description: Optional[str] = None
    highlights: Optional[List[str]] = None
    url: Optional[str] = None
    technologies: Optional[List[str]] = None
    skills: Optional[List[str]] = None


class BasicsSection(BaseModel):
    """Basics section containing basic information."""

    basics: Optional[Basics] = None


class WorkSection(BaseModel):
    """Work section containing a list of work experiences."""

    work: Optional[List[Work]] = None


class EducationSection(BaseModel):
    """Education section containing a list of education entries."""

    education: Optional[List[Education]] = None


class SkillsSection(BaseModel):
    """Skills section containing a list of skill categories."""

    skills: Optional[List[Skill]] = None


class ProjectsSection(BaseModel):
    """Projects section containing a list of projects."""

    projects: Optional[List[Project]] = None


class AwardsSection(BaseModel):
    """Awards section containing a list of awards."""

    awards: Optional[List[Award]] = None


class JSONResume(BaseModel):
    """Complete JSON Resume format model."""

    basics: Optional[Basics] = None
    work: Optional[List[Work]] = None
    volunteer: Optional[List[Volunteer]] = None
    education: Optional[List[Education]] = None
    awards: Optional[List[Award]] = None
    certificates: Optional[List[Certificate]] = None
    publications: Optional[List[Publication]] = None
    skills: Optional[List[Skill]] = None
    languages: Optional[List[Language]] = None
    interests: Optional[List[Interest]] = None
    references: Optional[List[Reference]] = None
    projects: Optional[List[Project]] = None


class CategoryScore(BaseModel):
    score: float = Field(ge=0, description="Score achieved in this category")
    max: int = Field(gt=0, description="Maximum possible score")
    evidence: str = Field(min_length=1, description="Evidence supporting the score")


class Scores(BaseModel):
    open_source: CategoryScore
    self_projects: CategoryScore
    production: CategoryScore
    technical_skills: CategoryScore


class BonusPoints(BaseModel):
    total: float = Field(ge=0, le=20, description="Total bonus points")
    breakdown: str = Field(description="Breakdown of bonus points")


class Deductions(BaseModel):
    total: float = Field(
        ge=0,
        description="Total deduction points (stored as positive, applied as negative)",
    )
    reasons: str = Field(description="Reasons for deductions")


class EvaluationData(BaseModel):
    scores: Scores
    bonus_points: BonusPoints
    deductions: Deductions
    key_strengths: List[str] = Field(min_items=1, max_items=5)
    areas_for_improvement: List[str] = Field(min_items=1, max_items=5)


class JobDescriptionData(BaseModel):
    job_title: str
    required_skills: List[str]
    preferred_skills: List[str] = []
    years_of_experience: Optional[float] = None
    education_requirements: Optional[str] = None
    must_have_qualifications: List[str] = []
    industry: Optional[str] = None


class JobCategoryScore(BaseModel):
    score: float = Field(ge=0, le=100, description="Score for this category out of 100")
    evidence: str = Field(min_length=1, description="Evidence from the resume supporting this score")


class JobScores(BaseModel):
    skills_match: JobCategoryScore
    experience_match: JobCategoryScore
    job_title_alignment: JobCategoryScore
    education: JobCategoryScore
    resume_quality: JobCategoryScore
    missing_critical_requirements: JobCategoryScore


class LLMJobScores(BaseModel):
    experience_match: JobCategoryScore
    job_title_alignment: JobCategoryScore
    education: JobCategoryScore
    resume_quality: JobCategoryScore
    missing_critical_requirements: JobCategoryScore


class LLMJobEvaluationResponse(BaseModel):
    scores: LLMJobScores
    key_strengths: List[str] = Field(min_items=1, max_items=5)
    areas_for_improvement: List[str] = Field(min_items=1, max_items=5)


class SeniorityAssessment(BaseModel):
    target_level: int
    target_label: str
    candidate_level: int
    candidate_label: str
    highest_level: int
    highest_label: str
    gap: int


class IndustryMatch(BaseModel):
    industry: str
    mention_count: int = 0
    matched_entries: List[str] = []


class MustHaveStatus(BaseModel):
    qualification: str
    status: Literal["found", "not_found", "unverifiable"]
    resolved: Optional[bool] = None


class SkillExperience(BaseModel):
    skill: str
    years: float = Field(ge=0)
    evidence: List[str] = []


class KeywordMatchResult(BaseModel):
    matched_required: List[str] = []
    missing_required: List[str] = []
    matched_preferred: List[str] = []
    missing_preferred: List[str] = []
    must_have_status: List[MustHaveStatus] = []
    coverage_score: float = Field(ge=0, le=100)
    gated: bool = False
    knockout_failed: bool = False
    skill_experience: Optional[List[SkillExperience]] = None
    estimated_total_years: Optional[float] = None


class ScoreSummary(BaseModel):
    summary: str


class JobEvaluationData(BaseModel):
    scores: JobScores
    semantic_match_score: float = Field(ge=0, le=100)
    weighted_total: float = Field(ge=0, le=100)
    key_strengths: List[str]
    areas_for_improvement: List[str]
    job_title: str
    keyword_match: Optional[KeywordMatchResult] = None
    seniority: Optional[SeniorityAssessment] = None
    jd_years_of_experience: Optional[float] = None
    weight_profile: str = "engineering"
    industry_match: Optional[IndustryMatch] = None
    score_summary: Optional[str] = None


class GitHubProfile(BaseModel):
    """Pydantic model for GitHub profile data."""

    username: str
    name: Optional[str] = None
    bio: Optional[str] = None
    location: Optional[str] = None
    company: Optional[str] = None
    public_repos: Optional[int] = None
    followers: Optional[int] = None
    following: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    avatar_url: Optional[str] = None
    blog: Optional[str] = None
    twitter_username: Optional[str] = None
    hireable: Optional[bool] = None


class OllamaProvider:
    """Ollama LLM provider implementation."""

    def __init__(self):
        import ollama

        self.client = ollama

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Dict[str, Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Send a chat request to Ollama."""

        ollama_options = options.copy() if options else {}

        # remove steam from ollama options
        ollama_options.pop("stream", None)

        # Add num_ctx 32K context window to options
        ollama_options["num_ctx"] = 32768

        # convert to chat params
        chat_params = {
            "model": model,
            "messages": messages,
            "options": ollama_options,
        }

        # add it to top level
        if "stream" in kwargs:
            chat_params["stream"] = kwargs["stream"]

        if "format" in kwargs:
            chat_params["format"] = kwargs["format"]

        return self.client.chat(**chat_params)


class GeminiDailyQuotaExceeded(Exception):
    """Raised when the local Gemini daily request budget is exhausted.

    Deliberately NOT a subclass of google.api_core.exceptions.ResourceExhausted,
    so GeminiProvider.chat()'s `except ResourceExhausted` retry loop can never
    catch it -- it propagates immediately with no retries and no pacing wait.
    """


_DAILY_QUOTA_ID_PATTERN = re.compile(r"GenerateRequestsPerDay|quota_id[^\n]*PerDay", re.IGNORECASE)


def _is_daily_quota_error(exc: Exception) -> bool:
    """Best-effort detection of a requests-per-DAY quota violation from a Gemini 429.

    Matches only the quota_id (e.g. GenerateRequestsPerDayPerProjectPerModel-FreeTier),
    never the shared metric name (generate_content_free_tier_requests), which also
    appears on recoverable per-minute violations -- matching the metric name would
    misclassify a transient per-minute 429 as fatal daily exhaustion. If Google's
    error format changes and this stops matching, calls simply fall back to the
    existing retry/backoff path (today's behavior) -- never worse.
    """
    return bool(_DAILY_QUOTA_ID_PATTERN.search(str(exc)))


class GeminiDailyQuotaLedger:
    """Persistent per-model, per-UTC-day counter of real Gemini API attempts.

    File-based (cache/gemini_quota_<model>_<YYYY-MM-DD>.json) so the count
    survives across separate `python score.py` invocations, unlike
    GeminiRateLimiter's in-memory pacing. Always active regardless of
    DEVELOPMENT_MODE, since quota consumption is real in every mode.

    Note: the UTC day boundary is an approximation -- Google's free-tier
    reset is commonly reported as midnight Pacific time, not UTC. This is
    self-correcting: if the local ledger resets early/late relative to the
    real quota, the next real 429 (if any) re-locks the ledger via
    mark_exhausted(), at the cost of at most one extra call.
    """

    def __init__(self, requests_per_day: int, cache_dir: str = "cache"):
        self.requests_per_day = requests_per_day
        self.cache_dir = cache_dir
        self._lock = threading.Lock()

    def _today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _path(self, model: str) -> str:
        safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", model)
        return os.path.join(self.cache_dir, f"gemini_quota_{safe_model}_{self._today()}.json")

    def _read_count(self, model: str) -> int:
        path = self._path(model)
        if not os.path.exists(path):
            return 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return int(data.get("count", 0))
        except Exception as e:
            logger.warning(f"Invalid Gemini quota ledger file {path}: {e}. Treating today's count as 0.")
            return 0

    def _write_count(self, model: str, count: int) -> None:
        path = self._path(model)
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            tmp_path = path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump({"date": self._today(), "model": model, "count": count}, f, indent=2)
            os.replace(tmp_path, path)
        except Exception as e:
            logger.warning(f"Failed to write Gemini quota ledger file {path}: {e}")

    def usage_line(self, model: str) -> str:
        count = self._read_count(model)
        if self.requests_per_day <= 0:
            return f"{count} Gemini requests used today for {model} (no daily limit configured)"
        return f"{count}/{self.requests_per_day} Gemini requests used today for {model}"

    def preflight(self, model: str) -> None:
        if self.requests_per_day <= 0:
            return
        with self._lock:
            count = self._read_count(model)
            if count >= self.requests_per_day:
                raise GeminiDailyQuotaExceeded(
                    f"Gemini daily request budget exhausted for {model}: "
                    f"{count}/{self.requests_per_day} used today (UTC date {self._today()}). "
                    "Not calling the API -- retrying cannot help until the daily quota resets "
                    "(~midnight UTC by this local ledger; Google's actual reset may be midnight "
                    "Pacific). Options: wait for the reset, set LLM_PROVIDER=ollama in .env for "
                    "further testing today, or adjust GEMINI_REQUESTS_PER_DAY if your real quota differs."
                )

    def record_request(self, model: str) -> int:
        with self._lock:
            count = self._read_count(model) + 1
            self._write_count(model, count)
        print(f"[GeminiProvider] Daily quota: {self.usage_line(model)}")
        return count

    def mark_exhausted(self, model: str) -> None:
        with self._lock:
            count = max(self._read_count(model), self.requests_per_day if self.requests_per_day > 0 else 0)
            self._write_count(model, count)


def _get_gemini_daily_ledger() -> GeminiDailyQuotaLedger:
    global _GEMINI_DAILY_LEDGER
    if _GEMINI_DAILY_LEDGER is None:
        from config import GEMINI_REQUESTS_PER_DAY

        _GEMINI_DAILY_LEDGER = GeminiDailyQuotaLedger(GEMINI_REQUESTS_PER_DAY)
    return _GEMINI_DAILY_LEDGER


_GEMINI_DAILY_LEDGER: Optional[GeminiDailyQuotaLedger] = None


def get_gemini_daily_usage_line(model: str) -> str:
    """Public accessor for the daily quota usage banner (used by score.py)."""
    return _get_gemini_daily_ledger().usage_line(model)


class GeminiRateLimiter:
    """Proactive client-side pacing for Gemini API calls.

    Sleeps before a request if it would arrive sooner than min_interval
    after the previous request, spreading calls out so the free-tier
    per-minute quota is rarely hit in the first place. This is separate
    from (and complementary to) the reactive 429 backoff in GeminiProvider.
    Also enforces the daily request budget via GeminiDailyQuotaLedger.
    """

    def __init__(self, requests_per_minute: float):
        self.min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._last_request_time: Optional[float] = None
        self._lock = threading.Lock()

    def acquire(self, model: str) -> None:
        _get_gemini_daily_ledger().preflight(model)

        with self._lock:
            if self.min_interval <= 0 or self._last_request_time is None:
                self._last_request_time = time.monotonic()
            else:
                wait = self._last_request_time + self.min_interval - time.monotonic()
                if wait > 0:
                    print(
                        f"[GeminiProvider] Pacing: waiting {wait:.1f}s to stay under "
                        f"{60.0 / self.min_interval:.1f} req/min"
                    )
                    time.sleep(wait)

                self._last_request_time = time.monotonic()

        _get_gemini_daily_ledger().record_request(model)


def _get_gemini_rate_limiter() -> GeminiRateLimiter:
    global _GEMINI_RATE_LIMITER
    if _GEMINI_RATE_LIMITER is None:
        from config import GEMINI_REQUESTS_PER_MINUTE

        _GEMINI_RATE_LIMITER = GeminiRateLimiter(GEMINI_REQUESTS_PER_MINUTE)
    return _GEMINI_RATE_LIMITER


_GEMINI_RATE_LIMITER: Optional[GeminiRateLimiter] = None


class GeminiProvider:
    """Google Gemini API provider implementation."""

    def __init__(self, api_key: str):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.client = genai

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Dict[str, Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Send a chat request to Google Gemini API."""
        import re
        import time
        import random
        from google.api_core.exceptions import ResourceExhausted

        MAX_RETRIES = 5
        BASE_DELAY = 10.0  # seconds — base for exponential backoff
        MAX_DELAY = 120.0  # cap so we never wait more than 2 minutes

        # Map options to Gemini parameters
        generation_config = {}
        if options:
            if "temperature" in options:
                generation_config["temperature"] = options["temperature"]
            if "top_p" in options:
                generation_config["top_p"] = options["top_p"]

        # Create a Gemini model
        gemini_model = self.client.GenerativeModel(
            model_name=model, generation_config=generation_config
        )

        # Convert messages to Gemini format
        gemini_messages = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            gemini_messages.append({"role": role, "parts": [msg["content"]]})

        for attempt in range(MAX_RETRIES):
            try:
                _get_gemini_rate_limiter().acquire(model)

                # Send the chat request
                response = gemini_model.generate_content(gemini_messages)

                # Convert Gemini response to Ollama-like format for compatibility
                return {"message": {"role": "assistant", "content": response.text}}

            except ResourceExhausted as e:
                if _is_daily_quota_error(e):
                    # Daily quota exhaustion cannot recover mid-retry -- lock the
                    # local ledger to the configured budget and fail immediately
                    # instead of burning the remaining retry attempts on a wall
                    # that will not move until the daily reset.
                    _get_gemini_daily_ledger().mark_exhausted(model)
                    raise GeminiDailyQuotaExceeded(
                        f"Gemini daily request quota for '{model}' is exhausted "
                        f"(confirmed by API). Retrying will not help -- the daily "
                        f"quota does not refill until the daily reset (~midnight "
                        f"UTC/Pacific). Local ledger set to the configured budget so "
                        f"subsequent calls today fail fast. Switch LLM_PROVIDER=ollama "
                        f"in .env to keep testing today."
                    ) from e

                if attempt == MAX_RETRIES - 1:
                    # All retries exhausted — re-raise the original exception.
                    # This surfaces unrecoverable quota errors (RPD, TPM, etc.)
                    # instead of silently failing or returning bad data.
                    raise

                # Parse the API-suggested retry delay from the error message
                match = re.search(r"retry[_ ]in\s+([\d.]+)s", str(e), re.IGNORECASE)
                api_hint = float(match.group(1)) if match else None

                # Exponential backoff: BASE_DELAY * 2^attempt, capped at MAX_DELAY
                exp_delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)

                # Prefer the API hint when it is shorter than our computed delay
                delay = api_hint if (api_hint and api_hint < exp_delay) else exp_delay

                # Add ±20% randomized jitter to avoid thundering herd
                sleep_time = round(delay * random.uniform(0.8, 1.2), 2)

                print(
                    f"[GeminiProvider] Rate limit hit "
                    f"(attempt {attempt + 1}/{MAX_RETRIES}). "
                    f"Retrying in {sleep_time}s..."
                )
                time.sleep(sleep_time)
