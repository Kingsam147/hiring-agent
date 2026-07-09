import threading
import time
import os
import re
import json
import logging
from datetime import datetime, timezone
from typing import (
    List,
    Optional,
    Dict,
    Tuple,
    Any,
    Literal,
    Protocol,
    runtime_checkable,
)
from pydantic import BaseModel, Field, field_validator
from enum import Enum

logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Enum for supported model providers."""

    OLLAMA = "ollama"
    GEMINI = "gemini"
    CLAUDE = "claude"


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Dict[str, Any] = None,
        **kwargs,
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
    evidence: str = Field(
        min_length=1, description="Evidence from the resume supporting this score"
    )


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
        **kwargs,
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


# Paid-tier Gemini pricing, USD per 1M tokens (verified at ai.google.dev/gemini-api/docs/pricing).
# gemini-2.5-pro's ">200k prompt" tier is intentionally not modeled -- this
# pipeline's prompts are a few thousand tokens at most. Prices change over
# time; re-verify against the pricing page if costs look off.
GEMINI_PRICING_PER_MILLION_TOKENS = {
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-3.5-flash": {"input": 1.50, "output": 9.00},
}


class GeminiSpendTracker:
    """Persistent per-model, per-UTC-day tracker of Gemini token usage and cost.

    File-based (cache/gemini_spend_<model>_<YYYY-MM-DD>.json) so totals survive
    across separate `python score.py` invocations. Purely for visibility --
    never blocks or delays a call, unlike the request-count guard this replaced
    (removed once billing was enabled, since the free-tier daily cap no longer
    applies).
    """

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        self._lock = threading.Lock()

    def _today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _path(self, model: str) -> str:
        safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", model)
        return os.path.join(
            self.cache_dir, f"gemini_spend_{safe_model}_{self._today()}.json"
        )

    def _read(self, model: str) -> Dict[str, Any]:
        path = self._path(model)
        default = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}
        if not os.path.exists(path):
            return default
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "input_tokens": int(data.get("input_tokens", 0)),
                "output_tokens": int(data.get("output_tokens", 0)),
                "cost": float(data.get("cost", 0.0)),
            }
        except Exception as e:
            logger.warning(
                f"Invalid Gemini spend file {path}: {e}. Treating today's totals as 0."
            )
            return default

    def _write(self, model: str, totals: Dict[str, Any]) -> None:
        path = self._path(model)
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            tmp_path = path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"date": self._today(), "model": model, **totals}, f, indent=2
                )
            os.replace(tmp_path, path)
        except Exception as e:
            logger.warning(f"Failed to write Gemini spend file {path}: {e}")

    @staticmethod
    def estimate_cost(
        model: str, input_tokens: int, output_tokens: int
    ) -> Optional[float]:
        pricing = GEMINI_PRICING_PER_MILLION_TOKENS.get(model)
        if pricing is None:
            return None
        return (input_tokens / 1_000_000) * pricing["input"] + (
            output_tokens / 1_000_000
        ) * pricing["output"]

    def record_usage(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """Record one call's token usage. Returns (this_call_cost, today's running totals)."""
        call_cost = self.estimate_cost(model, input_tokens, output_tokens)
        with self._lock:
            totals = self._read(model)
            totals["input_tokens"] += input_tokens
            totals["output_tokens"] += output_tokens
            totals["cost"] += call_cost or 0.0
            self._write(model, totals)
        return call_cost, totals

    def today_summary(self, model: str) -> str:
        totals = self._read(model)
        total_tokens = totals["input_tokens"] + totals["output_tokens"]
        if GEMINI_PRICING_PER_MILLION_TOKENS.get(model) is None:
            return f"{total_tokens} tokens used today for {model} (pricing unknown for this model)"
        return f"${totals['cost']:.4f} spent today ({total_tokens} tokens) for {model}"


def _get_gemini_spend_tracker() -> GeminiSpendTracker:
    global _GEMINI_SPEND_TRACKER
    if _GEMINI_SPEND_TRACKER is None:
        _GEMINI_SPEND_TRACKER = GeminiSpendTracker()
    return _GEMINI_SPEND_TRACKER


_GEMINI_SPEND_TRACKER: Optional[GeminiSpendTracker] = None


def get_gemini_daily_spend_line(model: str) -> str:
    """Public accessor for the daily spend banner (used by score.py)."""
    return _get_gemini_spend_tracker().today_summary(model)


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
        **kwargs,
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
                # Send the chat request
                response = gemini_model.generate_content(gemini_messages)

                # Record token usage and print per-call + running daily cost.
                usage = response.usage_metadata
                input_tokens = usage.prompt_token_count
                output_tokens = max(usage.total_token_count - input_tokens, 0)
                call_cost, _ = _get_gemini_spend_tracker().record_usage(
                    model, input_tokens, output_tokens
                )
                if call_cost is None:
                    print(
                        f"[GeminiProvider] {input_tokens + output_tokens} tokens used "
                        f"(pricing unknown for {model})"
                    )
                else:
                    print(
                        f"[GeminiProvider] This call: {input_tokens} in + {output_tokens} out tokens, "
                        f"~${call_cost:.4f}. Today so far: {_get_gemini_spend_tracker().today_summary(model)}"
                    )

                # Convert Gemini response to Ollama-like format for compatibility
                return {"message": {"role": "assistant", "content": response.text}}

            except ResourceExhausted as e:
                if attempt == MAX_RETRIES - 1:
                    # All retries exhausted — re-raise the original exception.
                    # This surfaces unrecoverable quota errors (RPD, TPM, etc.)
                    # instead of silently failing or returning bad data.
                    raise

                # Parse the API-suggested retry delay from the error message
                match = re.search(r"retry[_ ]in\s+([\d.]+)s", str(e), re.IGNORECASE)
                api_hint = float(match.group(1)) if match else None

                # Exponential backoff: BASE_DELAY * 2^attempt, capped at MAX_DELAY
                exp_delay = min(BASE_DELAY * (2**attempt), MAX_DELAY)

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


class ClaudeProvider:
    """Anthropic Claude API provider implementation.

    Matches the LLMProvider Protocol and the OllamaProvider/GeminiProvider
    return shape: {"message": {"role": "assistant", "content": text}}.

    Claude Sonnet 5 rejects non-default temperature/top_p/top_k, so the
    `options` sampling values are intentionally not forwarded. The Ollama-style
    `format` kwarg (JSON schema) is accepted but ignored — callers must
    instruct JSON output in the prompt and parse with
    extract_json_from_response, exactly as the Gemini path already does.
    """

    def __init__(self, api_key: str):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Dict[str, Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        system_parts = []
        conversation = []
        for message in messages:
            if message["role"] == "system":
                system_parts.append(message["content"])
            else:
                conversation.append(
                    {"role": message["role"], "content": message["content"]}
                )

        request_params = {
            "model": model,
            "max_tokens": 16000,
            "messages": conversation,
        }
        if system_parts:
            request_params["system"] = "\n\n".join(system_parts)

        response = self.client.messages.create(**request_params)

        if response.stop_reason == "max_tokens":
            logger.warning("[ClaudeProvider] Response truncated at max_tokens.")

        text_parts = [block.text for block in response.content if block.type == "text"]
        return {"message": {"role": "assistant", "content": "".join(text_parts)}}
