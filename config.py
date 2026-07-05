"""
Configuration settings for the hiring agent application.
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Global development mode flag
DEVELOPMENT_MODE = True


def _load_gemini_requests_per_minute() -> float:
    """Read GEMINI_REQUESTS_PER_MINUTE from the environment.

    A value <= 0 disables proactive pacing (the reactive 429 backoff in
    GeminiProvider still applies). Defaults to 8, conservative for the
    Gemini free tier, since Google's published RPM limits vary by model
    and change over time.
    """
    raw_value = os.getenv("GEMINI_REQUESTS_PER_MINUTE", "8")
    try:
        return float(raw_value)
    except ValueError:
        logger.warning(
            f"Invalid GEMINI_REQUESTS_PER_MINUTE value '{raw_value}'. Falling back to default of 8."
        )
        return 8.0


# Proactive client-side pacing for Gemini API calls, shared across every
# GeminiProvider instance in the process. Does not affect Ollama.
GEMINI_REQUESTS_PER_MINUTE = _load_gemini_requests_per_minute()


def _load_gemini_requests_per_day() -> int:
    """Read GEMINI_REQUESTS_PER_DAY from the environment.

    A value <= 0 disables the daily-budget pre-flight guard (usage is still
    recorded for visibility). Defaults to 20, matching the observed
    free-tier RPD cap for gemini-2.5-flash
    (quota_id GenerateRequestsPerDayPerProjectPerModel-FreeTier).
    """
    raw_value = os.getenv("GEMINI_REQUESTS_PER_DAY", "20")
    try:
        return int(float(raw_value))
    except ValueError:
        logger.warning(
            f"Invalid GEMINI_REQUESTS_PER_DAY value '{raw_value}'. Falling back to default of 20."
        )
        return 20


# Local daily budget for real Gemini API calls, tracked per model per UTC
# day in cache/gemini_quota_*.json so it survives across separate script
# invocations. Does not affect Ollama.
GEMINI_REQUESTS_PER_DAY = _load_gemini_requests_per_day()
