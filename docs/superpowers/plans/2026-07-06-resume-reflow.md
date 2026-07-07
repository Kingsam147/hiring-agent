# Resume Reflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `reflow.py` — a standalone script that reads the latest `result.md` gap analysis, uses Claude Sonnet 5 to rewrite only the wording (summary, skills, bullets) of `resume/resume_reflow/reflow_resume.py`, regrades each candidate with the existing `JobDescriptionEvaluator`, and writes `reflow_resume_tailored.py` when the best score clears 70% — after first simplifying `score.py` to a single, non-interactive Custom Job Description path.

**Architecture:** `score.py` drops both interactive prompts and Mode 1 entirely, so `result.md` always has the job-match shape. A new `ClaudeProvider` slots into the existing `LLMProvider` Protocol (`models.py` / `prompt.py` / `llm_utils.py`). `reflow.py` parses `result.md` by its literal Markdown headers, loads the cached `JSONResume` and the CONTENT block of `reflow_resume.py` (imported via `importlib` so the layout constants and `_wrap()` are reused, never reimplemented), runs a max-6-iteration tailor → validate → remap → regrade loop with a frozen knockout resolver captured once up front, and finally splices the best candidate's content into a byte-identical copy of `reflow_resume.py`.

**Tech Stack:** Python, Pydantic, Jinja2, `anthropic` SDK (new), ReportLab (already required at runtime by `reflow_resume.py`, now added to `requirements.txt`).

**Design spec:** `docs/superpowers/specs/2026-07-06-resume-reflow-design.md` (approved).

**Branch:** all work happens on a new `feature/resume-reflow` branch created from the current state of `feature/job-requirement-gate` (never on `main`, per repo rules).

**Note on testing:** Per this repo's `CLAUDE.md`, there is no automated test suite — validation is manual. Each task ends with a `python -c` import/behavior smoke check, and Task 10 is the full end-to-end manual validation run.

---

### Task 1: `score.py` simplification — remove Mode 1 and both interactive prompts

**Files:**
- Modify: `score.py:1-30` (imports)
- Modify: `score.py:68-95` (delete `select_mode`, `select_weight_profile`)
- Modify: `score.py:112-205` (delete `build_evaluation_markdown`)
- Modify: `score.py:364-380` (delete `_evaluate_resume`)
- Modify: `score.py:418-430` (`main()` header — drop mode logic, hardcode profile)
- Modify: `score.py:488-504` (gate block — remove the `mode == 2` guard)
- Modify: `score.py:559-607` (tail of `main()` — delete the mode-1 branch and its CSV block)

**What is kept, deliberately:** `ResumeEvaluator` and `EvaluationData` stay in `evaluator.py`/`models.py` untouched — only `score.py`'s entry point changes. Also kept in `score.py`: `find_resume_file`, `load_job_description`, `build_job_evaluation_markdown`, `build_flagged_report_markdown`, `write_result_markdown`, `_knockout_resolver`, `is_valid_resume_data`, `find_profile`, the resume/GitHub cache blocks, and the `job_evaluations.csv` writing branch (so the `csv` import stays).

- [ ] **Step 1: Replace the import block**

Find this exact block (`score.py` lines 1-28):

```python
import os
import sys
import json
import logging
import csv
from pdf import PDFHandler
from github import fetch_and_display_github_info
from models import (
    JSONResume,
    EvaluationData,
    JobEvaluationData,
    RequirementGateResult,
    ModelProvider,
    get_gemini_daily_spend_line,
)
from typing import Optional
from evaluator import ResumeEvaluator, JobDescriptionEvaluator
from pathlib import Path
from prompt import DEFAULT_MODEL, MODEL_PARAMETERS, MODEL_PROVIDER_MAPPING
from weight_profiles import WEIGHT_PROFILES, DEFAULT_PROFILE, suggest_profile
from transform import (
    transform_evaluation_response,
    transform_job_evaluation_response,
    convert_json_resume_to_text,
    convert_github_data_to_text,
    convert_blog_data_to_text,
)
from config import DEVELOPMENT_MODE
```

Replace with:

```python
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
    RequirementGateResult,
    ModelProvider,
    get_gemini_daily_spend_line,
)
from typing import Optional
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
```

Removed imports and why: `EvaluationData` (only used by `build_evaluation_markdown`/`_evaluate_resume`, both deleted below), `ResumeEvaluator` (only used by `_evaluate_resume`), `transform_evaluation_response` (only used by the mode-1 CSV branch), `convert_blog_data_to_text` (only used by `_evaluate_resume`). `Optional` stays (used by `_knockout_resolver`); `csv` stays (job_evaluations.csv branch survives); `WEIGHT_PROFILES`/`DEFAULT_PROFILE`/`suggest_profile` stay (used by `build_job_evaluation_markdown`).

- [ ] **Step 2: Delete `select_mode` and `select_weight_profile`**

Find this exact block (lines 68-95) and delete it entirely (leave `load_job_description` immediately after `find_resume_file`):

```python
def select_mode() -> int:
    print("\nChoose scoring mode:")
    print("  1. HackerRank Intern (original)")
    print("  2. Custom Job Description")
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice in ("1", "2"):
            return int(choice)
        print("Invalid choice. Please enter 1 or 2.")


def select_weight_profile() -> str:
    profile_names = list(WEIGHT_PROFILES.keys())
    print("\nChoose a weight profile (affects how category scores are combined):")
    for i, name in enumerate(profile_names, 1):
        default_marker = " (default)" if name == DEFAULT_PROFILE else ""
        print(f"  {i}. {name}{default_marker}")
    choice = input(f"Enter choice (1-{len(profile_names)}, Enter for default): ").strip()
    if not choice:
        return DEFAULT_PROFILE
    try:
        index = int(choice) - 1
        if 0 <= index < len(profile_names):
            return profile_names[index]
    except ValueError:
        pass
    print(f"Invalid choice. Using default profile '{DEFAULT_PROFILE}'.")
    return DEFAULT_PROFILE
```

- [ ] **Step 3: Delete `build_evaluation_markdown`**

Delete the whole function `def build_evaluation_markdown(...)` (lines 112-205, from `def build_evaluation_markdown(` through its final `return "\n".join(lines)` — the next top-level statement is `def build_job_evaluation_markdown(`). Nothing else references it after the mode-1 branch is removed in Step 6.

- [ ] **Step 4: Delete `_evaluate_resume`**

Find this exact block (lines 364-380) and delete it (the next statement is `def _knockout_resolver(`):

```python
def _evaluate_resume(
    resume_data: JSONResume, github_data: dict = None, blog_data: dict = None
) -> Optional[EvaluationData]:
    model_params = MODEL_PARAMETERS.get(DEFAULT_MODEL)
    evaluator = ResumeEvaluator(model_name=DEFAULT_MODEL, model_params=model_params)

    resume_text = convert_json_resume_to_text(resume_data)

    if github_data:
        github_text = convert_github_data_to_text(github_data)
        resume_text += github_text

    if blog_data:
        blog_text = convert_blog_data_to_text(blog_data)
        resume_text += blog_text

    return evaluator.evaluate_resume(resume_text)
```

- [ ] **Step 5: Simplify the head of `main()` and the gate block**

Find this exact block (lines 418-430):

```python
def main():
    pdf_path = find_resume_file()

    mode = select_mode()

    if MODEL_PROVIDER_MAPPING.get(DEFAULT_MODEL) == ModelProvider.GEMINI:
        print(f"Gemini spend so far today: {get_gemini_daily_spend_line(DEFAULT_MODEL)}")

    job_description = None
    weight_profile = DEFAULT_PROFILE
    if mode == 2:
        job_description = load_job_description()
        weight_profile = select_weight_profile()
```

Replace with:

```python
def main():
    pdf_path = find_resume_file()

    if MODEL_PROVIDER_MAPPING.get(DEFAULT_MODEL) == ModelProvider.GEMINI:
        print(f"Gemini spend so far today: {get_gemini_daily_spend_line(DEFAULT_MODEL)}")

    job_description = load_job_description()
    weight_profile = "engineering"
```

Then find this exact block (lines 488-504):

```python
    job_evaluator = None
    if mode == 2:
        model_params = MODEL_PARAMETERS.get(DEFAULT_MODEL)
        job_evaluator = JobDescriptionEvaluator(
            job_description=job_description,
            model_name=DEFAULT_MODEL,
            model_params=model_params,
            weight_profile=weight_profile,
        )
        gate_resume_text = convert_json_resume_to_text(resume_data)
        gate_result = job_evaluator.check_requirements(
            gate_resume_text, resume_data=resume_data, knockout_resolver=_knockout_resolver
        )
        if not gate_result.passed:
            markdown = build_flagged_report_markdown(gate_result, candidate_name)
            write_result_markdown(markdown)
            return gate_result
```

Replace with (guard removed, one dedent level):

```python
    model_params = MODEL_PARAMETERS.get(DEFAULT_MODEL)
    job_evaluator = JobDescriptionEvaluator(
        job_description=job_description,
        model_name=DEFAULT_MODEL,
        model_params=model_params,
        weight_profile=weight_profile,
    )
    gate_resume_text = convert_json_resume_to_text(resume_data)
    gate_result = job_evaluator.check_requirements(
        gate_resume_text, resume_data=resume_data, knockout_resolver=_knockout_resolver
    )
    if not gate_result.passed:
        markdown = build_flagged_report_markdown(gate_result, candidate_name)
        write_result_markdown(markdown)
        return gate_result
```

- [ ] **Step 6: Delete the mode-1 tail branch of `main()`**

Find this exact block (lines 559-607, from `if mode == 1:` through the end of the `else:` body):

```python
    if mode == 1:
        score = _evaluate_resume(resume_data, github_data)
        write_result_markdown(build_evaluation_markdown(score, candidate_name))

        if DEVELOPMENT_MODE:
            csv_row = transform_evaluation_response(
                file_name=os.path.basename(pdf_path),
                evaluation=score,
                resume_data=resume_data,
                github_data=github_data,
            )
            csv_path = "resume_evaluations.csv"
            file_exists = os.path.exists(csv_path)
            with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
                fieldnames = list(csv_row.keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(csv_row)

        return score

    else:
        resume_text = convert_json_resume_to_text(resume_data)
        if github_data:
            resume_text += convert_github_data_to_text(github_data)

        job_evaluation = job_evaluator.evaluate(
            resume_text, resume_data=resume_data, knockout_resolver=_knockout_resolver
        )
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
```

Replace with (former `else:` body promoted one level):

```python
    resume_text = convert_json_resume_to_text(resume_data)
    if github_data:
        resume_text += convert_github_data_to_text(github_data)

    job_evaluation = job_evaluator.evaluate(
        resume_text, resume_data=resume_data, knockout_resolver=_knockout_resolver
    )
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
```

- [ ] **Step 7: Verify no stale references remain and the module imports cleanly**

Run:
```bash
python -c "
import ast, pathlib
tree = ast.parse(pathlib.Path('score.py').read_text(encoding='utf-8'))
names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
stale = {'select_mode', 'select_weight_profile', '_evaluate_resume', 'build_evaluation_markdown', 'ResumeEvaluator', 'EvaluationData', 'transform_evaluation_response', 'convert_blog_data_to_text', 'mode'} & names
print('stale references:', stale or 'none')
import score
print('import OK')
"
```
Expected: `stale references: none` then `import OK`, no traceback.

- [ ] **Step 8: Commit**

```bash
git add score.py
git commit -m "refactor: drop Mode 1 and interactive prompts from score.py, hardcode engineering profile"
```

---

### Task 2: Claude Sonnet 5 provider plumbing

**Files:**
- Modify: `models.py:15-19` (`ModelProvider` enum)
- Modify: `models.py` (append `ClaudeProvider` after `GeminiProvider`, end of file)
- Modify: `prompt.py:28-67` (`MODEL_PARAMETERS`, `MODEL_PROVIDER_MAPPING`, API key)
- Modify: `llm_utils.py:7-8, 40-62` (imports + provider branch)
- Modify: `requirements.txt`
- Modify: `.env.example`

Key API facts baked into this design (verified against current Anthropic docs): the model ID is exactly `claude-sonnet-5` (no date suffix); Claude Sonnet 5 **rejects non-default `temperature`/`top_p`/`top_k` with a 400**, so `ClaudeProvider` must not forward the pipeline's `options` sampling values; the Anthropic Messages API takes the system prompt as a separate `system` parameter, not a `{"role": "system"}` message; responses arrive as content blocks, and only `text`-type blocks are joined into the returned string (Sonnet 5 runs adaptive thinking by default, so non-text blocks can appear).

- [ ] **Step 1: Add `ModelProvider.CLAUDE`**

In `models.py`, find:

```python
class ModelProvider(Enum):
    """Enum for supported model providers."""

    OLLAMA = "ollama"
    GEMINI = "gemini"
```

Replace with:

```python
class ModelProvider(Enum):
    """Enum for supported model providers."""

    OLLAMA = "ollama"
    GEMINI = "gemini"
    CLAUDE = "claude"
```

- [ ] **Step 2: Add `ClaudeProvider`**

Append to the end of `models.py` (after the `GeminiProvider` class):

```python
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
        **kwargs
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

        text_parts = [
            block.text for block in response.content if block.type == "text"
        ]
        return {"message": {"role": "assistant", "content": "".join(text_parts)}}
```

`List`, `Dict`, `Any`, and `logger` are already available at the top of `models.py` — no import changes needed there. `import anthropic` is inside `__init__` (same lazy-import pattern as `OllamaProvider`/`GeminiProvider`), so merely importing `models.py` never requires the SDK.

- [ ] **Step 3: Register the model in `prompt.py`**

Find the end of `MODEL_PARAMETERS` (line 44):

```python
    "gemini-3.5-flash": {"temperature": 0.1, "top_p": 0.9},
    "gemini-3.1-flash-lite": {"temperature": 0.1, "top_p": 0.9},
}
```

Replace with:

```python
    "gemini-3.5-flash": {"temperature": 0.1, "top_p": 0.9},
    "gemini-3.1-flash-lite": {"temperature": 0.1, "top_p": 0.9},
    # Anthropic Claude models. Claude Sonnet 5 rejects non-default sampling
    # parameters, so no temperature/top_p here — ClaudeProvider ignores them.
    "claude-sonnet-5": {},
}
```

Find the end of `MODEL_PROVIDER_MAPPING` (line 64):

```python
    "gemini-3.5-flash": ModelProvider.GEMINI,
    "gemini-3.1-flash-lite": ModelProvider.GEMINI,
}
```

Replace with:

```python
    "gemini-3.5-flash": ModelProvider.GEMINI,
    "gemini-3.1-flash-lite": ModelProvider.GEMINI,
    # Anthropic Claude models
    "claude-sonnet-5": ModelProvider.CLAUDE,
}
```

Find (line 67):

```python
# Get API keys from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
```

Replace with:

```python
# Get API keys from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")
```

- [ ] **Step 4: Add the branch in `llm_utils.initialize_llm_provider`**

Find (lines 7-8):

```python
from models import ModelProvider, OllamaProvider, GeminiProvider
from prompt import MODEL_PROVIDER_MAPPING, GEMINI_API_KEY
```

Replace with:

```python
from models import ModelProvider, OllamaProvider, GeminiProvider, ClaudeProvider
from prompt import MODEL_PROVIDER_MAPPING, GEMINI_API_KEY, CLAUDE_API_KEY
```

Find the body of `initialize_llm_provider` (lines 50-62):

```python
    # Default to Ollama provider
    provider = OllamaProvider()
    # If using Gemini and API key is available, use Gemini provider
    model_provider = MODEL_PROVIDER_MAPPING.get(model_name, ModelProvider.OLLAMA)
    if model_provider == ModelProvider.GEMINI:
        if not GEMINI_API_KEY:
            logger.warning("⚠️ Gemini API key not found. Falling back to Ollama.")
        else:
            logger.info(f"🔄 Using Google Gemini API provider with model {model_name}")
            provider = GeminiProvider(api_key=GEMINI_API_KEY)
    else:
        logger.info(f"🔄 Using Ollama provider with model {model_name}")
    return provider
```

Replace with:

```python
    # Default to Ollama provider
    provider = OllamaProvider()
    model_provider = MODEL_PROVIDER_MAPPING.get(model_name, ModelProvider.OLLAMA)
    if model_provider == ModelProvider.GEMINI:
        if not GEMINI_API_KEY:
            logger.warning("⚠️ Gemini API key not found. Falling back to Ollama.")
        else:
            logger.info(f"🔄 Using Google Gemini API provider with model {model_name}")
            provider = GeminiProvider(api_key=GEMINI_API_KEY)
    elif model_provider == ModelProvider.CLAUDE:
        if not CLAUDE_API_KEY:
            logger.warning("⚠️ Claude API key not found. Falling back to Ollama.")
        else:
            logger.info(f"🔄 Using Anthropic Claude API provider with model {model_name}")
            provider = ClaudeProvider(api_key=CLAUDE_API_KEY)
    else:
        logger.info(f"🔄 Using Ollama provider with model {model_name}")
    return provider
```

Note: `reflow.py` (Task 7 onward) calls `initialize_llm_provider("claude-sonnet-5")` directly — it never reads `DEFAULT_MODEL`. Regrading still uses `DEFAULT_MODEL` through the untouched `JobDescriptionEvaluator` path, so reported scores stay comparable to `score.py`. Because the Ollama fallback here would silently regrade-tailor with the wrong model, `reflow.py`'s `main()` hard-exits when `CLAUDE_API_KEY` is empty (Task 9).

- [ ] **Step 5: Update `requirements.txt` and `.env.example`**

`requirements.txt` — find:

```
black==25.9.0
sentence-transformers
```

Replace with:

```
black==25.9.0
sentence-transformers
anthropic
reportlab
```

`reportlab` is an addition beyond the design spec's list: `reflow_resume.py` imports it at module top (its docstring says `pip install reportlab` manually), and `reflow.py` imports that module for `_wrap()`/layout constants, making ReportLab a direct dependency of the new feature. Flag this in the PR description.

`.env.example` — append:

```
# Anthropic Claude API Key (required for reflow.py resume tailoring)
CLAUDE_API_KEY=your_claude_api_key_here
```

- [ ] **Step 6: Install and smoke-check**

Run:
```bash
python -m pip install anthropic reportlab
python -c "
from llm_utils import initialize_llm_provider
from models import ModelProvider, ClaudeProvider
from prompt import MODEL_PROVIDER_MAPPING
assert MODEL_PROVIDER_MAPPING['claude-sonnet-5'] == ModelProvider.CLAUDE
provider = initialize_llm_provider('claude-sonnet-5')
print(type(provider).__name__)
"
```
Expected: prints `ClaudeProvider` if `CLAUDE_API_KEY` is set in `.env`, otherwise `OllamaProvider` plus the fallback warning — either way, no traceback.

- [ ] **Step 7: Commit**

```bash
git add models.py prompt.py llm_utils.py requirements.txt .env.example
git commit -m "feat: add ClaudeProvider and register claude-sonnet-5 in the provider mapping"
```

---

### Task 3: Reflow prompt templates and skills-bank skeleton

**Files:**
- Create: `prompts/templates/resume_reflow_system_message.jinja`
- Create: `prompts/templates/resume_reflow_user_message.jinja`
- Modify: `prompts/template_manager.py:37-54` (register both)
- Create: `resume/resume_reflow/skills_bank.txt`

- [ ] **Step 1: Create the system message template**

Create `prompts/templates/resume_reflow_system_message.jinja`:

```
You are an expert resume writer tailoring one specific resume toward one specific job description. You rewrite ONLY the wording of the summary, the skills lines, and the experience/project bullet points. You never change facts, structure, or layout.

HARD CONSTRAINTS — every single one must hold in every response:

1. FIXED METRICS. These metrics must appear VERBATIM, character for character, in the bullets of the same entries they appear in today: "1,384ms" and "196ms"; "~$60" and "~$0.38"; "5-10x"; "32 REST endpoints"; "500+ stars"; "1,200+", "900+" and "300+"; "PR #283"; "PR #822". Never reword, round, or move them to a different entry.
2. NO EM DASHES. The em dash character (U+2014) must not appear anywhere in your output. En dashes in date ranges are part of the fixed metadata and are not yours to touch.
3. TRUTH ONLY. Only mention skills, tools, and projects that already appear in the current resume content or in the SKILLS BANK section of the prompt. Never invent experience, employers, projects, technologies, or numbers.
4. SAME STRUCTURE. Return exactly the same number of skills lines, the same experience entries and project entries with their "title" values copied VERBATIM, and exactly the same number of bullets per entry as you were given. Never add or remove a job, project, skills line, or bullet.
5. OUT OF SCOPE. Name, contact line, links, entry dates/links metadata, the education line, and the activities line are not part of your output and must never be referenced as changed.
6. ONE PAGE. Each bullet must be short enough to wrap to at most 2 printed lines: keep every bullet under about 220 characters. Keep the summary within about 5 printed lines (under about 450 characters). Each skills line ("label" plus "rest") must fit on one printed line: keep label+rest under about 115 characters.

You MUST respond with ONLY the JSON structure specified in the prompt. No markdown fences, no commentary.
```

- [ ] **Step 2: Create the user message template**

Create `prompts/templates/resume_reflow_user_message.jinja`:

```
Tailor the resume content below toward this job. Close as much of the listed gap as you truthfully can by reworking wording, emphasis, and terminology — surfacing matching skills from the SKILLS BANK where they are genuinely relevant.

## TARGET JOB

Job title: {{ job_title }}

Required skills the resume is MISSING:
{% for skill in missing_required_skills %}
- {{ skill }}
{% else %}
- (none)
{% endfor %}

Preferred skills the resume is missing:
{% for skill in missing_preferred_skills %}
- {{ skill }}
{% else %}
- (none)
{% endfor %}

Areas for improvement from the last evaluation:
{% for area in improvement_areas %}
- {{ area }}
{% else %}
- (none)
{% endfor %}

## CURRENT RESUME CONTENT

SUMMARY:
{{ summary }}

SKILLS (label + rest pairs — keep the same count and the same label wording unless a label rename is clearly justified by the job):
{% for skill in skills %}
- label: "{{ skill.label }}" rest: "{{ skill.rest }}"
{% endfor %}

EXPERIENCE (titles are fixed; bullet counts are fixed):
{% for entry in experience %}
### {{ entry.title }}
{% for bullet in entry.bullets %}
- {{ bullet }}
{% endfor %}
{% endfor %}

PROJECTS (titles are fixed; bullet counts are fixed):
{% for entry in projects %}
### {{ entry.title }}
{% for bullet in entry.bullets %}
- {{ bullet }}
{% endfor %}
{% endfor %}

## SKILLS BANK (additional truthful skills/projects you MAY draw on — this bank may be empty)

{{ skills_bank if skills_bank else "(empty)" }}

{% if retry_feedback %}
## PROBLEMS WITH YOUR PREVIOUS ATTEMPT — FIX THESE

{{ retry_feedback }}
{% endif %}

## OUTPUT

Return ONLY this JSON structure, nothing else:

{
    "summary": "string",
    "skills": [{"label": "string", "rest": "string"}],
    "experience": [{"title": "string", "bullets": ["string"]}],
    "projects": [{"title": "string", "bullets": ["string"]}]
}
```

- [ ] **Step 3: Register both templates**

In `prompts/template_manager.py`, find the end of the `template_files` dict (lines 52-54):

```python
            "requirement_recheck": "requirement_recheck.jinja",
            "requirement_recheck_system_message": "requirement_recheck_system_message.jinja",
        }
```

Replace with:

```python
            "requirement_recheck": "requirement_recheck.jinja",
            "requirement_recheck_system_message": "requirement_recheck_system_message.jinja",
            "resume_reflow_system_message": "resume_reflow_system_message.jinja",
            "resume_reflow_user_message": "resume_reflow_user_message.jinja",
        }
```

- [ ] **Step 4: Create the skills-bank skeleton**

Create `resume/resume_reflow/skills_bank.txt` (categories match the existing `SKILLS` labels in `reflow_resume.py` — Languages, Backend, Cloud & DevOps, Architecture & Security — plus Projects):

```
# Skills Bank for reflow.py
# List truthful skills and projects that are NOT (fully) reflected in the
# current resume wording. reflow.py hands this file to the tailoring model,
# which may only use items listed here or already present in the resume.
# One item per line under each header. Leave sections empty if nothing applies.

[Languages]

[Backend]

[Cloud & DevOps]

[Architecture & Security]

[Projects]
# Format: Project Name - one-line truthful description of what you built/did
```

- [ ] **Step 5: Verify templates load and render**

Run:
```bash
python -c "
from prompts.template_manager import TemplateManager
tm = TemplateManager()
print(tm.render_template('resume_reflow_system_message')[:60])
rendered = tm.render_template(
    'resume_reflow_user_message',
    job_title='Backend Engineer',
    missing_required_skills=['Kubernetes'],
    missing_preferred_skills=[],
    improvement_areas=['quantify impact'],
    summary='Backend engineer.',
    skills=[{'label': 'Languages: ', 'rest': 'Python'}],
    experience=[{'title': 'Acme | Engineer', 'bullets': ['Did things.']}],
    projects=[{'title': 'Tool', 'bullets': ['Built it.']}],
    skills_bank='',
    retry_feedback=None,
)
assert 'Kubernetes' in rendered and 'Acme | Engineer' in rendered and '(empty)' in rendered
print('render OK')
"
```
Expected: first line of the system message, then `render OK`, no `Template not found` / `Error rendering` messages.

Note: the template accesses `skill.label` / `entry.title` via Jinja attribute lookup, which also works for dicts (`skills=[{'label': ...}]`) and for the Pydantic objects `reflow.py` will pass — both are exercised before Task 7 wires the real call.

- [ ] **Step 6: Commit**

```bash
git add prompts/templates/resume_reflow_system_message.jinja prompts/templates/resume_reflow_user_message.jinja prompts/template_manager.py resume/resume_reflow/skills_bank.txt
git commit -m "feat: add resume reflow prompt templates and skills bank skeleton"
```

---

### Task 4: `reflow.py` — `result.md` parser

**Files:**
- Create: `reflow.py` (project root)

The parser matches the literal Markdown produced by `score.py` (this is the accepted trade-off in the design spec — if `build_job_evaluation_markdown` / `build_flagged_report_markdown` change format, this parser must be updated). Exact strings as they exist in `score.py` today:

| Field | Literal produced by `score.py` |
|---|---|
| Full-report first line | `# Job Match Evaluation: {candidate_name}` |
| Flagged-report first line | `# Requirement Gate: {candidate_name}` |
| Job title | `**Target Role:** {job_title}` |
| Weight profile (full report only) | `**Weight profile:** {profile}` |
| Missing required skills | line `**Required skills MISSING:**`, followed by one line of comma-joined skills or `None` |
| Missing preferred skills | line `**Preferred skills missing:**`, followed by one line of comma-joined skills or `None` (section may be absent) |
| Improvement areas (full report) | `## Areas for Improvement` section, `- item` lines |
| Missing requirements (flagged report) | `## Features to Add` section, `- item` lines |

- [ ] **Step 1: Create `reflow.py` with imports, constants, and the parser**

Create `reflow.py`:

```python
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
```

Note: importing `score` executes its module-level `logging.basicConfig(...)`, which also configures `reflow.py`'s logging — do not call `basicConfig` again here.

- [ ] **Step 2: Verify the parser against synthetic reports**

Run:
```bash
python -c "
from pathlib import Path
import reflow

Path('_test_full.md').write_text(
    '# Job Match Evaluation: Jane\n**Target Role:** Backend Engineer\n\n'
    '**Overall Match:** 62.0/100\n**Weight profile:** engineering\n\n'
    '**Required skills MISSING:**\nkubernetes, terraform\n\n'
    '**Preferred skills missing:**\nNone\n\n'
    '## Areas for Improvement\n- Quantify impact\n- Mention CI/CD\n', encoding='utf-8')
gap = reflow.parse_result_markdown('_test_full.md')
assert gap.job_title == 'Backend Engineer' and gap.weight_profile == 'engineering'
assert gap.missing_required_skills == ['kubernetes', 'terraform']
assert gap.missing_preferred_skills == [] and gap.improvement_areas == ['Quantify impact', 'Mention CI/CD']

Path('_test_flagged.md').write_text(
    '# Requirement Gate: Jane\n**Target Role:** Backend Engineer\n\n'
    '**Status:** FLAGGED\n\n## Features Kept\n- python\n\n'
    '## Features to Add\n- kubernetes\n- security clearance\n', encoding='utf-8')
flagged = reflow.parse_result_markdown('_test_flagged.md')
assert flagged.missing_required_skills == ['kubernetes', 'security clearance']
assert flagged.weight_profile == 'engineering'
print('parser OK')
"
python -c "import os; os.remove('_test_full.md'); os.remove('_test_flagged.md')"
```
Expected: `parser OK`.

Also verify the missing-file and wrong-shape errors:
```bash
python -c "import reflow; reflow.parse_result_markdown('_no_such_file.md')"
```
Expected: exits with the "Run 'python score.py' first" message (non-zero exit code).

- [ ] **Step 3: Commit**

```bash
git add reflow.py
git commit -m "feat: add reflow.py result.md gap-analysis parser"
```

---

### Task 5: `reflow.py` — resume and content loading

**Files:**
- Modify: `reflow.py` (append loaders after `parse_result_markdown`)

`resume/resume_reflow/` is not a Python package (no `__init__.py`) and lives under the data folder `resume/`, so `reflow_resume.py` is loaded via `importlib.util.spec_from_file_location`. Importing it executes only constants + function definitions (the `build()` call is guarded by `__main__`), but it does `import reportlab` at module top — hence the Task 2 `requirements.txt` addition.

- [ ] **Step 1: Append the loaders**

Append to `reflow.py`:

```python
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
```

`load_skills_bank` returns `""` for a missing file **and** for the untouched skeleton (headers/comments only), so the prompt renders `(empty)` instead of feeding the model boilerplate — this is the "skills bank empty on first run" behavior: tailoring proceeds using only content already in the resume.

- [ ] **Step 2: Verify loaders against the real files**

Run:
```bash
python -c "
import reflow
resume = reflow.load_cached_resume()
print('resume name:', resume.basics.name if resume.basics else None)
module = reflow.load_reflow_resume_module()
print('summary chars:', len(module.SUMMARY))
print('skills lines:', len(module.SKILLS))
print('experience entries:', len(module.EXPERIENCE), '| project entries:', len(module.PROJECTS))
print('wrap smoke:', len(module._wrap('word ' * 80, module.F_REG, module.SZ_BODY, module.TEXT_RIGHT - module.BULLET_TEXT_X)))
print('skills bank:', repr(reflow.load_skills_bank()[:40]))
"
```
Expected (with the current repo state): resume name printed from `cache/resumecache_Samuel_Darius_Resume_FixedIncome_SWE.json`, `skills lines: 4`, `experience entries: 2 | project entries: 3`, a wrap line count > 2, and `skills bank: ''` (skeleton only). No traceback (requires `reportlab` installed from Task 2).

- [ ] **Step 3: Commit**

```bash
git add reflow.py
git commit -m "feat: load cached JSONResume, reflow_resume CONTENT block, and skills bank in reflow.py"
```

---

### Task 6: `reflow.py` — layout-fit validator

**Files:**
- Modify: `reflow.py` (append the candidate models + layout validator)

This validator **imports** the renderer's own pieces — `module._wrap`, `module.LEADING`, the `GAP_*` constants, fonts/sizes, and the x-coordinates — and replays `build()`'s vertical accumulation exactly. Derivation, read directly from `reflow_resume.py:build()`:

- Wrap widths: summary lines wrap at `TEXT_RIGHT - BODY_X` (570.0 − 52.0 = 518pt); bullet continuation lines wrap at `TEXT_RIGHT - BULLET_TEXT_X` (570.0 − 60.0 = 510pt); both at `F_REG` / `SZ_BODY` (8.8pt).
- Skills lines are drawn with a single `drawString` each (bold label + regular rest) — they **never wrap**, they silently overflow the right edge. So the validator checks `stringWidth(label, F_BOLD, SZ_BODY) + stringWidth(rest, F_REG, SZ_BODY) <= TEXT_RIGHT - BODY_X` per line.
- Vertical accumulation (`top` is measured from the page top; page height `PAGE_H = 792.0`):
  1. `top = 75.0` (SUMMARY header position — the name/contact block above is fixed).
  2. SUMMARY: `y = top + GAP_HEADER_TO_BODY`; the last line's top is `y + (line_count - 1) * LEADING`; then `top = last + GAP_SECTION`.
  3. SKILLS: `y = top + GAP_HEADER_TO_SKILLS`; `last = y + (len(SKILLS) - 1) * LEADING`; `top = last + GAP_SECTION`.
  4. EXPERIENCE via `entries(...)` with `GAP_HEADER_TO_ENTRY_EXP`: `y = top + gap`; per entry, bullets start at `bt = y + GAP_ENTRY_TITLE_TO_BULLET` and each bullet advances `bt += line_count * LEADING` (that is `bullet()`'s return value); after each entry `last = bt - LEADING` and `y = last + GAP_ENTRY_TO_ENTRY`; after the loop `top = last + GAP_SECTION`.
  5. PROJECTS: identical with `GAP_HEADER_TO_ENTRY_PROJ`.
  6. EDUCATION: one fixed line at `y = top + GAP_HEADER_TO_BODY`; then `top = y + 16.2` — **16.2 is an inline literal in `build()`, not a named constant**, so the replay must hardcode it with a comment.
  7. ACTIVITIES: one fixed line at `y = top + GAP_HEADER_TO_SKILLS`. This `y` is the final top-of-line coordinate; the fit check requires `y + LEADING <= PAGE_H` (one full line height of room — slightly conservative, since the actual glyph bottom sits at baseline + descent, which is less than `LEADING` below `y`).

- [ ] **Step 1: Append the candidate models and validator**

Append to `reflow.py`:

```python
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
```

- [ ] **Step 2: Verify the validator accepts the current resume and rejects an overflow**

The current `reflow_resume.py` content renders to one page today, so it must validate clean; a candidate with an absurdly long bullet must fail both the 2-line check and (if long enough) the page check.

Run:
```bash
python -c "
import copy
import reflow
module = reflow.load_reflow_resume_module()
candidate = reflow.build_candidate_from_module(module)
problems = reflow.check_layout_fit(module, candidate)
print('current content problems:', problems or 'none')
assert problems == []

overflow = copy.deepcopy(candidate)
overflow.experience[0].bullets[0] = 'built and shipped things ' * 30
overflow_problems = reflow.check_layout_fit(module, overflow)
print('overflow problems:', len(overflow_problems))
assert any('wraps to' in p for p in overflow_problems)
print('layout validator OK')
"
```
Expected: `current content problems: none`, then at least one overflow problem, then `layout validator OK`. **If the current content does NOT validate clean, stop — the replay has drifted from `build()`; re-diff `_replay_vertical_layout` against `build()` before proceeding.**

- [ ] **Step 3: Commit**

```bash
git add reflow.py
git commit -m "feat: add layout-fit validator replaying reflow_resume build() geometry"
```

---

### Task 7: `reflow.py` — tailor call and guardrail validation

**Files:**
- Modify: `reflow.py` (append guardrail scans + the tailor/retry function)

Guardrails run **before** a candidate is ever scored: structure check (same counts, verbatim titles), em-dash scan, fixed-metric scan (each of the 8 metric groups must remain verbatim in the bullets of the same entry it lives in today — the entry mapping is derived at runtime from the original module, not hardcoded), and the Task 6 layout-fit check. Any failure re-prompts Claude with the specific offending fields, up to 3 retries; after that the iteration's candidate is discarded (`None`) and the loop moves on with the previous best.

- [ ] **Step 1: Append the guardrail scans**

Append to `reflow.py`:

```python
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
```

- [ ] **Step 2: Append the tailor call with the retry loop**

Append to `reflow.py`:

```python
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
```

- [ ] **Step 3: Verify the guardrails without any API call**

Run:
```bash
python -c "
import copy
import reflow
module = reflow.load_reflow_resume_module()
candidate = reflow.build_candidate_from_module(module)
assert reflow.validate_candidate(module, candidate) == []

homes = reflow._metric_home_entries(module)
assert len(homes) == 8, f'expected all 8 metric groups located, got {len(homes)}'

broken = copy.deepcopy(candidate)
broken.projects[0].bullets[-1] = broken.projects[0].bullets[-1].replace('1,384ms', '1384 ms')
broken.summary = 'A summary with an em dash — right here.'
problems = reflow.validate_candidate(module, broken)
assert any('fixed metric' in p for p in problems) and any('em dash' in p for p in problems)
print('guardrails OK')
"
```
Expected: `guardrails OK`. The `len(homes) == 8` assertion confirms every metric group is locatable in the current `reflow_resume.py` bullets (i.e., the `FIXED_METRIC_GROUPS` substrings match the file's actual text).

- [ ] **Step 4: Commit**

```bash
git add reflow.py
git commit -m "feat: add Claude tailoring call with structure, em-dash, fixed-metric, and layout guardrails"
```

---

### Task 8: `reflow.py` — JSONResume remapping and the regrade loop

**Files:**
- Modify: `reflow.py` (append remapping, frozen resolver, regrade, and the loop)

Design decisions carried through here:

- **Frozen resolver.** One interactive `check_requirements()` call happens up front (before the loop) on the original resume text, using a wrapper around `score._knockout_resolver` that records every answer. All later regrades use `frozen_resolver(qualification) -> captured.get(qualification)` — never interactive again. A qualification that never came up during setup resolves to `None` (treated as "skip"), which is the safe default.
- **Fresh evaluator per regrade.** Each iteration constructs a new `JobDescriptionEvaluator` (`_job_data`/`_keyword_result` start `None`) and runs `check_requirements()` **then** `evaluate()` on it — the same two-step flow `score.py` runs, so the LLM semantic recheck participates in the regraded coverage exactly as it would in a real run. `extract_job_requirements()` hits its `jdreqcache_*` disk cache, so the JD-extraction LLM call is never repeated; only the resume-dependent calls (recheck + scoring + summary) run fresh per new resume text, and `jobevalcache_*`/`reqcheckcache_*` still dedupe repeated texts across runs.
- **Regrade text.** Per the design spec, regrade text is `convert_json_resume_to_text(tailored_resume)` only — no GitHub enrichment text is appended. Caveat (accepted): `score.py` appends GitHub text before `evaluate()`, so a follow-up `score.py` run on the tailored resume can score slightly differently (mostly via the semantic-match component). The relative iteration-to-iteration comparison inside the loop is unaffected because every candidate is scored identically.
- **Best-candidate tracking.** The best-scoring candidate across all iterations wins (Claude output varies; a later iteration can regress). The next tailor call always builds on the best candidate so far. The loop stops early after 2 consecutive iterations with no score improvement (a discarded/guardrail-failed iteration counts as no improvement).

- [ ] **Step 1: Append remapping helpers**

Append to `reflow.py`:

```python
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
            keywords=[keyword.strip() for keyword in skill.rest.split(",") if keyword.strip()],
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
```

Title parsing follows the design: `EXPERIENCE` titles are `"{name} | {position}"` (matched to `work[].name`, falling back to `work[].position`); `PROJECTS` titles are `"{name}"` or `"{name} | {role}"` (matched to `projects[].name` with substring tolerance, because the PDF-extracted name and the generator's title can differ slightly — e.g. `"Pokemon Radical Red Platform & Damage Calculator"`). A miss logs a warning and skips that entry rather than aborting.

- [ ] **Step 2: Append the frozen resolver, regrade, and loop**

Append to `reflow.py`:

```python
def build_frozen_resolver(
    job_description: str,
    weight_profile: str,
    original_resume_text: str,
    original_resume: JSONResume,
):
    """Ask the user about unresolved must-haves exactly once, up front."""
    captured_answers: Dict[str, Optional[bool]] = {}

    def capturing_resolver(qualification: str) -> Optional[bool]:
        answer = _knockout_resolver(qualification)
        captured_answers[qualification] = answer
        return answer

    setup_evaluator = JobDescriptionEvaluator(
        job_description=job_description,
        model_name=DEFAULT_MODEL,
        model_params=MODEL_PARAMETERS.get(DEFAULT_MODEL),
        weight_profile=weight_profile,
    )
    setup_evaluator.check_requirements(
        original_resume_text,
        resume_data=original_resume,
        knockout_resolver=capturing_resolver,
    )

    def frozen_resolver(qualification: str) -> Optional[bool]:
        return captured_answers.get(qualification)

    return frozen_resolver


def regrade_candidate(
    job_description: str,
    weight_profile: str,
    tailored_resume: JSONResume,
    frozen_resolver,
) -> float:
    resume_text = convert_json_resume_to_text(tailored_resume)
    fresh_evaluator = JobDescriptionEvaluator(
        job_description=job_description,
        model_name=DEFAULT_MODEL,
        model_params=MODEL_PARAMETERS.get(DEFAULT_MODEL),
        weight_profile=weight_profile,
    )
    fresh_evaluator.check_requirements(
        resume_text, resume_data=tailored_resume, knockout_resolver=frozen_resolver
    )
    evaluation = fresh_evaluator.evaluate(
        resume_text, resume_data=tailored_resume, knockout_resolver=frozen_resolver
    )
    return evaluation.weighted_total


def run_reflow_loop(
    tailor_provider,
    template_manager: TemplateManager,
    gap: GapAnalysis,
    original_resume: JSONResume,
    reflow_module,
    skills_bank: str,
    job_description: str,
    frozen_resolver,
) -> Tuple[Optional[TailoredResume], float, List[float]]:
    best_candidate: Optional[TailoredResume] = None
    best_score = float("-inf")
    score_history: List[float] = []
    stagnant_iterations = 0
    current_content = build_candidate_from_module(reflow_module)

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n=== Iteration {iteration}/{MAX_ITERATIONS} ===")
        candidate = generate_tailored_candidate(
            tailor_provider, template_manager, gap, current_content, skills_bank, reflow_module
        )
        if candidate is None:
            print("No valid candidate this iteration (guardrails failed after retries).")
            stagnant_iterations += 1
            if stagnant_iterations >= MAX_STAGNANT_ITERATIONS:
                print("Stopping early: no improvement for "
                      f"{MAX_STAGNANT_ITERATIONS} consecutive iterations.")
                break
            continue

        tailored_resume = apply_candidate_to_resume(original_resume, candidate)
        iteration_score = regrade_candidate(
            job_description, gap.weight_profile, tailored_resume, frozen_resolver
        )
        score_history.append(iteration_score)
        print(f"Iteration {iteration} weighted total: {iteration_score}/100")

        if iteration_score > best_score:
            best_score = iteration_score
            best_candidate = candidate
            current_content = candidate
            stagnant_iterations = 0
        else:
            stagnant_iterations += 1

        if stagnant_iterations >= MAX_STAGNANT_ITERATIONS:
            print("Stopping early: no improvement for "
                  f"{MAX_STAGNANT_ITERATIONS} consecutive iterations.")
            break

    return best_candidate, best_score, score_history
```

- [ ] **Step 3: Verify the remapping against the real cached resume**

Run:
```bash
python -c "
import reflow
module = reflow.load_reflow_resume_module()
original_resume = reflow.load_cached_resume()
candidate = reflow.build_candidate_from_module(module)
candidate.experience[0].bullets[0] = 'MARKER tailored bullet for the regrade text.'
tailored = reflow.apply_candidate_to_resume(original_resume, candidate)
from transform import convert_json_resume_to_text
text = convert_json_resume_to_text(tailored)
assert 'MARKER tailored bullet' in text, 'work-entry title matching failed'
assert tailored.basics.summary == candidate.summary
assert original_resume.work[0].highlights != tailored.work[0].highlights or True
print('remap OK — no warnings expected above for matched entries')
"
```
Expected: `remap OK`. **Watch the log output**: any `No cached work entry matches title ...` / `No cached project matches title ...` warning for the current resume+generator pair means the title-matching heuristics need adjusting before proceeding (see Risks).

- [ ] **Step 4: Commit**

```bash
git add reflow.py
git commit -m "feat: add JSONResume remapping, frozen knockout resolver, and regrade loop to reflow.py"
```

---

### Task 9: `reflow.py` — banding, tailored-file output, and CLI entry point

**Files:**
- Modify: `reflow.py` (append banding, source splicing, `main()`, entry point)
- Modify: `.gitignore` (ignore the generated `reflow_resume_tailored.py`)

The tailored generator is produced by **splicing new literals into the original source text** — everything outside the four CONTENT assignments (`SUMMARY`, `SKILLS`, `EXPERIENCE`, `PROJECTS`) is carried over byte-identical, including the LAYOUT and RENDER ENGINE blocks, `NAME`, `CONTACT_PREFIX`, links, `EDUCATION_LINE`, `ACTIVITIES_LINE`, and each entry's `meta` (re-serialized via `repr()` from the imported module — `meta` values are plain tuples/strings, safe to `repr`). Splice boundaries use the exact anchor lines present in `reflow_resume.py` today: `SUMMARY = (`, `# (bold label, regular remainder)`, `SKILLS = [`, `EXPERIENCE = [`, `PROJECTS = [`, `EDUCATION_LINE = `.

- [ ] **Step 1: Append banding and the source splicer**

Append to `reflow.py`:

```python
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


def _replace_source_block(source: str, start_anchor: str, end_anchor: str, replacement: str) -> str:
    start_index = source.index("\n" + start_anchor) + 1
    end_index = source.index("\n" + end_anchor) + 1
    return source[:start_index] + replacement + source[end_index:]


def write_tailored_generator(candidate: TailoredResume, reflow_module) -> None:
    source = REFLOW_RESUME_PATH.read_text(encoding="utf-8")
    source = _replace_source_block(
        source, "SUMMARY = (", "# (bold label, regular remainder)",
        _serialize_summary(candidate.summary),
    )
    source = _replace_source_block(
        source, "SKILLS = [", "EXPERIENCE = [", _serialize_skills(candidate.skills)
    )
    source = _replace_source_block(
        source, "EXPERIENCE = [", "PROJECTS = [",
        _serialize_entries("EXPERIENCE", candidate.experience, reflow_module.EXPERIENCE),
    )
    source = _replace_source_block(
        source, "PROJECTS = [", "EDUCATION_LINE = ",
        _serialize_entries("PROJECTS", candidate.projects, reflow_module.PROJECTS),
    )
    TAILORED_RESUME_PATH.write_text(source, encoding="utf-8")
    print(f"Wrote {TAILORED_RESUME_PATH}")
```

Boundary note: the `# (bold label, regular remainder)` comment line is used as the SUMMARY block's end anchor, so it survives verbatim; the `SKILLS = [` block then ends at `EXPERIENCE = [`, and so on. Each `_serialize_*` result ends with `\n\n` so the anchor line that follows keeps a blank line above it, matching the original file's spacing.

- [ ] **Step 2: Append `main()` and the entry point**

Append to `reflow.py`:

```python
def main():
    if not CLAUDE_API_KEY:
        sys.exit(
            "Error: CLAUDE_API_KEY is not set. Add it to your .env "
            "(see .env.example) — reflow.py tailors with Claude Sonnet 5."
        )

    gap = parse_result_markdown()
    print(f"Target role: {gap.job_title} | weight profile: {gap.weight_profile}")
    print(f"Missing required skills: {gap.missing_required_skills or 'none'}")
    print(f"Missing preferred skills: {gap.missing_preferred_skills or 'none'}")

    job_description = load_job_description()
    original_resume = load_cached_resume()
    reflow_module = load_reflow_resume_module()
    skills_bank = load_skills_bank()
    if not skills_bank:
        print(
            f"Note: '{SKILLS_BANK_PATH}' is empty — tailoring will only rework "
            "content already in the resume."
        )

    original_resume_text = convert_json_resume_to_text(original_resume)
    print("\nResolving must-have qualifications once, up front (answers are "
          "frozen for every regrade in the loop):")
    frozen_resolver = build_frozen_resolver(
        job_description, gap.weight_profile, original_resume_text, original_resume
    )

    tailor_provider = initialize_llm_provider(TAILOR_MODEL)
    template_manager = TemplateManager()

    best_candidate, best_score, score_history = run_reflow_loop(
        tailor_provider,
        template_manager,
        gap,
        original_resume,
        reflow_module,
        skills_bank,
        job_description,
        frozen_resolver,
    )

    print("\n" + "=" * 60)
    print(f"Score history: {score_history or 'no scored candidates'}")

    band = resolve_band(best_score) if best_candidate is not None else None
    if band is None:
        print(
            f"Best score {best_score if best_candidate else 'n/a'}/100 is below 70. "
            "This job is not a compatible match for the current skills and "
            "experience — no tailored resume written."
        )
        return

    print(f"Best score: {best_score}/100 — band {band}")
    write_tailored_generator(best_candidate, reflow_module)
    print(
        f"Render it with: python {TAILORED_RESUME_PATH} "
        "(layout and render engine are byte-identical to reflow_resume.py)"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Ignore the generated file**

In `.gitignore`, find the line `result.md` (added by the previous feature) and replace with:

```
result.md
resume/resume_reflow/reflow_resume_tailored.py
```

- [ ] **Step 4: Verify the splicer produces a valid, byte-identical-elsewhere file**

Run (no API calls — splices the unmodified content back in and diffs the parts that matter):
```bash
python -c "
import ast
import reflow
module = reflow.load_reflow_resume_module()
candidate = reflow.build_candidate_from_module(module)
reflow.write_tailored_generator(candidate, module)

source = reflow.TAILORED_RESUME_PATH.read_text(encoding='utf-8')
ast.parse(source)  # must be valid Python

import importlib.util
spec = importlib.util.spec_from_file_location('reflow_resume_tailored', reflow.TAILORED_RESUME_PATH)
tailored_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tailored_module)
assert tailored_module.SUMMARY == module.SUMMARY
assert list(tailored_module.SKILLS) == list(module.SKILLS)
assert tailored_module.EXPERIENCE == module.EXPERIENCE
assert tailored_module.PROJECTS == module.PROJECTS
assert tailored_module.NAME == module.NAME and tailored_module.EDUCATION_LINE == module.EDUCATION_LINE
out = tailored_module.build('_splice_check.pdf')
print('splice round-trip OK, rendered', out)
"
python -c "import os; os.remove('_splice_check.pdf'); os.remove('resume/resume_reflow/reflow_resume_tailored.py')"
```
Expected: `splice round-trip OK, rendered _splice_check.pdf` — proving the spliced file parses, exposes semantically identical content, and renders through the untouched engine. (Round-tripping through `repr()` can change quote style/wrapping of the four content literals versus the original source text; "byte-identical" applies to everything **outside** the four replaced blocks, which the splicer never rewrites.)

- [ ] **Step 5: Commit**

```bash
git add reflow.py .gitignore
git commit -m "feat: add score banding, tailored generator output, and CLI entry point to reflow.py"
```

---

### Task 10: End-to-end manual validation

This project has no automated test suite (per `CLAUDE.md`: "There is no test suite. Validation is done manually"). Validate by running the real pipeline. Prerequisites: `.env` has `CLAUDE_API_KEY` plus the usual `LLM_PROVIDER`/`DEFAULT_MODEL` (and `GEMINI_API_KEY` if using Gemini); exactly one resume PDF in `resume/`; a real JD in `job_description.txt`.

**Files:** none (verification only)

- [ ] **Step 1: Produce a fresh `result.md`**

Run `python score.py`.

Expected: **no mode or weight-profile prompt appears at all** (the only possible interaction is the must-have y/n/skip fallback); the run ends with `Report written to result.md`; `result.md` starts with `# Job Match Evaluation:` (or `# Requirement Gate:` if the gate flagged — both shapes are valid input for `reflow.py`).

- [ ] **Step 2: Run the reflow loop**

Run `python reflow.py`.

Expected:
- The gap summary printed first matches `result.md`'s content (target role, missing skills).
- The empty-skills-bank note prints (first run, skeleton untouched).
- Must-have questions (if any) are asked **once**, before iteration 1 — never again mid-loop.
- Log line confirms `Using Anthropic Claude API provider with model claude-sonnet-5` for the tailor calls, while regrade logs show the configured `DEFAULT_MODEL` pipeline (JD extraction served from cache: `Loaded job requirements from cache ...`).
- Each iteration prints `Iteration N weighted total: X/100`; the loop stops at 6 iterations or after 2 consecutive non-improving iterations.
- Final output prints the score history list and either the not-compatible message (best < 70, and **no** `reflow_resume_tailored.py` exists) or the band + `Wrote resume/resume_reflow/reflow_resume_tailored.py`.

- [ ] **Step 3: Inspect the tailored generator diff**

Run: `git diff --no-index resume/resume_reflow/reflow_resume.py resume/resume_reflow/reflow_resume_tailored.py`

Expected: differences confined to the `SUMMARY`, `SKILLS`, `EXPERIENCE`, and `PROJECTS` assignments (wording of the summary/skills/bullets, plus `repr`-formatting of those four literals). Zero diff hunks in the docstring, LAYOUT (LOCKED) block, `NAME`/`CONTACT_PREFIX`/`GH`/`PORTFOLIO`, any `"meta"` value, `EDUCATION_LINE`, `ACTIVITIES_LINE`, or the RENDER ENGINE block.

- [ ] **Step 4: Render the tailored PDF and inspect it visually**

Run: `python resume/resume_reflow/reflow_resume_tailored.py`

Expected: `Wrote Samuel_Darius_Resume.pdf` with no traceback. Open the PDF and check: exactly one page; every bullet occupies at most 2 lines; no skills line runs past the right margin; the ACTIVITIES line sits above the bottom edge; name/contact/links/dates/education/activities are unchanged from the original PDF.

- [ ] **Step 5: Spot-check content guarantees in the file itself**

Run:
Save this as a throwaway script (avoids `$`-interpolation issues in PowerShell), run it with `python _spot_check.py`, then delete it:

```python
from pathlib import Path

source = Path("resume/resume_reflow/reflow_resume_tailored.py").read_text(encoding="utf-8")
dollar = chr(36)
metrics = [
    "1,384ms", "196ms", f"~{dollar}60", f"~{dollar}0.38", "5-10x",
    "32 REST endpoints", "500+ stars", "1,200+", "900+", "300+",
    "PR #283", "PR #822",
]
missing = [metric for metric in metrics if metric not in source]
print("missing fixed metrics:", missing or "none")
content_start = source.index("SUMMARY = (")
content_end = source.index("EDUCATION_LINE = ")
print("em dashes in tailored content:", source[content_start:content_end].count(chr(0x2014)))
```
Expected: `missing fixed metrics: none` and `em dashes in tailored content: 0`. (The docstring above the CONTENT block legitimately doesn't contain em dashes either, but only the CONTENT slice is asserted.)

- [ ] **Step 6: Confirm comparability with a real regrade**

Optional but recommended: temporarily point the pipeline at the tailored content by re-running `python reflow.py` a second time — the second run's iteration-1 score should land near the first run's best score (LLM variance aside), confirming the frozen resolver + caches keep regrades reproducible. Also confirm `git status` shows no unexpected tracked-file changes (`result.md` and `reflow_resume_tailored.py` must both be ignored).

- [ ] **Step 7: Commit any straggling fixes and stop on the branch**

```bash
git status --short
git add -A
git commit -m "chore: post-validation fixes for resume reflow"   # only if anything changed
```

Per repo rules: the feature branch stays open — **no merge to main** without an explicit command.

---

## Risks and edge cases (tracked across tasks)

- **Claude returns malformed JSON or drifts from the schema.** Handled by `extract_json_from_response` + Pydantic parse inside the per-iteration retry loop (Task 7); after 1+3 failed attempts the iteration is discarded and the loop continues from the previous best. A whole run of discarded iterations ends with "not a compatible match" and no file written.
- **Claude Sonnet 5 rejects sampling parameters.** `ClaudeProvider` never forwards `options` (Task 2). If other callers later reuse the provider with the shared evaluator plumbing, their `options` dicts are silently ignored — documented in the provider docstring.
- **`skills_bank.txt` empty on first run.** Detected in `load_skills_bank` (Task 5); the prompt renders `(empty)` and `main()` prints an explicit note; tailoring still runs on resume-only content.
- **All 6 iterations exhausted without reaching 70%.** `resolve_band` returns `None`; the script prints the not-compatible message and writes nothing (Task 9).
- **A `reflow_resume.py` title doesn't match any cached `JSONResume` entry.** `apply_candidate_to_resume` logs a warning and skips that entry instead of crashing (Task 8); the Task 8 Step 3 smoke check surfaces this for the current resume+generator pair before any API spend. If it fires, tighten `_find_work_entry`/`_find_project_entry` for the actual cached names.
- **Layout replay drift.** `_replay_vertical_layout` mirrors `build()` including the inline `16.2` literal; the Task 6 Step 2 check ("current content must validate clean") is the tripwire. Any future edit to `reflow_resume.py`'s render engine requires re-syncing the replay.
- **`result.md` format coupling.** The parser matches `score.py`'s literal headers (accepted design trade-off). Task 4 records the exact strings; if `build_job_evaluation_markdown`/`build_flagged_report_markdown` change, update the `reflow.py` constants in the same commit.
- **Regrade vs. real `score.py` score.** Regrades exclude GitHub-enrichment text (design-literal, Task 8 note), so a post-hoc `score.py` run on the tailored resume may differ modestly — mostly via semantic match. Iteration-to-iteration comparisons are unaffected.
- **Cost/latency per iteration.** Each iteration = 1-4 Claude tailor calls + fresh `DEFAULT_MODEL` recheck/scoring/summary calls + a fresh sentence-transformers load (a few seconds each). JD extraction is always cache-served. Gemini users see the existing per-call spend lines; worst case is 6 iterations × 4 tailor attempts.
- **`reportlab` was an undeclared runtime dependency** of `reflow_resume.py`; Task 2 adds it to `requirements.txt` (called out as a deviation-by-necessity from the spec's dependency list).

## Summary of files touched

- `score.py` — remove `select_mode`, `select_weight_profile`, `build_evaluation_markdown`, `_evaluate_resume`, the mode-1 branch + `resume_evaluations.csv` block, and dead imports; hardcode `weight_profile = "engineering"` on the always-on Custom JD path
- `models.py` — `ModelProvider.CLAUDE`, `ClaudeProvider`
- `prompt.py` — register `claude-sonnet-5` in `MODEL_PARAMETERS`/`MODEL_PROVIDER_MAPPING`, read `CLAUDE_API_KEY`
- `llm_utils.py` — `ModelProvider.CLAUDE` branch in `initialize_llm_provider`
- `requirements.txt` — add `anthropic`, `reportlab`
- `.env.example` — document `CLAUDE_API_KEY`
- `prompts/templates/resume_reflow_system_message.jinja` — new
- `prompts/templates/resume_reflow_user_message.jinja` — new
- `prompts/template_manager.py` — register the two reflow templates
- `resume/resume_reflow/skills_bank.txt` — new categorized skeleton
- `reflow.py` — new orchestrator (parser → loaders → layout validator → tailor+guardrails → remap+regrade loop → banding/output/CLI)
- `.gitignore` — ignore `resume/resume_reflow/reflow_resume_tailored.py`
