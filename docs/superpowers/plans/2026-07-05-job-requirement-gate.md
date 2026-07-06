# Job Requirement Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an early requirement gate to Custom Job Description mode that flags a resume missing hard requirements — using deterministic keyword matching, then an LLM semantic recheck, then an interactive fallback — before the expensive GitHub/embedding/LLM-scoring pipeline runs, and switch all of `score.py`'s report output from terminal printing to a single overwritten `result.md`.

**Architecture:** `JobDescriptionEvaluator` gains a `check_requirements()` method that runs three passes (deterministic match → batched LLM semantic recheck → interactive knockout fallback) and caches its result so the existing `evaluate()` doesn't repeat the work. `score.py` calls the gate right after resume parsing (before GitHub fetch); on failure it writes a rejection report and stops, on success it proceeds to the existing full-evaluation path. All three report builders in `score.py` are converted from `print()` calls to functions that return Markdown strings, written once to `result.md`.

**Tech Stack:** Python, Pydantic, Jinja2 (existing stack — no new dependencies).

**Note on testing:** Per this repo's `CLAUDE.md`, there is no automated test suite — validation is manual, running the pipeline on a real resume with the configured provider. Each task below ends with a manual verification step (an import smoke-check via `python -c`) instead of a pytest run, and the final task is a full end-to-end manual run of `score.py`.

---

### Task 1: Add new data models

**Files:**
- Modify: `models.py:337-340` (insert between `KeywordMatchResult` and `ScoreSummary`)

- [ ] **Step 1: Add `RequirementVerdict`, `RequirementRecheckResponse`, `RequirementGateResult`**

Open `models.py`. Find this exact block (lines 337-340):

```python
    skill_experience: Optional[List[SkillExperience]] = None
    estimated_total_years: Optional[float] = None


class ScoreSummary(BaseModel):
```

Replace it with:

```python
    skill_experience: Optional[List[SkillExperience]] = None
    estimated_total_years: Optional[float] = None


class RequirementVerdict(BaseModel):
    requirement: str
    status: Literal["met", "not_met", "uncertain"]
    reasoning: str


class RequirementRecheckResponse(BaseModel):
    verdicts: List[RequirementVerdict] = []


class RequirementGateResult(BaseModel):
    passed: bool
    job_title: str
    kept_required_skills: List[str] = []
    missing_required_skills: List[str] = []
    kept_must_haves: List[str] = []
    missing_must_haves: List[str] = []


class ScoreSummary(BaseModel):
```

`Literal` and `List` are already imported at the top of `models.py` (line 8) — no import changes needed.

- [ ] **Step 2: Verify the module still imports cleanly**

Run: `python -c "import models; print(models.RequirementGateResult(passed=True, job_title='x'))"`
Expected: prints `passed=True job_title='x' kept_required_skills=[] missing_required_skills=[] kept_must_haves=[] missing_must_haves=[]` with no traceback.

- [ ] **Step 3: Commit**

```bash
git add models.py
git commit -m "feat: add requirement-gate data models"
```

---

### Task 2: Fold LLM semantic-recheck verdicts into keyword matching

**Files:**
- Modify: `keyword_matching.py:11` (import)
- Modify: `keyword_matching.py:146-181` (`apply_knockout_resolutions`)
- Modify: `keyword_matching.py` (insert `apply_llm_recheck` after `apply_knockout_resolutions`, before `build_skills_evidence` at line 184)

- [ ] **Step 1: Import `RequirementVerdict`**

Find this line (line 11):

```python
from models import JSONResume, JobDescriptionData, KeywordMatchResult, MustHaveStatus, IndustryMatch
```

Replace with:

```python
from models import (
    JSONResume,
    JobDescriptionData,
    KeywordMatchResult,
    MustHaveStatus,
    IndustryMatch,
    RequirementVerdict,
)
```

- [ ] **Step 2: Extend `apply_knockout_resolutions` to escalate LLM-uncertain must-haves**

Find this exact block (lines 146-181):

```python
def apply_knockout_resolutions(
    result: KeywordMatchResult, resolver: Optional[Callable[[str], Optional[bool]]]
) -> KeywordMatchResult:
    if resolver is None:
        return result

    updated_status = []
    knockout_failed = False
    for status in result.must_have_status:
        if status.status != "unverifiable":
            updated_status.append(status)
            continue

        answer = resolver(status.qualification)
        if answer is None:
            updated_status.append(status)
            continue

        updated_status.append(
            MustHaveStatus(qualification=status.qualification, status=status.status, resolved=answer)
        )
        if answer is False:
            knockout_failed = True

    coverage_score = result.coverage_score
    gated = result.gated
    if knockout_failed:
        gated = True
        coverage_score = min(coverage_score, GATE_CAP)

    return result.model_copy(update={
        "must_have_status": updated_status,
        "gated": gated,
        "coverage_score": coverage_score,
        "knockout_failed": knockout_failed,
    })
```

Replace with:

```python
def apply_knockout_resolutions(
    result: KeywordMatchResult, resolver: Optional[Callable[[str], Optional[bool]]]
) -> KeywordMatchResult:
    if resolver is None:
        return result

    updated_status = []
    knockout_failed = False
    for status in result.must_have_status:
        if status.status == "found" or status.resolved is not None:
            updated_status.append(status)
            if status.resolved is False:
                knockout_failed = True
            continue

        answer = resolver(status.qualification)
        if answer is None:
            updated_status.append(status)
            continue

        updated_status.append(
            MustHaveStatus(qualification=status.qualification, status=status.status, resolved=answer)
        )
        if answer is False:
            knockout_failed = True

    coverage_score = result.coverage_score
    gated = result.gated
    if knockout_failed:
        gated = True
        coverage_score = min(coverage_score, GATE_CAP)

    return result.model_copy(update={
        "must_have_status": updated_status,
        "gated": gated,
        "coverage_score": coverage_score,
        "knockout_failed": knockout_failed,
    })
```

This changes only the skip condition — from "only `unverifiable` items go to the resolver" to "any item not already resolved (`found`, or an explicit LLM `met`/`not_met` verdict) goes to the resolver" — so a `not_found` must-have that the LLM couldn't confidently classify also gets the interactive y/n/skip fallback, matching the three-pass design. The cap value and the rest of the function's behavior are unchanged from today (the actual 30-point `KNOCKOUT_CAP` on the overall weighted total is still applied separately in `evaluator.py:evaluate()`, as it already was).

- [ ] **Step 3: Add `apply_llm_recheck`**

Find this line (start of `build_skills_evidence`, originally line 184):

```python
def build_skills_evidence(result: KeywordMatchResult) -> str:
```

Insert the following function immediately **before** it:

```python
def apply_llm_recheck(
    result: KeywordMatchResult, verdicts: Dict[str, RequirementVerdict]
) -> KeywordMatchResult:
    if not verdicts:
        return result

    matched_required = list(result.matched_required)
    missing_required = []
    for skill in result.missing_required:
        verdict = verdicts.get(skill)
        if verdict and verdict.status == "met":
            matched_required.append(skill)
        else:
            missing_required.append(skill)

    updated_status = []
    for status in result.must_have_status:
        verdict = verdicts.get(status.qualification)
        if verdict is None or status.status == "found":
            updated_status.append(status)
            continue
        if verdict.status == "met":
            updated_status.append(status.model_copy(update={"resolved": True}))
        elif verdict.status == "not_met":
            updated_status.append(status.model_copy(update={"resolved": False}))
        else:
            updated_status.append(status)

    required_total = len(matched_required) + len(missing_required)
    preferred_total = len(result.matched_preferred) + len(result.missing_preferred)
    if required_total and preferred_total:
        coverage = 100 * (
            REQUIRED_WEIGHT * (len(matched_required) / required_total)
            + PREFERRED_WEIGHT * (len(result.matched_preferred) / preferred_total)
        )
    elif required_total:
        coverage = 100 * (len(matched_required) / required_total)
    elif preferred_total:
        coverage = 100 * (len(result.matched_preferred) / preferred_total)
    else:
        coverage = 50.0

    gated = any(status.status != "found" and status.resolved is not True for status in updated_status)
    if gated:
        coverage = min(coverage, GATE_CAP)

    return result.model_copy(update={
        "matched_required": matched_required,
        "missing_required": missing_required,
        "must_have_status": updated_status,
        "coverage_score": round(coverage, 1),
        "gated": gated,
    })


```

- [ ] **Step 4: Verify the module still imports cleanly**

Run:
```bash
python -c "
from keyword_matching import apply_llm_recheck, apply_knockout_resolutions
from models import KeywordMatchResult, MustHaveStatus, RequirementVerdict

result = KeywordMatchResult(
    matched_required=[], missing_required=['docker'],
    must_have_status=[MustHaveStatus(qualification='security clearance', status='not_found')],
    coverage_score=0.0,
)
verdicts = {
    'docker': RequirementVerdict(requirement='docker', status='met', reasoning='resume mentions containerization with Docker'),
    'security clearance': RequirementVerdict(requirement='security clearance', status='uncertain', reasoning='resume does not mention clearance either way'),
}
updated = apply_llm_recheck(result, verdicts)
print(updated.matched_required, updated.missing_required, updated.must_have_status)
"
```
Expected: `['docker'] [] [MustHaveStatus(qualification='security clearance', status='not_found', resolved=None)]` (the `docker` skill moves to matched; the uncertain must-have is left with `resolved=None` so it will hit the interactive prompt next).

- [ ] **Step 5: Commit**

```bash
git add keyword_matching.py
git commit -m "feat: fold LLM semantic-recheck verdicts into keyword matching"
```

---

### Task 3: Add the semantic-recheck prompt templates

**Files:**
- Create: `prompts/templates/requirement_recheck_system_message.jinja`
- Create: `prompts/templates/requirement_recheck.jinja`
- Modify: `prompts/template_manager.py:37-52`

- [ ] **Step 1: Create the system message template**

Create `prompts/templates/requirement_recheck_system_message.jinja`:

```
You are a meticulous technical recruiter double-checking whether a resume actually satisfies specific job requirements that a deterministic keyword search could not confirm.

For each requirement, decide whether the resume provides clear evidence that it is met, clear evidence that it is not met, or whether the evidence is genuinely ambiguous.

Only answer "met" when the resume text clearly demonstrates the requirement, accounting for reasonable synonyms and phrasing differences (e.g. "B.S. Computer Science" satisfies "Bachelor's degree in Computer Science").
Only answer "not_met" when the resume text clearly contradicts or omits the requirement with no reasonable synonym present.
Answer "uncertain" whenever you cannot confidently decide either way — do not guess.

You MUST respond with ONLY the JSON structure specified in the prompt. Do not add explanatory text.
```

- [ ] **Step 2: Create the recheck prompt template**

Create `prompts/templates/requirement_recheck.jinja`:

```
Re-check the following requirements against the resume below. A deterministic keyword search could not confirm any of them — decide using semantic understanding instead.

## REQUIREMENTS TO RE-CHECK

{% for requirement in requirements %}
- {{ requirement }}
{% endfor %}

## CANDIDATE RESUME

{{ resume_text }}

## INSTRUCTIONS

For every requirement listed above, return exactly one verdict object. Use the requirement text EXACTLY as given above for the "requirement" field so it can be matched back programmatically.

Return ONLY this JSON structure, no other text:

{
    "verdicts": [
        {"requirement": "string", "status": "met", "reasoning": "string"}
    ]
}
```

- [ ] **Step 3: Register both templates**

Open `prompts/template_manager.py`. Find this exact block (lines 37-52):

```python
        template_files = {
            "basics": "basics.jinja",
            "work": "work.jinja",
            "education": "education.jinja",
            "skills": "skills.jinja",
            "projects": "projects.jinja",
            "awards": "awards.jinja",
            "system_message": "system_message.jinja",
            "github_project_selection": "github_project_selection.jinja",
            "resume_evaluation_criteria": "resume_evaluation_criteria.jinja",
            "resume_evaluation_system_message": "resume_evaluation_system_message.jinja",
            "job_description_extraction": "job_description_extraction.jinja",
            "job_evaluation_criteria": "job_evaluation_criteria.jinja",
            "job_evaluation_system_message": "job_evaluation_system_message.jinja",
            "why_this_score": "why_this_score.jinja",
        }
```

Replace with:

```python
        template_files = {
            "basics": "basics.jinja",
            "work": "work.jinja",
            "education": "education.jinja",
            "skills": "skills.jinja",
            "projects": "projects.jinja",
            "awards": "awards.jinja",
            "system_message": "system_message.jinja",
            "github_project_selection": "github_project_selection.jinja",
            "resume_evaluation_criteria": "resume_evaluation_criteria.jinja",
            "resume_evaluation_system_message": "resume_evaluation_system_message.jinja",
            "job_description_extraction": "job_description_extraction.jinja",
            "job_evaluation_criteria": "job_evaluation_criteria.jinja",
            "job_evaluation_system_message": "job_evaluation_system_message.jinja",
            "why_this_score": "why_this_score.jinja",
            "requirement_recheck": "requirement_recheck.jinja",
            "requirement_recheck_system_message": "requirement_recheck_system_message.jinja",
        }
```

- [ ] **Step 4: Verify both templates load and render**

Run:
```bash
python -c "
from prompts.template_manager import TemplateManager
tm = TemplateManager()
print(tm.render_template('requirement_recheck_system_message'))
print('---')
print(tm.render_template('requirement_recheck', requirements=['security clearance'], resume_text='John Doe, Software Engineer'))
"
```
Expected: both templates print rendered text with no `Template not found` or `Error rendering` messages, and the second output contains `security clearance` and `John Doe, Software Engineer`.

- [ ] **Step 5: Commit**

```bash
git add prompts/templates/requirement_recheck_system_message.jinja prompts/templates/requirement_recheck.jinja prompts/template_manager.py
git commit -m "feat: add requirement semantic-recheck prompt templates"
```

---

### Task 4: Add the requirement gate to `JobDescriptionEvaluator`

**Files:**
- Modify: `evaluator.py:1-30` (imports)
- Modify: `evaluator.py:157-215` (`__init__`, `_load_embedding_model`)
- Modify: `evaluator.py:347-354` (start of `evaluate`)
- Modify: `evaluator.py` (insert `_llm_recheck_requirements` and `check_requirements` methods)

- [ ] **Step 1: Add imports**

Find this block (lines 3-24):

```python
from models import (
    JSONResume,
    EvaluationData,
    JobDescriptionData,
    JobScores,
    JobCategoryScore,
    LLMJobEvaluationResponse,
    JobEvaluationData,
    KeywordMatchResult,
    SeniorityAssessment,
    ScoreSummary,
)
from llm_utils import initialize_llm_provider, extract_json_from_response
from keyword_matching import (
    compute_keyword_match,
    build_skills_evidence,
    compute_industry_mentions,
    apply_knockout_resolutions,
    KNOCKOUT_CAP,
)
from seniority import assess_seniority
from weight_profiles import get_profile, DEFAULT_PROFILE
from config import DEVELOPMENT_MODE
```

Replace with:

```python
from models import (
    JSONResume,
    EvaluationData,
    JobDescriptionData,
    JobScores,
    JobCategoryScore,
    LLMJobEvaluationResponse,
    JobEvaluationData,
    KeywordMatchResult,
    SeniorityAssessment,
    ScoreSummary,
    RequirementVerdict,
    RequirementRecheckResponse,
    RequirementGateResult,
)
from llm_utils import initialize_llm_provider, extract_json_from_response
from keyword_matching import (
    compute_keyword_match,
    build_skills_evidence,
    compute_industry_mentions,
    apply_knockout_resolutions,
    apply_llm_recheck,
    KNOCKOUT_CAP,
)
from seniority import assess_seniority
from weight_profiles import get_profile, DEFAULT_PROFILE
from config import DEVELOPMENT_MODE
```

Also find this line (line 42, the summary prompt version constant) and add a recheck prompt version right after it:

```python
# Bump whenever why_this_score.jinja changes materially.
SUMMARY_PROMPT_VERSION = "1"
```

Replace with:

```python
# Bump whenever why_this_score.jinja changes materially.
SUMMARY_PROMPT_VERSION = "1"

# Bump whenever requirement_recheck.jinja or its inputs change materially.
RECHECK_PROMPT_VERSION = "1"
```

- [ ] **Step 2: Make the embedding model lazy and add gate-result caching**

Find this exact block (lines 157-185):

```python
class JobDescriptionEvaluator:
    def __init__(
        self,
        job_description: str,
        model_name: str = DEFAULT_MODEL,
        model_params: dict = None,
        weight_profile: str = DEFAULT_PROFILE,
    ):
        if not job_description or not job_description.strip():
            raise ValueError("Job description cannot be empty")
        if not model_name:
            raise ValueError("Model name cannot be empty")

        self.job_description = job_description
        self.model_name = model_name
        self.model_params = model_params or MODEL_PARAMETERS.get(
            model_name, {"temperature": 0.1, "top_p": 0.9}
        )
        self.weight_profile = weight_profile
        self.weights = get_profile(weight_profile)
        self.template_manager = TemplateManager()
        self.provider = initialize_llm_provider(model_name)
        self._load_embedding_model()

    def _load_embedding_model(self):
        from sentence_transformers import SentenceTransformer
        logger.info("Loading Sentence Transformers model (all-MiniLM-L6-v2)...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
```

Replace with:

```python
class JobDescriptionEvaluator:
    def __init__(
        self,
        job_description: str,
        model_name: str = DEFAULT_MODEL,
        model_params: dict = None,
        weight_profile: str = DEFAULT_PROFILE,
    ):
        if not job_description or not job_description.strip():
            raise ValueError("Job description cannot be empty")
        if not model_name:
            raise ValueError("Model name cannot be empty")

        self.job_description = job_description
        self.model_name = model_name
        self.model_params = model_params or MODEL_PARAMETERS.get(
            model_name, {"temperature": 0.1, "top_p": 0.9}
        )
        self.weight_profile = weight_profile
        self.weights = get_profile(weight_profile)
        self.template_manager = TemplateManager()
        self.provider = initialize_llm_provider(model_name)
        self.embedding_model = None
        self._job_data: Optional[JobDescriptionData] = None
        self._keyword_result: Optional[KeywordMatchResult] = None

    def _load_embedding_model(self):
        from sentence_transformers import SentenceTransformer
        logger.info("Loading Sentence Transformers model (all-MiniLM-L6-v2)...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
```

- [ ] **Step 3: Add `_llm_recheck_requirements` and `check_requirements`**

Find this line (the start of `evaluate`, originally around line 347):

```python
    def evaluate(
        self,
        resume_text: str,
        resume_data: Optional[JSONResume] = None,
        knockout_resolver: Optional[Callable[[str], Optional[bool]]] = None,
    ) -> JobEvaluationData:
        logger.info("Extracting requirements from job description...")
        job_data = self.extract_job_requirements()
        logger.info(f"Job title: {job_data.job_title} | Required skills: {job_data.required_skills}")

        logger.info("Computing deterministic keyword match...")
        keyword_result = compute_keyword_match(job_data, resume_text, resume_data)
        logger.info(
            f"Keyword coverage: {keyword_result.coverage_score} | "
            f"Missing required: {keyword_result.missing_required}"
        )

        keyword_result = apply_knockout_resolutions(keyword_result, knockout_resolver)
        if keyword_result.knockout_failed:
            logger.info("A must-have qualification was rejected by the reviewer — capping score.")

        logger.info("Assessing job-title seniority...")
```

Replace with:

```python
    def _llm_recheck_requirements(
        self, resume_text: str, requirements: List[str]
    ) -> Dict[str, RequirementVerdict]:
        cache_path = (
            f"cache/reqcheckcache_{_hash_key(self.model_name, RECHECK_PROMPT_VERSION, resume_text, '|'.join(requirements))}.json"
        )
        cached = _read_llm_cache(cache_path, RequirementRecheckResponse)
        if cached is None:
            system_message = self.template_manager.render_template("requirement_recheck_system_message")
            if system_message is None:
                raise ValueError("Failed to render requirement_recheck_system_message template")

            prompt = self.template_manager.render_template(
                "requirement_recheck", requirements=requirements, resume_text=resume_text
            )
            if prompt is None:
                raise ValueError("Failed to render requirement_recheck template")

            chat_params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                "options": {
                    "stream": False,
                    "temperature": self.model_params.get("temperature", 0.1),
                    "top_p": self.model_params.get("top_p", 0.9),
                },
            }

            response = self.provider.chat(**chat_params, format=RequirementRecheckResponse.model_json_schema())
            response_text = extract_json_from_response(response["message"]["content"])
            cached = RequirementRecheckResponse(**json.loads(response_text))
            _write_llm_cache(cache_path, cached)

        return {verdict.requirement: verdict for verdict in cached.verdicts}

    def check_requirements(
        self,
        resume_text: str,
        resume_data: Optional[JSONResume] = None,
        knockout_resolver: Optional[Callable[[str], Optional[bool]]] = None,
    ) -> RequirementGateResult:
        logger.info("Extracting requirements from job description...")
        job_data = self.extract_job_requirements()
        logger.info(f"Job title: {job_data.job_title} | Required skills: {job_data.required_skills}")

        logger.info("Computing deterministic keyword match...")
        keyword_result = compute_keyword_match(job_data, resume_text, resume_data)

        recheck_candidates = list(keyword_result.missing_required) + [
            status.qualification
            for status in keyword_result.must_have_status
            if status.status in ("not_found", "unverifiable")
        ]
        if recheck_candidates:
            logger.info(f"Rechecking {len(recheck_candidates)} unresolved requirement(s) with the LLM...")
            verdicts = self._llm_recheck_requirements(resume_text, recheck_candidates)
            keyword_result = apply_llm_recheck(keyword_result, verdicts)

        keyword_result = apply_knockout_resolutions(keyword_result, knockout_resolver)
        if keyword_result.knockout_failed:
            logger.info("A must-have qualification was rejected — capping score.")

        self._job_data = job_data
        self._keyword_result = keyword_result

        missing_must_haves = [
            status.qualification
            for status in keyword_result.must_have_status
            if status.status != "found" and status.resolved is not True
        ]
        kept_must_haves = [
            status.qualification
            for status in keyword_result.must_have_status
            if status.status == "found" or status.resolved is True
        ]

        return RequirementGateResult(
            passed=not keyword_result.missing_required and not missing_must_haves,
            job_title=job_data.job_title,
            kept_required_skills=list(keyword_result.matched_required),
            missing_required_skills=list(keyword_result.missing_required),
            kept_must_haves=kept_must_haves,
            missing_must_haves=missing_must_haves,
        )

    def evaluate(
        self,
        resume_text: str,
        resume_data: Optional[JSONResume] = None,
        knockout_resolver: Optional[Callable[[str], Optional[bool]]] = None,
    ) -> JobEvaluationData:
        if self._job_data is not None and self._keyword_result is not None:
            logger.info("Reusing job requirements and keyword match from check_requirements().")
            job_data = self._job_data
            keyword_result = self._keyword_result
        else:
            logger.info("Extracting requirements from job description...")
            job_data = self.extract_job_requirements()
            logger.info(f"Job title: {job_data.job_title} | Required skills: {job_data.required_skills}")

            logger.info("Computing deterministic keyword match...")
            keyword_result = compute_keyword_match(job_data, resume_text, resume_data)
            keyword_result = apply_knockout_resolutions(keyword_result, knockout_resolver)

        logger.info(
            f"Keyword coverage: {keyword_result.coverage_score} | "
            f"Missing required: {keyword_result.missing_required}"
        )
        if keyword_result.knockout_failed:
            logger.info("A must-have qualification was rejected by the reviewer — capping score.")

        if self.embedding_model is None:
            self._load_embedding_model()

        logger.info("Assessing job-title seniority...")
```

Note: the rest of the original `evaluate` method body (seniority assessment through the final `return result`) is unchanged — this replacement only covers the method's opening lines up through the `logger.info("Assessing job-title seniority...")` line, so everything after it in the file stays exactly as it is today.

- [ ] **Step 4: Verify the module still imports cleanly**

Run: `python -c "import evaluator; print(evaluator.JobDescriptionEvaluator.check_requirements)"`
Expected: prints `<function JobDescriptionEvaluator.check_requirements at 0x...>` with no traceback (this will not load the embedding model or contact any LLM, since construction no longer does either).

- [ ] **Step 5: Commit**

```bash
git add evaluator.py
git commit -m "feat: add JobDescriptionEvaluator.check_requirements gate with lazy embedding load"
```

---

### Task 5: Wire the gate into `score.py` and switch report output to `result.md`

**Files:**
- Modify: `score.py:1-27` (imports)
- Modify: `score.py:109-353` (replace `print_evaluation_results` and `print_job_evaluation_results` with markdown builders; add `build_flagged_report_markdown` and `write_result_markdown`)
- Modify: `score.py:356-424` (remove `_evaluate_with_job_description`, keep `_evaluate_resume` and `_knockout_resolver`)
- Modify: `score.py:426-601` (`main()`)
- Modify: `.gitignore` (ignore `result.md`)

- [ ] **Step 1: Update imports**

Find this block (lines 1-27):

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

RESULT_FILE_PATH = "result.md"
```

- [ ] **Step 2: Replace `print_evaluation_results` with `build_evaluation_markdown`**

Find this exact block (lines 109-222):

```python
def print_evaluation_results(
    evaluation: EvaluationData, candidate_name: str = "Candidate"
):
    print("\n" + "=" * 80)
    print(f"📊 RESUME EVALUATION RESULTS FOR: {candidate_name}")
    print("=" * 80)

    if not evaluation:
        print("❌ No evaluation data available")
        return

    total_score = 0
    max_score = 0

    if hasattr(evaluation, "scores") and evaluation.scores:
        for category_name, category_data in evaluation.scores.model_dump().items():
            category_score = min(category_data["score"], category_data["max"])
            total_score += category_score
            max_score += category_data["max"]

            if category_score < category_data["score"]:
                print(
                    f"⚠️  Warning: {category_name} score capped from {category_data['score']} to {category_score} (max: {category_data['max']})"
                )

    if hasattr(evaluation, "bonus_points") and evaluation.bonus_points:
        total_score += evaluation.bonus_points.total

    if hasattr(evaluation, "deductions") and evaluation.deductions:
        total_score -= evaluation.deductions.total

    max_possible_score = max_score + 20
    if total_score > max_possible_score:
        total_score = max_possible_score
        print(f"⚠️  Warning: Total score capped at maximum possible value")

    print(f"\n🎯 OVERALL SCORE: {total_score:.1f}/{max_score}")

    print("\n📈 DETAILED SCORES:")
    print("-" * 60)

    if hasattr(evaluation, "scores") and evaluation.scores:
        category_maxes = {
            "open_source": 35,
            "self_projects": 30,
            "production": 25,
            "technical_skills": 10,
        }

        if hasattr(evaluation.scores, "open_source") and evaluation.scores.open_source:
            os_score = evaluation.scores.open_source
            capped_score = min(os_score.score, category_maxes["open_source"])
            print(f"🌐 Open Source:          {capped_score}/{os_score.max}")
            print(f"   Evidence: {os_score.evidence}")
            print()

        if (
            hasattr(evaluation.scores, "self_projects")
            and evaluation.scores.self_projects
        ):
            sp_score = evaluation.scores.self_projects
            capped_score = min(sp_score.score, category_maxes["self_projects"])
            print(f"🚀 Self Projects:        {capped_score}/{sp_score.max}")
            print(f"   Evidence: {sp_score.evidence}")
            print()

        if hasattr(evaluation.scores, "production") and evaluation.scores.production:
            prod_score = evaluation.scores.production
            capped_score = min(prod_score.score, category_maxes["production"])
            print(f"🏢 Production Experience: {capped_score}/{prod_score.max}")
            print(f"   Evidence: {prod_score.evidence}")
            print()

        if (
            hasattr(evaluation.scores, "technical_skills")
            and evaluation.scores.technical_skills
        ):
            tech_score = evaluation.scores.technical_skills
            capped_score = min(tech_score.score, category_maxes["technical_skills"])
            print(f"💻 Technical Skills:     {capped_score}/{tech_score.max}")
            print(f"   Evidence: {tech_score.evidence}")
            print()

    if hasattr(evaluation, "bonus_points") and evaluation.bonus_points:
        print(f"\n⭐ BONUS POINTS: {evaluation.bonus_points.total}")
        print("-" * 30)
        print(f"   {evaluation.bonus_points.breakdown}")

    if (
        hasattr(evaluation, "deductions")
        and evaluation.deductions
        and evaluation.deductions.total > 0
    ):
        print(f"\n⚠️  DEDUCTIONS: -{evaluation.deductions.total}")
        print("-" * 30)
        if evaluation.deductions.reasons:
            print(f"   {evaluation.deductions.reasons}")

    if hasattr(evaluation, "key_strengths") and evaluation.key_strengths:
        print(f"\n✅ KEY STRENGTHS:")
        print("-" * 30)
        for i, strength in enumerate(evaluation.key_strengths, 1):
            print(f"  {i}. {strength}")

    if (
        hasattr(evaluation, "areas_for_improvement")
        and evaluation.areas_for_improvement
    ):
        print(f"\n🔧 AREAS FOR IMPROVEMENT:")
        print("-" * 30)
        for i, area in enumerate(evaluation.areas_for_improvement, 1):
            print(f"  {i}. {area}")

    print("\n" + "=" * 80)
```

Replace with:

```python
def build_evaluation_markdown(
    evaluation: EvaluationData, candidate_name: str = "Candidate"
) -> str:
    lines = [f"# Resume Evaluation Results: {candidate_name}"]

    if not evaluation:
        lines.append("\nNo evaluation data available.")
        return "\n".join(lines)

    total_score = 0
    max_score = 0

    if hasattr(evaluation, "scores") and evaluation.scores:
        for category_name, category_data in evaluation.scores.model_dump().items():
            category_score = min(category_data["score"], category_data["max"])
            total_score += category_score
            max_score += category_data["max"]

            if category_score < category_data["score"]:
                lines.append(
                    f"\n> Warning: {category_name} score capped from {category_data['score']} "
                    f"to {category_score} (max: {category_data['max']})"
                )

    if hasattr(evaluation, "bonus_points") and evaluation.bonus_points:
        total_score += evaluation.bonus_points.total

    if hasattr(evaluation, "deductions") and evaluation.deductions:
        total_score -= evaluation.deductions.total

    max_possible_score = max_score + 20
    if total_score > max_possible_score:
        total_score = max_possible_score
        lines.append("\n> Warning: Total score capped at maximum possible value")

    lines.append(f"\n**Overall Score:** {total_score:.1f}/{max_score}")
    lines.append("\n## Detailed Scores")

    if hasattr(evaluation, "scores") and evaluation.scores:
        category_maxes = {
            "open_source": 35,
            "self_projects": 30,
            "production": 25,
            "technical_skills": 10,
        }

        if evaluation.scores.open_source:
            os_score = evaluation.scores.open_source
            capped_score = min(os_score.score, category_maxes["open_source"])
            lines.append(f"\n**Open Source:** {capped_score}/{os_score.max}")
            lines.append(f"- Evidence: {os_score.evidence}")

        if evaluation.scores.self_projects:
            sp_score = evaluation.scores.self_projects
            capped_score = min(sp_score.score, category_maxes["self_projects"])
            lines.append(f"\n**Self Projects:** {capped_score}/{sp_score.max}")
            lines.append(f"- Evidence: {sp_score.evidence}")

        if evaluation.scores.production:
            prod_score = evaluation.scores.production
            capped_score = min(prod_score.score, category_maxes["production"])
            lines.append(f"\n**Production Experience:** {capped_score}/{prod_score.max}")
            lines.append(f"- Evidence: {prod_score.evidence}")

        if evaluation.scores.technical_skills:
            tech_score = evaluation.scores.technical_skills
            capped_score = min(tech_score.score, category_maxes["technical_skills"])
            lines.append(f"\n**Technical Skills:** {capped_score}/{tech_score.max}")
            lines.append(f"- Evidence: {tech_score.evidence}")

    if hasattr(evaluation, "bonus_points") and evaluation.bonus_points:
        lines.append(f"\n## Bonus Points: {evaluation.bonus_points.total}")
        lines.append(f"- {evaluation.bonus_points.breakdown}")

    if (
        hasattr(evaluation, "deductions")
        and evaluation.deductions
        and evaluation.deductions.total > 0
    ):
        lines.append(f"\n## Deductions: -{evaluation.deductions.total}")
        if evaluation.deductions.reasons:
            lines.append(f"- {evaluation.deductions.reasons}")

    if hasattr(evaluation, "key_strengths") and evaluation.key_strengths:
        lines.append("\n## Key Strengths")
        for strength in evaluation.key_strengths:
            lines.append(f"- {strength}")

    if hasattr(evaluation, "areas_for_improvement") and evaluation.areas_for_improvement:
        lines.append("\n## Areas for Improvement")
        for area in evaluation.areas_for_improvement:
            lines.append(f"- {area}")

    return "\n".join(lines)
```

- [ ] **Step 3: Replace `print_job_evaluation_results` with `build_job_evaluation_markdown`**

Find this exact block (lines 225-353):

```python
def print_job_evaluation_results(
    evaluation: JobEvaluationData, candidate_name: str = "Candidate"
):
    print("\n" + "=" * 80)
    print(f"📊 JOB MATCH EVALUATION FOR: {candidate_name}")
    print(f"   Target Role: {evaluation.job_title}")
    print("=" * 80)

    print(f"\n🎯 OVERALL MATCH: {evaluation.weighted_total}/100")
    if evaluation.keyword_match and evaluation.keyword_match.knockout_failed:
        print("   [CAPPED at 30 — reviewer confirmed a must-have is not met]")
    print(f"   Weight profile: {evaluation.weight_profile}")

    if evaluation.score_summary:
        print("\n💬 WHY THIS SCORE:")
        print("-" * 30)
        print(evaluation.score_summary)

    weights = WEIGHT_PROFILES.get(evaluation.weight_profile, WEIGHT_PROFILES[DEFAULT_PROFILE])

    print("\n📈 CATEGORY BREAKDOWN:")
    print("-" * 60)

    categories = [
        (f"💻 Skills Match       ({weights['skills_match']:.0%})", evaluation.scores.skills_match),
        (f"🏢 Experience Match   ({weights['experience_match']:.0%})", evaluation.scores.experience_match),
        (f"📋 Title Alignment    ({weights['job_title_alignment']:.0%})", evaluation.scores.job_title_alignment),
        (f"🎓 Education          ({weights['education']:.0%})", evaluation.scores.education),
        (f"📝 Resume Quality     ({weights['resume_quality']:.0%})", evaluation.scores.resume_quality),
        (f"⚠️  Missing Critical   ({weights['missing_critical_requirements']:.0%})", evaluation.scores.missing_critical_requirements),
    ]

    for label, category in categories:
        print(f"{label}: {category.score:.0f}/100")
        print(f"   Evidence: {category.evidence}")
        if category is evaluation.scores.job_title_alignment and evaluation.seniority:
            seniority = evaluation.seniority
            print(
                f"   Seniority: target={seniority.target_label}, candidate={seniority.candidate_label} "
                f"(gap {seniority.gap:+d})"
            )
        print()

    print(f"🔍 Semantic Match     ({weights['semantic_match']:.0%}): {evaluation.semantic_match_score:.1f}/100")
    print("   Whole-document embedding similarity (all-MiniLM-L6-v2) — supplementary signal.")
    print()

    if evaluation.keyword_match:
        keyword_match = evaluation.keyword_match
        print("🔑 KEYWORD MATCH:")
        print("-" * 30)
        coverage_line = f"Keyword coverage: {keyword_match.coverage_score:.1f}/100"
        if keyword_match.gated:
            coverage_line += " [CAPPED — a must-have qualification was not found]"
        print(coverage_line)

        total_required = len(keyword_match.matched_required) + len(keyword_match.missing_required)
        print(f"\nRequired skills matched ({len(keyword_match.matched_required)}/{total_required}):")
        print(f"  {', '.join(keyword_match.matched_required) if keyword_match.matched_required else 'None'}")
        print(f"Required skills MISSING:")
        print(f"  {', '.join(keyword_match.missing_required) if keyword_match.missing_required else 'None'}")

        total_preferred = len(keyword_match.matched_preferred) + len(keyword_match.missing_preferred)
        if total_preferred:
            print(f"\nPreferred skills matched ({len(keyword_match.matched_preferred)}/{total_preferred}):")
            print(f"  {', '.join(keyword_match.matched_preferred) if keyword_match.matched_preferred else 'None'}")
            print(f"Preferred skills missing:")
            print(f"  {', '.join(keyword_match.missing_preferred) if keyword_match.missing_preferred else 'None'}")

        if keyword_match.must_have_status:
            print("\nMust-have qualifications:")
            status_labels = {
                "found": "found",
                "not_found": "NOT FOUND",
                "unverifiable": "could not be verified by keyword matching",
            }
            for status in keyword_match.must_have_status:
                if status.resolved is True:
                    label = "confirmed by reviewer"
                elif status.resolved is False:
                    label = "REJECTED by reviewer (knockout)"
                else:
                    label = status_labels[status.status]
                print(f"  - {status.qualification}: {label}")

        if keyword_match.skill_experience:
            print("\nSKILL TENURE (deterministic, from work-history dates):")
            for skill_exp in keyword_match.skill_experience:
                if skill_exp.years > 0:
                    print(f"  - {skill_exp.skill}: {skill_exp.years} yrs")
                else:
                    print(f"  - {skill_exp.skill}: no dated evidence")
            if evaluation.jd_years_of_experience is not None and keyword_match.estimated_total_years is not None:
                print(
                    f"  JD asks for {evaluation.jd_years_of_experience} yrs; candidate total "
                    f"~{keyword_match.estimated_total_years} yrs (from parseable work dates)"
                )

        if evaluation.industry_match:
            industry_match = evaluation.industry_match
            if industry_match.mention_count:
                print(f"\nIndustry ({industry_match.industry}): mentioned in {industry_match.mention_count} work entr" +
                      ("y" if industry_match.mention_count == 1 else "ies"))
            else:
                print(f"\nIndustry ({industry_match.industry}): no literal mentions (LLM judges domain fit within Experience Match)")

        suggested_profile = suggest_profile(
            evaluation.job_title, evaluation.industry_match.industry if evaluation.industry_match else None
        )
        if suggested_profile != evaluation.weight_profile:
            print(
                f"\nNote: this JD looks like a '{suggested_profile}' role; consider rerunning with that "
                "weight profile (no extra LLM cost)."
            )
        print()

    if evaluation.key_strengths:
        print("✅ KEY STRENGTHS:")
        print("-" * 30)
        for i, strength in enumerate(evaluation.key_strengths, 1):
            print(f"  {i}. {strength}")

    if evaluation.areas_for_improvement:
        print(f"\n🔧 AREAS FOR IMPROVEMENT:")
        print("-" * 30)
        for i, area in enumerate(evaluation.areas_for_improvement, 1):
            print(f"  {i}. {area}")

    print("\n" + "=" * 80)
```

Replace with:

```python
def build_job_evaluation_markdown(
    evaluation: JobEvaluationData, candidate_name: str = "Candidate"
) -> str:
    lines = [f"# Job Match Evaluation: {candidate_name}"]
    lines.append(f"**Target Role:** {evaluation.job_title}")

    lines.append(f"\n**Overall Match:** {evaluation.weighted_total}/100")
    if evaluation.keyword_match and evaluation.keyword_match.knockout_failed:
        lines.append("> CAPPED at 30 — reviewer confirmed a must-have is not met")
    lines.append(f"**Weight profile:** {evaluation.weight_profile}")

    if evaluation.score_summary:
        lines.append("\n## Why This Score")
        lines.append(evaluation.score_summary)

    weights = WEIGHT_PROFILES.get(evaluation.weight_profile, WEIGHT_PROFILES[DEFAULT_PROFILE])

    lines.append("\n## Category Breakdown")

    categories = [
        (f"Skills Match ({weights['skills_match']:.0%})", evaluation.scores.skills_match),
        (f"Experience Match ({weights['experience_match']:.0%})", evaluation.scores.experience_match),
        (f"Title Alignment ({weights['job_title_alignment']:.0%})", evaluation.scores.job_title_alignment),
        (f"Education ({weights['education']:.0%})", evaluation.scores.education),
        (f"Resume Quality ({weights['resume_quality']:.0%})", evaluation.scores.resume_quality),
        (f"Missing Critical ({weights['missing_critical_requirements']:.0%})", evaluation.scores.missing_critical_requirements),
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
    lines.append("- Whole-document embedding similarity (all-MiniLM-L6-v2) — supplementary signal.")

    if evaluation.keyword_match:
        keyword_match = evaluation.keyword_match
        lines.append("\n## Keyword Match")
        coverage_line = f"Keyword coverage: {keyword_match.coverage_score:.1f}/100"
        if keyword_match.gated:
            coverage_line += " [CAPPED — a must-have qualification was not found]"
        lines.append(coverage_line)

        total_required = len(keyword_match.matched_required) + len(keyword_match.missing_required)
        lines.append(f"\n**Required skills matched ({len(keyword_match.matched_required)}/{total_required}):**")
        lines.append(", ".join(keyword_match.matched_required) if keyword_match.matched_required else "None")
        lines.append("\n**Required skills MISSING:**")
        lines.append(", ".join(keyword_match.missing_required) if keyword_match.missing_required else "None")

        total_preferred = len(keyword_match.matched_preferred) + len(keyword_match.missing_preferred)
        if total_preferred:
            lines.append(f"\n**Preferred skills matched ({len(keyword_match.matched_preferred)}/{total_preferred}):**")
            lines.append(", ".join(keyword_match.matched_preferred) if keyword_match.matched_preferred else "None")
            lines.append("\n**Preferred skills missing:**")
            lines.append(", ".join(keyword_match.missing_preferred) if keyword_match.missing_preferred else "None")

        if keyword_match.must_have_status:
            lines.append("\n**Must-have qualifications:**")
            status_labels = {
                "found": "found",
                "not_found": "NOT FOUND",
                "unverifiable": "could not be verified by keyword matching",
            }
            for status in keyword_match.must_have_status:
                if status.resolved is True:
                    label = "confirmed met"
                elif status.resolved is False:
                    label = "REJECTED (knockout)"
                else:
                    label = status_labels[status.status]
                lines.append(f"- {status.qualification}: {label}")

        if keyword_match.skill_experience:
            lines.append("\n**Skill tenure (deterministic, from work-history dates):**")
            for skill_exp in keyword_match.skill_experience:
                if skill_exp.years > 0:
                    lines.append(f"- {skill_exp.skill}: {skill_exp.years} yrs")
                else:
                    lines.append(f"- {skill_exp.skill}: no dated evidence")
            if evaluation.jd_years_of_experience is not None and keyword_match.estimated_total_years is not None:
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
            evaluation.job_title, evaluation.industry_match.industry if evaluation.industry_match else None
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


def build_flagged_report_markdown(
    gate_result: RequirementGateResult, candidate_name: str = "Candidate"
) -> str:
    lines = [f"# Requirement Gate: {candidate_name}"]
    lines.append(f"**Target Role:** {gate_result.job_title}")
    lines.append("\n**Status:** FLAGGED — does not meet hard requirements")

    lines.append("\n## Features Kept")
    kept = gate_result.kept_required_skills + gate_result.kept_must_haves
    if kept:
        for item in kept:
            lines.append(f"- {item}")
    else:
        lines.append("- None")

    lines.append("\n## Features to Add")
    missing = gate_result.missing_required_skills + gate_result.missing_must_haves
    for item in missing:
        lines.append(f"- {item}")

    return "\n".join(lines)


def write_result_markdown(markdown: str) -> None:
    Path(RESULT_FILE_PATH).write_text(markdown, encoding="utf-8")
    print(f"Report written to {RESULT_FILE_PATH}")
```

- [ ] **Step 4: Remove `_evaluate_with_job_description`**

Find this exact block (originally lines 388-401):

```python
def _evaluate_with_job_description(
    resume_text: str,
    job_description: str,
    resume_data: Optional[JSONResume] = None,
    weight_profile: str = DEFAULT_PROFILE,
) -> Optional[JobEvaluationData]:
    model_params = MODEL_PARAMETERS.get(DEFAULT_MODEL)
    evaluator = JobDescriptionEvaluator(
        job_description=job_description,
        model_name=DEFAULT_MODEL,
        model_params=model_params,
        weight_profile=weight_profile,
    )
    return evaluator.evaluate(resume_text, resume_data=resume_data, knockout_resolver=_knockout_resolver)


def is_valid_resume_data(resume_data: JSONResume) -> bool:
```

Replace with:

```python
def is_valid_resume_data(resume_data: JSONResume) -> bool:
```

(This function is no longer needed — `main()` now builds a `JobDescriptionEvaluator` once and calls `.check_requirements()` then `.evaluate()` on the same instance, so the gate result is reused instead of re-derived.)

- [ ] **Step 5: Restructure `main()`**

Find this exact block (originally lines 426-601, the entire `main` function through end of file):

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

    resume_file_stem = os.path.splitext(os.path.basename(pdf_path))[0]
    cache_filename = f"cache/resumecache_{resume_file_stem}.json"
    github_cache_filename = f"cache/githubcache_{resume_file_stem}.json"

    resume_data = None
    cache_loaded = False

    if DEVELOPMENT_MODE and os.path.exists(cache_filename) and os.path.getmtime(cache_filename) >= os.path.getmtime(pdf_path):
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
                print(f"Failed to delete invalid cache file {cache_filename}: {delete_err}")

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
                print(f"Failed to delete invalid GitHub cache file {github_cache_filename}: {delete_err}")

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

    candidate_name = os.path.splitext(os.path.basename(pdf_path))[0]
    if (
        resume_data
        and hasattr(resume_data, "basics")
        and resume_data.basics
        and resume_data.basics.name
    ):
        candidate_name = resume_data.basics.name

    if mode == 1:
        score = _evaluate_resume(resume_data, github_data)
        print_evaluation_results(score, candidate_name)

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

        job_evaluation = _evaluate_with_job_description(
            resume_text, job_description, resume_data=resume_data, weight_profile=weight_profile
        )
        print_job_evaluation_results(job_evaluation, candidate_name)

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
```

Replace with:

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

    resume_file_stem = os.path.splitext(os.path.basename(pdf_path))[0]
    cache_filename = f"cache/resumecache_{resume_file_stem}.json"
    github_cache_filename = f"cache/githubcache_{resume_file_stem}.json"

    resume_data = None
    cache_loaded = False

    if DEVELOPMENT_MODE and os.path.exists(cache_filename) and os.path.getmtime(cache_filename) >= os.path.getmtime(pdf_path):
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
                print(f"Failed to delete invalid cache file {cache_filename}: {delete_err}")

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
                print(f"Failed to delete invalid GitHub cache file {github_cache_filename}: {delete_err}")

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


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Ignore the generated report file**

Open `.gitignore`. Find this line (line 11):

```
job_evaluations.csv
```

Replace with:

```
job_evaluations.csv
result.md
```

- [ ] **Step 7: Verify the module still imports cleanly**

Run: `python -c "import score; print(score.build_flagged_report_markdown, score.write_result_markdown)"`
Expected: prints both function objects with no traceback.

- [ ] **Step 8: Commit**

```bash
git add score.py .gitignore
git commit -m "feat: gate resumes on missing JD requirements before running the full pipeline, write reports to result.md"
```

---

### Task 6: End-to-end manual verification

This project has no automated test suite (see `CLAUDE.md`) — validate by actually running the pipeline, per existing project convention.

**Files:** none (verification only)

- [ ] **Step 1: Verify mode 1 (HackerRank Intern) still works and writes `result.md`**

Run: `python score.py`, choose mode `1` when prompted.

Expected: no traceback; console ends with `Report written to result.md`; `result.md` exists in the project root and starts with `# Resume Evaluation Results:`.

- [ ] **Step 2: Verify mode 2 flags a resume missing a hard requirement**

Temporarily edit `job_description.txt` to add an obviously-unmet requirement, e.g. append a line: `Must have an active TS/SCI security clearance.` (use a requirement you know the current resume in `resume/` does not mention). Save.

Run: `python score.py`, choose mode `2`, pick any weight profile (or press Enter for default). If prompted with `Must-have could not be auto-verified: "..."`, answer `n`.

Expected: run stops before any `Fetching GitHub data` message ever prints (confirming the gate short-circuits before GitHub enrichment); console ends with `Report written to result.md`; `result.md` starts with `# Requirement Gate:` and its `## Features to Add` section lists the security-clearance requirement (and any other missing required skills).

Revert `job_description.txt` back to its original content afterward (`git checkout -- job_description.txt` if it's tracked, or manually remove the appended line).

- [ ] **Step 3: Verify mode 2 runs the full evaluation when the gate passes**

Run: `python score.py` again, choose mode `2` with the original (reverted) `job_description.txt`, pick a weight profile.

Expected: this time `Fetching GitHub data` (or the GitHub cache-load message) prints, followed by the full evaluation pipeline; console ends with `Report written to result.md`; `result.md` starts with `# Job Match Evaluation:` and contains `## Category Breakdown`, `## Keyword Match`, and (if strengths/gaps were returned) `## Key Strengths` / `## Areas for Improvement` sections.

- [ ] **Step 4: Confirm no stray `print_evaluation_results` / `print_job_evaluation_results` / `_evaluate_with_job_description` references remain**

Run: `python -c "
import ast, pathlib
tree = ast.parse(pathlib.Path('score.py').read_text(encoding='utf-8'))
names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
stale = {'print_evaluation_results', 'print_job_evaluation_results', '_evaluate_with_job_description'} & names
print('stale references:', stale or 'none')
"`
Expected: `stale references: none`.

- [ ] **Step 5: Commit any leftover changes (e.g. a regenerated `result.md` if not gitignored correctly)**

```bash
git status --short
```
Expected: clean (or only untracked `result.md`, which should NOT appear if step 6 of Task 5 correctly added it to `.gitignore` — if it does appear, re-check `.gitignore`). No commit needed for this task unless something unexpected surfaced.

---

## Summary of files touched

- `models.py` — 3 new Pydantic models
- `keyword_matching.py` — 1 new function (`apply_llm_recheck`), 1 modified function (`apply_knockout_resolutions`)
- `prompts/templates/requirement_recheck.jinja` — new
- `prompts/templates/requirement_recheck_system_message.jinja` — new
- `prompts/template_manager.py` — register 2 new templates
- `evaluator.py` — 2 new methods (`_llm_recheck_requirements`, `check_requirements`) on `JobDescriptionEvaluator`, lazy embedding model, `evaluate()` reuses cached gate results
- `score.py` — imports, 2 print functions converted to markdown builders, 1 new markdown builder, 1 new file-writer, `_evaluate_with_job_description` removed, `main()` restructured
- `.gitignore` — ignore `result.md`
