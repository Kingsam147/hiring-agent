# Job Requirement Gate — Design

**Date:** 2026-07-05
**Status:** Approved

## Problem

In Custom Job Description mode (`score.py` mode 2), a resume that is missing a
hard requirement (a must-have qualification, or a required skill) still runs
the entire pipeline — GitHub API fetch, semantic embedding model load, full
LLM scoring call, score-summary generation — before the score is merely
capped at 30. This wastes time and LLM cost on resumes that should have been
rejected immediately.

## Goal

Add an early gate: right after the resume is parsed, check the job
description's hard requirements against the resume. If anything required is
missing, print a rejection report listing everything missing and stop —
skipping GitHub enrichment, semantic scoring, and full LLM evaluation
entirely. This only applies to mode 2; mode 1 (HackerRank Intern) has no job
description to gate against and is unaffected.

## Scope of "requirements"

Both of the following count as hard requirements:
- `must_have_qualifications` extracted from the JD
- `required_skills` extracted from the JD

If *any* item from either list is missing, the resume is flagged. The report
lists **every** missing item (not just the first found) before terminating.

## Verification method

Verification happens in three passes, each only running if the previous
pass left something unresolved — cheapest/most-deterministic first:

**Pass 1 — deterministic keyword match.** Reuses the existing
`compute_keyword_match()` — no LLM cost. Produces `matched_required`,
`missing_required`, and `must_have_status` (`found` / `not_found` /
`unverifiable`) exactly as today.

**Pass 2 — LLM semantic recheck (new).** Literal keyword matching
false-negatives on phrasing (e.g. JD says "Bachelor's degree in Computer
Science", resume says "B.S. Computer Science"). Anything Pass 1 could not
confirm — every `missing_required` skill, plus every must-have with status
`not_found` or `unverifiable` — is batched into a **single** LLM call (not
one call per item) that asks the model to judge each one against the resume
text and return a verdict: `met`, `not_met`, or `uncertain`, with a short
reasoning string. This is one cheap classification call, not the full
evaluation pipeline.

Verdicts of `met`/`not_met` are conclusive: the requirement moves to
"kept" or stays in "missing" respectively, and no further check happens
for it. Verdicts of `uncertain` fall through to Pass 3.

**Pass 3 — interactive fallback (existing, extended).** Anything still
unresolved after Pass 2 (i.e. LLM said `uncertain`) is escalated to the
existing `_knockout_resolver` prompt (y = meets it / n = does not / s =
skip) — but now only for must-have qualifications, per your answer.
Required skills that come back `uncertain` from Pass 2 have no interactive
fallback and conservatively stay in "missing" (there's no existing
mechanism to prompt for skills, and defaulting to missing is the safer
failure mode for a hiring gate).

Net effect: a requirement only reaches the user prompt if both the
deterministic match AND the LLM were unable to confirm it — this should
make Pass 3 prompts rarer than they are today, since Pass 2 resolves most
of the borderline cases the old code silently treated as failures.

## Pipeline placement

The gate runs in `score.py:main()` immediately after `resume_data` is
available (whether from cache or fresh PDF extraction), and **before** the
GitHub fetch block. `candidate_name` computation moves up to happen at this
point too, since the rejection report needs it.

Resume text used for the gate check is built from `resume_data` alone
(`convert_json_resume_to_text`), without GitHub enrichment — consistent with
running before GitHub data exists.

## Avoiding duplicate LLM calls / duplicate prompts

`JobDescriptionEvaluator.evaluate()` currently re-derives JD requirements and
keyword match internally every time it's called. To avoid doing this work
(and re-asking the interactive knockout questions) twice when the gate
passes and full evaluation proceeds:

- Add `JobDescriptionEvaluator.check_requirements(resume_text, resume_data,
  knockout_resolver) -> RequirementGateResult`. It runs the three verification
  passes above exactly once — `compute_keyword_match()`, the new LLM semantic
  recheck, then `apply_knockout_resolutions()` for whatever is still
  unresolved — and caches the resulting `job_data` and the corrected
  `keyword_result` on `self`.
- `evaluate()` is updated to reuse `self._job_data` / `self._keyword_result`
  if `check_requirements()` already populated them, instead of
  recomputing/re-prompting.
- The embedding model (`SentenceTransformer`), currently loaded eagerly in
  `__init__`, becomes lazy: loaded on first use inside `evaluate()` rather
  than at construction time. A flagged resume that never reaches `evaluate()`
  never pays that cost.

## New data models

`models.py` gets two new Pydantic models:

```python
class RequirementVerdict(BaseModel):
    requirement: str
    status: Literal["met", "not_met", "uncertain"]
    reasoning: str

class RequirementRecheckResponse(BaseModel):
    verdicts: List[RequirementVerdict]

class RequirementGateResult(BaseModel):
    passed: bool
    job_title: str
    kept_required_skills: List[str]
    missing_required_skills: List[str]
    kept_must_haves: List[str]
    missing_must_haves: List[str]
```

`RequirementVerdict`/`RequirementRecheckResponse` are the structured-output
shape for the Pass 2 LLM call. `passed` on `RequirementGateResult` is `True`
only when both missing lists are empty.

`keyword_matching.py` gets a new `apply_llm_recheck(result, verdicts)`
function (parallel to the existing `apply_knockout_resolutions`) that folds
Pass 2 verdicts into a `KeywordMatchResult`: moving skills between
`matched_required`/`missing_required`, setting `resolved` on the relevant
`must_have_status` entries for `met`/`not_met` verdicts, and recomputing
`coverage_score`/`gated` so the correction is reflected consistently
everywhere `keyword_result` is used downstream (not just the gate).

A new prompt template `prompts/templates/requirement_recheck.jinja` (plus a
short system-message template) sends the resume text and the batched list of
unresolved requirements to the LLM for Pass 2.

## Output on flag — explicit kept vs. added lists

A new `build_flagged_report_markdown(gate_result, candidate_name) -> str` in
`score.py` explicitly lists both sides, not just what's missing:
- Candidate name and JD job title
- **Features kept** — required skills and must-haves the resume already
  satisfies (`kept_required_skills` + `kept_must_haves`)
- **Features to add** — required skills and must-haves missing from the
  resume (`missing_required_skills` + `missing_must_haves`)
- A clear "FLAGGED — does not meet hard requirements" status line

No CSV row is written (per decision — a flagged resume never ran a real
evaluation, so there's nothing meaningful to record in
`job_evaluations.csv`). `main()` writes this report to `result.md` (see
below) and returns immediately.

## Output delivery — written to `result.md`, not the terminal

Per your follow-up, **all** of `score.py`'s report output moves from
`print()` to a single Markdown file, not just the new gate report:

- `print_evaluation_results` (mode 1) and `print_job_evaluation_results`
  (mode 2 full evaluation) are converted to `build_evaluation_markdown(...)
  -> str` and `build_job_evaluation_markdown(...) -> str` respectively —
  same content and structure they print today (scores, evidence, keyword
  match, strengths/improvements), reformatted as Markdown headers/bullets
  instead of ASCII banners.
- `build_flagged_report_markdown(...)` (new, above) covers the gate-rejection
  case.
- `main()` calls whichever of the three applies, then writes the returned
  string to `result.md` in the project root with a single overwrite
  (`Path("result.md").write_text(markdown, encoding="utf-8")`) — each run
  replaces the previous file entirely.
- A one-line console message (`Report written to result.md`) confirms
  completion so the terminal isn't silent; this is the only thing still
  printed.
- The Gemini daily-spend line and interactive prompts (mode/profile
  selection, knockout y/n/skip) still print to the terminal as before — only
  the *report* output moves to the file.

## Files touched

- **`models.py`** — add `RequirementVerdict`, `RequirementRecheckResponse`,
  `RequirementGateResult`.
- **`keyword_matching.py`** — add `apply_llm_recheck()`.
- **`prompts/templates/`** — add `requirement_recheck.jinja` and its system
  message template.
- **`evaluator.py`** — add `check_requirements()` (runs all three
  verification passes); add the Pass 2 LLM call method; make embedding model
  lazy; update `evaluate()` to reuse cached gate results when present.
- **`score.py`** — move `candidate_name` computation earlier; insert the
  gate check before the GitHub fetch block; convert `print_evaluation_results`
  / `print_job_evaluation_results` into `build_*_markdown` functions that
  return strings; add `build_flagged_report_markdown()`; write the resulting
  Markdown to `result.md` (overwrite) at the end of `main()`; construct the
  `JobDescriptionEvaluator` once and pass the same instance through to the
  full-evaluation path so gate results are reused.

## Out of scope

- No change to mode 1 (HackerRank Intern) evaluation *logic* — only its
  output destination changes (file instead of terminal).
- No change to the existing knockout-cap behavior for cases where the gate
  *passes* but a must-have was still borderline (that logic in
  `keyword_matching.py` / `evaluate()` stays as-is).
- No CSV tracking of flagged/rejected resumes (explicitly declined).
- No historical/append log of past `result.md` runs — each run overwrites.
