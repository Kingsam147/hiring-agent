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

Reuses the existing deterministic `compute_keyword_match()` — no LLM cost.

For must-have qualifications too long/free-form to keyword-match
(`status="unverifiable"`), fall back to the existing interactive
`_knockout_resolver` prompt (y = meets it / n = does not / s = skip),
identical to today's knockout UX:
- `n` → counts as missing (flag)
- `y` or skip → does not count as missing

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
  knockout_resolver) -> RequirementGateResult`. It calls
  `extract_job_requirements()` and `compute_keyword_match()` +
  `apply_knockout_resolutions()` once, and caches the resulting `job_data`
  and `keyword_result` on `self`.
- `evaluate()` is updated to reuse `self._job_data` / `self._keyword_result`
  if `check_requirements()` already populated them, instead of
  recomputing/re-prompting.
- The embedding model (`SentenceTransformer`), currently loaded eagerly in
  `__init__`, becomes lazy: loaded on first use inside `evaluate()` rather
  than at construction time. A flagged resume that never reaches `evaluate()`
  never pays that cost.

## New data model

`models.py` gets a new Pydantic model:

```python
class RequirementGateResult(BaseModel):
    passed: bool
    job_title: str
    missing_required_skills: List[str]
    missing_must_haves: List[str]
```

`passed` is `True` only when both lists are empty.

## Output on flag

A new `print_flagged_report(gate_result, candidate_name)` in `score.py`
prints:
- Candidate name
- JD job title
- Missing required skills (list)
- Missing must-have qualifications (list)
- A clear "FLAGGED — does not meet hard requirements" status line

No CSV row is written (per decision — a flagged resume never ran a real
evaluation, so there's nothing meaningful to record in
`job_evaluations.csv`). `main()` returns immediately after printing.

## Files touched

- **`models.py`** — add `RequirementGateResult`.
- **`evaluator.py`** — add `check_requirements()`; make embedding model lazy;
  update `evaluate()` to reuse cached gate results when present.
- **`score.py`** — move `candidate_name` computation earlier; insert the
  gate check before the GitHub fetch block; add `print_flagged_report()`;
  construct the `JobDescriptionEvaluator` once and pass the same instance
  through to the full-evaluation path so gate results are reused.

## Out of scope

- No change to mode 1 (HackerRank Intern) behavior.
- No change to the existing knockout-cap behavior for cases where the gate
  *passes* but a must-have was still borderline (that logic in
  `keyword_matching.py` / `evaluate()` stays as-is).
- No CSV tracking of flagged/rejected resumes (explicitly declined).
