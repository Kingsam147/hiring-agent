# Resume Reflow — Design

**Date:** 2026-07-06
**Status:** Approved

## Problem

`score.py` grades a resume against a job description but never acts on the
gap it finds. Closing that gap today means manually rewriting
`resume/resume_reflow/reflow_resume.py` by hand, re-running `score.py`, and
repeating until the score is acceptable — with no guardrail against
accidentally changing the fixed layout, inventing an unearned skill, or
altering one of the immutable metrics baked into the resume's bullets.

## Goal

A new standalone script, `reflow.py`, that:

1. Reads the most recent job-match gap analysis (`result.md`), the canonical
   resume generator (`resume/resume_reflow/reflow_resume.py`), and a bank of
   additional truthful skills/projects (`resume/resume_reflow/skills_bank.txt`).
2. Uses Claude Sonnet 5 to rewrite only the resume's wording (summary,
   skills, bullets) to close the gap — never the layout, fonts, spacing,
   fixed metrics, or em-dash rule already enforced by
   `reflow_resume.py`'s own editing rules.
3. Regrades each candidate with the existing `JobDescriptionEvaluator`
   pipeline (same score `score.py` would produce) and repeats until the
   score plateaus or a cap is hit.
4. Reports the highest score band reached, or tells the user the job isn't a
   compatible match if it can't clear 70% using only truthful content.

## Score bands

The regrade score is `JobEvaluationData.weighted_total` — the same "Overall
Match X/100" `score.py` already prints. Reported bands:

- `< 70%` — not compatible; exit without writing an output file.
- `70% ≤ x < 80%`
- `80% ≤ x < 90%`
- `≥ 90%`

Bands are reporting labels only, not stopping thresholds — the loop always
pushes for the highest score it can truthfully reach within its iteration
budget, not just the first band crossed.

## Prerequisite change: `score.py` simplification

Per direction received while scoping this feature, `score.py` drops its two
interactive prompts and Mode 1 entirely:

- `select_mode()` and the Mode 1 (HackerRank Intern) branch are removed.
  `main()` always runs the Custom Job Description path.
- `select_weight_profile()` is removed. `weight_profile` is hardcoded to
  `"engineering"` (already `weight_profiles.DEFAULT_PROFILE`).
- Now-dead code removed from `score.py`: `_evaluate_resume`,
  `build_evaluation_markdown`, the mode-1 CSV-writing branch, and the
  imports only mode 1 used (`ResumeEvaluator`, `EvaluationData`,
  `transform_evaluation_response`). `ResumeEvaluator`/`EvaluationData`
  themselves are left in `evaluator.py`/`models.py` — only the `score.py`
  entry point changes.
- Effect for `reflow.py`: `result.md` is now always the job-match format
  (`build_job_evaluation_markdown` or `build_flagged_report_markdown`), so
  `reflow.py` only ever needs to parse one shape.

## `reflow.py` pipeline

### One-time setup

1. Load `job_description.txt` (required — same file `score.py` reads).
2. Parse `result.md` for job title, weight profile, and the baseline gap
   (`**Required skills MISSING:**`, `**Preferred skills missing:**`,
   `## Areas for Improvement` from the full report; `## Features to Add`
   from the flagged-gate report). Exit with a clear message if `result.md`
   is missing or doesn't look like a Mode 2 report — the user must run
   `score.py` first.
3. Load the cached original resume, `cache/resumecache_<name>.json`
   (`JSONResume`), and the current CONTENT block of
   `resume/resume_reflow/reflow_resume.py` (`SUMMARY`, `SKILLS`,
   `EXPERIENCE`, `PROJECTS`).
4. Load `resume/resume_reflow/skills_bank.txt` — categorized truthful
   skills/projects not necessarily reflected in the current resume wording.
5. Construct one `JobDescriptionEvaluator` and call `check_requirements()`
   **once**, on the original (untailored) resume text, with the existing
   interactive `_knockout_resolver`. This is the only point in the whole run
   where the user is prompted. Every answer given is captured into a dict
   and wrapped in a frozen resolver function:
   `frozen_resolver(qualification) -> dict.get(qualification)` — reused for
   every later regrade so a candidate's real, unchanging qualifications are
   never re-asked about mid-loop.

### Loop (max 6 iterations; stop early after 2 consecutive iterations with no score improvement)

For each iteration:

1. **Tailor.** Call Claude Sonnet 5 (new `resume_reflow_system_message.jinja`
   / `resume_reflow_user_message.jinja` templates) with: the current best
   CONTENT block, `skills_bank.txt`, the gap list, and the hard constraints
   below. Structured JSON output covers only `summary`, `skills` (list of
   `{label, rest}`), `experience` (list of `{title, bullets}`), `projects`
   (list of `{title, bullets}`) — `title` values must match the original
   entries verbatim (same count of entries and bullets per entry; no adding
   or removing whole jobs/projects).

   Hard constraints given to the model every call:
   - Preserve the 8 fixed metrics verbatim (1,384ms→196ms, ~$60→~$0.38,
     5-10x, 32 endpoints, 500+ stars, 1,200+/900+/300+, PR #283, PR #822).
   - No em dashes anywhere.
   - Only use skills/projects that appear in the current resume or
     `skills_bank.txt` — never fabricate.
   - `NAME`, `CONTACT_PREFIX`, links, entry `meta` (dates/links),
     `EDUCATION_LINE`, `ACTIVITIES_LINE` are out of scope — untouched.

2. **Validate.** Before this candidate is scored:
   - Em-dash scan across all returned text.
   - Fixed-metric scan — each of the 8 metric substrings must still be
     present verbatim in the relevant bullet.
   - Layout-fit check, reusing `reflow_resume.py`'s own `_wrap()` function
     and gap constants directly (imported, not reimplemented) so the check
     always matches the real renderer: every bullet must wrap to ≤ 2 lines
     at the actual font/width, and replaying `build()`'s vertical-position
     accumulation must land within the 792pt page height.

   Any failure triggers a targeted re-prompt naming the specific offending
   field (up to 3 retries for that iteration); if still failing, the
   iteration's candidate is discarded and the loop moves on using the
   previous best.

3. **Remap onto `JSONResume`.** Deep-copy the cached original `JSONResume`
   and apply the validated candidate:
   - `basics.summary` ← tailored summary.
   - `skills` ← rebuilt from tailored `{label, rest}` pairs
     (`Skill(name=label, keywords=[...rest.split(",")])`).
   - `work[].highlights` ← tailored experience bullets, matched to the
     original `work` entries by parsing `title` as `"{name} | {position}"`.
   - `projects[].highlights` ← tailored project bullets, matched by
     `title` → `name`.
   - Everything else (contact info, dates, education, company names) is
     copied through from the original.

4. **Regrade.** Build resume text via `convert_json_resume_to_text()` on the
   remapped `JSONResume` and score it with a **fresh**
   `JobDescriptionEvaluator` instance (`_job_data`/`_keyword_result` start
   `None`, so `evaluate()` recomputes `keyword_result` for this iteration's
   text instead of reusing stale data) using the frozen knockout resolver.
   `extract_job_requirements()` still hits its on-disk cache, so this does
   not repeat the JD-extraction LLM call — only the resume-dependent scoring
   call runs fresh each iteration, exactly as it would in a real `score.py`
   run against this text.
5. Track the best-scoring candidate seen across all iterations (not just the
   last one — Claude's output can vary call to call, so a later iteration
   could technically regress).

### Outcome

- Best score `< 70%`: print that the job isn't a compatible match with the
  current skills/experience; do not write an output file.
- Best score `≥ 70%`: print the final band, the score achieved, and the
  score history across iterations; write
  `resume/resume_reflow/reflow_resume_tailored.py` — a full copy of
  `reflow_resume.py` with only `SUMMARY`, `SKILLS`, and `EXPERIENCE`/
  `PROJECTS` bullets replaced with the best candidate's content. Running
  `python reflow_resume_tailored.py` renders the tailored PDF with the
  identical, unmodified layout/render engine.

## Claude Sonnet 5 integration

Follows the existing `LLMProvider` Protocol pattern exactly (per
`CLAUDE.md`'s documented extension process):

- `models.py` — add `ModelProvider.CLAUDE = "claude"`; add `ClaudeProvider`
  (uses the `anthropic` SDK, reads `CLAUDE_API_KEY`, translates the
  system/user message shape into the Anthropic Messages API, returns
  `{"message": {"role": "assistant", "content": text}}` to match the
  existing provider return shape).
- `prompt.py` — register `"claude-sonnet-5"` in `MODEL_PROVIDER_MAPPING` and
  `MODEL_PARAMETERS`; read `CLAUDE_API_KEY` from the environment.
- `llm_utils.py` — add the `ModelProvider.CLAUDE` branch in
  `initialize_llm_provider`.
- `requirements.txt` — add `anthropic`.
- `.env.example` — document `CLAUDE_API_KEY`.

`reflow.py` calls `initialize_llm_provider("claude-sonnet-5")` directly — it
does not read `DEFAULT_MODEL`. Claude is used only for the tailoring calls;
regrading still uses whatever `DEFAULT_MODEL` the existing pipeline is
configured with (Ollama or Gemini), so the score `reflow.py` reports stays
directly comparable to what a real `score.py` run would produce.

## New files

- `reflow.py` (project root) — the orchestrator, run via `python reflow.py`.
- `resume/resume_reflow/skills_bank.txt` — categorized skeleton matching the
  existing `SKILLS` categories (Languages, Backend, Cloud & DevOps,
  Architecture & Security) plus a `Projects` section, left for the user to
  fill in.
- `prompts/templates/resume_reflow_system_message.jinja` and
  `resume_reflow_user_message.jinja`, registered in
  `TemplateManager._load_templates()` per the existing convention.

## Files touched

- **`score.py`** — remove `select_mode`, `select_weight_profile`, the Mode 1
  branch, and now-dead mode-1-only code/imports; hardcode
  `weight_profile = "engineering"` and the Custom Job Description path.
- **`models.py`** — add `ModelProvider.CLAUDE`, `ClaudeProvider`.
- **`prompt.py`** — register the Claude model/provider mapping and API key.
- **`llm_utils.py`** — add the Claude branch to `initialize_llm_provider`.
- **`requirements.txt`**, **`.env.example`** — Claude SDK dependency and key.
- **`prompts/template_manager.py`** — register the two new reflow templates.

## Out of scope

- No change to `reflow_resume.py`'s LAYOUT or RENDER ENGINE blocks — read
  and imported from, never edited.
- No change to `NAME`, `CONTACT_PREFIX`, links, entry `meta`,
  `EDUCATION_LINE`, or `ACTIVITIES_LINE` — tailoring only ever touches
  summary/skills/bullets.
- No fabrication guardrail beyond prompting + the fixed-metric/em-dash scan
  — there is no automated proof that every word Claude writes traces back to
  the resume or skills bank; the user is expected to review the tailored
  output before sending it, same as any AI-assisted rewrite.
- No multi-page support — `reflow_resume.py` renders a single page today,
  and `reflow.py`'s layout-fit check enforces staying within that same
  single page.

## Known trade-off (accepted)

`reflow.py` parses `result.md` by matching the literal Markdown headers
`build_job_evaluation_markdown`/`build_flagged_report_markdown` produce,
rather than reading a structured sidecar file. This keeps `score.py`
unchanged beyond the Mode 1/prompt removal above, at the cost of the parser
needing an update if those two functions' output format changes later.
