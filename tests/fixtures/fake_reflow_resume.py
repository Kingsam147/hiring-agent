# -*- coding: utf-8 -*-
"""
Fake ReFlow Resume Generator (test fixture)
============================================

A structurally faithful but entirely fictitious stand-in for
resume/resume_reflow/reflow_resume.py, used so tests never depend on (or
need to commit) a real person's resume content. The LAYOUT and RENDER
ENGINE blocks below are copied verbatim from the real generator (they are
generic PDF-layout code, not personal data). Only the CONTENT block is
fabricated, and it deliberately includes the same 8 fixed-metric groups
reflow.py's FIXED_METRIC_GROUPS checks for, so guardrail tests exercise
realistic pass/fail behavior.
"""

import sys
from reportlab.pdfgen import canvas
from reportlab.pdfbase.pdfmetrics import stringWidth

# ====================================================================
# LAYOUT (LOCKED)  -  do not edit; these define the template
# ====================================================================
PAGE_W, PAGE_H = 612.0, 792.0  # US Letter
LEFT = 42.0  # section headers + rule start
TEXT_RIGHT = 570.0  # body text wrap edge
META_RIGHT = 558.0  # right-aligned entry meta edge
RULE_RIGHT = 570.0  # section rule end

LINK = (0.203922, 0.313725, 0.419608)  # slate-blue hyperlink color
BLACK = (0, 0, 0)

F_REG, F_BOLD = "Helvetica", "Helvetica-Bold"
SZ_NAME, SZ_CONTACT, SZ_HEAD, SZ_ENTRY, SZ_BODY = 15.0, 8.5, 9.5, 9.0, 8.8

BODY_X = 52.0  # summary text + bullet marker x
BULLET_TEXT_X = 60.0  # text after the bullet / hanging-indent continuation
ENTRY_X = 54.0  # entry titles
LEADING = 12.0  # line height

GAP_HEADER_TO_BODY = 17.8
GAP_HEADER_TO_SKILLS = 16.8
GAP_HEADER_TO_ENTRY_EXP = 17.9
GAP_HEADER_TO_ENTRY_PROJ = 16.9
GAP_SECTION = 17.2
GAP_ENTRY_TITLE_TO_BULLET = 12.9
GAP_ENTRY_TO_ENTRY = 16.0
RULE_BELOW_HEADER = 8.9

# ====================================================================
# CONTENT (EDIT HERE) -- fabricated, not a real person
# ====================================================================
NAME = "TEST CANDIDATE"
CONTACT_PREFIX = "Testville, ZZ · 555-000-0000 · test.candidate@example.com · "
GH = "https://github.com/testcandidate"
PORTFOLIO = "https://testcandidate.example.com/"

SUMMARY = (
    "Fictional backend engineer used only to exercise the reflow test suite. "
    "No real achievements, employers, or metrics below describe an actual person."
)

# (bold label, regular remainder)
SKILLS = [
    ("Languages: ", "Python, Go, SQL"),
    ("Backend: ", "FastAPI, PostgreSQL, Redis"),
]

EXPERIENCE = [
    {
        "title": "Fixture Corp | Software Engineer",
        "meta": [("Jan 2024 – Present", None)],
        "bullets": [
            "Improved load performance 5-10x, reduced response times from 1,384ms to 196ms, and cut cloud costs from ~$60 to ~$0.38 through caching and query tuning.",
            "Designed 32 REST endpoints serving 1,200+ records, 900+ categories, and 300+ tags for the fixture platform.",
        ],
    },
]

PROJECTS = [
    {
        "title": "Fixture Open Source Contribution",
        "meta": [
            ("PR #283", "https://github.com/example/example/pull/283"),
            (" | ", None),
            ("PR #822", "https://github.com/example/example/pull/822"),
        ],
        "bullets": [
            "Contributed to a fictional open source project (500+ stars); PR #283 and PR #822 both merged upstream.",
        ],
    },
]

EDUCATION_LINE = "Fixture University | B.S. Computer Science, Class of 2025"
ACTIVITIES_LINE = "Fixture Club (2023-Present)"


# ====================================================================
# RENDER ENGINE (LOCKED)  -  do not edit
# ====================================================================
def _by(top, size):
    """Canvas baseline y from a top-of-line coordinate measured from page top."""
    return PAGE_H - (top + size * 0.80)


def _wrap(text, font, size, width):
    """Greedy word wrap to a pixel width; returns a list of lines."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = w if not cur else cur + " " + w
        if stringWidth(trial, font, size) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def build(path="fixture_resume.pdf"):
    c = canvas.Canvas(path, pagesize=(PAGE_W, PAGE_H))

    def header(top, label):
        c.setFont(F_BOLD, SZ_HEAD)
        c.setFillColorRGB(*BLACK)
        c.drawString(LEFT, _by(top, SZ_HEAD), label)
        ry = PAGE_H - (top + RULE_BELOW_HEADER)
        c.setLineWidth(0.6)
        c.setStrokeColorRGB(*BLACK)
        c.line(LEFT, ry, RULE_RIGHT, ry)

    def entry(top, title, meta):
        c.setFont(F_BOLD, SZ_ENTRY)
        c.setFillColorRGB(*BLACK)
        c.drawString(ENTRY_X, _by(top, SZ_ENTRY), title)
        mw = sum(stringWidth(t, F_BOLD, SZ_ENTRY) for t, _ in meta)
        mx = META_RIGHT - mw
        yb = _by(top, SZ_ENTRY)
        for t, link in meta:
            c.setFillColorRGB(*(LINK if link else BLACK))
            c.setFont(F_BOLD, SZ_ENTRY)
            c.drawString(mx, yb, t)
            w = stringWidth(t, F_BOLD, SZ_ENTRY)
            if link:
                c.linkURL(link, (mx, yb - 2, mx + w, yb + SZ_ENTRY), relative=0)
            mx += w

    def bullet(top, text):
        c.setFillColorRGB(*BLACK)
        c.setFont(F_REG, SZ_BODY)
        c.drawString(BODY_X, _by(top, SZ_BODY), "•")
        lines = _wrap(text, F_REG, SZ_BODY, TEXT_RIGHT - BULLET_TEXT_X)
        for i, ln in enumerate(lines):
            c.drawString(BULLET_TEXT_X, _by(top + i * LEADING, SZ_BODY), ln)
        return top + len(lines) * LEADING

    def entries(section, gap_header_to_entry):
        nonlocal top
        y = top + gap_header_to_entry
        last = y
        for e in section:
            entry(y, e["title"], e["meta"])
            bt = y + GAP_ENTRY_TITLE_TO_BULLET
            for b in e["bullets"]:
                bt = bullet(bt, b)
            last = bt - LEADING
            y = last + GAP_ENTRY_TO_ENTRY
        top = last + GAP_SECTION

    c.setFont(F_BOLD, SZ_NAME)
    c.setFillColorRGB(*BLACK)
    c.drawCentredString(PAGE_W / 2, _by(45.1, SZ_NAME), NAME)

    cy = _by(62.8, SZ_CONTACT)
    segs = [
        (CONTACT_PREFIX, None),
        ("GitHub", GH),
        (" · ", None),
        ("Portfolio", PORTFOLIO),
    ]
    x = PAGE_W / 2 - sum(stringWidth(t, F_REG, SZ_CONTACT) for t, _ in segs) / 2
    for t, link in segs:
        c.setFillColorRGB(*(LINK if link else BLACK))
        c.setFont(F_REG, SZ_CONTACT)
        c.drawString(x, cy, t)
        w = stringWidth(t, F_REG, SZ_CONTACT)
        if link:
            c.linkURL(link, (x, cy - 2, x + w, cy + SZ_CONTACT), relative=0)
        x += w

    top = 75.0

    header(top, "SUMMARY")
    y = top + GAP_HEADER_TO_BODY
    for ln in _wrap(SUMMARY, F_REG, SZ_BODY, TEXT_RIGHT - BODY_X):
        c.setFillColorRGB(*BLACK)
        c.setFont(F_REG, SZ_BODY)
        c.drawString(BODY_X, _by(y, SZ_BODY), ln)
        last = y
        y += LEADING
    top = last + GAP_SECTION

    header(top, "TECHNICAL SKILLS")
    y = top + GAP_HEADER_TO_SKILLS
    for label, rest in SKILLS:
        c.setFillColorRGB(*BLACK)
        c.setFont(F_BOLD, SZ_BODY)
        c.drawString(BODY_X, _by(y, SZ_BODY), label)
        c.setFont(F_REG, SZ_BODY)
        c.drawString(
            BODY_X + stringWidth(label, F_BOLD, SZ_BODY), _by(y, SZ_BODY), rest
        )
        last = y
        y += LEADING
    top = last + GAP_SECTION

    header(top, "EXPERIENCE")
    entries(EXPERIENCE, GAP_HEADER_TO_ENTRY_EXP)

    header(top, "PROJECTS")
    entries(PROJECTS, GAP_HEADER_TO_ENTRY_PROJ)

    header(top, "EDUCATION")
    y = top + GAP_HEADER_TO_BODY
    c.setFillColorRGB(*BLACK)
    c.setFont(F_REG, SZ_BODY)
    c.drawString(BODY_X, _by(y, SZ_BODY), EDUCATION_LINE)
    top = y + 16.2

    header(top, "ACTIVITIES")
    y = top + GAP_HEADER_TO_SKILLS
    c.setFillColorRGB(*BLACK)
    c.setFont(F_REG, SZ_BODY)
    c.drawString(BODY_X, _by(y, SZ_BODY), ACTIVITIES_LINE)

    c.showPage()
    c.save()
    return path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "fixture_resume.pdf"
    build(out)
    print("Wrote", out)
