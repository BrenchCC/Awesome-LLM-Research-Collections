---
name: repo-note-xhs-recommender
description: Turn an existing Quarto note in this repository into a concise, fact-checked Chinese XHS/RedNote recommendation delivered as copy-ready plain text. Use when the user asks to introduce, share, or recommend a file under notes/ on 小红书, XHS, or RedNote without changing repository content.
---

# Repo Note XHS Recommender

Create a natural Chinese recommendation from an existing repository note. Treat the task as read-only unless the user explicitly requests file changes.

## Workflow

1. Resolve the note path supplied by the user and read the complete file, including front matter, captions, conclusions, and references relevant to its claims.
2. Extract a compact fact sheet:
   - Exact note title and central question
   - One or two strongest quantitative findings
   - Main topics covered
   - Most important conclusion or caveat
   - Intended readers
   - Canonical published URL. Prefer a user-supplied link; otherwise read `website.site-url` from `_quarto.yml` and map the note's repository-relative `.qmd` path to `.html`
3. Check the scope and attribution of every claim. Distinguish tasks from trajectories, trials from unique samples, ranges from percentage-point changes, source experiments from the note author's analysis, and stability from correctness.
4. Draft the copy in this order:
   - A one-line hook
   - A short introduction naming the note and its central question
   - The strongest evidence in one or two compact paragraphs
   - The note's coverage and core judgment
   - A sentence identifying suitable readers
   - The full article link
5. Re-read the source and verify every number, unit, model count, comparison, attribution, title, and link before answering.

## Writing Rules

- Return only copy-ready Chinese text unless the user asks for explanation or alternatives. Do not add a preface, Markdown headings, bullets, or code fences.
- Default to 250–450 Chinese characters in four to six short paragraphs.
- Keep the tone conversational, technically precise, and restrained. Prefer a concrete tension or question over exaggerated promotion.
- Use short paragraphs that are easy to read on mobile.
- Treat any user-provided example as a tone and structure reference, not as the factual source.
- Preserve official English names for papers, models, benchmarks, and established technical terms when useful.
- Attribute third-party experiments as “原文实验” or “研究者”, rather than implying that the note author ran them.
- State limitations that materially change the interpretation of a striking number.
- Do not invent facts, broaden conclusions, or silently resolve inconsistent evidence.
- Do not add emoji, hashtags, engagement bait, or Markdown headings by default.

## Read-Only Boundary

Do not edit notes, README files, indexes, assets, generators, or site configuration. Do not run sync or render commands and do not create images. Text output completes the task unless the user explicitly requests publication or repository changes.
