---
name: repo-notes-site-curator
description: Maintain this repository's bilingual notes section by writing, translating, polishing, or updating paired notes under notes/en and notes/zh, normalizing note front matter, syncing README Notes sections, generating Quarto notes indexes, and validating the website. Use when the user wants to add, translate, improve, or publish a paper-reading or technical-reflection note.
---

# Repo Notes Site Curator

Update together:
`notes/en/**/*.qmd`
`notes/zh/**/*.qmd`
`notes/assets/`
`README.md`
`README.zh-CN.md`
`notes/en/index.qmd`
`notes/zh/index.qmd`

Also update `_quarto.yml` only when the notes render paths or navigation structure need to change.

## Note Structure

Use matching bilingual paths:

```text
notes/en/<topic>/<slug>.qmd
notes/zh/<topic>/<slug>.qmd
```

Required front matter:

```yaml
---
author: Brench
date: YYYY-MM-DD
title: "<localized title>"
description: "<localized one-sentence summary>"
order: 1
note_type: paper-reading
topic: <topic>
---
```

Allowed `note_type` values:
- `paper-reading`
- `technical-reflection`

Keep paired English and Chinese notes aligned on `date`, `author`, `order`, `note_type`, and `topic`. Titles and descriptions should be localized.

## Bilingual Writing and Translation

- When adding, translating, or polishing a note in either language, create or update the matching note in the other language in the same task.
- Preserve the same section structure, technical claims, formulas, links, image order, and conclusions across both versions.
- Write idiomatic localized prose instead of translating sentence by sentence. Keep official paper titles, model names, benchmark names, and established technical terms accurate.
- Localize headings, image alt text, callout labels, descriptions, and surrounding explanations.
- Treat a missing or stale paired translation as incomplete work unless the user explicitly requests a single-language draft.

## Asset Rules

- Put shared images under `notes/assets/<slug>/`.
- From `notes/<lang>/<topic>/<slug>.qmd`, reference shared assets with `../../assets/<slug>/<file>`.
- Do not leave per-language duplicate image folders unless the image is language-specific.

## Workflow

1. Read the source note and identify its topic, slug, language, note type, structure, and main claims.
2. Polish the requested language for clear, natural technical writing and Quarto-friendly formatting.
3. Create or update the paired note at the same relative path in the other language, preserving content alignment while writing idiomatic localized prose.
4. Normalize required front matter in both files.
5. Move shared images to `notes/assets/<slug>/` and update image links.
6. Check `.codex/project.local.json` for `conda_env`; run Python commands through `conda run -n <env> python ...`.
7. Generate README notes sections and notes index pages:

```bash
python scripts/sync_notes.py --write
```

8. Validate generated notes content:

```bash
python scripts/sync_notes.py
```

9. Validate the paper catalog still ignores notes:

```bash
python scripts/check_readme_qmd_sync.py
```

10. Render the website when requested or when navigation/render configuration changed:

```bash
quarto render
```

## Output Expectations

- Keep notes separate from the paper catalog.
- Deliver complete English and Chinese note pairs unless the user explicitly requests otherwise.
- Do not hand-edit generated README Notes sections after running the sync script.
- Mention any missing bilingual pair, missing metadata, or failed render clearly.
