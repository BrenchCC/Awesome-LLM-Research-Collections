---
name: repo-notes-site-curator
description: Maintain this repository's bilingual notes section by adding or updating paired notes under notes/en and notes/zh, normalizing note front matter, syncing README Notes sections, generating Quarto notes indexes, and validating the website. Use when the user has finished a paper-reading or technical-reflection note and wants it merged into the Awesome LLM Research Collections site.
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

## Asset Rules

- Put shared images under `notes/assets/<slug>/`.
- From `notes/<lang>/<topic>/<slug>.qmd`, reference shared assets with `../../assets/<slug>/<file>`.
- Do not leave per-language duplicate image folders unless the image is language-specific.

## Workflow

1. Read the finished note and identify its topic, slug, language, and note type.
2. Create or update the paired note at the same relative path in the other language.
3. Normalize required front matter in both files.
4. Move shared images to `notes/assets/<slug>/` and update image links.
5. Check `.codex/project.local.json` for `conda_env`; run Python commands through `conda run -n <env> python ...`.
6. Generate README notes sections and notes index pages:

```bash
python scripts/sync_notes.py --write
```

7. Validate generated notes content:

```bash
python scripts/sync_notes.py
```

8. Validate the paper catalog still ignores notes:

```bash
python scripts/check_readme_qmd_sync.py
```

9. Render the website when requested or when navigation/render configuration changed:

```bash
quarto render
```

## Output Expectations

- Keep notes separate from the paper catalog.
- Do not hand-edit generated README Notes sections after running the sync script.
- Mention any missing bilingual pair, missing metadata, or failed render clearly.
