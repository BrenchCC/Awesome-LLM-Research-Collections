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
`index.qmd`
`zh/index.qmd`
`notes/en/index.qmd`
`notes/zh/index.qmd`

Also update `_quarto.yml` only when the notes render paths or navigation structure need to change.

The generated homepage reads note counts and latest-note dates. Refresh it after notes generator writes.

## Shared README Stability

`README.md` and `README.zh-CN.md` are shared generated surfaces maintained by the paper, notes, and blogs workflows.

- Keep `# Contents` / `# 目录` ordering stable. Notes entries must remain immediately before Blogs entries.
- When changing a generator, ensure repeated runs are idempotent and do not reorder headings owned by another generator.
- After any command writes either README, run every shared README checker in CI order:

```bash
python scripts/sync_notes.py
python scripts/check_readme_qmd_sync.py
python scripts/sync_blog_shares.py
```

- If a checker reports README drift, run its `--write` command, then rerun all shared README checkers from the beginning.

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
date-modified: YYYY-MM-DD
title: "<localized title>"
description: "<localized one-sentence summary>"
order: 1
note_type: paper-reading
topic: <topic>
tags:
  - <tag>
---
```

Allowed `note_type` values:
- `paper-reading`
- `technical-reflection`

Treat `date` as the note's creation date. Set it when creating the bilingual note pair, keep it unchanged during later edits, and use it to order notes from newest creation to oldest creation.

Treat `date-modified` as the latest source-modification date. Set it equal to `date` for a new note; whenever either language version is later edited, update `date-modified` in both paired notes to the current `Asia/Shanghai` calendar date before running any generator. Refresh it after changing prose, structure, figures, formulas, links, or front matter.

Keep paired English and Chinese notes aligned on `date`, `date-modified`, `author`, `order`, `note_type`, `topic`, and `tags`. Titles and descriptions should be localized.

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
- Preserve high-resolution source images. Do not downsample an asset only to make it appear smaller on the rendered page.
- Control final layout in `.qmd` image references with Quarto `width` and `fig-align` attributes, for example: `![Alt text](../../assets/<slug>/<file>){width="90%" fig-align="center"}`.
- Center Mermaid diagrams explicitly with the Quarto cell option `%%| fig-align: center`; do not rely on the browser's default alignment. Keep this option aligned across bilingual notes, then inspect the rendered SVG to confirm it is horizontally centered.
- Choose display width according to the content and rendered legibility. As starting ranges, use about `70%`–`80%` for titles or compact text crops and `85%`–`95%` for wide figures or tables. Keep corresponding image order, `width`, and `fig-align` attributes aligned across bilingual notes.
- Treat file-size optimization and display sizing as separate concerns. When a source asset is unnecessarily large, prefer lossless optimization before considering any resolution reduction.

## Workflow

1. Read the source note and identify its topic, slug, language, note type, structure, and main claims.
2. Polish the requested language for clear, natural technical writing and Quarto-friendly formatting.
3. Create or update the paired note at the same relative path in the other language, preserving content alignment while writing idiomatic localized prose.
4. Normalize required front matter in both files. Preserve the creation `date`, update both `date-modified` fields to the current `Asia/Shanghai` date, and verify that both dates match across the pair.
5. Move shared images to `notes/assets/<slug>/` and update image links.
6. Check `.codex/project.local.json` for `conda_env`; run Python commands through `conda run -n <env> python ...`.
7. Generate README notes sections and notes index pages:

```bash
python scripts/sync_notes.py --write
```

8. Refresh generated homepage statistics:

```bash
python scripts/check_readme_qmd_sync.py --write
```

9. Validate generated notes content:

```bash
python scripts/sync_notes.py
```

10. Validate the paper catalog still ignores notes while keeping homepage stats current:

```bash
python scripts/check_readme_qmd_sync.py
```

11. Validate that the Blogs generator still accepts the shared README files:

```bash
python scripts/sync_blog_shares.py
```

12. Render the website when requested; when navigation/render configuration changed; or after adding or modifying images, SVGs, `width`, or `fig-align` attributes:

```bash
quarto render --no-execute
```

Inspect the rendered pages for image overflow, undersized content, blur, unwanted horizontal scrolling, and Mermaid diagrams that are not horizontally centered. Fix presentation by adjusting the `.qmd` layout options first instead of shrinking the high-resolution source image.

## Output Expectations

- Keep notes separate from the paper catalog.
- Deliver complete English and Chinese note pairs unless the user explicitly requests otherwise.
- Confirm that every edited note pair preserves its creation `date` and carries the current `date-modified` value before generating indexes.
- Confirm that bilingual image order and Quarto layout attributes match, and that the rendered image layout has been inspected.
- Do not hand-edit generated README Notes sections after running the sync script.
- Mention any missing bilingual pair, missing metadata, or failed render clearly.
