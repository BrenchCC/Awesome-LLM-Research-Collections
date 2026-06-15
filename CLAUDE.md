# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is a documentation-first bilingual repository containing curated collections of LLM research papers organized by topic, notes, and curated blogs. It is also a Quarto website deployed to GitHub Pages. The maintained paper catalog exists in `README.md`, `README.zh-CN.md`, and generated category pages under `papers/en/` and `papers/zh/`.
- Attention
- LLMs (Foundation Models, Inference)
- Multimodal LLMs (Vision-Language)
- Embeddings
- Training (SFT)
- Reinforcement Learning (Reasoning RL, Agentic RL, VLA RL)
- Agents Application (Memory)
- Auto-Prompt

## Development Commands

Use lightweight checks before committing:

```bash
# Quick file inventory
rg --files

# Inspect heading and paper-entry structure
rg -n "^#|^##|^- \\*\\*" README.md

# Regenerate and verify bilingual Quarto pages
python scripts/check_readme_qmd_sync.py --write
python scripts/check_readme_qmd_sync.py

# Regenerate and verify bilingual blog pages
python scripts/sync_blog_shares.py --write
python scripts/sync_blog_shares.py

# Render the site locally
quarto render

# Review only intended content edits
git diff -- README.md README.zh-CN.md index.qmd papers zh blogs data _quarto.yml scripts

# Check recent commit style
git log --oneline -n 10
```

## Local Tooling Configuration

Before running Python commands, check `.codex/project.local.json` for a `conda_env` value. If it exists, use that environment with `conda run -n <conda_env> python ...` and do not ask for the environment name again. If the file is missing or the value is empty, ask for the Conda environment before executing Python.

The local config file is intentionally git-ignored because it may contain machine-specific settings.

## Shared Generated README Safety

`README.md` and `README.zh-CN.md` are shared generated surfaces. Paper, Notes, and Blogs workflows all read or write portions of them.

- Generators must preserve sections and Contents entries owned by other generators.
- Keep `Notes` / `笔记` immediately before `Blogs` / `博客` in `# Contents` / `# 目录`.
- Generator writes must be idempotent: running the same generator twice must produce no second diff.
- After any generator writes either README, run the full CI-equivalent sequence:

```bash
python scripts/check_readme_qmd_sync.py
python scripts/sync_blog_shares.py
quarto render --no-execute
```

If either checker reports README drift, run that generator with `--write`, then restart the full sequence. Do not consider the task complete after validating only the generator that was directly edited.

## Project Structure

- `README.md`: Primary Markdown catalog and taxonomy
- `README.zh-CN.md`: Chinese catalog with matching papers and localized descriptions
- `_quarto.yml`: Quarto website configuration and navigation
- `index.qmd`, `papers/en/*.qmd`: English website pages generated from README
- `zh/index.qmd`, `papers/zh/*.qmd`: Chinese website pages generated from README.zh-CN
- `data/blog_shares.json`: Source data for curated blogs
- `blogs/en/index.qmd`, `blogs/zh/index.qmd`: Generated bilingual blog index pages
- `assets/icons/`: Local SVG icons for paper/project/code/model links
- `scripts/check_readme_qmd_sync.py`: Bilingual QMD regeneration and sync checker
- `scripts/sync_blog_shares.py`: Blogs README section and Quarto page generator/checker
- `.github/workflows/quarto-gh-pages.yml`: GitHub Pages artifact deployment workflow
- `LICENSE`: Project license
- `AGENTS.md`: Repository guidelines and contributor instructions
- `.codex/`, `.claude/`, `.omc/`, `.omx/`: Local tooling metadata; do not edit unless change is tooling-related

## Adding Papers

When contributing papers, edit the relevant section in both `README.md` and `README.zh-CN.md`, keep `# Contents` / `# 目录` aligned with heading changes, then regenerate the Quarto pages. Use the existing English paper entry format:

```markdown
- **Paper Title** (YYYY.MM) \
  **Description**: Concise description of the paper's contributions. \
  [[Paper](url)] [[Project](url)] [[Code](url)] [[Hugging Face](url)]
```

Ordering: Newer papers first within each section.

Use this Chinese entry format in `README.zh-CN.md`:

```markdown
- **Paper Title** (YYYY.MM) \
  **描述**: 中文论文贡献摘要。 \
  [[论文](url)]
  [[项目](url)]
  [[代码](url)]
  [[Hugging Face](url)]
```

Keep paper titles in official English wording in both languages.

After editing README content, run:

```bash
python scripts/check_readme_qmd_sync.py --write
python scripts/check_readme_qmd_sync.py
quarto render
```

Do not commit `_site/` or `.quarto/`.

## Adding Blogs

When adding technical blog posts or essays, edit `data/blog_shares.json` instead of hand-editing generated README sections or `blogs/` pages. Use these exact fields:

```json
{
  "slug": "stable-lowercase-slug",
  "date": "YYYY-MM-DD",
  "title_en": "Official or English title",
  "title_zh": "Localized Chinese title, or official title when better left untranslated",
  "description_en": "One concise English summary.",
  "description_zh": "一句简洁中文简介。",
  "blog_url": "https://example.com/post",
  "github_url": ""
}
```

`blog_url` is required. `github_url` is optional; leave it empty when the blog has no official linked GitHub repository. Generated output sorts by date descending.

After editing blog data, run:

```bash
python scripts/sync_blog_shares.py --write
python scripts/sync_blog_shares.py
quarto render
```

## Commit Style

Follow Conventional Commits:
- Preferred types: `docs`, `chore`, `style`
- Example: `docs: add <paper title> to reinforcement learning section`

## Content Validation

No automated tests. Verify manually:
- Links are canonical and point to paper/project/code roots
- Paper is placed in the best-matching category/subcategory in both languages
- `# Contents` and `# 目录` match actual headings after edits
- QMD pages match README via `python scripts/check_readme_qmd_sync.py`
- Blog README sections and pages match `data/blog_shares.json` via `python scripts/sync_blog_shares.py`
- Both shared README checks pass after any Notes, Blogs, paper catalog, Contents, or generator change
- Quarto renders successfully via `quarto render --no-execute`

## This is a README Curator Repository

This repository uses the `repo-readme-paper-curator` skill for managing paper entries. When adding papers, use that skill for consistent bilingual README placement and Quarto page synchronization.

This repository uses the `repo-blog-share-curator` skill for managing curated blogs. When adding blog posts, use that skill for consistent bilingual blog data, README sections, and Quarto page synchronization.
