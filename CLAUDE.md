# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is a documentation-first bilingual repository containing curated collections of LLM research papers organized by topic. It is also a Quarto website deployed to GitHub Pages. The maintained catalog exists in `README.md`, `README.zh-CN.md`, and generated category pages under `papers/en/` and `papers/zh/`.
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

# Render the site locally
quarto render

# Review only intended content edits
git diff -- README.md README.zh-CN.md index.qmd papers zh

# Check recent commit style
git log --oneline -n 10
```

## Local Tooling Configuration

Before running Python commands, check `.codex/project.local.json` for a `conda_env` value. If it exists, use that environment with `conda run -n <conda_env> python ...` and do not ask for the environment name again. If the file is missing or the value is empty, ask for the Conda environment before executing Python.

The local config file is intentionally git-ignored because it may contain machine-specific settings.

## Project Structure

- `README.md`: Primary Markdown catalog and taxonomy
- `README.zh-CN.md`: Chinese catalog with matching papers and localized descriptions
- `_quarto.yml`: Quarto website configuration and navigation
- `index.qmd`, `papers/en/*.qmd`: English website pages generated from README
- `zh/index.qmd`, `papers/zh/*.qmd`: Chinese website pages generated from README.zh-CN
- `assets/icons/`: Local SVG icons for paper/project/code/model links
- `scripts/check_readme_qmd_sync.py`: Bilingual QMD regeneration and sync checker
- `.github/workflows/quarto-gh-pages.yml`: GitHub Pages artifact deployment workflow
- `LICENSE`: Project license
- `AGENTS.md`: Repository guidelines and contributor instructions
- `.codex/`, `.omc/`, `.omx/`: Local tooling metadata; do not edit unless change is tooling-related

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
- Quarto renders successfully via `quarto render`

## This is a README Curator Repository

This repository uses the `repo-readme-paper-curator` skill for managing paper entries. When adding papers, use that skill for consistent bilingual README placement and Quarto page synchronization.
