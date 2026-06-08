# Repository Guidelines

## Project Structure & Module Organization
This repository is documentation-first, bilingual, and website-enabled. The primary maintained content is the paired paper catalog in `README.md`, `README.zh-CN.md`, the Quarto paper pages under `papers/en/` and `papers/zh/`, the notes section, and the blogs catalog.

- `README.md`: primary Markdown catalog and taxonomy.
- `README.zh-CN.md`: Chinese catalog with the same papers, Chinese descriptions, and localized link labels.
- `_quarto.yml`: Quarto website configuration and navigation.
- `index.qmd`, `papers/en/*.qmd`: generated English website pages.
- `zh/index.qmd`, `papers/zh/*.qmd`: generated Chinese website pages.
- `data/blog_shares.json`: source data for curated blogs.
- `blogs/en/index.qmd`, `blogs/zh/index.qmd`: generated bilingual blogs website pages.
- `assets/icons/`: local SVG resource icons for paper/project/code/model links.
- `scripts/check_readme_qmd_sync.py`: bilingual sync checker and qmd regeneration helper.
- `scripts/sync_blog_shares.py`: blogs README and Quarto page generator/checker.
- `LICENSE`: project license.
- `.codex/`, `.claude/`, `.omc/`, `.omx/`: local tooling metadata; do not edit unless your change is tooling-related.

When contributing papers, update the matching sections in both README files, keep both contents lists aligned with heading changes, then regenerate and verify both language versions of the Quarto pages.

When contributing blogs, update `data/blog_shares.json`, then regenerate and verify the generated README blog sections and `blogs/` index pages.

## Build, Test, and Development Commands
Use lightweight checks before committing:

- `rg --files` - quick file inventory.
- `rg -n "^#|^##|^- \\*\\*" README.md` - inspect heading and paper-entry structure.
- `python scripts/check_readme_qmd_sync.py --write` - regenerate English and Chinese qmd pages from both README files.
- `python scripts/check_readme_qmd_sync.py` - verify all qmd pages match the bilingual README sources.
- `python scripts/sync_blog_shares.py --write` - regenerate README blog sections and bilingual blog index pages.
- `python scripts/sync_blog_shares.py` - verify generated blog content is in sync.
- `quarto render` - render the website into `_site/`.
- `git diff -- README.md README.zh-CN.md index.qmd papers zh blogs data _quarto.yml scripts` - review intended content edits.
- `git log --oneline -n 10` - check recent commit style.

## Local Tooling Configuration
Before running Python commands, check `.codex/project.local.json` for a `conda_env` value. If it exists, use that environment with `conda run -n <conda_env> python ...` and do not ask for the environment name again. If the file is missing or the value is empty, ask for the Conda environment before executing Python.

The local config file is intentionally git-ignored because it may contain machine-specific settings.

## Coding Style & Naming Conventions
Markdown and qmd consistency are the core style requirements.

- Keep heading hierarchy stable (`#` for top sections, `##` for subsections).
- Use existing paper entry format consistently: title, date `(YYYY.MM)`, concise description, and links.
- Keep paper titles in official English wording in both README files.
- Use `**Description**` and English link labels in `README.md`; use `**描述**` and Chinese link labels (`论文`, `项目`, `代码`) in `README.zh-CN.md`.
- Preserve ordering rules within sections (newer papers first unless section policy says otherwise).
- Keep Quarto pages synchronized with both README files by running the sync script after paper edits.
- Keep blogs in `data/blog_shares.json` with exact fields: `slug`, `date`, `title_en`, `title_zh`, `description_en`, `description_zh`, `blog_url`, `github_url`.
- Blogs sort by `date` descending through `scripts/sync_blog_shares.py`; do not hand-edit generated Blogs sections.
- Leave `github_url` empty when a blog has no official linked GitHub repository.
- Keep README command snippets environment-agnostic (`python ...` / `pip ...`), not Conda-specific.
- Avoid unrelated reformatting or whitespace-only churn.

## Testing Guidelines
No automated test suite is configured. Treat review as content validation:

- Verify links are canonical and point to paper/project/code roots.
- Ensure the paper is placed in the best-matching category/subcategory in both languages.
- Confirm both `# Contents` / `# 目录` match actual headings after edits.
- Confirm `python scripts/check_readme_qmd_sync.py` passes.
- Confirm `python scripts/sync_blog_shares.py` passes after blog edits.
- Confirm `quarto render` succeeds before pushing website changes.

GitHub Pages deploys through `.github/workflows/quarto-gh-pages.yml` using GitHub Actions artifacts. Do not commit `_site/` or `.quarto/`.

## Commit & Pull Request Guidelines
Follow Conventional Commits seen in project history:

- Preferred types: `docs`, `chore`, `style`.
- Example: `docs: add <paper title> to reinforcement learning section`.

For PRs, include:

- What changed (sections touched).
- Why the placement is correct (classification rationale).
- Any taxonomy updates (new category/subcategory and bilingual contents updates).
- Whether bilingual Quarto pages were regenerated and rendered successfully.
- For blog changes, whether `scripts/sync_blog_shares.py --write` and the check command were run.

Keep PRs focused and small; one paper batch per PR is preferred.
