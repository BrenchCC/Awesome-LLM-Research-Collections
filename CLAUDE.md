# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is a documentation-first bilingual repository containing curated collections of LLM research papers organized by topic, notes, and curated blogs. It is also a Quarto website deployed to GitHub Pages. The maintained paper catalog exists in `README.md`, `README.zh-CN.md`, generated paper overview pages under `papers/en/index.qmd` and `papers/zh/index.qmd`, and generated category pages under `papers/en/` and `papers/zh/`.
- Attention
- LLMs (Foundation Models, Inference)
- Multimodal LLMs (Vision-Language)
- Embeddings
- Training (SFT)
- Reinforcement Learning (Reasoning RL, Agentic RL, VLA RL)
- Agents Application (Memory)
- Auto-Prompt

## Topic Classification

Keep the `agents` and `reinforcement-learning` categories distinct for both papers and notes:

- `agents` is for concept, system, evaluation, and application analysis. It focuses on how agents are designed or used and normally does not cover training methods.
- `reinforcement-learning` is for reinforcement-learning training, including reasoning RL and Agentic RL. Rollouts, rewards, policy optimization, trajectory data, token-level masks or log probabilities, and credit assignment belong here, including when they are used to train agents.

For material spanning both areas, use the main question as the tie-breaker: agent behavior or applications go under `agents`; reinforcement-learning training of agents goes under `reinforcement-learning`.

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

# Regenerate and verify bilingual notes pages
python scripts/sync_notes.py --write
python scripts/sync_notes.py

# Validate explicitly authorized note downloads
python scripts/check_note_attachments.py

# Regenerate and verify bilingual blog pages
python scripts/sync_blog_shares.py --write
python scripts/sync_blog_shares.py

# Test and locally validate the Feishu Wiki mirror
python -m unittest discover -s tests -v
python scripts/sync_feishu_wiki.py --check

# Render the site locally
quarto render

# Review only intended content edits
git diff -- README.md README.zh-CN.md index.qmd papers zh blogs data assets _quarto.yml styles.css scripts

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

## Feishu Wiki Mirror

GitHub `main` is the only source of truth for the managed private Wiki space `Awesome LLM Research Collections`. The production mirror is already initialized with the selected existing Feishu application. Routine content work must not recreate the space, replace the application, or repeat the one-time membership setup.

### Content and ownership

- Papers come from `README.md` and `README.zh-CN.md`, notes from `notes/en/` and `notes/zh/`, and blogs from `data/blog_shares.json`.
- `scripts/sync_feishu_wiki.py` builds the bilingual Wiki tree, converts QMD constructs and media, rewrites internal links, and tracks content/media hashes and Docx revisions.
- Every managed page starts with a reader-facing description of its subject and purpose. Do not mention synchronization in that introduction; keep source path and source commit in the neutral document-information footer. Manual Feishu body edits may still be overwritten.
- The Wiki homepage stores the versioned ownership manifest and is finalized only after successful writes. Never hand-edit the manifest, infer ownership from matching titles, or remove unknown nodes.
- Duplicate titles, corrupt manifests, moved managed nodes, revision conflicts, and managed deletion targets with unknown children must fail closed.

### Commands and credentials

```bash
# Local parsing and conversion only; no Feishu access
python scripts/sync_feishu_wiki.py --check

# Read-only remote diff
python scripts/sync_feishu_wiki.py --plan

# Serialized incremental update
python scripts/sync_feishu_wiki.py --apply
```

Remote execution requires lark-cli `1.0.86`, `rsvg-convert`, and these environment variables: `LARKSUITE_CLI_APP_ID`, `LARKSUITE_CLI_APP_SECRET`, `LARKSUITE_CLI_BRAND=feishu`, and `FEISHU_WIKI_SPACE_ID`. GitHub Actions stores them as Secrets `FEISHU_APP_ID`, `FEISHU_APP_SECRET`, and Variable `FEISHU_WIKI_SPACE_ID`. Never print or commit credentials, and never place the App Secret in command arguments.

Use `--plan` before a local `--apply`. The complete setup, permissions, recovery, and rotation procedure lives in `docs/feishu-wiki-sync.md`; consult it rather than guessing lark-cli flags.

### Automation and triggering

- `.github/workflows/feishu-wiki-sync.yml` runs `apply` after relevant mirrored-content changes reach `main`.
- UTC cron `0 4 * * *` targets 12:00 Asia/Shanghai; GitHub may start scheduled jobs a few minutes late.
- Manual **Sync Feishu Wiki** runs accept `plan` or `apply`. Prefer a `plan` run, inspect its log, then launch `apply`.
- The equivalent CLI trigger is `gh workflow run feishu-wiki-sync.yml --ref main -f mode=plan`; use `mode=apply` after reviewing the plan.
- Local uncommitted or unpushed changes do not trigger the mirror. Push and scheduled events always use `apply`.
- Writes are intentionally serial and use bounded retries to respect Feishu Docx limits. Always verify the final Actions result and remote manifest status; local tests alone are insufficient.

## Project Structure

- `README.md`: Primary Markdown catalog and taxonomy
- `README.zh-CN.md`: Chinese catalog with matching papers and localized descriptions
- `_quarto.yml`: Quarto website configuration and navigation
- `index.qmd`, `zh/index.qmd`: Generated bilingual homepage entry surfaces for Papers, Notes, and Blogs
- `papers/en/index.qmd`, `papers/zh/index.qmd`: Generated bilingual paper overview pages
- `papers/en/*.qmd`, `papers/zh/*.qmd`: Generated bilingual paper category pages
- `data/blog_shares.json`: Source data for curated blogs
- `blogs/en/index.qmd`, `blogs/zh/index.qmd`: Generated bilingual blog index pages
- `assets/favicon.svg`: Local SVG site icon and visual identity source
- `assets/icons/`: Local SVG icons for paper/project/code/model links
- `styles.css`: Shared Quarto website styling for homepage, overview, notes, blogs, and paper pages
- `scripts/check_readme_qmd_sync.py`: Bilingual QMD regeneration and sync checker
- `scripts/sync_blog_shares.py`: Blogs README section and Quarto page generator/checker
- `scripts/sync_feishu_wiki.py`: Bilingual Feishu Wiki converter and incremental synchronizer
- `tests/test_sync_feishu_wiki.py`: Synchronizer conversion and state-machine tests
- `.github/workflows/feishu-wiki-sync.yml`: Push, manual, and daily Wiki synchronization
- `docs/feishu-wiki-sync.md`: Feishu setup and operations runbook
- `.github/workflows/quarto-gh-pages.yml`: GitHub Pages artifact deployment workflow
- `LICENSE`: Project license
- `AGENTS.md`: Repository guidelines and contributor instructions
- `.codex/`, `.claude/`, `.omc/`, `.omx/`: Local tooling metadata; do not edit unless change is tooling-related

The homepage is generated by `scripts/check_readme_qmd_sync.py` and must remain a three-entry hub for Papers, Notes, and Blogs. Paper category browsing and recent-paper lists belong on `papers/en/index.qmd` and `papers/zh/index.qmd`.

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

## Adding or Updating Notes

Maintain notes as localized pairs under `notes/en/<topic>/<slug>.qmd` and `notes/zh/<topic>/<slug>.qmd`. Keep their structure, claims, formulas, links, figures, conclusions, and shared front matter aligned.

Use two front-matter dates. `date` is the immutable creation date and controls newest-created-first ordering. `date-modified` records the latest source modification. Set both to the current `Asia/Shanghai` date when creating a bilingual note pair; on later edits, preserve `date` and update `date-modified` in both language versions before regenerating Notes indexes and homepage statistics.

TeX/PDF downloads are opt-in and may be enabled only when the user explicitly requests them. File presence is not authorization. Add each requested local attachment to both `resources` and `other-links`, keep normalized paths and order aligned across the bilingual pair, and localize only the link labels. Do not add global resource globs, auto-scan attachment directories, or expose downloads in README/Notes cards. Preserve original attachment bytes unless the user requests a translation, rebuild, copy, or replacement, and retain previously authorized links during unrelated edits unless removal is requested.

The Feishu mirror must create attachment resources only from the same explicit metadata. GitHub Actions validates and packages only those authorized files; neither path may infer downloads from files merely present in the repository.

After editing a note, run:

```bash
python scripts/sync_notes.py --write
python scripts/check_readme_qmd_sync.py --write
python scripts/sync_notes.py
python scripts/check_readme_qmd_sync.py
python scripts/sync_blog_shares.py
python scripts/check_note_attachments.py
quarto render --no-execute
```

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

After editing notes or blog data, rerun `python scripts/check_readme_qmd_sync.py --write` so homepage stats stay current.

## Commit Style

Follow Conventional Commits:
- Preferred types: `docs`, `chore`, `style`
- Example: `docs: add <paper title> to reinforcement learning section`

## Content Validation

The Feishu synchronizer has automated unittests. Catalog and generated-site validation still requires these checks:

- Links are canonical and point to paper/project/code roots
- Paper is placed in the best-matching category/subcategory in both languages
- `# Contents` and `# 目录` match actual headings after edits
- QMD pages match README via `python scripts/check_readme_qmd_sync.py`
- Edited English and Chinese note pairs preserve matching creation `date` values and carry matching current `date-modified` values
- Notes README sections and pages match note front matter via `python scripts/sync_notes.py`
- Blog README sections and pages match `data/blog_shares.json` via `python scripts/sync_blog_shares.py`
- Homepage stats are refreshed after note or blog generator writes
- Both shared README checks pass after any Notes, Blogs, paper catalog, Contents, or generator change
- Feishu tests pass via `python -m unittest discover -s tests -v`
- Local Feishu conversion passes via `python scripts/sync_feishu_wiki.py --check`
- Remote changes are previewed with `--plan` before local `--apply`, and the final GitHub Actions result is inspected
- Quarto renders successfully via `quarto render --no-execute`

## This is a README Curator Repository

This repository uses the `repo-readme-paper-curator` skill for managing paper entries. When adding papers, use that skill for consistent bilingual README placement and Quarto page synchronization.

This repository uses the `repo-blog-share-curator` skill for managing curated blogs. When adding blog posts, use that skill for consistent bilingual blog data, README sections, and Quarto page synchronization.
