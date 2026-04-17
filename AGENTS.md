# Repository Guidelines

## Project Structure & Module Organization
This repository is documentation-first. The main artifact is `README.md`, which stores curated LLM paper entries organized by topic (for example `LLMs`, `Multimodal LLMs`, `Reinforcement Learning`).

- `README.md`: primary content and taxonomy.
- `LICENSE`: project license.
- `.codex/`, `.omc/`, `.omx/`: local tooling metadata; do not edit unless your change is tooling-related.

When contributing papers, edit only the relevant section in `README.md` and keep `# Contents` aligned with heading changes.

## Build, Test, and Development Commands
There is no build pipeline for this repo. Use lightweight checks before committing:

- `rg --files` - quick file inventory.
- `rg -n "^#|^##|^- \\*\\*" README.md` - inspect heading and paper-entry structure.
- `git diff -- README.md` - review only intended edits.
- `git log --oneline -n 10` - check recent commit style.

## Coding Style & Naming Conventions
Markdown consistency is the core style requirement.

- Keep heading hierarchy stable (`#` for top sections, `##` for subsections).
- Use existing paper entry format consistently: title, date `(YYYY.MM)`, concise description, and links.
- Preserve ordering rules within sections (newer papers first unless section policy says otherwise).
- Avoid unrelated reformatting or whitespace-only churn.

## Testing Guidelines
No automated test suite is configured. Treat review as content validation:

- Verify links are canonical and point to paper/project/code roots.
- Ensure the paper is placed in the best-matching category/subcategory.
- Confirm `# Contents` matches actual headings after edits.

## Commit & Pull Request Guidelines
Follow Conventional Commits seen in project history:

- Preferred types: `docs`, `chore`, `style`.
- Example: `docs: add <paper title> to reinforcement learning section`.

For PRs, include:

- What changed (sections touched).
- Why the placement is correct (classification rationale).
- Any taxonomy updates (new category/subcategory and `# Contents` updates).

Keep PRs focused and small; one paper batch per PR is preferred.
