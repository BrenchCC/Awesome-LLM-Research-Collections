# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is a documentation-first repository containing curated collections of LLM research papers organized by topic. The primary artifact is `README.md`, which stores paper entries under categories like:
- Attention
- LLMs (Foundation Models, Inference)
- Multimodal LLMs (Vision-Language)
- Embeddings
- Training (SFT)
- Reinforcement Learning (Reasoning RL, Agentic RL, VLA RL)
- Agents Application (Memory)
- Auto-Prompt

## Development Commands

This repository has no build pipeline. Use lightweight checks before committing:

```bash
# Quick file inventory
rg --files

# Inspect heading and paper-entry structure
rg -n "^#|^##|^- \\*\\*" README.md

# Review only intended edits
git diff -- README.md

# Check recent commit style
git log --oneline -n 10
```

## Project Structure

- `README.md`: Primary content and taxonomy
- `LICENSE`: Project license
- `AGENTS.md`: Repository guidelines and contributor instructions
- `.codex/`, `.omc/`, `.omx/`: Local tooling metadata; do not edit unless change is tooling-related

## Adding Papers

When contributing papers, edit only the relevant section in `README.md` and keep `# Contents` aligned with heading changes. Use the existing paper entry format:

```markdown
- **Paper Title** (YYYY.MM) \
  **Description**: Concise description of the paper's contributions. \
  [[Paper](url)] [[Project](url)] [[Code](url)] [[Hugging Face](url)]
```

Ordering: Newer papers first within each section.

## Commit Style

Follow Conventional Commits:
- Preferred types: `docs`, `chore`, `style`
- Example: `docs: add <paper title> to reinforcement learning section`

## Content Validation

No automated tests. Verify manually:
- Links are canonical and point to paper/project/code roots
- Paper is placed in the best-matching category/subcategory
- `# Contents` matches actual headings after edits

## This is a README Curator Repository

This repository uses the `repo-readme-paper-curator` skill for managing paper entries. When adding papers, use that skill for consistent formatting and placement.