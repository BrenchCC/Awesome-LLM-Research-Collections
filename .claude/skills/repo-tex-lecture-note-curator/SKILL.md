---
name: repo-tex-lecture-note-curator
description: Convert a new local TeX lecture into this repository's faithful bilingual Quarto note pair, preserve or build requested source downloads, and validate the website, GitHub Actions artifact, and Feishu mirror. Use when publishing lecture-style notes from TeX in Awesome-LLM-Research-Collections; do not use for ordinary QMD edits without a TeX source.
---

# Repository TeX Lecture Note Curator

Turn one TeX source into a complete bilingual note pair without silently changing its technical content. Use `repo-notes-site-curator` for the repository's bilingual Notes synchronization rules and `repo-to-feishu-wiki` when the requested delivery includes the Feishu mirror.

Before editing, read [references/workflow.md](references/workflow.md), the repository `AGENTS.md`, and the source TeX completely. Inventory sections, subsections, formulas, cross-references, tables, custom environments, TikZ figures, bibliography entries, and companion files before conversion.

Treat attachment downloads as opt-in. Enable the `resources` plus `other-links` interface only when the user explicitly requests downloads. A file's presence never authorizes publishing it. Do not translate, duplicate, rebuild, or replace a source attachment unless the user requests that operation.

Preserve existing uncommitted work. Do not commit, push, trigger remote Actions, or run Feishu `apply` unless the user explicitly includes those operations. Local conversion and validation do not prove remote delivery.

