---
name: repo-readme-paper-curator
description: Maintain this repository README.md by appending paper entries from one or multiple arXiv links using the existing or newly created category/subcategory structure and strict output format. Use when the user provides one or more arXiv URLs and wants title/date/description/paper/project/code/huggingface links organized into the correct README sections for this repository only.
---

# Repo README Paper Curator

Update only:
`README.md`

Do not create new files unless explicitly requested.

## Required Output Format

Use this exact list style for each paper entry:

```markdown
- **<Paper Title>** (YYYY.MM) \
  **Description**: <1-2 sentence summary based on abstract> \
  [[Paper](<arxiv_abs_url>)]
  [[Project](<project_url>)]            # optional
  [[Code](<github_url>)]                # optional
  [[Hugging Face](<hf_url>)]]           # optional
```

Rules:
- Keep field order exactly: `Paper`, `Project`, `Code`, `Hugging Face`.
- Use the paper title as the display title.
- Use publication month from arXiv as `YYYY.MM`.
- Keep existing repository style (headings and markdown layout) unchanged.
- Always include `Description`, generated from the paper abstract.
- Keep `Description` concise in English, 1-2 sentences, focus on problem and core contribution.
- Always include `Paper`.
- `Project` / `Code` / `Hugging Face` are optional. If unavailable, omit those lines instead of writing `N/A`.

## Markdown Layout Rules

- Use 2 spaces indentation for all lines under one paper bullet item.
- Use `\` at the end of the title line and `Description` line for forced line breaks.
- Keep one blank line between two paper bullet items.
- Do not leave trailing spaces at line ends.

## Workflow

1. Read current `README.md` and identify existing top-level and second-level categories.
2. Normalize input into a list of arXiv links (single or multiple).
3. For each arXiv link, parse and extract:
- Canonical arXiv abstract URL (`https://arxiv.org/abs/<id>`).
- Title from the paper metadata (match the official arXiv title).
- Submission date month for `YYYY.MM`.
4. Generate `Description` from abstract (1-2 concise English sentences).
5. Resolve `Project` URL first, then discover missing links from it.
6. Collect companion links in this order:
- `Project`: official project page (arXiv links first, then author/org pages).
- `Code`: GitHub repository URL (direct from arXiv/project page/footer/nav/buttons).
- `Hugging Face`: model/dataset/space URL (from arXiv/project page/footer/nav/buttons).
7. If `Code` or `Hugging Face` is missing, crawl the `Project` page and one hop of internal links likely named `resources`, `links`, `demo`, `github`, `model`, `weights`, `download`.
8. Choose the best category/subcategory in `README.md`. Create missing category or subcategory when no existing section fits.
9. Insert each entry under that section based on date ordering (newer first).
10. If one of `Project` / `Code` / `Hugging Face` is unavailable after discovery, omit that line and keep only available fields.
11. After all entries are inserted, update `# Contents` once so category directory remains consistent.

## Multi-Link Input Rules

- Accept one or multiple arXiv URLs in one request.
- Process links independently, then apply edits in a single README update.
- De-duplicate by canonical arXiv ID before writing.
- If an entry with the same arXiv ID already exists, skip insertion and keep existing record unless user asks to refresh fields.

## Category Selection Policy (Repository-Specific)

Prioritize existing sections in this README:
- `Attention`
- `LLMs`
- `Multimodal LLMs`
- `Embeddings`
- `Reinforcement Learning`
- `Agents Application`

Heuristic:
- Vision-language / image-grounded reasoning -> `Multimodal LLMs`
- Attention mechanism, KV/cache/architecture internals -> `Attention`
- RLHF/GRPO/PPO/RFT policy learning -> `Reinforcement Learning`
- Agent workflow/tool-use/app integration -> `Agents Application`
- Embedding model/training/retrieval representation -> `Embeddings`
- General LLM modeling/pretraining/inference not fitting above -> `LLMs`

When uncertain, choose the closest existing section and briefly state the assumption in the response.

## Auto Create Category/Subcategory

When no existing section is a good fit:
- Create a new top-level category as `# <Category Name>`.
- Create a new second-level subcategory as `## <Subcategory Name>`.
- Keep naming concise and consistent with current style.
- Insert new categories near semantically similar sections; otherwise append at the end.
- Under new subcategory, append paper entries using the required list format.

Creation rules:
- Reuse existing headings when similarity is high; avoid near-duplicate headings.
- Prefer singular naming style already used in README.
- Do not create more than one new top-level category for a single paper unless user explicitly asks.

## Link Discovery and Validation

Use this source priority:
1. arXiv abstract page metadata and external links
2. official project page
3. official project page one-hop internal links
4. author/org release pages explicitly referenced by 1-3

Validation rules:
- Prefer canonical HTTPS URLs.
- For GitHub, prefer repository root URL over issue/pull/blob paths.
- For Hugging Face, prefer canonical repo root (`https://huggingface.co/<org_or_user>/<repo>`), dataset, or space root.
- Remove tracking query parameters when possible.
- Deduplicate links; keep one best URL per field.

## Editing Constraints

- Modify only `README.md`.
- Preserve existing heading hierarchy.
- Keep additions minimal and preserve existing style.
- Do not rewrite unrelated content.

## Contents Hygiene

- Keep `# Contents` aligned with actual headings in README.
- Add new top-level category links when new categories are created.
- Add second-level links in Contents only when the README currently follows that pattern for the same section.
- Remove stale Contents links that no longer have matching headings.

## Time Ordering Rule

- Treat time as paper attribute in `(YYYY.MM)`.
- Within the same category/subcategory, sort entries by date descending.
- Newer papers must appear above older papers.
- If two papers share the same month, keep existing relative order and place the new one after existing same-month entries.
