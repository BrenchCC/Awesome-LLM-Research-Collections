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
8. Choose the best category/subcategory in `README.md`, preferring an existing second-level subcategory before falling back to a top-level section. Create a missing category or subcategory when no existing section fits.
9. Insert each entry under the chosen subcategory when available; otherwise under the chosen top-level section, based on date ordering (newer first).
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
- `Training`
- `Reinforcement Learning`
- `Agents Application`
- `Auto-Prompt`

### Core Principle

Do not classify by surface keywords alone.
Classify by the paper's **primary research contribution** in this order:
1. Main task / problem setting
2. Training paradigm or optimization objective
3. Input-output modality
4. System form (agent / workflow / evaluator / tool-use)
5. Model component or architectural focus

Use the section that best matches the paper's **central novelty**, not just the model it happens to use.

### Section Mapping Rules

#### `Attention`
Choose `Attention` when the main contribution is about transformer internals or attention-specific system design, including:
- attention variants or sparse / linear / lightning-style attention
- KV cache, memory compression, routing within attention, long-context attention efficiency
- layer interaction mechanisms that are primarily attention-architecture innovations

Do **not** use `Attention` if attention is only one ingredient inside a broader LLM, MLLM, or training paper.

#### `LLMs`
Choose `LLMs` for general language-model papers where the main contribution is about:
- base LLM architecture, pretraining, post-training, scaling, inference, reasoning, coding, or long-context language modeling
- mixture-of-experts, decoder-only model design, multilingual or general-purpose text foundation models

Use this as the default fallback for text-first foundation model work when no more specific section below is a better fit.

#### `Multimodal LLMs`
Choose `Multimodal LLMs` when the core contribution is multimodal understanding or generation, such as:
- vision-language models, image-grounded reasoning, video-language models, audio-language models
- multimodal alignment, perception-reasoning integration, visual grounding, multimodal long-context modeling
- VLM / MLLM capability papers where multimodal reasoning is the main point

Important boundary:
- If the paper is about **multimodal agents or VLA models**, but the primary novelty is still multimodal perception/reasoning/modeling, keep it in `Multimodal LLMs`.
- If the primary novelty is **RL algorithm design**, move it to `Reinforcement Learning` instead.

#### `Embeddings`
Choose `Embeddings` when the central topic is representation learning for retrieval or semantic matching, including:
- text embeddings, multimodal embeddings, retrieval representations, contrastive representation learning
- embedding training objectives, similarity learning, reranking representations, dense retrieval encoders

Do not use `Embeddings` for general LLM hidden-state analysis unless the deliverable is an actual embedding model or retrieval representation method.

#### `Training`
Choose `Training` when the core novelty is a general training method rather than a model family or RL method, including:
- supervised fine-tuning methodology
- data selection, curriculum learning, token/sample weighting, distillation, optimization recipes
- training efficiency methods that are not specifically attention-centric and not primarily RL

Use `Training` especially when the paper proposes a reusable training recipe applicable across multiple model types.

#### `Reinforcement Learning`
Choose `Reinforcement Learning` when the main contribution is policy optimization or reward-driven learning, including:
- RLHF, RLAIF, GRPO, PPO, DPO-like RL variants when framed as RL or online policy improvement
- reward modeling tightly coupled to RL training
- reasoning RL, agent RL, VLA RL, robotic RL post-training, exploration or credit assignment methods

Important boundary:
- A **vision-language-action** paper is **not automatically** `Multimodal LLMs`.
- If its main novelty is RL formulation, reward construction, trajectory optimization, or policy improvement, classify it as `Reinforcement Learning`.
- If its main novelty is multimodal architecture or perception-reasoning modeling without RL as the central contribution, classify it as `Multimodal LLMs`.

#### `Agents Application`
Choose `Agents Application` when the paper's main contribution is agent systems or tool-use workflows, including:
- autonomous agents, browser/computer-use agents, tool-using LLM systems
- multi-agent orchestration, planning-execution-reflection workflows, application frameworks
- software agents, web agents, productivity agents, embodied agents at the system/workflow level

Important boundary:
- If the work is mostly an **agent application/framework**, use `Agents Application`.
- If it is mostly an **agent training or RL method**, prefer `Reinforcement Learning`.
- If it is mostly a **foundation model capability paper** that includes agent benchmarks, prefer `LLMs` or `Multimodal LLMs`.

#### `Auto-Prompt`
Choose `Auto-Prompt` when the main contribution is automatic prompt construction, prompt optimization, evaluator prompt evolution, or prompt ensemble/search.

### Decision Procedure

For every paper, explicitly run this decision order before inserting:
1. Is the main novelty an **RL / reward / policy optimization** method?
   - Yes -> `Reinforcement Learning`
2. Is the main novelty a **general training recipe** such as SFT/data selection/distillation/curriculum?
   - Yes -> `Training`
3. Is the main novelty **automatic prompt optimization or judge prompting**?
   - Yes -> `Auto-Prompt`
4. Is the main novelty an **agent system / tool-use workflow / orchestration framework**?
   - Yes -> `Agents Application`
5. Is the paper primarily about **embedding / retrieval representations**?
   - Yes -> `Embeddings`
6. Is the paper primarily about **attention-specific architecture or efficiency**?
   - Yes -> `Attention`
7. Is the paper primarily about **multimodal modeling, perception, or multimodal reasoning**?
   - Yes -> `Multimodal LLMs`
8. Otherwise -> `LLMs`


### Subcategory Reuse Policy

After choosing the top-level category, always try to place the paper into the most specific existing second-level subcategory under that category before creating anything new.

Use this subcategory preference order:
1. existing second-level subcategory with clear semantic match
2. newly created second-level subcategory under the chosen top-level category
3. top-level category directly, only if the section intentionally has no subcategory structure

Subcategory matching should use the paper's **primary contribution**, not lexical overlap alone.
Examples:
- a general technical report should prefer `LLMs -> Foundation Models`
- a visual reasoning paper should prefer `Multimodal LLMs -> Vision-Language`
- a VLA policy optimization paper should prefer `Reinforcement Learning -> VLA RL`
- a token weighting SFT paper should prefer `Training -> SFT`

If a top-level category already contains second-level subcategories, prefer inserting into one of them instead of mixing some papers directly under the top-level heading.

### Tie-Breaking Rules

When multiple sections seem plausible, break ties with these priorities:
- **training objective beats modality**
  - Example: a VLA paper centered on policy optimization -> `Reinforcement Learning`, not `Multimodal LLMs`
- **system contribution beats benchmark coverage**
  - Example: a general LLM paper evaluated on agent tasks -> `LLMs`, not `Agents Application`
- **core reusable method beats downstream application domain**
  - Example: a generic data selection method for LLM post-training -> `Training`, not the domain it was tested on
- **architectural primitive beats generic model family** only when the primitive itself is the novelty
  - Example: KV-cache compression method -> `Attention`; otherwise keep broader papers in `LLMs` / `Multimodal LLMs`

### Response Requirement for Ambiguous Cases

If classification is ambiguous, briefly state the applied rule in the response, for example:
- "Classified under Reinforcement Learning because the main novelty is the policy optimization method rather than the multimodal architecture."
- "Classified under Training because the paper's main contribution is a general SFT recipe, not a new foundation model."

### Quick Reference: Paper Type -> Recommended Placement

Use this quick table as a fast default, then verify with the full decision procedure above.

- base text foundation model / technical report -> `LLMs`
- MoE LLM / long-context LLM / reasoning LLM / coding LLM -> `LLMs`
- vision-language model / video-language model / audio-language model -> `Multimodal LLMs`
- multimodal reasoning / grounding / perception-language integration -> `Multimodal LLMs`
- VLA model with architecture or perception as the main novelty -> `Multimodal LLMs` with optional subcategory `## VLA`
- VLA model with RL or reward design as the main novelty -> `Reinforcement Learning` with optional subcategory `## VLA RL`
- RLHF / GRPO / PPO / online policy optimization / credit assignment -> `Reinforcement Learning`
- SFT recipe / data selection / curriculum / distillation / token weighting -> `Training`
- retrieval encoder / embedding model / reranker representation learning -> `Embeddings`
- tool-use agent / browser agent / computer-use / multi-agent workflow -> `Agents Application`
- prompt search / prompt optimization / LLM judge prompt evolution -> `Auto-Prompt`
- attention variant / KV cache / long-context attention efficiency -> `Attention`

### Suggested Reusable Subcategories

Prefer these names when they fit, to keep the repository taxonomy consistent over time:

- under `LLMs`: `## Foundation Models`, `## Reasoning`, `## Coding`, `## Long Context`, `## MoE`, `## Inference`
- under `Multimodal LLMs`: `## Vision-Language`, `## Video`, `## Audio`, `## VLA`
- under `Reinforcement Learning`: `## RLHF`, `## Reasoning RL`, `## Agent RL`, `## VLA RL`
- under `Agents Application`: `## Computer Use`, `## Web Agents`, `## Multi-Agent`, `## Memory`
- under `Training`: `## SFT`, `## Data Selection`, `## Distillation`
- under `Embeddings`: `## Text Embeddings`, `## Multimodal Embeddings`, `## Retrieval`

## Auto Create Category/Subcategory

Creation is allowed by default when existing sections are too coarse or clearly mismatched.
Do **not** force a paper into an existing section if that would make the repository taxonomy less accurate.

### When to Create a New Category

Create a new top-level category when **all** of the following are true:
- no existing top-level section matches the paper's primary contribution well
- placing it into the nearest existing section would be misleading for future browsing
- the new category is likely reusable for future papers, not a one-off label

Examples of valid reasons to create a new top-level category:
- the repository starts accumulating papers on a major area not covered by current sections
- a paper family is repeatedly being misfiled under a too-broad bucket
- the distinction is meaningful at browsing time, such as `Reasoning`, `Inference`, `Robotics`, or `Evaluation`, if these become recurring themes in the repo

### When to Create a New Subcategory

Prefer creating a new **second-level subcategory** before creating a new top-level category when the paper clearly belongs under a broad existing area but needs finer organization.

Examples:
- under `LLMs`: `## Reasoning`, `## Inference`, `## Long Context`, `## MoE`
- under `Multimodal LLMs`: `## Vision-Language`, `## Video`, `## Audio`, `## VLA`
- under `Reinforcement Learning`: `## RLHF`, `## Reasoning RL`, `## VLA RL`
- under `Agents Application`: `## Computer Use`, `## Web Agents`, `## Multi-Agent`

### Creation Rules

When creating new categories or subcategories:
- Reuse existing headings when similarity is high; avoid near-duplicate headings.
- Prefer concise names with broad reuse value.
- Prefer noun phrases over long descriptive headings.
- Prefer singular/plural style consistent with nearby sections.
- Create at most one new top-level category for a single paper unless explicitly requested otherwise.
- If a broad section already exists, prefer adding a subcategory under it instead of creating a sibling top-level section.

### Placement Rules

- Insert a new top-level category near the most semantically similar existing section; otherwise append at the end.
- Insert a new subcategory under the chosen parent section and keep sibling subcategories semantically grouped.
- Under the new section/subsection, append paper entries using the required list format.

### Decision Bias

Use this preference order:
1. existing subcategory
2. new subcategory under an existing top-level category
3. existing top-level category
4. new top-level category

If classification is still fuzzy, prefer creating a reusable subcategory rather than misclassifying the paper into an over-broad bucket.

### Contents Update Requirement

Whenever a new top-level category or second-level subcategory is created, update `# Contents` in the same edit so links stay aligned with the heading structure.

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
