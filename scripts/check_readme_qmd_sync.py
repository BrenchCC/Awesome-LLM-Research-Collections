import re
import html
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


README_TITLE_EN = "Awesome-LLM-Research-Collections"
README_TITLE_ZH = "Awesome LLM 研究论文合集"

CATEGORY_SLUGS = {
    "Attention": "attention",
    "注意力机制": "attention",
    "LLMs": "llms",
    "大语言模型": "llms",
    "Multimodal LLMs": "multimodal-llms",
    "多模态大模型": "multimodal-llms",
    "Embeddings": "embeddings",
    "嵌入模型": "embeddings",
    "Training": "training",
    "训练": "training",
    "Reinforcement Learning": "reinforcement-learning",
    "强化学习": "reinforcement-learning",
    "Agents Application": "agents-application",
    "智能体应用": "agents-application",
    "Vision": "vision",
    "视觉": "vision",
    "Auto-Prompt": "auto-prompt",
    "自动提示": "auto-prompt",
    "Attention Architecture": "attention-architecture",
    "注意力架构": "attention-architecture",
    "Foundation Models": "foundation-models",
    "基础模型": "foundation-models",
    "Inference": "inference",
    "推理": "inference",
    "Detection": "detection",
    "检测": "detection",
    "Vision-Language": "vision-language",
    "视觉语言": "vision-language",
    "Multimodal Reasoning": "multimodal-reasoning",
    "多模态推理": "multimodal-reasoning",
    "VLA": "vla",
    "视觉-语言-动作": "vla",
    "SFT": "sft",
    "监督微调": "sft",
    "SFT Methods": "sft-methods",
    "监督微调方法": "sft-methods",
    "Optimization": "optimization",
    "优化": "optimization",
    "Distillation": "distillation",
    "蒸馏": "distillation",
    "Policy Optimization": "policy-optimization",
    "策略优化": "policy-optimization",
    "Reward Modeling": "reward-modeling",
    "奖励建模": "reward-modeling",
    "Video Generation RL": "video-generation-rl",
    "视频生成强化学习": "video-generation-rl",
    "Reasoning RL": "reasoning-rl",
    "推理强化学习": "reasoning-rl",
    "Agentic RL": "agentic-rl",
    "智能体强化学习": "agentic-rl",
    "VLA RL": "vla-rl",
    "视觉-语言-动作强化学习": "vla-rl",
    "AI Research": "ai-research",
    "AI 研究": "ai-research",
    "Tool Use": "tool-use",
    "工具调用": "tool-use",
    "Agent Skills": "agent-skills",
    "智能体技能": "agent-skills",
    "Agent Development": "agent-development",
    "智能体开发": "agent-development",
    "Memory": "memory",
    "记忆": "memory",
    "Object Detection": "object-detection",
    "目标检测": "object-detection",
    "Prompt Optimization": "prompt-optimization",
    "提示优化": "prompt-optimization",
    "Judge Prompting": "judge-prompting",
    "评测器提示": "judge-prompting",
    "Featured": "featured",
    "精选": "featured",
}

LINK_ICONS = {
    "Paper": "arxiv.svg",
    "论文": "arxiv.svg",
    "Project": "project.svg",
    "项目": "project.svg",
    "Code": "github.svg",
    "代码": "github.svg",
    "Hugging Face": "huggingface.svg",
}

PAPER_LABELS = {"Paper", "论文"}


@dataclass
class Link:
    label: str
    url: str


@dataclass
class Paper:
    title: str
    date: str
    description: str
    links: List[Link]
    category: str
    subcategory: Optional[str]


@dataclass
class LanguageConfig:
    key: str
    readme_path: str
    index_path: str
    papers_dir: str
    site_title: str
    site_description: str
    description_label: str
    skipped_headings: List[str]
    category_descriptions: Dict[str, str]
    labels: Dict[str, str]
    asset_prefix: str


LABELS_EN = {
    "home_eyebrow": "Curated LLM research map",
    "home_lede": "A browsable Quarto edition of the repository README, organized for fast scanning across papers, projects, code, and model resources.",
    "papers": "Papers",
    "categories": "Categories",
    "resource_links": "Resource links",
    "latest_month": "Latest month",
    "browse": "Browse",
    "research_categories": "Research Categories",
    "recent_eyebrow": "Fresh index",
    "recent_papers": "Recent Papers",
    "research_category": "Research category",
    "subcategories": "Subcategories",
    "direct_collection": "Direct collection",
    "no_papers": "No papers yet",
    "empty_note": "New papers will appear here after curation.",
    "empty_state": "No papers have been curated in this category yet.",
    "featured": "Featured",
    "paper_singular": "paper",
    "paper_plural": "papers",
    "language_href": "zh/index.html",
    "language_label": "中文",
}

LABELS_ZH = {
    "home_eyebrow": "LLM 研究地图",
    "home_lede": "LLM 研究论文、项目、代码与模型资源精选合集。",
    "papers": "论文",
    "categories": "分类",
    "resource_links": "资源链接",
    "latest_month": "最新月份",
    "browse": "浏览",
    "research_categories": "研究分类",
    "recent_eyebrow": "最新索引",
    "recent_papers": "近期论文",
    "research_category": "研究分类",
    "subcategories": "子分类",
    "direct_collection": "直接收录",
    "no_papers": "暂无论文",
    "empty_note": "新论文会在整理后显示在这里。",
    "empty_state": "该分类暂未收录论文。",
    "featured": "精选",
    "paper_singular": "篇论文",
    "paper_plural": "篇论文",
    "language_href": "../index.html",
    "language_label": "English",
}

CATEGORY_DESCRIPTIONS_EN = {
    "Attention": "Transformer internals, attention variants, KV/cache behavior, and depth-wise information flow.",
    "LLMs": "Foundation model reports, inference methods, long-context language modeling, coding, and reasoning systems.",
    "Multimodal LLMs": "Vision-language, video-language, and VLA research that connects perception with language reasoning.",
    "Embeddings": "Representation learning, retrieval, semantic matching, and embedding model research.",
    "Training": "Reusable training recipes, SFT methods, data selection, distillation, and optimization practice.",
    "SFT": "Supervised fine-tuning methods, data recipes, token weighting, and reasoning generalization studies.",
    "Reinforcement Learning": "Reward modeling, RLHF-style optimization, reasoning RL, agent RL, and VLA policy learning.",
    "Agents Application": "Agent systems, tool use, memory, AI research workflows, and reusable skill ecosystems.",
    "Vision": "Computer vision methods that are useful background for modern multimodal systems.",
    "Auto-Prompt": "Prompt optimization, evaluator prompting, prompt ensembles, and test-time prompt learning.",
}

CATEGORY_DESCRIPTIONS_ZH = {
    "注意力机制": "围绕 Transformer 内部机制、注意力变体、KV 缓存行为和跨层信息流的研究。",
    "大语言模型": "基础模型报告、推理方法、长上下文语言建模、代码与推理系统研究。",
    "多模态大模型": "连接视觉、视频、动作与语言推理的多模态理解和生成研究。",
    "嵌入模型": "面向检索、语义匹配、表示学习和嵌入模型训练的研究。",
    "训练": "可复用训练配方、监督微调、数据选择、蒸馏和优化实践。",
    "监督微调": "监督微调方法、数据配方、token 加权和推理泛化研究。",
    "强化学习": "奖励建模、RLHF 类优化、推理强化学习、智能体强化学习和 VLA 策略学习。",
    "智能体应用": "智能体系统、工具调用、记忆、AI 研究工作流和可复用技能生态。",
    "视觉": "对现代多模态系统有参考价值的计算机视觉方法。",
    "自动提示": "提示词优化、评测器提示、提示集成和测试时提示学习。",
}

LANGUAGE_CONFIGS = {
    "en": LanguageConfig(
        key = "en",
        readme_path = "README.md",
        index_path = "index.qmd",
        papers_dir = "papers/en",
        site_title = "Awesome LLM Research Collections",
        site_description = "A curated collection of LLM research papers, projects, code, and model resources.",
        description_label = "Description",
        skipped_headings = [README_TITLE_EN, "Contents", "Notes", "Blogs"],
        category_descriptions = CATEGORY_DESCRIPTIONS_EN,
        labels = LABELS_EN,
        asset_prefix = "../../"
    ),
    "zh": LanguageConfig(
        key = "zh",
        readme_path = "README.zh-CN.md",
        index_path = "zh/index.qmd",
        papers_dir = "papers/zh",
        site_title = README_TITLE_ZH,
        site_description = "LLM 论文、项目、代码与模型资源的中文精选合集。",
        description_label = "描述",
        skipped_headings = [README_TITLE_ZH, "目录", "笔记", "博客"],
        category_descriptions = CATEGORY_DESCRIPTIONS_ZH,
        labels = LABELS_ZH,
        asset_prefix = "../../"
    ),
}


def parse_args():
    """Parse command-line arguments.

    No parameters.
    """
    parser = argparse.ArgumentParser(
        description = "Check or regenerate bilingual Quarto qmd pages from README paper entries."
    )
    parser.add_argument(
        "--write",
        action = "store_true",
        help = "Write generated qmd pages instead of only checking them."
    )
    parser.add_argument(
        "--language",
        choices = ["all", "en", "zh"],
        default = "all",
        help = "Language set to check or regenerate."
    )
    parser.add_argument(
        "--readme",
        default = None,
        help = "Optional custom README path for legacy single-language checks."
    )
    return parser.parse_args()


def slugify(text):
    """Create a stable lowercase slug.

    Parameters:
        text: Source text to convert into a URL/file slug.
    """
    if text in CATEGORY_SLUGS:
        return CATEGORY_SLUGS[text]

    slug = text.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


def yaml_quote(text):
    """Quote text for a simple YAML scalar.

    Parameters:
        text: Text value to quote.
    """
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def parse_readme(readme_path, config):
    """Parse README paper entries into category collections.

    Parameters:
        readme_path: Path to the source README file.
        config: Language-specific parsing configuration.
    """
    category_order = []
    papers_by_category = {}
    current_category = None
    current_subcategory = None
    current_paper = None
    lines = readme_path.read_text(encoding = "utf-8").splitlines()
    description_pattern = (
        r"^\s+\*\*" + re.escape(config.description_label) + r"\*\*: (.+) \\\s*$"
    )

    for line in lines:
        top_match = re.match(r"^# (.+)$", line)
        if top_match:
            heading = top_match.group(1).strip()
            current_paper = None
            current_subcategory = None
            if heading in config.skipped_headings:
                current_category = None
                continue

            current_category = heading
            if current_category not in papers_by_category:
                papers_by_category[current_category] = []
                category_order.append(current_category)
            continue

        if current_category is None:
            continue

        sub_match = re.match(r"^## (.+)$", line)
        if sub_match:
            current_subcategory = sub_match.group(1).strip()
            current_paper = None
            continue

        title_match = re.match(r"^- \*\*(.+)\*\* \((\d{4}\.\d{2})\) \\\s*$", line)
        if title_match:
            current_paper = Paper(
                title = title_match.group(1).strip(),
                date = title_match.group(2),
                description = "",
                links = [],
                category = current_category,
                subcategory = current_subcategory
            )
            papers_by_category[current_category].append(current_paper)
            continue

        if current_paper is None:
            continue

        desc_match = re.match(description_pattern, line)
        if desc_match:
            current_paper.description = desc_match.group(1).strip()
            continue

        link_match = re.match(r"^\s+\[\[([^\]]+)\]\(([^)]+)\)\]\]?\s*$", line)
        icon_link_match = re.match(
            r"^\s+\[!\[([^\]]+)\]\(assets/icons/[^)]+\)\]\(([^)]+)\)\s*$",
            line
        )
        html_icon_link_match = re.match(
            r'^\s+<a href="([^"]+)"><img src="assets/icons/[^"]+" alt="([^"]+)" width="\d+"></a>\s*$',
            line
        )
        if link_match or icon_link_match or html_icon_link_match:
            match = link_match or icon_link_match
            label = match.group(1).strip() if match else html_icon_link_match.group(2).strip()
            url = match.group(2).strip() if match else html_icon_link_match.group(1).strip()
            current_paper.links.append(
                Link(
                    label = label,
                    url = url
                )
            )

    return category_order, papers_by_category


def all_papers(category_order, papers_by_category):
    """Flatten category-indexed papers.

    Parameters:
        category_order: Ordered list of top-level category names.
        papers_by_category: Mapping from category name to paper entries.
    """
    papers = []
    for category in category_order:
        papers.extend(papers_by_category.get(category, []))
    return papers


def paper_count_text(count, config):
    """Format a localized paper-count label.

    Parameters:
        count: Number of papers.
        config: Language-specific rendering configuration.
    """
    if config.key == "zh":
        return f"{count} {config.labels['paper_plural']}"

    unit = config.labels["paper_singular"] if count == 1 else config.labels["paper_plural"]
    return f"{count} {unit}"


def link_icon(link):
    """Choose the local icon for a resource link.

    Parameters:
        link: Parsed resource link.
    """
    if link.label in PAPER_LABELS and "arxiv.org" not in link.url:
        return "paper.svg"
    return LINK_ICONS.get(link.label, "project.svg")


def render_resource_link(link, prefix):
    """Render one paper resource link as HTML.

    Parameters:
        link: Parsed resource link.
        prefix: Relative path prefix from the current qmd file to repository root.
    """
    icon = link_icon(link)
    label = html.escape(link.label)
    url = html.escape(link.url, quote = True)
    icon_path = f"{prefix}assets/icons/{icon}"
    return (
        f'<a class="resource-link" href="{url}" target="_blank" rel="noopener">'
        f'<img src="{icon_path}" alt="" aria-hidden="true" />{label}</a>'
    )


def render_language_switch(config, href):
    """Render a compact language switch link.

    Parameters:
        config: Language-specific rendering configuration.
        href: Relative URL to the alternate-language page.
    """
    return (
        '<div class="language-switch">'
        f'<a href="{html.escape(href, quote = True)}">{html.escape(config.labels["language_label"])}</a>'
        '</div>'
    )


def render_paper_card(paper, prefix):
    """Render one paper card.

    Parameters:
        paper: Parsed paper entry.
        prefix: Relative path prefix from the current qmd file to repository root.
    """
    title = html.escape(paper.title)
    description = html.escape(paper.description)
    section = html.escape(paper.subcategory or paper.category)
    date = html.escape(paper.date)
    links = "\n".join(render_resource_link(link, prefix) for link in paper.links)

    return f"""<article class="paper-card" id="{slugify(paper.title)}">
  <div class="paper-meta">
    <span>{date}</span>
    <span>{section}</span>
  </div>
  <h3>{title}</h3>
  <p>{description}</p>
  <div class="paper-links">
{links}
  </div>
</article>"""


def category_index_href(category, config):
    """Build a category URL from the language homepage.

    Parameters:
        category: Top-level category name.
        config: Language-specific rendering configuration.
    """
    slug = slugify(category)
    if config.key == "zh":
        return f"../papers/zh/{slug}.html"
    return f"papers/en/{slug}.html"


def render_category_card(category, papers, config):
    """Render one homepage category card.

    Parameters:
        category: Top-level category name.
        papers: Papers under the category.
        config: Language-specific rendering configuration.
    """
    description = config.category_descriptions.get(category, config.site_description)
    subcategories = sorted({paper.subcategory for paper in papers if paper.subcategory})
    subcategory_text = "、".join(subcategories) if config.key == "zh" else ", ".join(subcategories)
    subcategory_text = subcategory_text if subcategory_text else config.labels["direct_collection"]
    latest = max((paper.date for paper in papers), default = config.labels["no_papers"])
    href = category_index_href(category, config)
    return f"""<a class="category-card" href="{href}">
  <span class="category-count">{paper_count_text(len(papers), config)}</span>
  <h3>{html.escape(category)}</h3>
  <p>{html.escape(description)}</p>
  <div class="category-foot">
    <span>{html.escape(latest)}</span>
    <span>{html.escape(subcategory_text)}</span>
  </div>
</a>"""


def render_recent_item(paper, config):
    """Render one recent-paper row.

    Parameters:
        paper: Parsed paper entry.
        config: Language-specific rendering configuration.
    """
    category = html.escape(paper.category)
    title = html.escape(paper.title)
    date = html.escape(paper.date)
    paper_slug = slugify(paper.title)
    href = f"{category_index_href(paper.category, config)}#{paper_slug}"
    return f"""<a class="recent-row" href="{href}">
  <span>{date}</span>
  <strong>{title}</strong>
  <em>{category}</em>
</a>"""


def generate_index(category_order, papers_by_category, config):
    """Generate the Quarto homepage.

    Parameters:
        category_order: Ordered list of top-level category names.
        papers_by_category: Mapping from category name to paper entries.
        config: Language-specific rendering configuration.
    """
    papers = all_papers(category_order, papers_by_category)
    recent_papers = sorted(papers, key = lambda paper: paper.date, reverse = True)[:8]
    latest_month = recent_papers[0].date if recent_papers else "N/A"
    category_cards = "\n".join(
        render_category_card(category, papers_by_category.get(category, []), config)
        for category in category_order
    )
    recent_rows = "\n".join(render_recent_item(paper, config) for paper in recent_papers)
    resource_count = sum(len(paper.links) for paper in papers)
    switch = render_language_switch(config, config.labels["language_href"])

    return f"""---
title: {yaml_quote(config.site_title)}
page-layout: full
toc: false
---

```{{=html}}
<section class="home-intro">
  {switch}
  <p class="eyebrow">{html.escape(config.labels["home_eyebrow"])}</p>
  <p class="lede">{html.escape(config.labels["home_lede"])}</p>
  <div class="stat-strip">
    <div><strong>{len(papers)}</strong><span>{html.escape(config.labels["papers"])}</span></div>
    <div><strong>{len(category_order)}</strong><span>{html.escape(config.labels["categories"])}</span></div>
    <div><strong>{resource_count}</strong><span>{html.escape(config.labels["resource_links"])}</span></div>
    <div><strong>{latest_month}</strong><span>{html.escape(config.labels["latest_month"])}</span></div>
  </div>
</section>

<section class="section-block">
  <div class="section-heading">
    <p class="eyebrow">{html.escape(config.labels["browse"])}</p>
    <h2>{html.escape(config.labels["research_categories"])}</h2>
  </div>
  <div class="category-grid">
{category_cards}
  </div>
</section>

<section class="section-block">
  <div class="section-heading">
    <p class="eyebrow">{html.escape(config.labels["recent_eyebrow"])}</p>
    <h2>{html.escape(config.labels["recent_papers"])}</h2>
  </div>
  <div class="recent-list">
{recent_rows}
  </div>
</section>
```
"""


def group_by_subcategory(papers, config):
    """Group papers by visible section name.

    Parameters:
        papers: Papers from one top-level category.
        config: Language-specific rendering configuration.
    """
    groups = []
    group_index = {}
    for paper in papers:
        name = paper.subcategory or config.labels["featured"]
        if name not in group_index:
            group_index[name] = []
            groups.append((name, group_index[name]))
        group_index[name].append(paper)
    return groups


def category_language_href(category, config):
    """Build the alternate-language URL for a category page.

    Parameters:
        category: Top-level category name.
        config: Language-specific rendering configuration.
    """
    slug = slugify(category)
    if config.key == "zh":
        return f"../en/{slug}.html"
    return f"../zh/{slug}.html"


def generate_category_page(category, papers, config):
    """Generate one category Quarto page.

    Parameters:
        category: Top-level category name.
        papers: Papers under the category.
        config: Language-specific rendering configuration.
    """
    description = config.category_descriptions.get(category, config.site_description)
    groups = group_by_subcategory(papers, config)
    total_links = sum(len(paper.links) for paper in papers)
    latest = max((paper.date for paper in papers), default = config.labels["no_papers"])
    nav_links = "\n".join(
        f'<a href="#{slugify(group_name)}">{html.escape(group_name)}</a>'
        for group_name, _ in groups
    )
    if not nav_links:
        nav_links = f'<span class="empty-note">{html.escape(config.labels["empty_note"])}</span>'

    sections = []
    for group_name, group_papers in groups:
        cards = "\n".join(render_paper_card(paper, config.asset_prefix) for paper in group_papers)
        sections.append(
            f"""<section class="paper-section" id="{slugify(group_name)}">
  <div class="section-heading compact">
    <p class="eyebrow">{paper_count_text(len(group_papers), config)}</p>
    <h2>{html.escape(group_name)}</h2>
  </div>
  <div class="paper-grid">
{cards}
  </div>
</section>"""
        )

    if not sections:
        sections.append(
            f"""<section class="paper-section">
  <p class="empty-state">{html.escape(config.labels["empty_state"])}</p>
</section>"""
        )

    switch = render_language_switch(config, category_language_href(category, config))

    return f"""---
title: {yaml_quote(category)}
description: {yaml_quote(description)}
page-layout: full
toc: false
---

```{{=html}}
<section class="category-hero">
  {switch}
  <p class="eyebrow">{html.escape(config.labels["research_category"])}</p>
  <p class="category-summary">{html.escape(description)}</p>
  <div class="stat-strip compact-strip">
    <div><strong>{len(papers)}</strong><span>{html.escape(config.labels["papers"])}</span></div>
    <div><strong>{total_links}</strong><span>{html.escape(config.labels["resource_links"])}</span></div>
    <div><strong>{latest}</strong><span>{html.escape(config.labels["latest_month"])}</span></div>
  </div>
  <nav class="section-nav" aria-label="{html.escape(config.labels["subcategories"])}">
    {nav_links}
  </nav>
</section>

{chr(10).join(sections)}
```
"""


def expected_files_for_config(config):
    """Build the expected qmd file map for one language.

    Parameters:
        config: Language-specific rendering configuration.
    """
    readme_path = Path(config.readme_path)
    category_order, papers_by_category = parse_readme(readme_path, config)
    files = {config.index_path: generate_index(category_order, papers_by_category, config)}
    for category in category_order:
        files[f"{config.papers_dir}/{slugify(category)}.qmd"] = generate_category_page(
            category = category,
            papers = papers_by_category.get(category, []),
            config = config
        )
    return files


def selected_configs(args):
    """Select language configurations for the current command.

    Parameters:
        args: Parsed command-line arguments.
    """
    if args.readme:
        config = LANGUAGE_CONFIGS["en"]
        return [
            LanguageConfig(
                key = config.key,
                readme_path = args.readme,
                index_path = config.index_path,
                papers_dir = config.papers_dir,
                site_title = config.site_title,
                site_description = config.site_description,
                description_label = config.description_label,
                skipped_headings = config.skipped_headings,
                category_descriptions = config.category_descriptions,
                labels = config.labels,
                asset_prefix = config.asset_prefix
            )
        ]

    if args.language == "all":
        return [LANGUAGE_CONFIGS["en"], LANGUAGE_CONFIGS["zh"]]

    return [LANGUAGE_CONFIGS[args.language]]


def expected_files(args):
    """Build the expected qmd file map.

    Parameters:
        args: Parsed command-line arguments.
    """
    files = {}
    for config in selected_configs(args):
        files.update(expected_files_for_config(config))
    return files


def write_or_check(files, write):
    """Write files or report out-of-sync paths.

    Parameters:
        files: Mapping from relative path to expected content.
        write: Whether to write expected content to disk.
    """
    out_of_sync = []
    for relative_path, expected in files.items():
        path = Path(relative_path)
        if write:
            path.parent.mkdir(parents = True, exist_ok = True)
            path.write_text(expected, encoding = "utf-8")
            continue

        if not path.exists():
            out_of_sync.append(relative_path)
            continue

        actual = path.read_text(encoding = "utf-8")
        if actual != expected:
            out_of_sync.append(relative_path)

    return out_of_sync


def main():
    """Run the sync checker.

    No parameters.
    """
    args = parse_args()
    files = expected_files(args)
    out_of_sync = write_or_check(
        files = files,
        write = args.write
    )

    if args.write:
        logger.info("Wrote %d qmd files", len(files))
        return 0

    if out_of_sync:
        logger.error("QMD pages are out of sync: %s", ", ".join(out_of_sync))
        return 1

    logger.info("QMD pages are in sync")
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    raise SystemExit(main())
