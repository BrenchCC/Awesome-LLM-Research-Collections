import re
import html
import json
import logging
import argparse
from typing import List
from pathlib import Path
from datetime import date
from urllib.parse import urlparse
from dataclasses import dataclass

logger = logging.getLogger(__name__)


DATA_PATH = Path("data/blog_shares.json")

REQUIRED_FIELDS = [
    "slug",
    "date",
    "title_en",
    "title_zh",
    "description_en",
    "description_zh",
    "blog_url",
    "github_url",
]


@dataclass
class BlogShare:
    slug: str
    date: str
    title_en: str
    title_zh: str
    description_en: str
    description_zh: str
    blog_url: str
    github_url: str


@dataclass
class LanguageConfig:
    key: str
    readme_path: Path
    index_path: Path
    site_title: str
    hero_eyebrow: str
    hero_summary: str
    latest_label: str
    entries_label: str
    source_label: str
    browse_label: str
    section_label: str
    readme_heading: str
    readme_description_label: str
    readme_empty_label: str
    readme_blog_label: str
    readme_github_label: str
    contents_lines: List[str]
    card_type_label: str


LANGUAGE_CONFIGS = {
    "en": LanguageConfig(
        key = "en",
        readme_path = Path("README.md"),
        index_path = Path("blogs/en/index.qmd"),
        site_title = "Blog Shares",
        hero_eyebrow = "Curated blog shares",
        hero_summary = "Selected technical blog posts and long-form essays worth tracking alongside the paper catalog.",
        latest_label = "Latest date",
        entries_label = "Blog shares",
        source_label = "Sources",
        browse_label = "Browse",
        section_label = "Latest Blog Shares",
        readme_heading = "# Blog Shares",
        readme_description_label = "Description",
        readme_empty_label = "No blog shares yet.",
        readme_blog_label = "Blog",
        readme_github_label = "GitHub",
        contents_lines = [
            "- [Blog Shares](#blog-shares)",
        ],
        card_type_label = "Blog Share"
    ),
    "zh": LanguageConfig(
        key = "zh",
        readme_path = Path("README.zh-CN.md"),
        index_path = Path("blogs/zh/index.qmd"),
        site_title = "博客分享",
        hero_eyebrow = "博客分享",
        hero_summary = "与论文目录并行整理的技术博客、长文和研究分享。",
        latest_label = "最新日期",
        entries_label = "博客分享",
        source_label = "来源",
        browse_label = "浏览",
        section_label = "最新博客分享",
        readme_heading = "# 博客分享",
        readme_description_label = "描述",
        readme_empty_label = "暂无博客分享。",
        readme_blog_label = "博客",
        readme_github_label = "GitHub",
        contents_lines = [
            "- [博客分享](#博客分享)",
        ],
        card_type_label = "博客分享"
    ),
}


def parse_args():
    """Parse command-line arguments.

    No parameters.
    """
    parser = argparse.ArgumentParser(
        description = "Generate and check bilingual blog-share README sections and Quarto indexes."
    )
    parser.add_argument(
        "--write",
        action = "store_true",
        help = "Write generated blog-share files instead of only checking them."
    )
    return parser.parse_args()


def validate_url(value, field, slug):
    """Validate one URL field.

    Parameters:
        value: URL string to validate.
        field: Field name used in error messages.
        slug: Blog-share slug used in error messages.
    """
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{slug} has invalid {field}: {value}")


def validate_item(item, index):
    """Validate one raw blog-share record.

    Parameters:
        item: Raw JSON object for one blog share.
        index: Item position used in error messages.
    """
    if not isinstance(item, dict):
        raise ValueError(f"Blog-share item {index} must be an object")

    item_fields = set(item.keys())
    required_fields = set(REQUIRED_FIELDS)
    missing = sorted(required_fields - item_fields)
    extra = sorted(item_fields - required_fields)
    if missing:
        raise ValueError(f"Blog-share item {index} is missing fields: {', '.join(missing)}")
    if extra:
        raise ValueError(f"Blog-share item {index} has unknown fields: {', '.join(extra)}")

    slug = item["slug"]
    if not re.match(r"^[a-z0-9]+(?:-[a-z0-9]+)*$", slug):
        raise ValueError(f"Blog-share item {index} has invalid slug: {slug}")

    try:
        date.fromisoformat(item["date"])
    except ValueError as error:
        raise ValueError(f"{slug} has invalid date: {item['date']}") from error

    text_fields = [
        "title_en",
        "title_zh",
        "description_en",
        "description_zh",
        "blog_url",
    ]
    for field in text_fields:
        if not isinstance(item[field], str) or not item[field].strip():
            raise ValueError(f"{slug} has empty {field}")

    if not isinstance(item["github_url"], str):
        raise ValueError(f"{slug} has non-string github_url")

    validate_url(
        value = item["blog_url"],
        field = "blog_url",
        slug = slug
    )
    if item["github_url"]:
        validate_url(
            value = item["github_url"],
            field = "github_url",
            slug = slug
        )
        if "github.com" not in urlparse(item["github_url"]).netloc.lower():
            raise ValueError(f"{slug} github_url must point to github.com")

    return BlogShare(
        slug = slug,
        date = item["date"],
        title_en = item["title_en"].strip(),
        title_zh = item["title_zh"].strip(),
        description_en = item["description_en"].strip(),
        description_zh = item["description_zh"].strip(),
        blog_url = item["blog_url"].strip(),
        github_url = item["github_url"].strip()
    )


def load_blog_shares():
    """Load and validate blog shares from JSON.

    No parameters.
    """
    raw_data = json.loads(DATA_PATH.read_text(encoding = "utf-8"))
    if not isinstance(raw_data, list):
        raise ValueError(f"{DATA_PATH} must contain a list")

    blog_shares = [
        validate_item(
            item = item,
            index = index
        )
        for index, item in enumerate(raw_data, start = 1)
    ]

    seen_slugs = set()
    seen_blog_urls = set()
    for blog_share in blog_shares:
        if blog_share.slug in seen_slugs:
            raise ValueError(f"Duplicate blog-share slug: {blog_share.slug}")
        if blog_share.blog_url in seen_blog_urls:
            raise ValueError(f"Duplicate blog_url: {blog_share.blog_url}")
        seen_slugs.add(blog_share.slug)
        seen_blog_urls.add(blog_share.blog_url)

    return sorted(blog_shares, key = lambda blog_share: blog_share.date, reverse = True)


def title_for_language(blog_share, config):
    """Return the localized title for one blog share.

    Parameters:
        blog_share: Blog-share metadata record.
        config: Language-specific rendering configuration.
    """
    return blog_share.title_zh if config.key == "zh" else blog_share.title_en


def description_for_language(blog_share, config):
    """Return the localized description for one blog share.

    Parameters:
        blog_share: Blog-share metadata record.
        config: Language-specific rendering configuration.
    """
    return blog_share.description_zh if config.key == "zh" else blog_share.description_en


def render_readme_entry(blog_share, config):
    """Render one README blog-share entry.

    Parameters:
        blog_share: Blog-share metadata record.
        config: Language-specific rendering configuration.
    """
    lines = [
        f"- **{title_for_language(blog_share, config)}** ({blog_share.date}) \\",
        f"  **{config.readme_description_label}**: {description_for_language(blog_share, config)} \\",
        f"  [[{config.readme_blog_label}]({blog_share.blog_url})]",
    ]
    if blog_share.github_url:
        lines.append(f"  [[{config.readme_github_label}]({blog_share.github_url})]")
    return "\n".join(lines)


def generate_readme_section(blog_shares, config):
    """Generate one README blog-share section.

    Parameters:
        blog_shares: Validated blog-share metadata records.
        config: Language-specific rendering configuration.
    """
    lines = [config.readme_heading, ""]
    if not blog_shares:
        lines.append(config.readme_empty_label)
        return "\n".join(lines).rstrip() + "\n"

    entries = [
        render_readme_entry(
            blog_share = blog_share,
            config = config
        )
        for blog_share in blog_shares
    ]
    lines.append("\n\n".join(entries))
    return "\n".join(lines).rstrip() + "\n"


def update_contents_section(content, config):
    """Update one README contents list with blog-share links.

    Parameters:
        content: Original README content.
        config: Language-specific rendering configuration.
    """
    lines = content.splitlines()
    contents_heading = "# Contents" if config.key == "en" else "# 目录"
    try:
        start_index = lines.index(contents_heading)
    except ValueError:
        raise ValueError(f"{config.readme_path} is missing {contents_heading}") from None

    end_index = None
    for index in range(start_index + 1, len(lines)):
        if lines[index].startswith("# "):
            end_index = index
            break

    if end_index is None:
        raise ValueError(f"{config.readme_path} has no content after {contents_heading}")

    old_block = lines[start_index + 1:end_index]
    remove_set = set(config.contents_lines)
    new_block = [line for line in old_block if line not in remove_set]
    while new_block and new_block[-1] == "":
        new_block.pop()
    new_block.extend(config.contents_lines)
    new_block.append("")

    updated = lines[:start_index + 1] + new_block + lines[end_index:]
    return "\n".join(updated).rstrip() + "\n"


def replace_readme_section(content, blog_section, config):
    """Replace or append one README blog-share section.

    Parameters:
        content: README content after contents-list normalization.
        blog_section: Generated blog-share section.
        config: Language-specific rendering configuration.
    """
    lines = content.splitlines()
    try:
        start_index = lines.index(config.readme_heading)
    except ValueError:
        prefix = "\n".join(lines).rstrip()
        return prefix + "\n\n" + blog_section

    end_index = len(lines)
    for index in range(start_index + 1, len(lines)):
        if lines[index].startswith("# "):
            end_index = index
            break

    prefix = lines[:start_index]
    while prefix and prefix[-1] == "":
        prefix.pop()
    suffix = lines[end_index:]
    joined = "\n".join(prefix).rstrip()
    if joined:
        joined += "\n\n"
    joined += blog_section.rstrip()
    if suffix:
        joined += "\n\n" + "\n".join(suffix).rstrip()
    return joined.rstrip() + "\n"


def expected_readme(blog_shares, config):
    """Build the expected README content for one language.

    Parameters:
        blog_shares: Validated blog-share metadata records.
        config: Language-specific rendering configuration.
    """
    content = config.readme_path.read_text(encoding = "utf-8")
    content = update_contents_section(
        content = content,
        config = config
    )
    blog_section = generate_readme_section(
        blog_shares = blog_shares,
        config = config
    )
    return replace_readme_section(
        content = content,
        blog_section = blog_section,
        config = config
    )


def render_resource_link(label, url, icon, prefix):
    """Render one resource link for a blog card.

    Parameters:
        label: Visible link label.
        url: Link URL.
        icon: Local icon filename.
        prefix: Relative path prefix from the index page to the repository root.
    """
    icon_path = f"{prefix}assets/icons/{icon}"
    return (
        f'<a class="resource-link" href="{html.escape(url, quote = True)}" target="_blank" rel="noopener">'
        f'<img src="{html.escape(icon_path, quote = True)}" alt="" aria-hidden="true" />{html.escape(label)}</a>'
    )


def render_blog_card(blog_share, config):
    """Render one blog card for the Quarto index page.

    Parameters:
        blog_share: Blog-share metadata record.
        config: Language-specific rendering configuration.
    """
    links = [
        render_resource_link(
            label = config.readme_blog_label,
            url = blog_share.blog_url,
            icon = "project.svg",
            prefix = "../../"
        )
    ]
    if blog_share.github_url:
        links.append(
            render_resource_link(
                label = config.readme_github_label,
                url = blog_share.github_url,
                icon = "github.svg",
                prefix = "../../"
            )
        )

    return f"""<article class="paper-card" id="{html.escape(blog_share.slug, quote = True)}">
  <div class="paper-meta">
    <span>{html.escape(blog_share.date)}</span>
    <span>{html.escape(config.card_type_label)}</span>
  </div>
  <h3>{html.escape(title_for_language(blog_share, config))}</h3>
  <p>{html.escape(description_for_language(blog_share, config))}</p>
  <div class="paper-links">
{chr(10).join(links)}
  </div>
</article>"""


def generate_index(blog_shares, config):
    """Generate one blog-share Quarto index page.

    Parameters:
        blog_shares: Validated blog-share metadata records.
        config: Language-specific rendering configuration.
    """
    latest = max((blog_share.date for blog_share in blog_shares), default = "N/A")
    cards = "\n".join(
        render_blog_card(
            blog_share = blog_share,
            config = config
        )
        for blog_share in blog_shares
    )
    content = (
        f'<p class="empty-state">{html.escape(config.readme_empty_label)}</p>'
        if not cards
        else f'<div class="paper-grid">\n{cards}\n  </div>'
    )

    return f"""---
title: "{config.site_title}"
page-layout: full
toc: false
---

```{{=html}}
<section class="category-hero">
  <p class="eyebrow">{html.escape(config.hero_eyebrow)}</p>
  <p class="category-summary">{html.escape(config.hero_summary)}</p>
  <div class="stat-strip compact-strip">
    <div><strong>{len(blog_shares)}</strong><span>{html.escape(config.entries_label)}</span></div>
    <div><strong>{len(blog_shares)}</strong><span>{html.escape(config.source_label)}</span></div>
    <div><strong>{html.escape(latest)}</strong><span>{html.escape(config.latest_label)}</span></div>
  </div>
</section>

<section class="paper-section">
  <div class="section-heading compact">
    <p class="eyebrow">{html.escape(config.browse_label)}</p>
    <h2>{html.escape(config.section_label)}</h2>
  </div>
  {content}
</section>
```
"""


def expected_files(blog_shares):
    """Build the expected generated file map.

    Parameters:
        blog_shares: Validated blog-share metadata records.
    """
    files = {}
    for config in LANGUAGE_CONFIGS.values():
        files[config.index_path] = generate_index(
            blog_shares = blog_shares,
            config = config
        )
        files[config.readme_path] = expected_readme(
            blog_shares = blog_shares,
            config = config
        )
    return files


def write_or_check(files, write):
    """Write generated files or report out-of-sync paths.

    Parameters:
        files: Mapping from path to expected content.
        write: Whether generated content should be written.
    """
    out_of_sync = []
    for path, expected in files.items():
        if write:
            path.parent.mkdir(parents = True, exist_ok = True)
            path.write_text(expected, encoding = "utf-8")
            continue

        if not path.exists():
            out_of_sync.append(path.as_posix())
            continue

        actual = path.read_text(encoding = "utf-8")
        if actual != expected:
            out_of_sync.append(path.as_posix())

    return out_of_sync


def main():
    """Run the blog-share sync checker.

    No parameters.
    """
    args = parse_args()
    blog_shares = load_blog_shares()
    files = expected_files(blog_shares)
    out_of_sync = write_or_check(
        files = files,
        write = args.write
    )

    if args.write:
        logger.info("Wrote %d generated blog-share files", len(files))
        return 0

    if out_of_sync:
        logger.error("Blog-share files are out of sync: %s", ", ".join(out_of_sync))
        return 1

    logger.info("Blog-share files are in sync")
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    raise SystemExit(main())
