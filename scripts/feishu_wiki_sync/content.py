"""Content generation and rendering for Feishu Wiki synchronization."""

import os
import re
import sys
import json
import shutil
import hashlib
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

# Add project root to Python path
sys.path.append(os.getcwd())
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sync_notes import NOTE_TYPE_ORDER
from sync_notes import LANGUAGE_CONFIGS as NOTE_CONFIGS
from sync_notes import scan_notes, validate_bilingual_pairs
from sync_blog_shares import load_blog_shares
from check_readme_qmd_sync import LANGUAGE_CONFIGS as PAPER_CONFIGS
from check_readme_qmd_sync import parse_readme, slugify
from .models import ASSET_DIR
from .models import HOME_TITLE
from .models import MAX_MEDIA_BYTES
from .models import MANIFEST_MARKER
from .models import PageSpec
from .models import REPOSITORY_URL
from .models import repo_relative, stable_hash


SUPPORTED_LOCAL_IMAGE_SUFFIXES = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".svg",
    ".tif",
    ".tiff",
    ".webp",
}


def strip_front_matter(text, source_path):
    """Remove YAML front matter from a QMD document.

    Parameters:
        text: Full QMD text.
        source_path: Source path used in validation errors.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{source_path} is missing YAML front matter")
    for index, line in enumerate(lines[1:], start = 1):
        if line.strip() == "---":
            return "\n".join(lines[index + 1:]).strip() + "\n"
    raise ValueError(f"{source_path} has unterminated YAML front matter")


def convert_callouts(text):
    """Convert Quarto callout containers into Markdown blockquotes.

    Parameters:
        text: QMD body without front matter.
    """
    output = []
    callout = None
    pattern = re.compile(
        r'^:{3,4}\s+\{\.callout-([a-z-]+)(?:\s+title="([^"]+)")?\}\s*$'
    )
    icons = {
        "note": "📝",
        "tip": "💡",
        "warning": "⚠️",
        "important": "❗",
        "caution": "🚨",
    }
    for line in text.splitlines():
        opening = pattern.match(line)
        if opening:
            callout = opening.group(1)
            title = opening.group(2) or callout.replace("-", " ").title()
            output.append(f"> **{icons.get(callout, 'ℹ️')} {title}**")
            output.append(">")
            continue
        if callout and re.match(r"^:{3,4}\s*$", line):
            callout = None
            output.append("")
            continue
        if callout:
            output.append(">" if not line else f"> {line}")
        else:
            output.append(line)
    if callout:
        raise ValueError("Unterminated Quarto callout")
    return "\n".join(output)


def default_svg_converter(source, destination):
    """Convert one SVG file to PNG with rsvg-convert.

    Parameters:
        source: Source SVG path.
        destination: Destination PNG path.
    """
    if shutil.which("rsvg-convert") is None:
        raise ValueError("rsvg-convert is required to convert SVG note images")
    destination.parent.mkdir(parents = True, exist_ok = True)
    subprocess.run(
        ["rsvg-convert", str(source), "-o", str(destination)],
        check = True,
        capture_output = True,
        text = True
    )


def convert_internal_link(url, source_path, source_key_by_path):
    """Convert a repository QMD link into a stable Wiki-link placeholder.

    Parameters:
        url: Markdown link destination.
        source_path: QMD file containing the link.
        source_key_by_path: Mapping from repository source paths to page keys.
    """
    parsed = urlsplit(url)
    if parsed.scheme or parsed.netloc or not parsed.path.endswith(".qmd"):
        return url
    target = (source_path.parent / unquote(parsed.path)).resolve()
    target_relative = repo_relative(target)
    key = source_key_by_path.get(target_relative)
    if key is None:
        raise ValueError(f"Unmapped internal QMD link in {source_path}: {url}")
    return urlunsplit(("feishu-wiki", key, "", "", parsed.fragment))


def convert_qmd_body(
    note,
    source_key_by_path,
    svg_converter = default_svg_converter
):
    """Convert one note QMD body into lark-cli Markdown.

    Parameters:
        note: Note metadata and source path.
        source_key_by_path: Mapping from repository paths to stable page keys.
        svg_converter: Callable used to rasterize SVG files.
    """
    source_path = note.source_path.resolve()
    body = strip_front_matter(
        text = source_path.read_text(encoding = "utf-8"),
        source_path = source_path
    )
    body = convert_callouts(body)
    body = re.sub(r"^```\{([a-zA-Z0-9_-]+)\}\s*$", r"```\1", body, flags = re.MULTILINE)
    body = body.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    body = re.sub(r"(?m)^\s*\{[^{}]+\}\s*$", "", body)
    body = re.sub(r"(!\[[^\]]*\]\([^)]+\))\{[^{}]*\}", r"\1", body)
    media_paths = []

    image_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

    def replace_image(match):
        """Rewrite one local Markdown image reference.

        Parameters:
            match: Regular-expression image match.
        """
        alt = match.group(1)
        raw_url = match.group(2).strip()
        parsed = urlsplit(raw_url)
        if parsed.scheme or parsed.netloc or raw_url.startswith("data:"):
            return match.group(0)
        source = (source_path.parent / unquote(parsed.path)).resolve()
        relative = repo_relative(source)
        if not source.is_file():
            raise ValueError(f"Missing note image: {relative}")
        if source.suffix.lower() not in SUPPORTED_LOCAL_IMAGE_SUFFIXES:
            raise ValueError(
                f"Unsupported note image format for Feishu: {relative}"
            )
        if source.stat().st_size > MAX_MEDIA_BYTES:
            raise ValueError(f"Note image exceeds 20 MB: {relative}")
        media_source = source
        if source.suffix.lower() == ".svg":
            svg_digest = hashlib.sha256(source.read_bytes()).hexdigest()[:20]
            media_source = (ASSET_DIR / f"{svg_digest}.png").resolve()
            if not media_source.exists():
                svg_converter(source, media_source)
            if media_source.stat().st_size > MAX_MEDIA_BYTES:
                raise ValueError(f"Converted note image exceeds 20 MB: {relative}")
        media_paths.append(media_source)
        media_relative = repo_relative(media_source)
        return f"![{alt}](@./{media_relative})"

    body = image_pattern.sub(replace_image, body)
    link_pattern = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)")

    def replace_link(match):
        """Rewrite one Markdown link when it targets a repository QMD file.

        Parameters:
            match: Regular-expression link match.
        """
        label = match.group(1)
        target = convert_internal_link(
            url = match.group(2).strip(),
            source_path = source_path,
            source_key_by_path = source_key_by_path
        )
        return f"[{label}]({target})"

    body = link_pattern.sub(replace_link, body)
    date_labels = {
        "en": ("Created", "Last modified"),
        "zh": ("创建日期", "最后更新"),
    }
    created_label, modified_label = date_labels[note.language]
    header = [
        f"# {note.title}",
        "",
        f"**{created_label}:** {note.date}  ",
        f"**{modified_label}:** {note.date_modified}  ",
        f"**Author:** {note.author}  ",
        f"**Topic:** {note.topic}  ",
        f"**Tags:** {', '.join(note.tags)}",
        "",
    ]
    return "\n".join(header) + body.strip() + "\n", media_paths


def wiki_url(node_token):
    """Return a tenant-resolving Feishu Wiki URL for a node token.

    Parameters:
        node_token: Wiki node token.
    """
    return f"https://feishu.cn/wiki/{node_token}"


def resolve_wiki_links(text, node_tokens):
    """Resolve stable Wiki-link placeholders after all nodes exist.

    Parameters:
        text: Markdown containing feishu-wiki placeholders.
        node_tokens: Stable page key to node token mapping.
    """
    pattern = re.compile(r"feishu-wiki://([^/?#)]+(?:/[^?#)]*)?)(#[^)\s]*)?")

    def replace(match):
        """Replace one internal placeholder with a Wiki URL.

        Parameters:
            match: Regular-expression placeholder match.
        """
        key = match.group(1).rstrip("/")
        if key not in node_tokens:
            raise ValueError(f"Missing Wiki token for internal link: {key}")
        return wiki_url(node_tokens[key]) + (match.group(2) or "")

    return pattern.sub(replace, text)


def internal_link(label, key):
    """Build a Markdown link using a stable Wiki placeholder.

    Parameters:
        label: Visible link label.
        key: Stable target page key.
    """
    return f"[{label}](feishu-wiki://{key})"


def render_link_list(items, empty_text):
    """Render a simple Markdown navigation list.

    Parameters:
        items: Sequence of label and stable-key pairs.
        empty_text: Text used when no items exist.
    """
    if not items:
        return empty_text + "\n"
    return "\n".join(
        f"- {internal_link(label, key)}"
        for label, key in items
    ) + "\n"


def render_paper_page(category, papers, language):
    """Render one clean paper-category Markdown page.

    Parameters:
        category: Top-level paper category title.
        papers: Parsed paper entries in the category.
        language: en or zh.
    """
    description_label = "描述" if language == "zh" else "Description"
    empty_text = "暂无论文。" if language == "zh" else "No papers yet."
    lines = [f"# {category}", ""]
    if not papers:
        return "\n".join(lines + [empty_text, ""])

    current_subcategory = object()
    for paper in papers:
        if paper.subcategory != current_subcategory:
            current_subcategory = paper.subcategory
            if current_subcategory:
                lines.extend([f"## {current_subcategory}", ""])
        lines.append(f"### {paper.title} ({paper.date})")
        lines.extend(["", f"**{description_label}:** {paper.description}", ""])
        if paper.links:
            links = " · ".join(
                f"[{link.label}]({link.url})"
                for link in paper.links
            )
            lines.extend([links, ""])
    return "\n".join(lines).rstrip() + "\n"


def render_blog_page(blog_shares, language):
    """Render one localized blog-share index page.

    Parameters:
        blog_shares: Validated blog-share records.
        language: en or zh.
    """
    title = "博客分享" if language == "zh" else "Blog Shares"
    description_label = "描述" if language == "zh" else "Description"
    blog_label = "原文" if language == "zh" else "Blog"
    lines = [f"# {title}", ""]
    for blog_share in blog_shares:
        item_title = blog_share.title_zh if language == "zh" else blog_share.title_en
        description = (
            blog_share.description_zh
            if language == "zh"
            else blog_share.description_en
        )
        lines.extend(
            [
                f"## {item_title} ({blog_share.date})",
                "",
                f"**{description_label}:** {description}",
                "",
                f"[{blog_label}]({blog_share.blog_url})",
            ]
        )
        if blog_share.github_url:
            lines.append(f"[GitHub]({blog_share.github_url})")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def note_page_key(note):
    """Return the stable key for one note page.

    Parameters:
        note: Note metadata record.
    """
    relative = note.relative_path.with_suffix("").as_posix()
    return f"notes/{note.language}/page/{relative}"


def note_topic_key(note):
    """Return the stable key for one note topic container.

    Parameters:
        note: Note metadata record.
    """
    return f"notes/{note.language}/{note.note_type}/{slugify(note.topic)}"


def build_source_key_map(notes_by_language, paper_categories):
    """Map repository QMD paths to their mirrored Wiki page keys.

    Parameters:
        notes_by_language: Scanned notes keyed by language.
        paper_categories: Parsed paper category names keyed by language.
    """
    mapping = {}
    for notes in notes_by_language.values():
        for note in notes:
            mapping[repo_relative(note.source_path)] = note_page_key(note)
    for language, categories in paper_categories.items():
        for category in categories:
            path = Path("papers") / language / f"{slugify(category)}.qmd"
            mapping[path.as_posix()] = f"papers/{language}/{slugify(category)}"
        mapping[f"papers/{language}/index.qmd"] = f"papers/{language}"
        mapping[f"notes/{language}/index.qmd"] = f"notes/{language}"
        mapping[f"blogs/{language}/index.qmd"] = f"blogs/{language}/index"
    return mapping


def add_spec(specs, spec):
    """Add one page specification while enforcing stable-key uniqueness.

    Parameters:
        specs: Mutable page mapping.
        spec: Page specification to add.
    """
    if spec.key in specs:
        raise ValueError(f"Duplicate PageSpec key: {spec.key}")
    if spec.parent_key == spec.key:
        raise ValueError(f"PageSpec cannot parent itself: {spec.key}")
    specs[spec.key] = spec


def build_page_specs(svg_converter = default_svg_converter):
    """Build and validate the complete desired bilingual Wiki tree.

    Parameters:
        svg_converter: Callable used to rasterize SVG note images.
    """
    specs = {}
    paper_categories = {}
    papers_by_language = {}
    notes_by_language = {}

    for language in ["zh", "en"]:
        paper_config = PAPER_CONFIGS[language]
        categories, papers_by_category = parse_readme(
            readme_path = Path(paper_config.readme_path),
            config = paper_config
        )
        paper_categories[language] = categories
        papers_by_language[language] = papers_by_category
        notes_by_language[language] = scan_notes(NOTE_CONFIGS[language])

    validate_bilingual_pairs(notes_by_language)
    source_key_by_path = build_source_key_map(
        notes_by_language = notes_by_language,
        paper_categories = paper_categories
    )

    add_spec(
        specs,
        PageSpec(
            key = "home",
            parent_key = None,
            title = HOME_TITLE,
            source_path = "README.md, README.zh-CN.md",
            body = ""
        )
    )
    root_definitions = [
        ("papers", "论文 / Papers", "README.md, README.zh-CN.md"),
        ("notes", "笔记 / Notes", "notes/"),
        ("blogs", "博客 / Blogs", "data/blog_shares.json"),
    ]
    for key, title, source_path in root_definitions:
        add_spec(
            specs,
            PageSpec(
                key = key,
                parent_key = None,
                title = title,
                source_path = source_path,
                body = ""
            )
        )

    language_titles = {"zh": "中文", "en": "English"}
    for language in ["zh", "en"]:
        paper_language_key = f"papers/{language}"
        category_items = [
            (category, f"papers/{language}/{slugify(category)}")
            for category in paper_categories[language]
        ]
        add_spec(
            specs,
            PageSpec(
                key = paper_language_key,
                parent_key = "papers",
                title = language_titles[language],
                source_path = PAPER_CONFIGS[language].readme_path,
                body = (
                    f"# {language_titles[language]}\n\n"
                    + render_link_list(category_items, "暂无分类。")
                )
            )
        )
        for category in paper_categories[language]:
            key = f"papers/{language}/{slugify(category)}"
            add_spec(
                specs,
                PageSpec(
                    key = key,
                    parent_key = paper_language_key,
                    title = category,
                    source_path = PAPER_CONFIGS[language].readme_path,
                    body = render_paper_page(
                        category = category,
                        papers = papers_by_language[language][category],
                        language = language
                    )
                )
            )

    for language in ["zh", "en"]:
        config = NOTE_CONFIGS[language]
        language_key = f"notes/{language}"
        type_items = [
            (config.section_labels[note_type], f"notes/{language}/{note_type}")
            for note_type in NOTE_TYPE_ORDER
        ]
        add_spec(
            specs,
            PageSpec(
                key = language_key,
                parent_key = "notes",
                title = language_titles[language],
                source_path = f"notes/{language}/",
                body = (
                    f"# {language_titles[language]}\n\n"
                    + render_link_list(type_items, "暂无笔记。")
                )
            )
        )
        notes_by_type = {
            note_type: [
                note
                for note in notes_by_language[language]
                if note.note_type == note_type
            ]
            for note_type in NOTE_TYPE_ORDER
        }
        for note_type in NOTE_TYPE_ORDER:
            type_key = f"notes/{language}/{note_type}"
            topics = sorted({note.topic for note in notes_by_type[note_type]})
            topic_items = [
                (topic, f"{type_key}/{slugify(topic)}")
                for topic in topics
            ]
            add_spec(
                specs,
                PageSpec(
                    key = type_key,
                    parent_key = language_key,
                    title = config.section_labels[note_type],
                    source_path = f"notes/{language}/",
                    body = (
                        f"# {config.section_labels[note_type]}\n\n"
                        + render_link_list(topic_items, "暂无主题。")
                    )
                )
            )
            for topic in topics:
                topic_notes = [
                    note
                    for note in notes_by_type[note_type]
                    if note.topic == topic
                ]
                topic_key = f"{type_key}/{slugify(topic)}"
                note_items = [
                    (note.title, note_page_key(note))
                    for note in sorted(
                        topic_notes,
                        key = lambda item: (item.order, item.title)
                    )
                ]
                add_spec(
                    specs,
                    PageSpec(
                        key = topic_key,
                        parent_key = type_key,
                        title = topic,
                        source_path = f"notes/{language}/{topic}/",
                        body = f"# {topic}\n\n" + render_link_list(note_items, "暂无笔记。")
                    )
                )
                for note in topic_notes:
                    body, media_paths = convert_qmd_body(
                        note = note,
                        source_key_by_path = source_key_by_path,
                        svg_converter = svg_converter
                    )
                    add_spec(
                        specs,
                        PageSpec(
                            key = note_page_key(note),
                            parent_key = topic_key,
                            title = note.title,
                            source_path = repo_relative(note.source_path),
                            body = body,
                            media_paths = media_paths
                        )
                    )

    blog_shares = load_blog_shares()
    for language in ["zh", "en"]:
        language_key = f"blogs/{language}"
        index_key = f"{language_key}/index"
        page_title = "博客分享" if language == "zh" else "Blog Shares"
        add_spec(
            specs,
            PageSpec(
                key = language_key,
                parent_key = "blogs",
                title = language_titles[language],
                source_path = "data/blog_shares.json",
                body = (
                    f"# {language_titles[language]}\n\n"
                    + render_link_list([(page_title, index_key)], "暂无博客。")
                )
            )
        )
        add_spec(
            specs,
            PageSpec(
                key = index_key,
                parent_key = language_key,
                title = page_title,
                source_path = "data/blog_shares.json",
                body = render_blog_page(blog_shares, language)
            )
        )

    specs["papers"].body = "# 论文 / Papers\n\n" + render_link_list(
        [("中文", "papers/zh"), ("English", "papers/en")],
        ""
    )
    specs["notes"].body = "# 笔记 / Notes\n\n" + render_link_list(
        [("中文", "notes/zh"), ("English", "notes/en")],
        ""
    )
    specs["blogs"].body = "# 博客 / Blogs\n\n" + render_link_list(
        [("中文", "blogs/zh"), ("English", "blogs/en")],
        ""
    )
    specs["home"].body = "\n".join(
        [
            f"# {HOME_TITLE}",
            "",
            "**Awesome LLM Research Collections**",
            "",
            "GitHub 仓库内容的飞书双语镜像。",
            "",
            f"- {internal_link('论文 / Papers', 'papers')}",
            f"- {internal_link('笔记 / Notes', 'notes')}",
            f"- {internal_link('博客 / Blogs', 'blogs')}",
            f"- [GitHub]({REPOSITORY_URL})",
            "",
        ]
    )

    for key, spec in specs.items():
        if spec.parent_key is not None and spec.parent_key not in specs:
            raise ValueError(f"PageSpec has unknown parent: {key} -> {spec.parent_key}")
    return specs, {
        "paper_counts": {
            language: sum(
                len(items)
                for items in papers_by_language[language].values()
            )
            for language in ["zh", "en"]
        },
        "note_counts": {
            language: len(notes_by_language[language])
            for language in ["zh", "en"]
        },
        "blog_count": len(blog_shares),
    }


def render_managed_page(spec, body, source_commit):
    """Add source provenance and an edit warning to a page body.

    Parameters:
        spec: Page specification.
        body: Resolved canonical Markdown body.
        source_commit: Git commit displayed on the page.
    """
    banner = "\n".join(
        [
            "> ⚠️ 自动同步，请勿直接编辑 / Automatically synchronized; do not edit directly.",
            ">",
            f"> Source: `{spec.source_path}`  ",
            f"> Git commit: `{source_commit}`",
            "",
        ]
    )
    return banner + body.rstrip() + "\n"


def render_manifest_block(manifest):
    """Render the machine-readable manifest block stored on the homepage.

    Parameters:
        manifest: Synchronization manifest.
    """
    payload = json.dumps(
        manifest.to_dict(),
        ensure_ascii = False,
        indent = 2,
        sort_keys = True
    )
    return "\n".join(
        [
            "## 同步清单 / Sync manifest",
            "",
            MANIFEST_MARKER,
            "",
            "```json",
            payload,
            "```",
            "",
        ]
    )


def render_homepage(spec, resolved_body, source_commit, manifest):
    """Render homepage navigation, status, and the synchronization manifest.

    Parameters:
        spec: Homepage specification.
        resolved_body: Homepage body with Wiki links resolved.
        source_commit: Current run commit.
        manifest: Manifest to embed.
    """
    state_label = "同步完成 / Complete" if manifest.status == "complete" else "同步进行中 / In progress"
    status = "\n".join(
        [
            "## 同步状态 / Sync status",
            "",
            f"- **Status:** {state_label}",
            f"- **Git commit:** `{manifest.commit}`",
            f"- **Updated at:** `{manifest.updated_at}`",
            "",
        ]
    )
    managed = render_managed_page(spec, resolved_body, source_commit)
    return managed + "\n" + status + "\n" + render_manifest_block(manifest)


def content_hash_for_spec(spec, node_tokens):
    """Hash one resolved page body and its media payload.

    Parameters:
        spec: Desired page specification.
        node_tokens: Stable page key to node token mapping.
    """
    resolved = resolve_wiki_links(spec.body, node_tokens)
    return stable_hash(resolved, spec.media_paths)
