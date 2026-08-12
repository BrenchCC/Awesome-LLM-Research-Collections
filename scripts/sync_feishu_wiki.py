#!/usr/bin/env python3
"""Mirror the repository's bilingual collections into a Feishu Wiki space."""

import os
import re
import sys
import json
import time
import shutil
import hashlib
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import asdict, dataclass, field
from urllib.parse import unquote, urlsplit, urlunsplit

# Add project root to Python path
sys.path.append(os.getcwd())
sys.path.append(str(Path(__file__).resolve().parent))

from sync_notes import NOTE_TYPE_ORDER
from sync_notes import LANGUAGE_CONFIGS as NOTE_CONFIGS
from sync_notes import scan_notes, validate_bilingual_pairs
from sync_blog_shares import load_blog_shares
from check_readme_qmd_sync import LANGUAGE_CONFIGS as PAPER_CONFIGS
from check_readme_qmd_sync import parse_readme, slugify

logger = logging.getLogger(__name__)


REPOSITORY_URL = "https://github.com/BrenchCC/Awesome-LLM-Research-Collections"
HOME_TITLE = "首页 / Home"
MANIFEST_MARKER = "FEISHU_SYNC_MANIFEST_V1"
MANIFEST_SCHEMA_VERSION = 1
MAX_MEDIA_BYTES = 20 * 1024 * 1024
RUNTIME_DIR = Path(".feishu-wiki-sync")
CONTENT_DIR = RUNTIME_DIR / "content"
ASSET_DIR = RUNTIME_DIR / "assets"
TRANSIENT_CODES = {429, 99991400}
TRANSIENT_TEXT = (
    "connection reset",
    "unexpected eof",
    " eof",
    "temporarily unavailable",
    "timed out",
    "timeout",
    "rate limit",
    "too many requests",
    "bad gateway",
    "service unavailable",
)
CREDENTIAL_ENV_KEYS = [
    "LARKSUITE_CLI_APP_ID",
    "LARKSUITE_CLI_APP_SECRET",
    "LARKSUITE_CLI_BRAND",
    "LARKSUITE_CLI_STRICT_MODE",
]


@dataclass
class PageSpec:
    """Describe one desired Wiki page.

    Parameters:
        key: Stable repository-owned page identifier.
        parent_key: Stable parent identifier, or None for a root node.
        title: Desired Wiki node title.
        source_path: Repository source path shown on the managed page.
        body: Canonical Markdown body without the managed banner.
        media_paths: Local media files referenced by the Markdown body.
    """

    key: str
    parent_key: Optional[str]
    title: str
    source_path: str
    body: str
    media_paths: List[Path] = field(default_factory = list)

    @property
    def depth(self):
        """Return the page depth derived from its stable key.

        No parameters.
        """
        return self.key.count("/")


@dataclass
class RemotePage:
    """Record one repository-managed remote Wiki page.

    Parameters:
        key: Stable repository-owned page identifier.
        parent_key: Stable parent identifier, or None for a root node.
        node_token: Wiki node token.
        obj_token: Backing Docx token.
        title: Last successfully synchronized title.
        content_hash: Hash of canonical content and referenced media.
        revision_id: Last observed document revision, or -1 for the homepage.
        source_path: Repository source path.
        source_commit: Commit displayed on the page.
    """

    key: str
    parent_key: Optional[str]
    node_token: str
    obj_token: str
    title: str
    content_hash: str
    revision_id: int
    source_path: str
    source_commit: str


@dataclass
class SyncManifest:
    """Persist the synchronization state inside the homepage.

    Parameters:
        schema_version: Manifest schema version.
        repository: Canonical repository URL.
        space_id: Target Wiki space identifier.
        status: Either in_progress or complete.
        commit: Git commit of the synchronization run.
        updated_at: UTC completion or checkpoint time.
        pending_create_key: Page key about to be created, if any.
        pages: Managed page records keyed by stable page identifier.
    """

    schema_version: int
    repository: str
    space_id: str
    status: str
    commit: str
    updated_at: str
    pending_create_key: Optional[str]
    pages: Dict[str, RemotePage]

    def to_dict(self):
        """Return a JSON-serializable manifest mapping.

        No parameters.
        """
        data = asdict(self)
        data["pages"] = {
            key: asdict(page)
            for key, page in sorted(self.pages.items())
        }
        return data

    @classmethod
    def from_dict(cls, data):
        """Build and validate a manifest from decoded JSON.

        Parameters:
            data: Decoded manifest mapping.
        """
        required = {
            "schema_version",
            "repository",
            "space_id",
            "status",
            "commit",
            "updated_at",
            "pending_create_key",
            "pages",
        }
        if not isinstance(data, dict) or set(data) != required:
            raise SafetyError("同步清单字段不完整或包含未知字段")
        if data["schema_version"] != MANIFEST_SCHEMA_VERSION:
            raise SafetyError("同步清单版本不受支持")
        if data["repository"] != REPOSITORY_URL:
            raise SafetyError("同步清单属于另一个 GitHub 仓库")
        if data["status"] not in {"in_progress", "complete"}:
            raise SafetyError("同步清单状态无效")
        if not isinstance(data["pages"], dict):
            raise SafetyError("同步清单 pages 字段必须是对象")

        page_fields = {
            "key",
            "parent_key",
            "node_token",
            "obj_token",
            "title",
            "content_hash",
            "revision_id",
            "source_path",
            "source_commit",
        }
        pages = {}
        seen_nodes = set()
        for key, raw_page in data["pages"].items():
            if not isinstance(raw_page, dict) or set(raw_page) != page_fields:
                raise SafetyError(f"同步清单页面字段无效: {key}")
            if raw_page["key"] != key:
                raise SafetyError(f"同步清单页面键不一致: {key}")
            if raw_page["node_token"] in seen_nodes:
                raise SafetyError("同步清单包含重复 node token")
            seen_nodes.add(raw_page["node_token"])
            pages[key] = RemotePage(**raw_page)

        return cls(
            schema_version = data["schema_version"],
            repository = data["repository"],
            space_id = data["space_id"],
            status = data["status"],
            commit = data["commit"],
            updated_at = data["updated_at"],
            pending_create_key = data["pending_create_key"],
            pages = pages
        )


@dataclass
class TreeNode:
    """Represent one node discovered in the remote Wiki tree.

    Parameters:
        node_token: Wiki node token.
        obj_token: Backing document token.
        parent_node_token: Remote parent node token or an empty string.
        title: Current remote title.
        has_child: Whether the node reports children.
        obj_type: Backing object type reported by the Wiki API.
    """

    node_token: str
    obj_token: str
    parent_node_token: str
    title: str
    has_child: bool
    obj_type: str = "docx"


@dataclass
class SyncAction:
    """Describe one proposed synchronization operation.

    Parameters:
        operation: create, update, rename, delete, or skip.
        key: Stable page key.
        detail: Human-readable reason.
    """

    operation: str
    key: str
    detail: str


class SyncError(RuntimeError):
    """Base error for synchronization failures."""


class SafetyError(SyncError):
    """Error raised when remote state is ambiguous or unsafe to mutate."""


class LarkCliError(SyncError):
    """Error raised when lark-cli returns a permanent failure."""


def parse_args():
    """Parse command-line arguments.

    No parameters.
    """
    parser = argparse.ArgumentParser(
        description = "Mirror papers, notes, and blogs into a Feishu Wiki space."
    )
    mode = parser.add_mutually_exclusive_group(required = True)
    mode.add_argument(
        "--check",
        action = "store_true",
        help = "Parse, convert, and validate local content without contacting Feishu."
    )
    mode.add_argument(
        "--plan",
        action = "store_true",
        help = "Read Feishu state and print the proposed changes without writing."
    )
    mode.add_argument(
        "--apply",
        action = "store_true",
        help = "Apply the synchronization plan serially."
    )
    return parser.parse_args()


def utc_now():
    """Return a stable UTC timestamp string.

    No parameters.
    """
    return datetime.now(timezone.utc).replace(microsecond = 0).isoformat()


def stable_hash(text, media_paths = None):
    """Hash normalized text and optional media bytes.

    Parameters:
        text: Canonical Markdown text.
        media_paths: Optional local media paths included in the page.
    """
    normalized = text.replace("\r\n", "\n").rstrip() + "\n"
    digest = hashlib.sha256(normalized.encode("utf-8"))
    for path in sorted(media_paths or [], key = lambda item: item.as_posix()):
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def get_git_commit():
    """Return GITHUB_SHA or the current Git HEAD.

    No parameters.
    """
    github_sha = os.environ.get("GITHUB_SHA", "").strip()
    if github_sha:
        return github_sha
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check = True,
        capture_output = True,
        text = True
    )
    return result.stdout.strip()


def repo_relative(path):
    """Return a repository-relative POSIX path after containment validation.

    Parameters:
        path: Path to validate and relativize.
    """
    root = Path.cwd().resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"Path escapes the repository: {path}") from error


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
    header = [
        f"# {note.title}",
        "",
        f"**Date:** {note.date}  ",
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
        fragment = match.group(2) or ""
        if key not in node_tokens:
            raise ValueError(f"Missing Wiki token for internal link: {key}")
        return wiki_url(node_tokens[key]) + fragment

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


def extract_json(text):
    """Decode a JSON object from lark-cli output with optional progress lines.

    Parameters:
        text: Captured standard output or error text.
    """
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        starts = [index for index, character in enumerate(stripped) if character == "{"]
        decoder = json.JSONDecoder()
        for start in starts:
            try:
                value, _ = decoder.raw_decode(stripped[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    return None


def find_values(value, key):
    """Recursively collect values stored under one JSON key.

    Parameters:
        value: Nested JSON-compatible value.
        key: Object key to collect.
    """
    found = []
    if isinstance(value, dict):
        for item_key, item_value in value.items():
            if item_key == key:
                found.append(item_value)
            found.extend(find_values(item_value, key))
    elif isinstance(value, list):
        for item in value:
            found.extend(find_values(item, key))
    return found


def first_value(value, key, default = None):
    """Return the first recursively discovered JSON key value.

    Parameters:
        value: Nested JSON-compatible value.
        key: Object key to find.
        default: Value returned when the key is absent.
    """
    values = find_values(value, key)
    return values[0] if values else default


def error_code(payload):
    """Extract an integer API error code from a lark-cli response.

    Parameters:
        payload: Decoded lark-cli JSON response.
    """
    value = first_value(payload, "code")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class LarkCliExecutor:
    """Run official lark-cli commands with bounded transient retries."""

    def __init__(self, max_attempts = 4, sleep = time.sleep):
        """Initialize the executor.

        Parameters:
            max_attempts: Maximum attempts for a transient failure.
            sleep: Delay callable injected by tests.
        """
        self.max_attempts = max_attempts
        self.sleep = sleep

    def run(self, arguments):
        """Run one bot command and return its successful JSON response.

        Parameters:
            arguments: lark-cli arguments excluding identity and output format.
        """
        command = ["lark-cli"] + arguments + ["--as", "bot", "--format", "json"]
        command_environment = os.environ.copy()
        for key in CREDENTIAL_ENV_KEYS:
            command_environment.pop(key, None)
        for attempt in range(1, self.max_attempts + 1):
            result = subprocess.run(
                command,
                capture_output = True,
                text = True,
                env = command_environment
            )
            payload = extract_json(result.stdout) or extract_json(result.stderr)
            if result.returncode == 0 and isinstance(payload, dict) and payload.get("ok") is True:
                return payload

            combined = (result.stdout + "\n" + result.stderr).lower()
            code = error_code(payload)
            transient = code in TRANSIENT_CODES or any(
                marker in combined
                for marker in TRANSIENT_TEXT
            )
            if transient and attempt < self.max_attempts:
                delay = 2 ** (attempt - 1)
                logger.warning(
                    "lark-cli transient failure; retrying in %s second(s) (%s/%s)",
                    delay,
                    attempt,
                    self.max_attempts
                )
                self.sleep(delay)
                continue

            message = first_value(payload, "message") if payload else None
            safe_message = message or f"lark-cli exited with code {result.returncode}"
            raise LarkCliError(f"{' '.join(arguments[:2])}: {safe_message}")
        raise LarkCliError("lark-cli retry loop exhausted")

    def list_nodes(self, space_id, parent_node_token = None):
        """List all direct Wiki children under one parent.

        Parameters:
            space_id: Target Wiki space identifier.
            parent_node_token: Parent token, or None for root nodes.
        """
        arguments = ["wiki", "+node-list", "--space-id", space_id, "--page-all"]
        if parent_node_token:
            arguments.extend(["--parent-node-token", parent_node_token])
        payload = self.run(arguments)
        nodes = first_value(payload, "nodes", [])
        if not isinstance(nodes, list):
            raise LarkCliError("wiki +node-list returned invalid nodes data")
        return nodes

    def fetch_document(self, obj_token):
        """Fetch a document's Markdown content and revision.

        Parameters:
            obj_token: Backing Docx token.
        """
        return self.run(
            [
                "docs",
                "+fetch",
                "--doc",
                obj_token,
                "--doc-format",
                "markdown",
                "--detail",
                "simple",
            ]
        )

    def create_node(self, space_id, title, parent_node_token = None):
        """Create a Docx Wiki node.

        Parameters:
            space_id: Target Wiki space identifier.
            title: New node title.
            parent_node_token: Parent node token, or None for a root node.
        """
        arguments = [
            "wiki",
            "+node-create",
            "--space-id",
            space_id,
            "--title",
            title,
        ]
        if parent_node_token:
            arguments.extend(["--parent-node-token", parent_node_token])
        return self.run(arguments)

    def rename_node(self, node_token, title):
        """Rename a Wiki node in place.

        Parameters:
            node_token: Wiki node token.
            title: New node title.
        """
        return self.run(
            [
                "drive",
                "+update-title",
                "--token",
                node_token,
                "--type",
                "wiki",
                "--title",
                title,
            ]
        )

    def overwrite_document(self, obj_token, revision_id, content_path):
        """Overwrite one document from a repository-relative Markdown file.

        Parameters:
            obj_token: Backing Docx token.
            revision_id: Current remote document revision.
            content_path: Repository-relative Markdown content path.
        """
        return self.run(
            [
                "docs",
                "+update",
                "--doc",
                obj_token,
                "--command",
                "overwrite",
                "--doc-format",
                "markdown",
                "--content",
                f"@./{content_path.as_posix()}",
                "--revision-id",
                str(revision_id),
            ]
        )

    def delete_node(self, space_id, node_token):
        """Delete one empty managed Wiki node without deleting children.

        Parameters:
            space_id: Target Wiki space identifier.
            node_token: Wiki node token to delete.
        """
        return self.run(
            [
                "wiki",
                "+node-delete",
                "--space-id",
                space_id,
                "--node-token",
                node_token,
                "--obj-type",
                "wiki",
                "--include-children=false",
                "--yes",
            ]
        )


def tree_node_from_json(raw_node):
    """Validate and normalize one node-list response object.

    Parameters:
        raw_node: Raw node mapping from lark-cli.
    """
    required = ["node_token", "obj_token", "title"]
    missing = [key for key in required if not raw_node.get(key)]
    if missing:
        raise LarkCliError(f"Wiki node is missing fields: {', '.join(missing)}")
    return TreeNode(
        node_token = raw_node["node_token"],
        obj_token = raw_node["obj_token"],
        parent_node_token = raw_node.get("parent_node_token", ""),
        title = raw_node["title"],
        has_child = bool(raw_node.get("has_child", False)),
        obj_type = raw_node.get("obj_type", "docx")
    )


def discover_tree(executor, space_id):
    """Recursively discover the target Wiki tree.

    Parameters:
        executor: Lark CLI executor.
        space_id: Target Wiki space identifier.
    """
    nodes = {}
    queue = [None]
    while queue:
        parent = queue.pop(0)
        raw_nodes = executor.list_nodes(space_id, parent)
        sibling_titles = {}
        for raw_node in raw_nodes:
            node = tree_node_from_json(raw_node)
            if node.node_token in nodes:
                raise SafetyError(f"Wiki tree contains duplicate token: {node.node_token}")
            sibling_titles.setdefault(node.title, []).append(node.node_token)
            nodes[node.node_token] = node
            if node.has_child:
                queue.append(node.node_token)
        duplicates = [title for title, tokens in sibling_titles.items() if len(tokens) > 1]
        if duplicates:
            parent_label = parent or "<root>"
            raise SafetyError(
                f"Wiki parent {parent_label} contains duplicate titles: {', '.join(duplicates)}"
            )
    return nodes


def document_content(payload):
    """Extract Markdown content from a docs +fetch response.

    Parameters:
        payload: Successful lark-cli response.
    """
    for key in ["content", "markdown", "text"]:
        values = find_values(payload, key)
        for value in values:
            if isinstance(value, str):
                return value
    raise LarkCliError("docs +fetch response did not contain Markdown content")


def document_revision(payload):
    """Extract an integer revision from a document response.

    Parameters:
        payload: Successful lark-cli document response.
    """
    revision = first_value(payload, "revision_id")
    try:
        return int(revision)
    except (TypeError, ValueError) as error:
        raise LarkCliError("Document response did not contain a revision_id") from error


def created_tokens(payload):
    """Extract node and document tokens from node-create output.

    Parameters:
        payload: Successful node-create response.
    """
    node_token = first_value(payload, "node_token")
    obj_token = first_value(payload, "obj_token")
    if not node_token or not obj_token:
        raise LarkCliError("wiki +node-create response did not contain node tokens")
    return str(node_token), str(obj_token)


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


def parse_manifest(content):
    """Parse and validate a manifest embedded in homepage Markdown.

    Parameters:
        content: Homepage Markdown fetched from Feishu.
    """
    if MANIFEST_MARKER not in content:
        raise SafetyError("同名首页存在，但未包含受管同步清单")
    marker_index = content.index(MANIFEST_MARKER)
    remainder = content[marker_index + len(MANIFEST_MARKER):]
    match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        remainder,
        flags = re.DOTALL | re.IGNORECASE
    )
    if match is None:
        raise SafetyError("同步清单 JSON 缺失或损坏")
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError as error:
        raise SafetyError("同步清单 JSON 无法解析") from error
    return SyncManifest.from_dict(data)


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


def new_manifest(space_id, commit, status = "in_progress"):
    """Create an empty manifest for a synchronization run.

    Parameters:
        space_id: Target Wiki space identifier.
        commit: Git commit of the synchronization run.
        status: Initial synchronization status.
    """
    return SyncManifest(
        schema_version = MANIFEST_SCHEMA_VERSION,
        repository = REPOSITORY_URL,
        space_id = space_id,
        status = status,
        commit = commit,
        updated_at = utc_now(),
        pending_create_key = None,
        pages = {}
    )


def load_remote_state(executor, space_id):
    """Discover the Wiki tree and read its managed manifest when present.

    Parameters:
        executor: Lark CLI executor.
        space_id: Target Wiki space identifier.
    """
    tree = discover_tree(executor, space_id)
    home_nodes = [
        node
        for node in tree.values()
        if node.parent_node_token == "" and node.title == HOME_TITLE
    ]
    if len(home_nodes) > 1:
        raise SafetyError("知识空间根目录包含重复首页")
    manifest_candidates = []
    if home_nodes:
        home_node = home_nodes[0]
        if home_node.obj_type != "docx":
            raise SafetyError("同名首页不是 Docx 节点")
        payload = executor.fetch_document(home_node.obj_token)
        manifest = parse_manifest(document_content(payload))
        manifest_candidates.append((home_node, manifest, payload))
    else:
        root_docx_nodes = [
            node
            for node in tree.values()
            if node.parent_node_token == "" and node.obj_type == "docx"
        ]
        for node in root_docx_nodes:
            payload = executor.fetch_document(node.obj_token)
            content = document_content(payload)
            if MANIFEST_MARKER not in content:
                continue
            manifest_candidates.append((node, parse_manifest(content), payload))
        if not manifest_candidates:
            return tree, None, None
    if len(manifest_candidates) > 1:
        raise SafetyError("知识空间包含多个同步清单，停止同步")
    home_node, manifest, payload = manifest_candidates[0]
    if manifest.space_id != space_id:
        raise SafetyError("首页同步清单属于另一个知识空间")
    home_page = manifest.pages.get("home")
    if home_page is None:
        raise SafetyError("同步清单缺少 home 页面")
    if home_page.node_token != home_node.node_token:
        raise SafetyError("同步清单中的首页 token 与远端不一致")
    if home_page.obj_token != home_node.obj_token:
        raise SafetyError("同步清单中的首页 document token 与远端不一致")
    return tree, manifest, document_revision(payload)


def validate_manifest_tree(manifest, tree):
    """Validate that manifest tokens still identify the expected remote nodes.

    Parameters:
        manifest: Stored synchronization manifest.
        tree: Discovered remote tree keyed by node token.
    """
    for key, page in manifest.pages.items():
        node = tree.get(page.node_token)
        if node is None:
            continue
        if node.obj_token != page.obj_token:
            raise SafetyError(f"受管节点 document token 已变化: {key}")
        if node.obj_type != "docx":
            raise SafetyError(f"受管节点不再是 Docx: {key}")
        if page.parent_key is None:
            expected_parent = ""
        else:
            parent = manifest.pages.get(page.parent_key)
            if parent is None:
                raise SafetyError(f"同步清单缺少父节点: {key} -> {page.parent_key}")
            expected_parent = parent.node_token
        if node.parent_node_token != expected_parent:
            raise SafetyError(f"受管节点被移动，停止同步: {key}")


def children_by_parent(tree):
    """Group remote tree nodes by parent token.

    Parameters:
        tree: Discovered remote tree keyed by node token.
    """
    grouped = {}
    for node in tree.values():
        grouped.setdefault(node.parent_node_token, []).append(node)
    return grouped


def expected_parent_token(spec, pages):
    """Resolve a desired parent key into a remote token.

    Parameters:
        spec: Desired page specification.
        pages: Known managed pages keyed by stable key.
    """
    if spec.parent_key is None:
        return ""
    parent = pages.get(spec.parent_key)
    return parent.node_token if parent else None


def staging_title(spec):
    """Return a deterministic temporary title used for crash-safe creation.

    Parameters:
        spec: Desired page specification.
    """
    key_hash = hashlib.sha256(spec.key.encode("utf-8")).hexdigest()[:12]
    return f"⏳ {spec.title[:80]} [sync:{key_hash}]"


def find_staging_node(spec, pages, tree):
    """Find a uniquely marked node left by an interrupted create operation.

    Parameters:
        spec: Desired page specification.
        pages: Known managed pages keyed by stable key.
        tree: Discovered remote tree keyed by node token.
    """
    parent_token = expected_parent_token(spec, pages)
    if parent_token is None:
        return None
    candidates = [
        node
        for node in tree.values()
        if node.parent_node_token == parent_token
        and node.title == staging_title(spec)
    ]
    if len(candidates) > 1:
        raise SafetyError(f"目标位置存在重复同步暂存节点: {spec.key}")
    return candidates[0] if candidates else None


def find_title_collision(spec, pages, tree):
    """Find an unmanaged sibling that already uses a desired title.

    Parameters:
        spec: Desired page specification.
        pages: Known managed pages keyed by stable key.
        tree: Discovered remote tree keyed by node token.
    """
    parent_token = expected_parent_token(spec, pages)
    if parent_token is None:
        return None
    managed_token = pages[spec.key].node_token if spec.key in pages else None
    candidates = [
        node
        for node in tree.values()
        if node.parent_node_token == parent_token
        and node.title == spec.title
        and node.node_token != managed_token
    ]
    if len(candidates) > 1:
        raise SafetyError(f"目标位置存在重复同名节点: {spec.key}")
    return candidates[0] if candidates else None


def fetch_managed_revisions(executor, manifest, tree):
    """Fetch current revisions for all existing managed documents.

    Parameters:
        executor: Lark CLI executor.
        manifest: Stored synchronization manifest.
        tree: Discovered remote tree keyed by node token.
    """
    revisions = {}
    for key, page in manifest.pages.items():
        if key == "home" or page.node_token not in tree:
            continue
        payload = executor.fetch_document(page.obj_token)
        revisions[key] = document_revision(payload)
    return revisions


def placeholder_tokens(specs, pages, tree):
    """Build deterministic tokens for plan-time hashing of missing pages.

    Parameters:
        specs: Desired page specifications.
        pages: Known managed pages.
        tree: Discovered remote tree keyed by node token.
    """
    tokens = {}
    for key in specs:
        if key in pages and pages[key].node_token in tree:
            tokens[key] = pages[key].node_token
        else:
            tokens[key] = "pending-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    return tokens


def content_hash_for_spec(spec, node_tokens):
    """Hash one resolved page body and its referenced media.

    Parameters:
        spec: Desired page specification.
        node_tokens: Stable key to remote token mapping.
    """
    resolved = resolve_wiki_links(spec.body, node_tokens)
    return stable_hash(resolved, spec.media_paths)


def compute_plan(specs, manifest, tree, revisions):
    """Compute a read-only synchronization diff.

    Parameters:
        specs: Desired page specifications.
        manifest: Stored manifest, or None for first synchronization.
        tree: Discovered remote tree keyed by node token.
        revisions: Current managed document revisions.
    """
    pages = manifest.pages if manifest else {}
    actions = []
    tokens = placeholder_tokens(specs, pages, tree)
    for spec in sorted(specs.values(), key = lambda item: (item.depth, item.key)):
        old = pages.get(spec.key)
        exists = old is not None and old.node_token in tree
        collision = find_title_collision(spec, pages, tree)
        staging = find_staging_node(spec, pages, tree)
        pending_recovery = (
            manifest is not None
            and manifest.status == "in_progress"
            and manifest.pending_create_key == spec.key
            and not exists
        )
        if collision and not pending_recovery:
            raise SafetyError(f"同名冲突，停止同步: {spec.key} ({spec.title})")
        if not exists:
            operation = "recover" if staging or (collision and pending_recovery) else "create"
            actions.append(SyncAction(operation, spec.key, spec.title))
            continue
        node = tree[old.node_token]
        if node.title != spec.title:
            actions.append(SyncAction("rename", spec.key, f"{node.title} -> {spec.title}"))
        desired_hash = content_hash_for_spec(spec, tokens)
        revision_changed = (
            spec.key != "home"
            and spec.key in revisions
            and old.revision_id != revisions[spec.key]
        )
        if spec.key == "home":
            continue
        if desired_hash != old.content_hash or revision_changed:
            reason = "source/media changed" if desired_hash != old.content_hash else "remote revision changed"
            actions.append(SyncAction("update", spec.key, reason))

    stale = set(pages) - set(specs)
    for key in sorted(stale, key = lambda item: (-item.count("/"), item)):
        actions.append(SyncAction("delete", key, pages[key].title))
    if "home" in pages and pages["home"].node_token in tree:
        actions.append(SyncAction("update", "home", "sync status and manifest"))
    return actions


def write_runtime_markdown(key, content):
    """Write deterministic temporary Markdown for lark-cli upload.

    Parameters:
        key: Stable page key used in the temporary filename.
        content: Complete managed Markdown content.
    """
    CONTENT_DIR.mkdir(parents = True, exist_ok = True)
    filename = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20] + ".md"
    path = CONTENT_DIR / filename
    path.write_text(content, encoding = "utf-8")
    return path


def overwrite_page(executor, page, revision_id, content):
    """Overwrite one remote document and return its new revision.

    Parameters:
        executor: Lark CLI executor.
        page: Managed remote page record.
        revision_id: Current remote revision.
        content: Complete Markdown document.
    """
    content_path = write_runtime_markdown(page.key, content)
    payload = executor.overwrite_document(
        obj_token = page.obj_token,
        revision_id = revision_id,
        content_path = content_path
    )
    return document_revision(payload)


def stabilize_document_revision(
    executor,
    obj_token,
    starting_revision,
    max_attempts = 5,
    delay_seconds = 2
):
    """Wait for asynchronous Docx conversion to stop changing revision.

    Parameters:
        executor: Lark CLI executor.
        obj_token: Backing Docx token.
        starting_revision: Revision returned by the overwrite request.
        max_attempts: Maximum post-write revision reads.
        delay_seconds: Delay between post-write reads.
    """
    previous_revision = starting_revision
    sleep = getattr(executor, "sleep", time.sleep)
    for _ in range(max_attempts):
        sleep(delay_seconds)
        payload = executor.fetch_document(obj_token)
        current_revision = document_revision(payload)
        if current_revision == previous_revision:
            return current_revision
        previous_revision = current_revision
    raise LarkCliError(
        f"Document revision did not stabilize after overwrite: {obj_token}"
    )


def checkpoint_home(executor, home_spec, manifest, home_revision):
    """Persist an honest in-progress checkpoint on the managed homepage.

    Parameters:
        executor: Lark CLI executor.
        home_spec: Homepage specification.
        manifest: In-progress synchronization manifest.
        home_revision: Current homepage document revision.
    """
    manifest.status = "in_progress"
    manifest.updated_at = utc_now()
    checkpoint_body = "\n".join(
        [
            f"# {HOME_TITLE}",
            "",
            "同步正在进行；若任务中断，下次 `--apply` 将从此检查点恢复。",
            "",
        ]
    )
    content = render_homepage(
        spec = home_spec,
        resolved_body = checkpoint_body,
        source_commit = manifest.commit,
        manifest = manifest
    )
    return overwrite_page(
        executor = executor,
        page = manifest.pages["home"],
        revision_id = home_revision,
        content = content
    )


def page_from_created(spec, payload, commit, revision_id):
    """Build a managed page record from node-create output.

    Parameters:
        spec: Desired page specification.
        payload: Successful node-create response.
        commit: Current Git commit.
        revision_id: Current blank document revision.
    """
    node_token, obj_token = created_tokens(payload)
    return RemotePage(
        key = spec.key,
        parent_key = spec.parent_key,
        node_token = node_token,
        obj_token = obj_token,
        title = spec.title,
        content_hash = "",
        revision_id = revision_id,
        source_path = spec.source_path,
        source_commit = commit
    )


def ensure_home(executor, space_id, specs, tree, manifest, home_revision, commit):
    """Create and checkpoint the homepage for a first synchronization.

    Parameters:
        executor: Lark CLI executor.
        space_id: Target Wiki space identifier.
        specs: Desired page specifications.
        tree: Discovered remote tree keyed by node token.
        manifest: Existing manifest, or None.
        home_revision: Existing homepage revision, or None.
        commit: Current Git commit.
    """
    if manifest is not None:
        was_complete = manifest.status == "complete"
        manifest.status = "in_progress"
        manifest.commit = commit
        if was_complete:
            manifest.pending_create_key = None
        home_revision = checkpoint_home(
            executor = executor,
            home_spec = specs["home"],
            manifest = manifest,
            home_revision = home_revision
        )
        return manifest, home_revision

    collision = find_title_collision(specs["home"], {}, tree)
    if collision:
        raise SafetyError("同名首页存在但没有同步清单，停止初始化")
    manifest = new_manifest(space_id, commit)
    staging = find_staging_node(specs["home"], {}, tree)
    if staging:
        node_token = staging.node_token
        obj_token = staging.obj_token
        fetched = executor.fetch_document(obj_token)
        home_revision = document_revision(fetched)
    else:
        payload = executor.create_node(space_id, staging_title(specs["home"]))
        node_token, obj_token = created_tokens(payload)
        fetched = executor.fetch_document(obj_token)
        home_revision = document_revision(fetched)
    manifest.pages["home"] = RemotePage(
        key = "home",
        parent_key = None,
        node_token = node_token,
        obj_token = obj_token,
        title = staging.title if staging else staging_title(specs["home"]),
        content_hash = "",
        revision_id = -1,
        source_path = specs["home"].source_path,
        source_commit = commit
    )
    tree[node_token] = TreeNode(
        node_token = node_token,
        obj_token = obj_token,
        parent_node_token = "",
        title = staging.title if staging else staging_title(specs["home"]),
        has_child = False,
        obj_type = "docx"
    )
    home_revision = checkpoint_home(
        executor = executor,
        home_spec = specs["home"],
        manifest = manifest,
        home_revision = home_revision
    )
    return manifest, home_revision


def adopt_pending_node(spec, manifest, collision, executor, commit):
    """Adopt only the exact node recorded by an interrupted create checkpoint.

    Parameters:
        spec: Desired page specification.
        manifest: In-progress synchronization manifest.
        collision: Matching remote tree node.
        executor: Lark CLI executor.
        commit: Current Git commit.
    """
    is_staging = collision.title == staging_title(spec)
    is_pending = (
        manifest.status == "in_progress"
        and manifest.pending_create_key == spec.key
        and collision.title == spec.title
    )
    if not is_staging and not is_pending:
        raise SafetyError(f"同名冲突，不能认领未知节点: {spec.key}")
    fetched = executor.fetch_document(collision.obj_token)
    return RemotePage(
        key = spec.key,
        parent_key = spec.parent_key,
        node_token = collision.node_token,
        obj_token = collision.obj_token,
        title = collision.title,
        content_hash = "",
        revision_id = document_revision(fetched),
        source_path = spec.source_path,
        source_commit = commit
    )


def create_missing_pages(
    executor,
    space_id,
    specs,
    tree,
    manifest,
    home_revision,
    commit
):
    """Create missing nodes parent-first with resumable homepage checkpoints.

    Parameters:
        executor: Lark CLI executor.
        space_id: Target Wiki space identifier.
        specs: Desired page specifications.
        tree: Mutable remote tree.
        manifest: Mutable in-progress manifest.
        home_revision: Current homepage revision.
        commit: Current Git commit.
    """
    for spec in sorted(specs.values(), key = lambda item: (item.depth, item.key)):
        if spec.key == "home":
            continue
        old = manifest.pages.get(spec.key)
        if old is not None and old.node_token in tree:
            continue
        collision = find_title_collision(spec, manifest.pages, tree)
        staging = find_staging_node(spec, manifest.pages, tree)
        if collision or staging:
            page = adopt_pending_node(
                spec = spec,
                manifest = manifest,
                collision = collision or staging,
                executor = executor,
                commit = commit
            )
        else:
            parent_token = expected_parent_token(spec, manifest.pages)
            if spec.parent_key is not None and parent_token is None:
                raise SafetyError(f"创建节点时父节点缺失: {spec.key}")
            payload = executor.create_node(
                space_id = space_id,
                title = staging_title(spec),
                parent_node_token = parent_token or None
            )
            node_token, obj_token = created_tokens(payload)
            fetched = executor.fetch_document(obj_token)
            page = page_from_created(
                spec = spec,
                payload = payload,
                commit = commit,
                revision_id = document_revision(fetched)
            )
            tree[node_token] = TreeNode(
                node_token = node_token,
                obj_token = obj_token,
                parent_node_token = parent_token or "",
                title = staging_title(spec),
                has_child = False,
                obj_type = "docx"
            )
        manifest.pages[spec.key] = page
        manifest.pending_create_key = None
    home_revision = checkpoint_home(
        executor = executor,
        home_spec = specs["home"],
        manifest = manifest,
        home_revision = home_revision
    )
    return home_revision


def rename_pages(executor, specs, tree, manifest):
    """Restore desired titles in place after checking sibling collisions.

    Parameters:
        executor: Lark CLI executor.
        specs: Desired page specifications.
        tree: Mutable remote tree.
        manifest: Mutable synchronization manifest.
    """
    for spec in sorted(specs.values(), key = lambda item: (item.depth, item.key)):
        page = manifest.pages[spec.key]
        node = tree[page.node_token]
        collision = find_title_collision(spec, manifest.pages, tree)
        if collision:
            raise SafetyError(f"重命名目标存在同名节点: {spec.key}")
        if node.title == spec.title:
            continue
        executor.rename_node(page.node_token, spec.title)
        node.title = spec.title


def synchronize_contents(
    executor,
    specs,
    manifest,
    tree,
    commit,
    revisions = None
):
    """Overwrite pages whose source hash or remote revision changed.

    Parameters:
        executor: Lark CLI executor.
        specs: Desired page specifications.
        manifest: Mutable synchronization manifest.
        tree: Discovered remote tree.
        commit: Current Git commit.
        revisions: Optional already-fetched revisions keyed by page key.
    """
    if revisions is None:
        revisions = fetch_managed_revisions(executor, manifest, tree)
    node_tokens = {
        key: manifest.pages[key].node_token
        for key in specs
    }
    synchronized = {}
    for spec in sorted(specs.values(), key = lambda item: (item.depth, item.key)):
        if spec.key == "home":
            continue
        old = manifest.pages[spec.key]
        resolved_body = resolve_wiki_links(spec.body, node_tokens)
        desired_hash = stable_hash(resolved_body, spec.media_paths)
        current_revision = revisions[spec.key]
        needs_update = (
            desired_hash != old.content_hash
            or old.revision_id != current_revision
        )
        source_commit = commit if needs_update else old.source_commit
        if needs_update:
            content = render_managed_page(spec, resolved_body, source_commit)
            current_revision = overwrite_page(
                executor = executor,
                page = old,
                revision_id = current_revision,
                content = content
            )
            current_revision = stabilize_document_revision(
                executor = executor,
                obj_token = old.obj_token,
                starting_revision = current_revision
            )
        synchronized[spec.key] = RemotePage(
            key = spec.key,
            parent_key = spec.parent_key,
            node_token = old.node_token,
            obj_token = old.obj_token,
            title = spec.title,
            content_hash = desired_hash,
            revision_id = current_revision,
            source_path = spec.source_path,
            source_commit = source_commit
        )
    return synchronized


def delete_stale_pages(executor, space_id, specs, tree, manifest):
    """Delete only stale manifest-owned nodes, deepest first and child-free.

    Parameters:
        executor: Lark CLI executor.
        space_id: Target Wiki space identifier.
        specs: Desired page specifications.
        tree: Mutable remote tree.
        manifest: Mutable synchronization manifest.
    """
    stale = set(manifest.pages) - set(specs)
    stale.discard("home")
    for key in sorted(stale, key = lambda item: (-item.count("/"), item)):
        page = manifest.pages[key]
        node = tree.get(page.node_token)
        if node is None:
            manifest.pages.pop(key)
            continue
        children = [
            child
            for child in tree.values()
            if child.parent_node_token == node.node_token
        ]
        if children:
            child_titles = ", ".join(child.title for child in children)
            raise SafetyError(f"受管失效节点仍有子节点，拒绝删除 {key}: {child_titles}")
        executor.delete_node(space_id, page.node_token)
        tree.pop(page.node_token)
        manifest.pages.pop(key)


def finalize_homepage(executor, specs, manifest, home_revision, synchronized, commit):
    """Write the final successful homepage and complete manifest last.

    Parameters:
        executor: Lark CLI executor.
        specs: Desired page specifications.
        manifest: Mutable synchronization manifest.
        home_revision: Current homepage document revision.
        synchronized: Successfully synchronized non-home pages.
        commit: Current Git commit.
    """
    home = manifest.pages["home"]
    node_tokens = {
        key: page.node_token
        for key, page in synchronized.items()
    }
    node_tokens["home"] = home.node_token
    resolved_body = resolve_wiki_links(specs["home"].body, node_tokens)
    home_hash = stable_hash(resolved_body, specs["home"].media_paths)
    synchronized["home"] = RemotePage(
        key = "home",
        parent_key = None,
        node_token = home.node_token,
        obj_token = home.obj_token,
        title = specs["home"].title,
        content_hash = home_hash,
        revision_id = -1,
        source_path = specs["home"].source_path,
        source_commit = commit
    )
    manifest.pages = synchronized
    manifest.status = "complete"
    manifest.commit = commit
    manifest.updated_at = utc_now()
    manifest.pending_create_key = None
    content = render_homepage(
        spec = specs["home"],
        resolved_body = resolved_body,
        source_commit = commit,
        manifest = manifest
    )
    return overwrite_page(
        executor = executor,
        page = synchronized["home"],
        revision_id = home_revision,
        content = content
    )


def print_actions(actions):
    """Print a deterministic synchronization plan without secrets.

    Parameters:
        actions: Proposed synchronization actions.
    """
    for action in actions:
        print(f"{action.operation.upper():8} {action.key} — {action.detail}")
    counts = {}
    for action in actions:
        counts[action.operation] = counts.get(action.operation, 0) + 1
    summary = ", ".join(
        f"{key}={value}"
        for key, value in sorted(counts.items())
    )
    print(f"Summary: {summary or 'no changes'}")


def require_remote_environment():
    """Validate bot credentials and return the target Wiki space ID.

    No parameters.
    """
    required = [
        "LARKSUITE_CLI_APP_ID",
        "LARKSUITE_CLI_APP_SECRET",
        "FEISHU_WIKI_SPACE_ID",
    ]
    missing = [name for name in required if not os.environ.get(name, "").strip()]
    if missing:
        raise SyncError("Missing required environment variables: " + ", ".join(missing))
    brand = os.environ.setdefault("LARKSUITE_CLI_BRAND", "feishu")
    if brand != "feishu":
        raise SyncError("LARKSUITE_CLI_BRAND must be feishu")
    strict_mode = os.environ.setdefault("LARKSUITE_CLI_STRICT_MODE", "bot")
    if strict_mode != "bot":
        raise SyncError("LARKSUITE_CLI_STRICT_MODE must be bot")
    if shutil.which("lark-cli") is None:
        raise SyncError("lark-cli is not installed or not available on PATH")
    return os.environ["FEISHU_WIKI_SPACE_ID"].strip()


def run_check(specs, statistics):
    """Report successful local conversion statistics.

    Parameters:
        specs: Validated desired page specifications.
        statistics: Parsed collection counts.
    """
    media_paths = {
        path.resolve()
        for spec in specs.values()
        for path in spec.media_paths
    }
    print("Feishu Wiki local validation passed")
    print(
        "Papers: "
        f"zh={statistics['paper_counts']['zh']}, "
        f"en={statistics['paper_counts']['en']}"
    )
    print(
        "Notes: "
        f"zh={statistics['note_counts']['zh']}, "
        f"en={statistics['note_counts']['en']}"
    )
    print(f"Blogs: zh={statistics['blog_count']}, en={statistics['blog_count']}")
    print(f"Wiki pages: {len(specs)}; local media: {len(media_paths)}")


def run_plan(specs, executor, space_id):
    """Read remote state and print a non-mutating synchronization diff.

    Parameters:
        specs: Desired page specifications.
        executor: Lark CLI executor.
        space_id: Target Wiki space identifier.
    """
    tree, manifest, _ = load_remote_state(executor, space_id)
    revisions = {}
    if manifest is not None:
        validate_manifest_tree(manifest, tree)
        revisions = fetch_managed_revisions(executor, manifest, tree)
    actions = compute_plan(specs, manifest, tree, revisions)
    print_actions(actions)


def run_apply(specs, executor, space_id, commit):
    """Apply the synchronization serially and commit the homepage manifest last.

    Parameters:
        specs: Desired page specifications.
        executor: Lark CLI executor.
        space_id: Target Wiki space identifier.
        commit: Current Git commit.
    """
    tree, manifest, home_revision = load_remote_state(executor, space_id)
    revisions = {}
    if manifest is not None:
        validate_manifest_tree(manifest, tree)
        revisions = fetch_managed_revisions(executor, manifest, tree)
    actions = compute_plan(specs, manifest, tree, revisions)
    print_actions(actions)

    manifest, home_revision = ensure_home(
        executor = executor,
        space_id = space_id,
        specs = specs,
        tree = tree,
        manifest = manifest,
        home_revision = home_revision,
        commit = commit
    )
    home_revision = create_missing_pages(
        executor = executor,
        space_id = space_id,
        specs = specs,
        tree = tree,
        manifest = manifest,
        home_revision = home_revision,
        commit = commit
    )
    rename_pages(
        executor = executor,
        specs = specs,
        tree = tree,
        manifest = manifest
    )
    current_revisions = {
        key: revisions.get(key, page.revision_id)
        for key, page in manifest.pages.items()
        if key != "home"
    }
    synchronized = synchronize_contents(
        executor = executor,
        specs = specs,
        manifest = manifest,
        tree = tree,
        commit = commit,
        revisions = current_revisions
    )
    delete_stale_pages(
        executor = executor,
        space_id = space_id,
        specs = specs,
        tree = tree,
        manifest = manifest
    )
    finalize_homepage(
        executor = executor,
        specs = specs,
        manifest = manifest,
        home_revision = home_revision,
        synchronized = synchronized,
        commit = commit
    )
    print(f"Apply complete: {len(specs)} managed pages at commit {commit}")


def main():
    """Run the selected local check, remote plan, or remote apply mode.

    No parameters.
    """
    args = parse_args()
    specs, statistics = build_page_specs()
    if args.check:
        run_check(specs, statistics)
        return

    space_id = require_remote_environment()
    executor = LarkCliExecutor()
    if args.plan:
        run_plan(specs, executor, space_id)
        return
    run_apply(
        specs = specs,
        executor = executor,
        space_id = space_id,
        commit = get_git_commit()
    )


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    try:
        main()
    except (OSError, ValueError, SyncError, subprocess.CalledProcessError) as error:
        logger.error("Feishu Wiki synchronization failed: %s", error)
        raise SystemExit(1) from error
