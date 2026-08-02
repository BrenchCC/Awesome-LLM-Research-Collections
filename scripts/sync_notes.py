import html
import logging
import argparse
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


REQUIRED_METADATA = [
    "title",
    "date",
    "description",
    "author",
    "order",
    "note_type",
    "topic",
    "tags",
]

NOTE_TYPE_ORDER = [
    "paper-reading",
    "technical-reflection",
]


@dataclass
class Note:
    language: str
    source_path: Path
    relative_path: Path
    title: str
    date: str
    description: str
    author: str
    order: int
    note_type: str
    topic: str
    tags: List[str]


@dataclass
class LanguageConfig:
    key: str
    notes_dir: Path
    index_path: Path
    readme_path: Path
    site_title: str
    hero_eyebrow: str
    hero_summary: str
    notes_label: str
    topics_label: str
    latest_label: str
    browse_label: str
    no_notes_label: str
    readme_heading: str
    readme_description_label: str
    readme_empty_label: str
    readme_note_label: str
    readme_alt_label: str
    contents_lines: List[str]
    section_labels: Dict[str, str]


LANGUAGE_CONFIGS = {
    "en": LanguageConfig(
        key = "en",
        notes_dir = Path("notes/en"),
        index_path = Path("notes/en/index.qmd"),
        readme_path = Path("README.md"),
        site_title = "Notes",
        hero_eyebrow = "Research notes",
        hero_summary = "Bilingual notes for paper readings and technical reflections around LLM research and engineering.",
        notes_label = "Notes",
        topics_label = "Topics",
        latest_label = "Latest date",
        browse_label = "Browse",
        no_notes_label = "No notes yet.",
        readme_heading = "# Notes",
        readme_description_label = "Description",
        readme_empty_label = "No notes yet.",
        readme_note_label = "Note",
        readme_alt_label = "中文",
        contents_lines = [
            "- [Notes](#notes)",
            "  - [Paper Readings](#paper-readings)",
            "  - [Technical Reflections](#technical-reflections)",
        ],
        section_labels = {
            "paper-reading": "Paper Readings",
            "technical-reflection": "Technical Reflections",
        }
    ),
    "zh": LanguageConfig(
        key = "zh",
        notes_dir = Path("notes/zh"),
        index_path = Path("notes/zh/index.qmd"),
        readme_path = Path("README.zh-CN.md"),
        site_title = "笔记",
        hero_eyebrow = "研究笔记",
        hero_summary = "围绕 LLM 研究与工程实践整理的双语论文解读和技术思考。",
        notes_label = "笔记",
        topics_label = "主题",
        latest_label = "最新日期",
        browse_label = "浏览",
        no_notes_label = "暂无笔记。",
        readme_heading = "# 笔记",
        readme_description_label = "描述",
        readme_empty_label = "暂无笔记。",
        readme_note_label = "笔记",
        readme_alt_label = "English",
        contents_lines = [
            "- [笔记](#笔记)",
            "  - [论文解读](#论文解读)",
            "  - [技术思考](#技术思考)",
        ],
        section_labels = {
            "paper-reading": "论文解读",
            "technical-reflection": "技术思考",
        }
    ),
}


def parse_args():
    """Parse command-line arguments.

    No parameters.
    """
    parser = argparse.ArgumentParser(
        description = "Generate and check bilingual notes indexes and README notes sections."
    )
    parser.add_argument(
        "--write",
        action = "store_true",
        help = "Write generated notes indexes and README notes sections."
    )
    return parser.parse_args()


def strip_yaml_value(value):
    """Strip a simple YAML scalar value.

    Parameters:
        value: Raw scalar value after the front matter key delimiter.
    """
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def parse_front_matter(path):
    """Parse simple front matter from a qmd file.

    Parameters:
        path: Source qmd path to read.
    """
    lines = path.read_text(encoding = "utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{path} is missing YAML front matter")

    end_index = None
    for index, line in enumerate(lines[1:], start = 1):
        if line.strip() == "---":
            end_index = index
            break

    if end_index is None:
        raise ValueError(f"{path} has unterminated YAML front matter")

    metadata = {}
    list_key = None
    for line in lines[1:end_index]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if list_key and line.startswith("  - "):
            metadata[list_key].append(strip_yaml_value(line[4:]))
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = strip_yaml_value(value)
        if value:
            metadata[key] = value
            list_key = None
        else:
            metadata[key] = []
            list_key = key

    missing = [key for key in REQUIRED_METADATA if key not in metadata]
    if missing:
        raise ValueError(f"{path} is missing note metadata: {', '.join(missing)}")

    if metadata["note_type"] not in NOTE_TYPE_ORDER:
        allowed = ", ".join(NOTE_TYPE_ORDER)
        raise ValueError(f"{path} has unsupported note_type: {metadata['note_type']} (allowed: {allowed})")

    try:
        metadata["order"] = int(metadata["order"])
    except ValueError as error:
        raise ValueError(f"{path} has non-integer order: {metadata['order']}") from error

    if not metadata["tags"]:
        raise ValueError(f"{path} has no tags")

    return metadata


def scan_notes(config):
    """Scan note qmd files for one language.

    Parameters:
        config: Language-specific notes configuration.
    """
    notes = []
    if not config.notes_dir.exists():
        return notes

    for path in sorted(config.notes_dir.rglob("*.qmd")):
        if path.name == "index.qmd":
            continue
        relative_path = path.relative_to(config.notes_dir)
        metadata = parse_front_matter(path)
        notes.append(
            Note(
                language = config.key,
                source_path = path,
                relative_path = relative_path,
                title = metadata["title"],
                date = metadata["date"],
                description = metadata["description"],
                author = metadata["author"],
                order = metadata["order"],
                note_type = metadata["note_type"],
                topic = metadata["topic"],
                tags = metadata["tags"]
            )
        )

    return notes


def grouped_notes(notes):
    """Group notes by note type.

    Parameters:
        notes: List of note metadata records.
    """
    groups = {note_type: [] for note_type in NOTE_TYPE_ORDER}
    for note in notes:
        groups[note.note_type].append(note)

    for note_type in NOTE_TYPE_ORDER:
        groups[note_type] = sorted(
            groups[note_type],
            key = lambda note: note.date,
            reverse = True
        )

    return groups


def validate_bilingual_pairs(notes_by_language):
    """Validate that English and Chinese notes are paired.

    Parameters:
        notes_by_language: Mapping from language key to scanned note records.
    """
    en_notes = {note.relative_path: note for note in notes_by_language["en"]}
    zh_notes = {note.relative_path: note for note in notes_by_language["zh"]}

    missing_en = sorted(zh_notes.keys() - en_notes.keys())
    missing_zh = sorted(en_notes.keys() - zh_notes.keys())
    if missing_en or missing_zh:
        details = []
        if missing_en:
            details.append("missing English: " + ", ".join(path.as_posix() for path in missing_en))
        if missing_zh:
            details.append("missing Chinese: " + ", ".join(path.as_posix() for path in missing_zh))
        raise ValueError("Bilingual note pairs are incomplete: " + "; ".join(details))

    for relative_path, en_note in en_notes.items():
        zh_note = zh_notes[relative_path]
        comparable_fields = ["date", "author", "order", "note_type", "topic", "tags"]
        for field in comparable_fields:
            if getattr(en_note, field) != getattr(zh_note, field):
                raise ValueError(
                    f"Bilingual note metadata mismatch for {relative_path}: {field}"
                )


def note_href(note):
    """Build an HTML href for a note from its index page.

    Parameters:
        note: Note metadata record.
    """
    return note.relative_path.with_suffix(".html").as_posix()


def render_note_card(note, config):
    """Render one note card for a notes index page.

    Parameters:
        note: Note metadata record.
        config: Language-specific notes configuration.
    """
    href = html.escape(note_href(note), quote = True)
    title = html.escape(note.title)
    description = html.escape(note.description)
    topic = html.escape(note.topic.upper())
    section = html.escape(config.section_labels[note.note_type])
    date = html.escape(note.date)
    tags = "\n".join(
        f'    <span class="note-tag">{html.escape(tag)}</span>'
        for tag in note.tags
    )
    return f"""<a class="category-card" href="{href}">
  <span class="category-count">{date}</span>
  <h3>{title}</h3>
  <p>{description}</p>
  <div class="note-tags">
{tags}
  </div>
  <div class="category-foot">
    <span>{topic}</span>
    <span>{section}</span>
  </div>
</a>"""


def render_note_section(note_type, notes, config):
    """Render one grouped note section.

    Parameters:
        note_type: Note type key for the section.
        notes: Notes belonging to the section.
        config: Language-specific notes configuration.
    """
    section_label = config.section_labels[note_type]
    section_id = section_label.lower().replace(" ", "-")
    cards = "\n".join(render_note_card(note, config) for note in notes)
    content = (
        f'<p class="empty-state">{html.escape(config.no_notes_label)}</p>'
        if not cards
        else f'<div class="category-grid">\n{cards}\n  </div>'
    )
    return f"""<section class="section-block" id="{html.escape(section_id, quote = True)}">
  <div class="section-heading">
    <p class="eyebrow">{html.escape(config.browse_label)}</p>
    <h2>{html.escape(section_label)}</h2>
  </div>
  {content}
</section>"""


def generate_index(notes, config):
    """Generate one notes index qmd page.

    Parameters:
        notes: Notes for one language.
        config: Language-specific notes configuration.
    """
    groups = grouped_notes(notes)
    topics = sorted({note.topic for note in notes})
    latest = max((note.date for note in notes), default = "N/A")
    nav_links = "\n".join(
        f'<a href="#{html.escape(config.section_labels[note_type].lower().replace(" ", "-"), quote = True)}">'
        f'{html.escape(config.section_labels[note_type])}</a>'
        for note_type in NOTE_TYPE_ORDER
    )
    sections = "\n\n".join(
        render_note_section(note_type, groups[note_type], config)
        for note_type in NOTE_TYPE_ORDER
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
    <div><strong>{len(notes)}</strong><span>{html.escape(config.notes_label)}</span></div>
    <div><strong>{len(topics)}</strong><span>{html.escape(config.topics_label)}</span></div>
    <div><strong>{html.escape(latest)}</strong><span>{html.escape(config.latest_label)}</span></div>
  </div>
  <nav class="section-nav" aria-label="{html.escape(config.browse_label)}">
    {nav_links}
  </nav>
</section>

{sections}
```
"""


def readme_note_path(note):
    """Build a repository-relative qmd path for a README note link.

    Parameters:
        note: Note metadata record.
    """
    return f"notes/{note.language}/{note.relative_path.as_posix()}"


def render_readme_note(note, paired_note, config):
    """Render one README note entry.

    Parameters:
        note: Note metadata record for the current language.
        paired_note: Matching note metadata record in the other language.
        config: Language-specific notes configuration.
    """
    return "\n".join(
        [
            f"- **{note.title}** ({note.date}) \\",
            f"  **{config.readme_description_label}**: {note.description} \\",
            f"  [[{config.readme_note_label}]({readme_note_path(note)})]",
            f"  [[{config.readme_alt_label}]({readme_note_path(paired_note)})]",
        ]
    )


def generate_readme_notes_section(notes, paired_notes, config):
    """Generate the README notes section for one language.

    Parameters:
        notes: Notes for the current language.
        paired_notes: Matching notes for the other language, keyed by relative path.
        config: Language-specific notes configuration.
    """
    groups = grouped_notes(notes)
    lines = [config.readme_heading, ""]
    for note_type in NOTE_TYPE_ORDER:
        lines.append(f"## {config.section_labels[note_type]}")
        section_notes = groups[note_type]
        if not section_notes:
            lines.extend(["", config.readme_empty_label, ""])
            continue

        lines.append("")
        entries = [
            render_readme_note(note, paired_notes[note.relative_path], config)
            for note in section_notes
        ]
        lines.append("\n\n".join(entries))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def update_contents_section(content, config):
    """Update the README contents list with notes entries.

    Parameters:
        content: Original README content.
        config: Language-specific notes configuration.
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

    blogs_line = "- [Blogs](#blogs)" if config.key == "en" else "- [博客](#博客)"
    try:
        blogs_index = new_block.index(blogs_line)
    except ValueError:
        blogs_index = len(new_block)
    new_block[blogs_index:blogs_index] = config.contents_lines
    new_block.append("")

    updated = lines[:start_index + 1] + new_block + lines[end_index:]
    return "\n".join(updated).rstrip() + "\n"


def replace_notes_section(content, notes_section, config):
    """Replace or append the generated README notes section.

    Parameters:
        content: README content after contents-list normalization.
        notes_section: Generated notes section for the current language.
        config: Language-specific notes configuration.
    """
    lines = content.splitlines()
    try:
        start_index = lines.index(config.readme_heading)
    except ValueError:
        prefix = "\n".join(lines).rstrip()
        return prefix + "\n\n" + notes_section

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
    joined += notes_section.rstrip()
    if suffix:
        joined += "\n\n" + "\n".join(suffix).rstrip()
    return joined.rstrip() + "\n"


def expected_readme(config, notes, paired_notes):
    """Build expected README content with generated notes sections.

    Parameters:
        config: Language-specific notes configuration.
        notes: Notes for the current language.
        paired_notes: Matching notes for the other language, keyed by relative path.
    """
    content = config.readme_path.read_text(encoding = "utf-8")
    content = update_contents_section(content, config)
    notes_section = generate_readme_notes_section(
        notes = notes,
        paired_notes = paired_notes,
        config = config
    )
    return replace_notes_section(content, notes_section, config)


def expected_files(notes_by_language):
    """Build the expected generated file map.

    Parameters:
        notes_by_language: Mapping from language key to scanned note records.
    """
    zh_note_map = {note.relative_path: note for note in notes_by_language["zh"]}
    en_note_map = {note.relative_path: note for note in notes_by_language["en"]}
    files = {}
    for language, config in LANGUAGE_CONFIGS.items():
        notes = notes_by_language[language]
        paired_notes = zh_note_map if language == "en" else en_note_map
        files[config.index_path] = generate_index(notes, config)
        files[config.readme_path] = expected_readme(config, notes, paired_notes)
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
    """Run the notes sync checker.

    No parameters.
    """
    args = parse_args()
    notes_by_language = {
        language: scan_notes(config)
        for language, config in LANGUAGE_CONFIGS.items()
    }
    validate_bilingual_pairs(notes_by_language)
    files = expected_files(notes_by_language)
    out_of_sync = write_or_check(
        files = files,
        write = args.write
    )

    if args.write:
        logger.info("Wrote %d generated notes files", len(files))
        return 0

    if out_of_sync:
        logger.error("Notes files are out of sync: %s", ", ".join(out_of_sync))
        return 1

    logger.info("Notes files are in sync")
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    raise SystemExit(main())
