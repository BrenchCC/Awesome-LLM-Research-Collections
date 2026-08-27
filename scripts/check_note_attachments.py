"""Validate and package explicitly authorized note downloads."""

import os
import sys
import json
import shutil
import hashlib
import logging
import argparse
from pathlib import Path

# Add project root to Python path
sys.path.append(os.getcwd())
sys.path.append(str(Path(__file__).resolve().parent))

from sync_notes import LANGUAGE_CONFIGS
from sync_notes import scan_notes, validate_bilingual_pairs
from feishu_wiki_sync.content import parse_note_downloads
from feishu_wiki_sync.content import validate_bilingual_downloads
from feishu_wiki_sync.models import repo_relative

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description = "Validate and package opt-in PDF/TeX note downloads."
    )
    parser.add_argument(
        "--manifest",
        type = Path,
        help = "Optional JSON manifest output path."
    )
    parser.add_argument(
        "--bundle-dir",
        type = Path,
        help = "Optional directory receiving authorized download files."
    )
    return parser.parse_args()


def collect_note_downloads():
    """Collect validated download metadata from all bilingual notes."""
    notes_by_language = {
        language: scan_notes(config)
        for language, config in LANGUAGE_CONFIGS.items()
    }
    validate_bilingual_pairs(notes_by_language)
    validate_bilingual_downloads(notes_by_language)
    entries = []
    for language in ["zh", "en"]:
        for note in notes_by_language[language]:
            downloads = parse_note_downloads(
                text = note.source_path.read_text(encoding = "utf-8"),
                source_path = note.source_path.resolve()
            )
            for label, download_path in downloads:
                entries.append(
                    {
                        "language": language,
                        "note": repo_relative(note.source_path.resolve()),
                        "label": label,
                        "path": repo_relative(download_path),
                        "size": download_path.stat().st_size,
                        "sha256": hashlib.sha256(download_path.read_bytes()).hexdigest(),
                    }
                )
    return entries


def write_bundle(entries, bundle_dir):
    """Copy each unique authorized file into an artifact directory.

    Parameters:
        entries: Validated attachment manifest entries.
        bundle_dir: Destination artifact directory.
    """
    bundle_dir.mkdir(parents = True, exist_ok = True)
    for relative in sorted({entry["path"] for entry in entries}):
        source = Path(relative)
        destination = bundle_dir / relative
        destination.parent.mkdir(parents = True, exist_ok = True)
        shutil.copy2(source, destination)


def write_manifest(entries, manifest_path):
    """Write a deterministic attachment manifest.

    Parameters:
        entries: Validated attachment manifest entries.
        manifest_path: Destination JSON path.
    """
    manifest_path.parent.mkdir(parents = True, exist_ok = True)
    manifest = {
        "schema_version": 1,
        "attachment_count": len(entries),
        "unique_file_count": len({entry["path"] for entry in entries}),
        "attachments": entries,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii = False, indent = 2, sort_keys = True) + "\n",
        encoding = "utf-8"
    )


def main():
    """Validate note downloads and optionally create CI artifacts."""
    args = parse_args()
    entries = collect_note_downloads()
    if args.bundle_dir:
        write_bundle(entries, args.bundle_dir)
    if args.manifest:
        write_manifest(entries, args.manifest)
    logger.info(
        "Validated %d authorized note-download links across %d unique files.",
        len(entries),
        len({entry["path"] for entry in entries})
    )


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    main()
