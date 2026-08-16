"""Data models and manifest parsing for Feishu Wiki synchronization."""

import os
import re
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

# Add project root to Python path
sys.path.append(os.getcwd())


REPOSITORY_URL = "https://github.com/BrenchCC/Awesome-LLM-Research-Collections"
HOME_TITLE = "首页 / Home"
MANIFEST_MARKER = "FEISHU_SYNC_MANIFEST_V1"
MANIFEST_SCHEMA_VERSION = 2
MAX_MEDIA_BYTES = 20 * 1024 * 1024
RUNTIME_DIR = Path(".feishu-wiki-sync")
CONTENT_DIR = RUNTIME_DIR / "content"
ASSET_DIR = RUNTIME_DIR / "assets"


class SyncError(RuntimeError):
    """Base error for synchronization failures."""


class SafetyError(SyncError):
    """Error raised when remote state is ambiguous or unsafe to mutate."""


class LarkCliError(SyncError):
    """Error raised when lark-cli returns a permanent failure."""


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
        obj_edit_time: Last known remote object edit time, or None.
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
    obj_edit_time: Optional[str]


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
        _validate_manifest_envelope(data)
        schema_version = data["schema_version"]
        if schema_version == 1:
            return _manifest_from_v1(data)
        if schema_version == MANIFEST_SCHEMA_VERSION:
            return _manifest_from_v2(data)
        raise SafetyError("同步清单版本不受支持")


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
        obj_edit_time: Latest object edit time reported by the Wiki API.
    """

    node_token: str
    obj_token: str
    parent_node_token: str
    title: str
    has_child: bool
    obj_type: str = "docx"
    obj_edit_time: Optional[str] = None


@dataclass
class SyncAction:
    """Describe one proposed synchronization operation.

    Parameters:
        operation: create, update, rename, delete, or recover.
        key: Stable page key.
        detail: Human-readable reason.
    """

    operation: str
    key: str
    detail: str


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


def _validate_manifest_envelope(data):
    """Validate the common manifest envelope before schema-specific parsing.

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
    if data["repository"] != REPOSITORY_URL:
        raise SafetyError("同步清单属于另一个 GitHub 仓库")
    if data["status"] not in {"in_progress", "complete"}:
        raise SafetyError("同步清单状态无效")
    if not isinstance(data["pages"], dict):
        raise SafetyError("同步清单 pages 字段必须是对象")


def _manifest_from_v1(data):
    """Parse a strict v1 manifest and migrate it to the v2 in-memory shape.

    Parameters:
        data: Decoded v1 manifest mapping.
    """
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
    pages = _parse_pages(
        raw_pages = data["pages"],
        page_fields = page_fields,
        default_obj_edit_time = None
    )
    return SyncManifest(
        schema_version = MANIFEST_SCHEMA_VERSION,
        repository = data["repository"],
        space_id = data["space_id"],
        status = data["status"],
        commit = data["commit"],
        updated_at = data["updated_at"],
        pending_create_key = data["pending_create_key"],
        pages = pages
    )


def _manifest_from_v2(data):
    """Parse a strict v2 manifest without allowing unknown page fields.

    Parameters:
        data: Decoded v2 manifest mapping.
    """
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
        "obj_edit_time",
    }
    pages = _parse_pages(
        raw_pages = data["pages"],
        page_fields = page_fields,
        default_obj_edit_time = "__missing__"
    )
    return SyncManifest(
        schema_version = data["schema_version"],
        repository = data["repository"],
        space_id = data["space_id"],
        status = data["status"],
        commit = data["commit"],
        updated_at = data["updated_at"],
        pending_create_key = data["pending_create_key"],
        pages = pages
    )


def _parse_pages(raw_pages, page_fields, default_obj_edit_time):
    """Validate and convert manifest page records.

    Parameters:
        raw_pages: Raw page mapping from manifest JSON.
        page_fields: Exact field set allowed for each page entry.
        default_obj_edit_time: Placeholder for v1 migration or strict v2 parsing.
    """
    pages = {}
    seen_nodes = set()
    for key, raw_page in raw_pages.items():
        if not isinstance(raw_page, dict) or set(raw_page) != page_fields:
            raise SafetyError(f"同步清单页面字段无效: {key}")
        if raw_page["key"] != key:
            raise SafetyError(f"同步清单页面键不一致: {key}")
        if raw_page["node_token"] in seen_nodes:
            raise SafetyError("同步清单包含重复 node token")
        seen_nodes.add(raw_page["node_token"])
        page_data = dict(raw_page)
        if default_obj_edit_time != "__missing__":
            page_data["obj_edit_time"] = default_obj_edit_time
        _validate_remote_page(page_data)
        pages[key] = RemotePage(**page_data)
    return pages


def _validate_remote_page(page_data):
    """Validate one decoded remote page record before dataclass construction.

    Parameters:
        page_data: Mutable page mapping in the target v2 shape.
    """
    if page_data["parent_key"] is not None and not isinstance(page_data["parent_key"], str):
        raise SafetyError(f"同步清单页面 parent_key 无效: {page_data['key']}")
    if not isinstance(page_data["revision_id"], int):
        raise SafetyError(f"同步清单页面 revision_id 无效: {page_data['key']}")
    obj_edit_time = page_data.get("obj_edit_time")
    if obj_edit_time is not None and not isinstance(obj_edit_time, str):
        raise SafetyError(f"同步清单页面 obj_edit_time 无效: {page_data['key']}")
    string_fields = [
        "key",
        "node_token",
        "obj_token",
        "title",
        "content_hash",
        "source_path",
        "source_commit",
    ]
    for field_name in string_fields:
        if not isinstance(page_data[field_name], str):
            raise SafetyError(f"同步清单页面 {field_name} 无效: {page_data['key']}")
