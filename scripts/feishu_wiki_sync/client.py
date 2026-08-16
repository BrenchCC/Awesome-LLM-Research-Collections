"""lark-cli adapter layer for Feishu Wiki synchronization."""

import os
import sys
import json
import time
import logging
import subprocess

# Add project root to Python path
sys.path.append(os.getcwd())

from feishu_wiki_sync.models import LarkCliError
from feishu_wiki_sync.models import TreeNode

logger = logging.getLogger(__name__)


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


def tree_node_from_json(raw_node):
    """Validate and normalize one Wiki node object.

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
        obj_type = raw_node.get("obj_type", "docx"),
        obj_edit_time = raw_node.get("obj_edit_time")
    )


def node_from_payload(payload):
    """Extract one normalized Wiki node from a raw API payload.

    Parameters:
        payload: Successful get-node response.
    """
    raw_node = first_value(payload, "node")
    if not isinstance(raw_node, dict):
        raise LarkCliError("wiki get_node response did not contain node data")
    return tree_node_from_json(raw_node)


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

    def run_api(self, method, path, params = None):
        """Run one raw OpenAPI request through lark-cli.

        Parameters:
            method: HTTP method such as GET or POST.
            path: OpenAPI path beginning with /open-apis/.
            params: Optional query-parameter mapping.
        """
        arguments = ["api", method, path]
        if params is not None:
            arguments.extend(
                [
                    "--params",
                    json.dumps(params, ensure_ascii = False, separators = (",", ":"))
                ]
            )
        return self.run(arguments)

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

    def fetch_document_revision(self, obj_token):
        """Fetch only a document revision through the lightweight document API.

        Parameters:
            obj_token: Backing Docx token.
        """
        payload = self.run_api(
            "GET",
            f"/open-apis/docx/v1/documents/{obj_token}"
        )
        return document_revision(payload)

    def get_node(self, node_token):
        """Fetch one Wiki node so callers can refresh obj_edit_time.

        Parameters:
            node_token: Wiki node token.
        """
        payload = self.run_api(
            "GET",
            "/open-apis/wiki/v2/spaces/get_node",
            params = {"token": node_token}
        )
        return node_from_payload(payload)

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
