"""Discover remote Wiki state and compute deterministic synchronization plans."""

import os
import sys
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

# Add project root to Python path
sys.path.append(os.getcwd())

from feishu_wiki_sync.client import document_content, document_revision
from feishu_wiki_sync.client import tree_node_from_json
from feishu_wiki_sync.content import content_hash_for_spec
from feishu_wiki_sync.models import HOME_TITLE, MANIFEST_MARKER
from feishu_wiki_sync.models import parse_manifest
from feishu_wiki_sync.models import SafetyError, SyncAction, SyncManifest, TreeNode


MAX_READ_WORKERS = 4


@dataclass
class RemoteSnapshot:
    """Contain one consistent read of the managed Wiki state.

    Parameters:
        tree: Discovered Wiki nodes keyed by node token.
        manifest: Parsed managed manifest, or None before initialization.
        home_revision: Current homepage Docx revision, or None before initialization.
    """

    tree: Dict[str, TreeNode]
    manifest: Optional[SyncManifest]
    home_revision: Optional[int]


@dataclass
class SyncPlan:
    """Describe the complete read-only decision used by apply.

    Parameters:
        actions: Ordered user-visible synchronization actions.
        desired_hashes: Resolved desired content hashes keyed by page key.
        current_revisions: Current revisions for existing non-home pages.
        current_edit_times: Current Wiki object edit times for existing pages.
        audited_revision_count: Number of lightweight revision reads performed.
    """

    actions: List[SyncAction]
    desired_hashes: Dict[str, str]
    current_revisions: Dict[str, int]
    current_edit_times: Dict[str, Optional[str]]
    audited_revision_count: int

    @property
    def has_actions(self):
        """Return whether the plan contains any remote mutation.

        No parameters.
        """
        return bool(self.actions)

    @property
    def has_non_home_actions(self):
        """Return whether any action mutates a non-home page.

        No parameters.
        """
        return any(action.key != "home" for action in self.actions)


def _list_frontier(executor, space_id, parents, max_workers):
    """List one breadth-first frontier with bounded concurrency.

    Parameters:
        executor: Lark CLI executor.
        space_id: Target Wiki space identifier.
        parents: Parent node tokens, using None for the root.
        max_workers: Maximum concurrent read commands.
    """
    if len(parents) == 1:
        return [executor.list_nodes(space_id, parents[0])]
    worker_count = min(max_workers, len(parents))
    with ThreadPoolExecutor(max_workers = worker_count) as pool:
        futures = [
            pool.submit(executor.list_nodes, space_id, parent)
            for parent in parents
        ]
        return [future.result() for future in futures]


def discover_tree(executor, space_id, max_workers = MAX_READ_WORKERS):
    """Recursively discover the Wiki tree with per-level parallel reads.

    Parameters:
        executor: Lark CLI executor.
        space_id: Target Wiki space identifier.
        max_workers: Maximum concurrent node-list commands.
    """
    nodes = {}
    frontier = [None]
    while frontier:
        raw_groups = _list_frontier(
            executor = executor,
            space_id = space_id,
            parents = frontier,
            max_workers = max_workers
        )
        next_frontier = []
        for parent, raw_nodes in zip(frontier, raw_groups):
            sibling_titles = {}
            for raw_node in raw_nodes:
                node = tree_node_from_json(raw_node)
                if node.node_token in nodes:
                    raise SafetyError(
                        f"Wiki tree contains duplicate token: {node.node_token}"
                    )
                sibling_titles.setdefault(node.title, []).append(node.node_token)
                nodes[node.node_token] = node
                if node.has_child:
                    next_frontier.append(node.node_token)
            duplicates = [
                title
                for title, tokens in sibling_titles.items()
                if len(tokens) > 1
            ]
            if duplicates:
                parent_label = parent or "<root>"
                raise SafetyError(
                    f"Wiki parent {parent_label} contains duplicate titles: "
                    + ", ".join(duplicates)
                )
        frontier = next_frontier
    return nodes


def load_remote_state(executor, space_id, max_workers = MAX_READ_WORKERS):
    """Discover the Wiki tree and read the managed homepage manifest.

    Parameters:
        executor: Lark CLI executor.
        space_id: Target Wiki space identifier.
        max_workers: Maximum concurrent tree-discovery reads.
    """
    tree = discover_tree(executor, space_id, max_workers = max_workers)
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
        manifest_candidates.append(
            (home_node, parse_manifest(document_content(payload)), payload)
        )
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
            return RemoteSnapshot(tree = tree, manifest = None, home_revision = None)

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
    return RemoteSnapshot(
        tree = tree,
        manifest = manifest,
        home_revision = document_revision(payload)
    )


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
    """Return the deterministic title used for crash-safe creation.

    Parameters:
        spec: Desired page specification.
    """
    key_hash = hashlib.sha256(spec.key.encode("utf-8")).hexdigest()[:12]
    return f"⏳ {spec.title[:80]} [sync:{key_hash}]"


def find_staging_node(spec, pages, tree):
    """Find a uniquely marked node left by an interrupted create.

    Parameters:
        spec: Desired page specification.
        pages: Known managed pages.
        tree: Discovered remote tree.
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
        pages: Known managed pages.
        tree: Discovered remote tree.
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


def placeholder_tokens(specs, pages, tree):
    """Build deterministic tokens for hashing pages not created yet.

    Parameters:
        specs: Desired page specifications.
        pages: Known managed pages.
        tree: Discovered remote tree.
    """
    tokens = {}
    for key in specs:
        if key in pages and pages[key].node_token in tree:
            tokens[key] = pages[key].node_token
        else:
            digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
            tokens[key] = "pending-" + digest
    return tokens


def _fetch_candidate_revisions(
    executor,
    pages,
    candidate_keys,
    max_workers
):
    """Fetch lightweight Docx revisions for candidate pages.

    Parameters:
        executor: Lark CLI executor.
        pages: Managed pages keyed by stable key.
        candidate_keys: Page keys requiring a revision check.
        max_workers: Maximum concurrent revision reads.
    """
    ordered_keys = sorted(candidate_keys)
    if not ordered_keys:
        return {}
    worker_count = min(max_workers, len(ordered_keys))
    with ThreadPoolExecutor(max_workers = worker_count) as pool:
        futures = [
            pool.submit(executor.fetch_document_revision, pages[key].obj_token)
            for key in ordered_keys
        ]
        return {
            key: future.result()
            for key, future in zip(ordered_keys, futures)
        }


def _compute_desired_hashes(specs, pages, tree):
    """Compute each desired page hash exactly once for the current plan.

    Parameters:
        specs: Desired page specifications.
        pages: Known managed pages.
        tree: Discovered remote tree.
    """
    node_tokens = placeholder_tokens(specs, pages, tree)
    return {
        key: content_hash_for_spec(spec, node_tokens)
        for key, spec in specs.items()
    }


def _revision_audit(
    executor,
    specs,
    manifest,
    tree,
    desired_hashes,
    max_workers
):
    """Audit only pages whose source or remote metadata may have changed.

    Wiki ``obj_edit_time`` is the remote change-detector contract for the
    zero-revision-read fast path. A missing value is always treated
    conservatively and keeps that page in the revision audit set.

    Parameters:
        executor: Lark CLI executor.
        specs: Desired page specifications.
        manifest: Existing synchronization manifest.
        tree: Discovered remote tree.
        desired_hashes: Precomputed desired hashes.
        max_workers: Maximum concurrent revision reads.
    """
    current_edit_times = {}
    candidates = set()
    for key, old in manifest.pages.items():
        if key == "home" or key not in specs or old.node_token not in tree:
            continue
        node = tree[old.node_token]
        current_edit_times[key] = node.obj_edit_time
        source_changed = desired_hashes[key] != old.content_hash
        metadata_unknown = old.obj_edit_time is None or node.obj_edit_time is None
        metadata_changed = old.obj_edit_time != node.obj_edit_time
        if source_changed or metadata_unknown or metadata_changed:
            candidates.add(key)

    fetched = _fetch_candidate_revisions(
        executor = executor,
        pages = manifest.pages,
        candidate_keys = candidates,
        max_workers = max_workers
    )
    current_revisions = {
        key: fetched.get(key, old.revision_id)
        for key, old in manifest.pages.items()
        if key != "home" and key in specs and old.node_token in tree
    }
    return current_revisions, current_edit_times, len(candidates)


def build_sync_plan(
    specs,
    snapshot,
    executor,
    commit,
    max_workers = MAX_READ_WORKERS
):
    """Build the complete plan and the audited state needed by apply.

    Parameters:
        specs: Desired page specifications.
        snapshot: Previously loaded remote snapshot.
        executor: Lark CLI executor.
        commit: Current source commit.
        max_workers: Maximum concurrent revision reads.
    """
    manifest = snapshot.manifest
    tree = snapshot.tree
    pages = manifest.pages if manifest else {}
    desired_hashes = _compute_desired_hashes(specs, pages, tree)
    current_revisions = {}
    current_edit_times = {}
    audited_revision_count = 0
    if manifest is not None:
        validate_manifest_tree(manifest, tree)
        (
            current_revisions,
            current_edit_times,
            audited_revision_count,
        ) = _revision_audit(
            executor = executor,
            specs = specs,
            manifest = manifest,
            tree = tree,
            desired_hashes = desired_hashes,
            max_workers = max_workers
        )

    actions = []
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
            actions.append(
                SyncAction("rename", spec.key, f"{node.title} -> {spec.title}")
            )
        if spec.key == "home":
            continue
        source_changed = desired_hashes[spec.key] != old.content_hash
        revision_changed = current_revisions[spec.key] != old.revision_id
        if source_changed or revision_changed:
            reason = "source/media changed" if source_changed else "remote revision changed"
            actions.append(SyncAction("update", spec.key, reason))

    stale = set(pages) - set(specs)
    for key in sorted(stale, key = lambda item: (-item.count("/"), item)):
        actions.append(SyncAction("delete", key, pages[key].title))

    home = pages.get("home")
    home_exists = home is not None and home.node_token in tree
    if home_exists:
        metadata_changed = any(
            pages[key].obj_edit_time != edit_time
            for key, edit_time in current_edit_times.items()
        )
        home_changed = desired_hashes["home"] != home.content_hash
        state_changed = manifest.status != "complete" or manifest.commit != commit
        other_actions = bool(actions)
        if metadata_changed or home_changed or state_changed or other_actions:
            actions.append(SyncAction("update", "home", "sync status and manifest"))

    return SyncPlan(
        actions = actions,
        desired_hashes = desired_hashes,
        current_revisions = current_revisions,
        current_edit_times = current_edit_times,
        audited_revision_count = audited_revision_count
    )
