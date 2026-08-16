"""Apply synchronization plans with crash-safe and bounded mutations."""

import os
import sys
import time
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to Python path
sys.path.append(os.getcwd())

from feishu_wiki_sync.client import created_tokens, document_revision
from feishu_wiki_sync.content import render_homepage, render_managed_page
from feishu_wiki_sync.content import resolve_wiki_links
from feishu_wiki_sync.models import CONTENT_DIR, HOME_TITLE, MANIFEST_SCHEMA_VERSION
from feishu_wiki_sync.models import LarkCliError, RemotePage, SafetyError, TreeNode
from feishu_wiki_sync.models import new_manifest, stable_hash, utc_now
from feishu_wiki_sync.planner import expected_parent_token, find_staging_node
from feishu_wiki_sync.planner import find_title_collision, staging_title

logger = logging.getLogger(__name__)


MAX_WRITE_WORKERS = 2


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
        revision_id: Current remote document revision.
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
        current_revision = executor.fetch_document_revision(obj_token)
        if current_revision == previous_revision:
            return current_revision
        previous_revision = current_revision
    raise LarkCliError(
        f"Document revision did not stabilize after overwrite: {obj_token}"
    )


def checkpoint_home(executor, home_spec, manifest, home_revision):
    """Persist an in-progress checkpoint on the managed homepage.

    Parameters:
        executor: Lark CLI executor.
        home_spec: Homepage specification.
        manifest: Mutable in-progress manifest.
        home_revision: Current homepage revision.
    """
    manifest.schema_version = MANIFEST_SCHEMA_VERSION
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


def _created_page(spec, payload, commit, revision_id):
    """Build a managed page record from node-create output.

    Parameters:
        spec: Desired page specification.
        payload: Successful node-create response.
        commit: Current source commit.
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
        source_commit = commit,
        obj_edit_time = None
    )


def ensure_home(
    executor,
    space_id,
    specs,
    tree,
    manifest,
    home_revision,
    commit
):
    """Create or checkpoint the homepage before non-home mutations.

    Parameters:
        executor: Lark CLI executor.
        space_id: Target Wiki space identifier.
        specs: Desired page specifications.
        tree: Mutable remote tree.
        manifest: Existing manifest, or None.
        home_revision: Current homepage revision, or None.
        commit: Current source commit.
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
        home_revision = executor.fetch_document_revision(obj_token)
    else:
        payload = executor.create_node(space_id, staging_title(specs["home"]))
        node_token, obj_token = created_tokens(payload)
        home_revision = executor.fetch_document_revision(obj_token)
    current_title = staging.title if staging else staging_title(specs["home"])
    manifest.pages["home"] = RemotePage(
        key = "home",
        parent_key = None,
        node_token = node_token,
        obj_token = obj_token,
        title = current_title,
        content_hash = "",
        revision_id = -1,
        source_path = specs["home"].source_path,
        source_commit = commit,
        obj_edit_time = None
    )
    tree[node_token] = TreeNode(
        node_token = node_token,
        obj_token = obj_token,
        parent_node_token = "",
        title = current_title,
        has_child = False,
        obj_type = "docx",
        obj_edit_time = None
    )
    home_revision = checkpoint_home(
        executor = executor,
        home_spec = specs["home"],
        manifest = manifest,
        home_revision = home_revision
    )
    return manifest, home_revision


def _adopt_pending_node(spec, manifest, collision, executor, commit):
    """Adopt only a deterministic staging or explicitly pending node.

    Parameters:
        spec: Desired page specification.
        manifest: In-progress manifest.
        collision: Matching remote tree node.
        executor: Lark CLI executor.
        commit: Current source commit.
    """
    is_staging = collision.title == staging_title(spec)
    is_pending = (
        manifest.status == "in_progress"
        and manifest.pending_create_key == spec.key
        and collision.title == spec.title
    )
    if not is_staging and not is_pending:
        raise SafetyError(f"同名冲突，不能认领未知节点: {spec.key}")
    return RemotePage(
        key = spec.key,
        parent_key = spec.parent_key,
        node_token = collision.node_token,
        obj_token = collision.obj_token,
        title = collision.title,
        content_hash = "",
        revision_id = executor.fetch_document_revision(collision.obj_token),
        source_path = spec.source_path,
        source_commit = commit,
        obj_edit_time = collision.obj_edit_time
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
    """Create missing nodes parent-first and checkpoint their tokens.

    Parameters:
        executor: Lark CLI executor.
        space_id: Target Wiki space identifier.
        specs: Desired page specifications.
        tree: Mutable remote tree.
        manifest: Mutable in-progress manifest.
        home_revision: Current homepage revision.
        commit: Current source commit.
    """
    created_any = False
    for spec in sorted(specs.values(), key = lambda item: (item.depth, item.key)):
        if spec.key == "home":
            continue
        old = manifest.pages.get(spec.key)
        if old is not None and old.node_token in tree:
            continue
        collision = find_title_collision(spec, manifest.pages, tree)
        staging = find_staging_node(spec, manifest.pages, tree)
        if collision or staging:
            page = _adopt_pending_node(
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
            revision_id = executor.fetch_document_revision(obj_token)
            page = _created_page(
                spec = spec,
                payload = payload,
                commit = commit,
                revision_id = revision_id
            )
            tree[node_token] = TreeNode(
                node_token = node_token,
                obj_token = obj_token,
                parent_node_token = parent_token or "",
                title = staging_title(spec),
                has_child = False,
                obj_type = "docx",
                obj_edit_time = None
            )
        manifest.pages[spec.key] = page
        manifest.pending_create_key = None
        created_any = True

    if created_any:
        home_revision = checkpoint_home(
            executor = executor,
            home_spec = specs["home"],
            manifest = manifest,
            home_revision = home_revision
        )
    return home_revision


def rename_pages(executor, specs, tree, manifest):
    """Restore desired titles serially after collision checks.

    Parameters:
        executor: Lark CLI executor.
        specs: Desired page specifications.
        tree: Mutable remote tree.
        manifest: Mutable synchronization manifest.
    """
    renamed = set()
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
        renamed.add(spec.key)
    return renamed


def _updated_page(
    executor,
    spec,
    old,
    current_revision,
    current_edit_time,
    resolved_body,
    desired_hash,
    commit
):
    """Overwrite and stabilize one independent non-home document.

    Parameters:
        executor: Lark CLI executor.
        spec: Desired page specification.
        old: Existing managed page record.
        current_revision: Audited current remote revision.
        current_edit_time: Remote edit time observed before the overwrite.
        resolved_body: Desired body with Wiki links resolved.
        desired_hash: Desired content and media hash.
        commit: Current source commit.
    """
    content = render_managed_page(spec, resolved_body, commit)
    updated_revision = overwrite_page(
        executor = executor,
        page = old,
        revision_id = current_revision,
        content = content
    )
    updated_revision = stabilize_document_revision(
        executor = executor,
        obj_token = old.obj_token,
        starting_revision = updated_revision
    )
    node = executor.get_node(old.node_token)
    obj_edit_time = node.obj_edit_time
    if (
        current_edit_time is not None
        and obj_edit_time == current_edit_time
        and updated_revision != old.revision_id
    ):
        logger.warning(
            "Wiki obj_edit_time did not advance after updating %s; "
            "disabling its revision-read fast path",
            spec.key
        )
        obj_edit_time = None
    return RemotePage(
        key = spec.key,
        parent_key = spec.parent_key,
        node_token = old.node_token,
        obj_token = old.obj_token,
        title = spec.title,
        content_hash = desired_hash,
        revision_id = updated_revision,
        source_path = spec.source_path,
        source_commit = commit,
        obj_edit_time = obj_edit_time
    )


def synchronize_contents(
    executor,
    specs,
    manifest,
    tree,
    plan,
    commit,
    max_workers = MAX_WRITE_WORKERS
):
    """Synchronize non-home contents with bounded cross-document writes.

    Parameters:
        executor: Lark CLI executor.
        specs: Desired page specifications.
        manifest: Mutable synchronization manifest.
        tree: Current remote tree.
        plan: Audited synchronization plan.
        commit: Current source commit.
        max_workers: Maximum concurrent document overwrites.
    """
    node_tokens = {
        key: manifest.pages[key].node_token
        for key in specs
    }
    synchronized = {}
    update_inputs = []
    for spec in sorted(specs.values(), key = lambda item: (item.depth, item.key)):
        if spec.key == "home":
            continue
        old = manifest.pages[spec.key]
        resolved_body = resolve_wiki_links(spec.body, node_tokens)
        desired_hash = stable_hash(resolved_body, spec.media_paths)
        current_revision = plan.current_revisions.get(spec.key, old.revision_id)
        current_edit_time = plan.current_edit_times.get(
            spec.key,
            tree[old.node_token].obj_edit_time
        )
        needs_update = (
            desired_hash != old.content_hash
            or old.revision_id != current_revision
        )
        if needs_update:
            update_inputs.append(
                (
                    spec,
                    old,
                    current_revision,
                    current_edit_time,
                    resolved_body,
                    desired_hash,
                )
            )
            continue
        edit_time = plan.current_edit_times.get(
            spec.key,
            tree[old.node_token].obj_edit_time
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
            source_commit = old.source_commit,
            obj_edit_time = edit_time
        )

    if update_inputs:
        worker_count = min(max_workers, len(update_inputs))
        logger.info(
            "Updating %s document(s) with at most %s concurrent writer(s)",
            len(update_inputs),
            worker_count
        )
        with ThreadPoolExecutor(max_workers = worker_count) as pool:
            future_to_key = {
                pool.submit(
                    _updated_page,
                    executor,
                    spec,
                    old,
                    current_revision,
                    current_edit_time,
                    resolved_body,
                    desired_hash,
                    commit
                ): spec.key
                for (
                    spec,
                    old,
                    current_revision,
                    current_edit_time,
                    resolved_body,
                    desired_hash,
                ) in update_inputs
            }
            try:
                for future in as_completed(future_to_key):
                    key = future_to_key[future]
                    synchronized[key] = future.result()
            except Exception:
                for future in future_to_key:
                    future.cancel()
                raise

    return {
        key: synchronized[key]
        for key in sorted(synchronized)
    }


def delete_stale_pages(executor, space_id, specs, tree, manifest):
    """Delete only stale manifest-owned nodes, deepest and child-free first.

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


def finalize_homepage(
    executor,
    specs,
    manifest,
    home_revision,
    synchronized,
    commit
):
    """Write the final successful homepage and v2 manifest last.

    Parameters:
        executor: Lark CLI executor.
        specs: Desired page specifications.
        manifest: Mutable synchronization manifest.
        home_revision: Current homepage revision.
        synchronized: Successfully synchronized non-home pages.
        commit: Current source commit.
    """
    home = manifest.pages["home"]
    node_tokens = {
        key: page.node_token
        for key, page in synchronized.items()
    }
    node_tokens["home"] = home.node_token
    resolved_body = resolve_wiki_links(specs["home"].body, node_tokens)
    synchronized["home"] = RemotePage(
        key = "home",
        parent_key = None,
        node_token = home.node_token,
        obj_token = home.obj_token,
        title = specs["home"].title,
        content_hash = stable_hash(resolved_body, specs["home"].media_paths),
        revision_id = -1,
        source_path = specs["home"].source_path,
        source_commit = commit,
        obj_edit_time = None
    )
    manifest.schema_version = MANIFEST_SCHEMA_VERSION
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


def apply_sync_plan(
    executor,
    space_id,
    specs,
    snapshot,
    plan,
    commit,
    max_workers = MAX_WRITE_WORKERS
):
    """Apply one audited plan and commit the homepage manifest last.

    Parameters:
        executor: Lark CLI executor.
        space_id: Target Wiki space identifier.
        specs: Desired page specifications.
        snapshot: Remote snapshot used to build the plan.
        plan: Audited synchronization plan.
        commit: Current source commit.
        max_workers: Maximum concurrent non-home document writes.
    """
    if not plan.has_actions:
        logger.info("No remote changes detected; skipping all writes")
        return snapshot.home_revision

    tree = snapshot.tree
    manifest = snapshot.manifest
    home_revision = snapshot.home_revision
    if manifest is None or plan.has_non_home_actions:
        manifest, home_revision = ensure_home(
            executor = executor,
            space_id = space_id,
            specs = specs,
            tree = tree,
            manifest = manifest,
            home_revision = home_revision,
            commit = commit
        )
    else:
        manifest.commit = commit

    home_revision = create_missing_pages(
        executor = executor,
        space_id = space_id,
        specs = specs,
        tree = tree,
        manifest = manifest,
        home_revision = home_revision,
        commit = commit
    )
    rename_pages(executor, specs, tree, manifest)
    synchronized = synchronize_contents(
        executor = executor,
        specs = specs,
        manifest = manifest,
        tree = tree,
        plan = plan,
        commit = commit,
        max_workers = max_workers
    )
    delete_stale_pages(executor, space_id, specs, tree, manifest)
    return finalize_homepage(
        executor = executor,
        specs = specs,
        manifest = manifest,
        home_revision = home_revision,
        synchronized = synchronized,
        commit = commit
    )
