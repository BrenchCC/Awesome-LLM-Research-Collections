"""Tests for Feishu Wiki planning, concurrency, and mutation recovery."""

import os
import sys
import time
import threading
import unittest
from pathlib import Path

# Add project root to Python path
sys.path.append(os.getcwd())
sys.path.append(str(Path.cwd() / "scripts"))

from feishu_wiki_sync.content import content_hash_for_spec
from feishu_wiki_sync.engine import apply_sync_plan, delete_stale_pages
from feishu_wiki_sync.engine import stabilize_document_revision
from feishu_wiki_sync.models import HOME_TITLE, MANIFEST_SCHEMA_VERSION
from feishu_wiki_sync.models import LarkCliError, PageSpec, RemotePage, SafetyError
from feishu_wiki_sync.models import SyncManifest, TreeNode
from feishu_wiki_sync.planner import RemoteSnapshot, build_sync_plan, discover_tree
from feishu_wiki_sync.planner import staging_title


class FakeExecutor:
    """Model independent Docx documents and record bounded concurrency."""

    def __init__(self, delay = 0):
        """Initialize the fake remote service.

        Parameters:
            delay: Artificial delay applied to read and write calls.
        """
        self.delay = delay
        self.sleep = lambda _: None
        self.nodes = {}
        self.documents = {}
        self.updated = []
        self.deleted = []
        self.renamed = []
        self.created = []
        self.revision_reads = 0
        self.active_reads = 0
        self.max_active_reads = 0
        self.active_writes = 0
        self.max_active_writes = 0
        self.fail_obj_token = None
        self.lock = threading.Lock()

    def add_page(self, page, edit_time):
        """Add one managed page to the fake remote service.

        Parameters:
            page: Managed page fixture.
            edit_time: Current Wiki object edit time.
        """
        self.documents[page.obj_token] = {
            "content": "",
            "revision_id": page.revision_id if page.revision_id >= 0 else 1,
        }
        self.nodes[page.node_token] = TreeNode(
            node_token = page.node_token,
            obj_token = page.obj_token,
            parent_node_token = "",
            title = page.title,
            has_child = False,
            obj_type = "docx",
            obj_edit_time = edit_time
        )

    def fetch_document_revision(self, obj_token):
        """Return one revision while recording read concurrency.

        Parameters:
            obj_token: Backing Docx token.
        """
        with self.lock:
            self.revision_reads += 1
            self.active_reads += 1
            self.max_active_reads = max(self.max_active_reads, self.active_reads)
        try:
            if self.delay:
                time.sleep(self.delay)
            return self.documents[obj_token]["revision_id"]
        finally:
            with self.lock:
                self.active_reads -= 1

    def overwrite_document(self, obj_token, revision_id, content_path):
        """Overwrite one document while recording write concurrency.

        Parameters:
            obj_token: Backing Docx token.
            revision_id: Expected current revision.
            content_path: Runtime Markdown path.
        """
        with self.lock:
            self.active_writes += 1
            self.max_active_writes = max(self.max_active_writes, self.active_writes)
        try:
            if self.delay:
                time.sleep(self.delay)
            if obj_token == self.fail_obj_token:
                raise LarkCliError("forced update failure")
            document = self.documents[obj_token]
            if document["revision_id"] != revision_id:
                raise LarkCliError("revision conflict")
            document["content"] = content_path.read_text(encoding = "utf-8")
            document["revision_id"] += 1
            self.updated.append(obj_token)
            return {
                "ok": True,
                "data": {"revision_id": document["revision_id"]},
            }
        finally:
            with self.lock:
                self.active_writes -= 1

    def get_node(self, node_token):
        """Return refreshed node metadata after an update.

        Parameters:
            node_token: Wiki node token.
        """
        node = self.nodes[node_token]
        node.obj_edit_time = "refreshed"
        return node

    def rename_node(self, node_token, title):
        """Record one title update.

        Parameters:
            node_token: Wiki node token.
            title: Desired title.
        """
        self.renamed.append((node_token, title))
        self.nodes[node_token].title = title
        return {"ok": True}

    def delete_node(self, space_id, node_token):
        """Record one safe Wiki node deletion.

        Parameters:
            space_id: Target Wiki space identifier.
            node_token: Wiki node token.
        """
        self.deleted.append((space_id, node_token))
        return {"ok": True}

    def create_node(self, space_id, title, parent_node_token = None):
        """Create one blank fake Docx Wiki node.

        Parameters:
            space_id: Target Wiki space identifier.
            title: Initial node title.
            parent_node_token: Optional parent node token.
        """
        index = len(self.created) + 1
        node_token = f"created-{index}"
        obj_token = f"doc-created-{index}"
        self.created.append((space_id, title, parent_node_token, node_token))
        self.documents[obj_token] = {"content": "", "revision_id": 1}
        self.nodes[node_token] = TreeNode(
            node_token = node_token,
            obj_token = obj_token,
            parent_node_token = parent_node_token or "",
            title = title,
            has_child = False,
            obj_edit_time = None
        )
        return {
            "ok": True,
            "data": {"node_token": node_token, "obj_token": obj_token},
        }


def remote_page(
    key,
    node_token,
    title,
    content_hash,
    revision_id,
    obj_edit_time
):
    """Build one concise v2 managed page fixture.

    Parameters:
        key: Stable page key.
        node_token: Wiki node token.
        title: Current page title.
        content_hash: Stored canonical content hash.
        revision_id: Stored Docx revision.
        obj_edit_time: Stored Wiki object edit time.
    """
    return RemotePage(
        key = key,
        parent_key = None,
        node_token = node_token,
        obj_token = f"doc-{node_token}",
        title = title,
        content_hash = content_hash,
        revision_id = revision_id,
        source_path = "source.md",
        source_commit = "commit-a",
        obj_edit_time = obj_edit_time
    )


def build_state(child_count = 1, changed_source = False, delay = 0):
    """Build matched local and remote state for planner tests.

    Parameters:
        child_count: Number of non-home pages.
        changed_source: Whether desired child bodies differ from stored hashes.
        delay: Artificial fake remote delay.
    """
    specs = {
        "home": PageSpec("home", None, HOME_TITLE, "README.md", "home body"),
    }
    pages = {}
    tree = {}
    executor = FakeExecutor(delay = delay)
    for index in range(child_count):
        key = f"child-{index}"
        body = f"body-{index}"
        specs[key] = PageSpec(key, None, f"Child {index}", "source.md", body)
        stored_body = "old body" if changed_source else body
        content_hash = content_hash_for_spec(specs[key], {key: f"node-{index}"})
        if changed_source:
            content_hash = content_hash_for_spec(
                PageSpec(key, None, f"Child {index}", "source.md", stored_body),
                {key: f"node-{index}"}
            )
        page = remote_page(
            key = key,
            node_token = f"node-{index}",
            title = f"Child {index}",
            content_hash = content_hash,
            revision_id = 1,
            obj_edit_time = "10"
        )
        pages[key] = page
        executor.add_page(page, "10")
        tree[page.node_token] = executor.nodes[page.node_token]

    node_tokens = {key: page.node_token for key, page in pages.items()}
    node_tokens["home"] = "home-token"
    home_hash = content_hash_for_spec(specs["home"], node_tokens)
    home = remote_page(
        key = "home",
        node_token = "home-token",
        title = HOME_TITLE,
        content_hash = home_hash,
        revision_id = -1,
        obj_edit_time = None
    )
    pages["home"] = home
    executor.add_page(home, "10")
    tree[home.node_token] = executor.nodes[home.node_token]
    manifest = SyncManifest(
        schema_version = MANIFEST_SCHEMA_VERSION,
        repository = "https://github.com/BrenchCC/Awesome-LLM-Research-Collections",
        space_id = "space",
        status = "complete",
        commit = "commit-a",
        updated_at = "2026-08-15T00:00:00+00:00",
        pending_create_key = None,
        pages = pages
    )
    snapshot = RemoteSnapshot(
        tree = tree,
        manifest = manifest,
        home_revision = 1
    )
    return specs, snapshot, executor


class PlannerFastPathTests(unittest.TestCase):
    """Verify metadata filtering and bounded read concurrency."""

    def test_v2_no_change_uses_zero_revision_reads_and_zero_writes(self):
        """Skip every per-page call when hashes and edit times match.

        Parameters:
            self: Current test case.
        """
        specs, snapshot, executor = build_state(child_count = 95)
        self.assertEqual(len(specs), 96)
        plan = build_sync_plan(specs, snapshot, executor, "commit-a")
        self.assertEqual(plan.actions, [])
        self.assertEqual(plan.audited_revision_count, 0)
        apply_sync_plan(executor, "space", specs, snapshot, plan, "commit-a")
        self.assertEqual(executor.revision_reads, 0)
        self.assertEqual(executor.updated, [])

    def test_edit_time_change_with_same_revision_refreshes_only_manifest(self):
        """Persist metadata drift without overwriting unchanged content.

        Parameters:
            self: Current test case.
        """
        specs, snapshot, executor = build_state()
        snapshot.tree["node-0"].obj_edit_time = "11"
        plan = build_sync_plan(specs, snapshot, executor, "commit-a")
        self.assertEqual(plan.audited_revision_count, 1)
        self.assertEqual(
            [(action.operation, action.key) for action in plan.actions],
            [("update", "home")]
        )
        apply_sync_plan(executor, "space", specs, snapshot, plan, "commit-a")
        self.assertEqual(executor.updated, ["doc-home-token"])
        self.assertEqual(snapshot.manifest.pages["child-0"].obj_edit_time, "11")

    def test_unchanged_edit_time_is_the_zero_read_change_signal(self):
        """Use matching Wiki edit metadata as the documented fast-path contract.

        Parameters:
            self: Current test case.
        """
        specs, snapshot, executor = build_state()
        executor.documents["doc-node-0"]["revision_id"] = 2
        plan = build_sync_plan(specs, snapshot, executor, "commit-a")
        self.assertEqual(plan.audited_revision_count, 0)
        self.assertEqual(executor.revision_reads, 0)
        self.assertEqual(plan.actions, [])

    def test_missing_edit_time_stays_on_the_conservative_audit_path(self):
        """Audit a page every run when Wiki edit metadata is unavailable.

        Parameters:
            self: Current test case.
        """
        specs, snapshot, executor = build_state()
        snapshot.manifest.pages["child-0"].obj_edit_time = None
        snapshot.tree["node-0"].obj_edit_time = None
        plan = build_sync_plan(specs, snapshot, executor, "commit-a")
        self.assertEqual(plan.audited_revision_count, 1)
        self.assertEqual(executor.revision_reads, 1)

    def test_candidate_revision_reads_use_at_most_four_workers(self):
        """Audit many metadata candidates with bounded read concurrency.

        Parameters:
            self: Current test case.
        """
        specs, snapshot, executor = build_state(child_count = 8, delay = 0.02)
        for key, page in snapshot.manifest.pages.items():
            if key != "home":
                page.obj_edit_time = None
        plan = build_sync_plan(specs, snapshot, executor, "commit-a")
        self.assertEqual(plan.audited_revision_count, 8)
        self.assertGreaterEqual(executor.max_active_reads, 2)
        self.assertLessEqual(executor.max_active_reads, 4)

    def test_tree_discovery_parallelizes_each_breadth_first_frontier(self):
        """List independent child-bearing parents concurrently.

        Parameters:
            self: Current test case.
        """
        class TreeExecutor:
            """Return one root frontier followed by delayed empty children."""

            def __init__(self):
                """Initialize concurrency counters.

                No parameters.
                """
                self.active = 0
                self.maximum = 0
                self.lock = threading.Lock()

            def list_nodes(self, space_id, parent_node_token = None):
                """Return root parents or an empty child list.

                Parameters:
                    space_id: Target Wiki space identifier.
                    parent_node_token: Current parent node token.
                """
                if parent_node_token is None:
                    return [
                        {
                            "node_token": f"parent-{index}",
                            "obj_token": f"doc-parent-{index}",
                            "title": f"Parent {index}",
                            "has_child": True,
                            "obj_edit_time": "10",
                        }
                        for index in range(4)
                    ]
                with self.lock:
                    self.active += 1
                    self.maximum = max(self.maximum, self.active)
                try:
                    time.sleep(0.02)
                    return []
                finally:
                    with self.lock:
                        self.active -= 1

        executor = TreeExecutor()
        tree = discover_tree(executor, "space")
        self.assertEqual(len(tree), 4)
        self.assertGreaterEqual(executor.maximum, 2)
        self.assertLessEqual(executor.maximum, 4)

    def test_remote_revision_drift_schedules_source_overwrite(self):
        """Treat a changed remote revision as a content update candidate.

        Parameters:
            self: Current test case.
        """
        specs, snapshot, executor = build_state()
        snapshot.tree["node-0"].obj_edit_time = "11"
        executor.documents["doc-node-0"]["revision_id"] = 2
        plan = build_sync_plan(specs, snapshot, executor, "commit-b")
        self.assertIn(
            ("update", "child-0", "remote revision changed"),
            [
                (action.operation, action.key, action.detail)
                for action in plan.actions
            ]
        )
        apply_sync_plan(executor, "space", specs, snapshot, plan, "commit-b")
        self.assertIn("doc-node-0", executor.updated)
        self.assertEqual(snapshot.manifest.pages["child-0"].revision_id, 3)

    def test_unmanaged_title_collision_fails_closed(self):
        """Never claim an unmanaged same-name sibling by title alone.

        Parameters:
            self: Current test case.
        """
        specs, snapshot, executor = build_state()
        child = snapshot.manifest.pages.pop("child-0")
        snapshot.tree.pop(child.node_token)
        executor.nodes.pop(child.node_token)
        executor.documents.pop(child.obj_token)
        unknown = TreeNode(
            node_token = "unknown",
            obj_token = "doc-unknown",
            parent_node_token = "",
            title = specs["child-0"].title,
            has_child = False,
            obj_edit_time = "10"
        )
        snapshot.tree[unknown.node_token] = unknown
        with self.assertRaises(SafetyError):
            build_sync_plan(specs, snapshot, executor, "commit-b")


class EngineConcurrencyTests(unittest.TestCase):
    """Verify bounded writes, finalization ordering, and recovery."""

    def test_content_updates_use_at_most_two_workers(self):
        """Update independent documents with two concurrent workers.

        Parameters:
            self: Current test case.
        """
        specs, snapshot, executor = build_state(
            child_count = 4,
            changed_source = True,
            delay = 0.02
        )
        plan = build_sync_plan(specs, snapshot, executor, "commit-b")
        apply_sync_plan(executor, "space", specs, snapshot, plan, "commit-b")
        self.assertGreaterEqual(executor.max_active_writes, 2)
        self.assertLessEqual(executor.max_active_writes, 2)
        self.assertEqual(snapshot.manifest.status, "complete")
        self.assertEqual(executor.updated[-1], "doc-home-token")

    def test_worker_failure_leaves_in_progress_manifest(self):
        """Never commit a complete manifest after a content worker fails.

        Parameters:
            self: Current test case.
        """
        specs, snapshot, executor = build_state(
            child_count = 2,
            changed_source = True,
            delay = 0.01
        )
        executor.fail_obj_token = "doc-node-0"
        plan = build_sync_plan(specs, snapshot, executor, "commit-b")
        with self.assertRaises(LarkCliError):
            apply_sync_plan(executor, "space", specs, snapshot, plan, "commit-b")
        self.assertEqual(snapshot.manifest.status, "in_progress")
        self.assertIn("内容更新正在进行", executor.documents["doc-home-token"]["content"])

    def test_stabilization_uses_lightweight_revision_reads(self):
        """Wait until successive lightweight revision reads agree.

        Parameters:
            self: Current test case.
        """
        revisions = iter([3, 4, 4])

        class RevisionExecutor:
            """Return a controlled asynchronous revision sequence."""

            sleep = staticmethod(lambda _: None)

            def fetch_document_revision(self, obj_token):
                """Return the next controlled revision.

                Parameters:
                    obj_token: Backing document token.
                """
                return next(revisions)

        revision = stabilize_document_revision(
            executor = RevisionExecutor(),
            obj_token = "doc-token",
            starting_revision = 2,
            delay_seconds = 0
        )
        self.assertEqual(revision, 4)

    def test_missing_page_is_staged_renamed_and_synchronized(self):
        """Create a missing page through the crash-safe staging flow.

        Parameters:
            self: Current test case.
        """
        specs, snapshot, executor = build_state()
        child = snapshot.manifest.pages.pop("child-0")
        snapshot.tree.pop(child.node_token)
        executor.nodes.pop(child.node_token)
        executor.documents.pop(child.obj_token)
        plan = build_sync_plan(specs, snapshot, executor, "commit-b")
        apply_sync_plan(executor, "space", specs, snapshot, plan, "commit-b")
        self.assertEqual(len(executor.created), 1)
        self.assertEqual(executor.created[0][1], staging_title(specs["child-0"]))
        created_token = executor.created[0][3]
        self.assertIn((created_token, specs["child-0"].title), executor.renamed)
        self.assertIn("doc-created-1", executor.updated)

    def test_staging_page_is_adopted_after_interrupted_create(self):
        """Recover only the deterministic staging node after a create crash.

        Parameters:
            self: Current test case.
        """
        specs, snapshot, executor = build_state()
        child = snapshot.manifest.pages.pop("child-0")
        snapshot.tree.pop(child.node_token)
        executor.nodes.pop(child.node_token)
        executor.documents.pop(child.obj_token)
        staged = TreeNode(
            node_token = "orphan",
            obj_token = "doc-orphan",
            parent_node_token = "",
            title = staging_title(specs["child-0"]),
            has_child = False,
            obj_edit_time = "10"
        )
        snapshot.tree[staged.node_token] = staged
        executor.nodes[staged.node_token] = staged
        executor.documents[staged.obj_token] = {"content": "", "revision_id": 1}
        snapshot.manifest.status = "in_progress"
        plan = build_sync_plan(specs, snapshot, executor, "commit-b")
        self.assertIn(
            ("recover", "child-0"),
            [(action.operation, action.key) for action in plan.actions]
        )
        apply_sync_plan(executor, "space", specs, snapshot, plan, "commit-b")
        self.assertEqual(executor.created, [])
        self.assertEqual(snapshot.manifest.pages["child-0"].node_token, "orphan")

    def test_static_edit_time_after_write_disables_future_fast_path(self):
        """Store unknown metadata when a successful write does not advance it.

        Parameters:
            self: Current test case.
        """
        specs, snapshot, executor = build_state(changed_source = True)
        snapshot.tree["node-0"].obj_edit_time = "11"
        executor.get_node = lambda node_token: executor.nodes[node_token]
        plan = build_sync_plan(specs, snapshot, executor, "commit-b")
        apply_sync_plan(executor, "space", specs, snapshot, plan, "commit-b")
        self.assertIsNone(snapshot.manifest.pages["child-0"].obj_edit_time)

        next_snapshot = RemoteSnapshot(
            tree = snapshot.tree,
            manifest = snapshot.manifest,
            home_revision = snapshot.home_revision
        )
        next_plan = build_sync_plan(specs, next_snapshot, executor, "commit-b")
        self.assertEqual(next_plan.audited_revision_count, 1)

    def test_delete_refuses_unknown_children(self):
        """Preserve the fail-closed deletion boundary.

        Parameters:
            self: Current test case.
        """
        specs, snapshot, executor = build_state()
        child = snapshot.manifest.pages["child-0"]
        snapshot.tree["unknown"] = TreeNode(
            node_token = "unknown",
            obj_token = "doc-unknown",
            parent_node_token = child.node_token,
            title = "User page",
            has_child = False
        )
        with self.assertRaises(SafetyError) as context:
            delete_stale_pages(
                executor,
                "space",
                {"home": specs["home"]},
                snapshot.tree,
                snapshot.manifest
            )
        self.assertIn("仍有子节点", str(context.exception))
        self.assertEqual(executor.deleted, [])


if __name__ == "__main__":
    unittest.main()
