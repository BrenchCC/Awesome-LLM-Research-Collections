import os
import json
import shutil
import tempfile
import unittest
import subprocess
from pathlib import Path
from unittest.mock import patch

# Add project root to Python path
os.sys.path.append(os.getcwd())
os.sys.path.append(str(Path.cwd() / "scripts"))

from sync_notes import Note  # noqa: E402
from sync_feishu_wiki import HOME_TITLE  # noqa: E402
from sync_feishu_wiki import LarkCliError, LarkCliExecutor  # noqa: E402
from sync_feishu_wiki import PageSpec, RemotePage, SafetyError  # noqa: E402
from sync_feishu_wiki import SyncManifest, TreeNode, build_page_specs  # noqa: E402
from sync_feishu_wiki import synchronize_contents  # noqa: E402
from sync_feishu_wiki import compute_plan, convert_qmd_body, create_missing_pages  # noqa: E402
from sync_feishu_wiki import delete_stale_pages, rename_pages, staging_title  # noqa: E402
from sync_feishu_wiki import render_manifest_block, parse_manifest, stable_hash  # noqa: E402
from sync_feishu_wiki import stabilize_document_revision  # noqa: E402


class FakeExecutor:
    """Record remote mutations for synchronization unit tests."""

    def __init__(self):
        """Initialize empty fake call logs.

        No parameters.
        """
        self.deleted = []
        self.created = []
        self.renamed = []
        self.updated = []
        self.documents = {}
        self.force_conflict = False
        self.sleep = lambda _: None

    def delete_node(self, space_id, node_token):
        """Record one fake deletion.

        Parameters:
            space_id: Target Wiki space identifier.
            node_token: Deleted Wiki node token.
        """
        self.deleted.append((space_id, node_token))
        return {"ok": True}

    def create_node(self, space_id, title, parent_node_token = None):
        """Create and record one fake Wiki node.

        Parameters:
            space_id: Target Wiki space identifier.
            title: New node title.
            parent_node_token: Optional parent node token.
        """
        index = len(self.created) + 1
        node_token = f"created-{index}"
        obj_token = f"doc-created-{index}"
        self.created.append((space_id, title, parent_node_token, node_token))
        self.documents[obj_token] = {"content": "", "revision_id": 1}
        return {
            "ok": True,
            "data": {"node_token": node_token, "obj_token": obj_token},
        }

    def fetch_document(self, obj_token):
        """Return one fake document and its revision.

        Parameters:
            obj_token: Backing Docx token.
        """
        document = self.documents[obj_token]
        return {
            "ok": True,
            "data": {
                "content": document["content"],
                "revision_id": document["revision_id"],
            },
        }

    def overwrite_document(self, obj_token, revision_id, content_path):
        """Overwrite one fake document with optimistic revision checking.

        Parameters:
            obj_token: Backing Docx token.
            revision_id: Expected document revision.
            content_path: Local Markdown content path.
        """
        document = self.documents[obj_token]
        if self.force_conflict or document["revision_id"] != revision_id:
            raise LarkCliError("revision conflict")
        document["content"] = content_path.read_text(encoding = "utf-8")
        document["revision_id"] += 1
        self.updated.append(obj_token)
        return {"ok": True, "data": {"revision_id": document["revision_id"]}}

    def rename_node(self, node_token, title):
        """Record one fake in-place node rename.

        Parameters:
            node_token: Wiki node token.
            title: New node title.
        """
        self.renamed.append((node_token, title))
        return {"ok": True}


def remote_page(
    key,
    parent_key,
    node_token,
    title,
    content_hash = "hash",
    revision_id = 1
):
    """Build a concise managed page fixture.

    Parameters:
        key: Stable page key.
        parent_key: Stable parent key.
        node_token: Wiki node token.
        title: Wiki page title.
        content_hash: Stored source content hash.
        revision_id: Stored Docx revision.
    """
    return RemotePage(
        key = key,
        parent_key = parent_key,
        node_token = node_token,
        obj_token = f"doc-{node_token}",
        title = title,
        content_hash = content_hash,
        revision_id = revision_id,
        source_path = "source.md",
        source_commit = "abc123"
    )


def manifest_with_pages(pages, status = "complete", pending_create_key = None):
    """Build a manifest fixture from managed pages.

    Parameters:
        pages: Managed page records.
        status: Synchronization status.
        pending_create_key: Optional interrupted create key.
    """
    return SyncManifest(
        schema_version = 1,
        repository = "https://github.com/BrenchCC/Awesome-LLM-Research-Collections",
        space_id = "space",
        status = status,
        commit = "abc123",
        updated_at = "2026-08-13T04:00:00+00:00",
        pending_create_key = pending_create_key,
        pages = {page.key: page for page in pages}
    )


def tree_node(page, parent_token = "", title = None):
    """Build a remote tree fixture from a managed page.

    Parameters:
        page: Managed page fixture.
        parent_token: Remote parent token.
        title: Optional current remote title.
    """
    return TreeNode(
        node_token = page.node_token,
        obj_token = page.obj_token,
        parent_node_token = parent_token,
        title = title or page.title,
        has_child = False,
        obj_type = "docx"
    )


class LocalConversionTests(unittest.TestCase):
    """Verify repository parsing and QMD conversion behavior."""

    def test_repository_baseline_counts(self):
        """Verify the accepted bilingual repository baseline.

        Parameters:
            self: Current test case.
        """
        specs, statistics = build_page_specs()
        self.assertEqual(
            statistics["paper_counts"]["zh"],
            statistics["paper_counts"]["en"]
        )
        self.assertEqual(
            statistics["note_counts"]["zh"],
            statistics["note_counts"]["en"]
        )
        self.assertGreater(statistics["paper_counts"]["zh"], 0)
        self.assertGreater(statistics["note_counts"]["zh"], 0)
        self.assertGreater(statistics["blog_count"], 0)
        self.assertEqual(specs["home"].title, HOME_TITLE)
        self.assertGreater(len(specs), 1)

    def test_qmd_callout_formula_image_and_internal_link_conversion(self):
        """Verify representative note constructs survive clean conversion.

        Parameters:
            self: Current test case.
        """
        temporary_root = Path(tempfile.mkdtemp(dir = Path.cwd()))
        try:
            source_dir = temporary_root / "notes" / "en" / "topic"
            asset_dir = temporary_root / "notes" / "assets"
            source_dir.mkdir(parents = True)
            asset_dir.mkdir(parents = True)
            image_path = asset_dir / "figure.png"
            image_path.write_bytes(b"png")
            target_path = source_dir / "target.qmd"
            target_path.write_text("---\ntitle: Target\n---\n\nTarget\n", encoding = "utf-8")
            source_path = source_dir / "source.qmd"
            source_path.write_text(
                "\n".join(
                    [
                        "---",
                        "title: Source",
                        "---",
                        "",
                        '::: {.callout-warning title="Boundary"}',
                        "Keep this warning.",
                        ":::",
                        "",
                        "$$",
                        "x = y + 1",
                        "$$",
                        "",
                        "![Figure](../../assets/figure.png){fig-alt=\"Figure\"}",
                        "",
                        "[Target](target.qmd)",
                        "",
                    ]
                ),
                encoding = "utf-8"
            )
            note = Note(
                language = "en",
                source_path = source_path,
                relative_path = Path("topic/source.qmd"),
                title = "Source",
                date = "2026-08-13",
                date_modified = "2026-08-15",
                description = "Description",
                author = "Brench",
                order = 1,
                note_type = "paper-reading",
                topic = "topic",
                tags = ["Tag"]
            )
            target_relative = target_path.resolve().relative_to(Path.cwd().resolve()).as_posix()
            body, media = convert_qmd_body(
                note = note,
                source_key_by_path = {target_relative: "notes/en/page/topic/target"}
            )
            self.assertIn("> **⚠️ Boundary**", body)
            self.assertIn("$$\nx = y + 1\n$$", body)
            self.assertIn("![Figure](@./", body)
            self.assertIn("feishu-wiki://notes/en/page/topic/target", body)
            self.assertIn("**Created:** 2026-08-13", body)
            self.assertIn("**Last modified:** 2026-08-15", body)
            self.assertEqual(media, [image_path.resolve()])
            self.assertNotIn("fig-alt", body)
        finally:
            shutil.rmtree(temporary_root)

    def test_svg_is_rasterized_to_deterministic_runtime_path(self):
        """Verify local SVG media is converted before upload.

        Parameters:
            self: Current test case.
        """
        temporary_root = Path(tempfile.mkdtemp(dir = Path.cwd()))
        created = []
        try:
            source_dir = temporary_root / "notes" / "zh" / "topic"
            source_dir.mkdir(parents = True)
            svg_path = source_dir / "figure.svg"
            svg_path.write_text(
                f"<svg xmlns=\"http://www.w3.org/2000/svg\"><title>{temporary_root.name}</title></svg>",
                encoding = "utf-8"
            )
            source_path = source_dir / "source.qmd"
            source_path.write_text(
                "---\ntitle: Source\n---\n\n![图](figure.svg)\n",
                encoding = "utf-8"
            )
            note = Note(
                language = "zh",
                source_path = source_path,
                relative_path = Path("topic/source.qmd"),
                title = "Source",
                date = "2026-08-13",
                date_modified = "2026-08-15",
                description = "Description",
                author = "Brench",
                order = 1,
                note_type = "paper-reading",
                topic = "topic",
                tags = ["Tag"]
            )

            def fake_converter(source, destination):
                """Write deterministic fake PNG bytes.

                Parameters:
                    source: Source SVG path.
                    destination: Destination PNG path.
                """
                self.assertEqual(source, svg_path.resolve())
                destination.parent.mkdir(parents = True, exist_ok = True)
                destination.write_bytes(b"fake-png")
                created.append(destination)

            body, media = convert_qmd_body(
                note = note,
                source_key_by_path = {},
                svg_converter = fake_converter
            )
            self.assertIn(".feishu-wiki-sync/assets/", body)
            self.assertIn("**创建日期:** 2026-08-13", body)
            self.assertIn("**最后更新:** 2026-08-15", body)
            self.assertEqual(media, created)
            self.assertTrue(media[0].is_file())
        finally:
            shutil.rmtree(temporary_root)

    def test_stable_hash_tracks_media_bytes(self):
        """Verify media changes affect the page content hash.

        Parameters:
            self: Current test case.
        """
        temporary_root = Path(tempfile.mkdtemp(dir = Path.cwd()))
        try:
            image_path = temporary_root / "image.png"
            image_path.write_bytes(b"first")
            first = stable_hash("body", [image_path])
            image_path.write_bytes(b"second")
            second = stable_hash("body", [image_path])
            self.assertNotEqual(first, second)
            self.assertEqual(stable_hash("body\n"), stable_hash("body\r\n"))
        finally:
            shutil.rmtree(temporary_root)


class ManifestAndPlanTests(unittest.TestCase):
    """Verify manifest validation and conservative diff behavior."""

    def test_manifest_round_trip_and_corruption_stop(self):
        """Verify manifests round-trip and corrupted input fails closed.

        Parameters:
            self: Current test case.
        """
        home = remote_page("home", None, "home-token", HOME_TITLE, revision_id = -1)
        manifest = manifest_with_pages([home])
        content = render_manifest_block(manifest)
        restored = parse_manifest(content)
        self.assertEqual(restored.pages["home"].node_token, "home-token")
        with self.assertRaises(SafetyError):
            parse_manifest(content.replace('"schema_version": 1', '"schema_version": 2'))

    def test_plan_covers_create_rename_revision_update_and_delete(self):
        """Verify the read-only diff classifies every supported operation.

        Parameters:
            self: Current test case.
        """
        home = remote_page("home", None, "home-token", HOME_TITLE, revision_id = -1)
        keep = remote_page("keep", None, "keep-token", "Old title", content_hash = stable_hash("same"))
        stale = remote_page("stale", None, "stale-token", "Stale")
        manifest = manifest_with_pages([home, keep, stale])
        specs = {
            "home": PageSpec("home", None, HOME_TITLE, "README.md", "home"),
            "keep": PageSpec("keep", None, "New title", "source.md", "same"),
            "new": PageSpec("new", None, "New", "source.md", "new"),
        }
        tree = {
            home.node_token: tree_node(home),
            keep.node_token: tree_node(keep, title = "Old title"),
            stale.node_token: tree_node(stale),
        }
        actions = compute_plan(specs, manifest, tree, {"keep": 2, "stale": 1})
        operations = {(action.operation, action.key) for action in actions}
        self.assertIn(("create", "new"), operations)
        self.assertIn(("rename", "keep"), operations)
        self.assertIn(("update", "keep"), operations)
        self.assertIn(("delete", "stale"), operations)
        self.assertIn(("update", "home"), operations)

    def test_duplicate_title_stops_without_guessing(self):
        """Verify an unmanaged same-name sibling blocks creation.

        Parameters:
            self: Current test case.
        """
        specs = {"new": PageSpec("new", None, "Duplicate", "source.md", "body")}
        unknown = TreeNode("unknown", "doc-unknown", "", "Duplicate", False)
        with self.assertRaises(SafetyError):
            compute_plan(specs, None, {"unknown": unknown}, {})

    def test_pending_create_is_recoverable_but_not_silently_adopted(self):
        """Verify only an explicit interrupted-create checkpoint enables adoption.

        Parameters:
            self: Current test case.
        """
        home = remote_page("home", None, "home-token", HOME_TITLE, revision_id = -1)
        specs = {
            "home": PageSpec("home", None, HOME_TITLE, "README.md", "home"),
            "new": PageSpec("new", None, "Recover me", "source.md", "body"),
        }
        tree = {
            "home-token": tree_node(home),
            "orphan": TreeNode("orphan", "doc-orphan", "", "Recover me", False),
        }
        manifest = manifest_with_pages(
            [home],
            status = "in_progress",
            pending_create_key = "new"
        )
        actions = compute_plan(specs, manifest, tree, {})
        self.assertIn(("recover", "new"), {(item.operation, item.key) for item in actions})
        manifest.pending_create_key = None
        with self.assertRaises(SafetyError):
            compute_plan(specs, manifest, tree, {})

    def test_managed_delete_refuses_unknown_children(self):
        """Verify deletion never removes or reparents an unknown child.

        Parameters:
            self: Current test case.
        """
        stale = remote_page("stale", None, "stale-token", "Stale")
        manifest = manifest_with_pages([stale])
        tree = {
            "stale-token": tree_node(stale),
            "unknown": TreeNode("unknown", "doc-unknown", "stale-token", "User page", False),
        }
        executor = FakeExecutor()
        with self.assertRaises(SafetyError):
            delete_stale_pages(executor, "space", {}, tree, manifest)
        self.assertEqual(executor.deleted, [])

    def test_managed_leaf_delete_is_explicit_and_child_safe(self):
        """Verify a manifest-owned leaf can be deleted without recursive deletion.

        Parameters:
            self: Current test case.
        """
        stale = remote_page("stale", None, "stale-token", "Stale")
        manifest = manifest_with_pages([stale])
        tree = {"stale-token": tree_node(stale)}
        executor = FakeExecutor()
        delete_stale_pages(executor, "space", {}, tree, manifest)
        self.assertEqual(executor.deleted, [("space", "stale-token")])
        self.assertEqual(tree, {})


class LarkCliRetryTests(unittest.TestCase):
    """Verify bounded retry behavior for transient CLI failures."""

    def test_rate_limit_is_retried_with_bounded_backoff(self):
        """Verify a rate-limit response is retried and then succeeds.

        Parameters:
            self: Current test case.
        """
        failures = subprocess.CompletedProcess(
            args = ["lark-cli"],
            returncode = 1,
            stdout = json.dumps(
                {"ok": False, "error": {"code": 99991400, "message": "rate limited"}}
            ),
            stderr = ""
        )
        success = subprocess.CompletedProcess(
            args = ["lark-cli"],
            returncode = 0,
            stdout = json.dumps({"ok": True, "data": {"nodes": []}}),
            stderr = ""
        )
        delays = []
        executor = LarkCliExecutor(max_attempts = 2, sleep = delays.append)
        with patch("sync_feishu_wiki.subprocess.run", side_effect = [failures, success]):
            payload = executor.run(["wiki", "+node-list", "--space-id", "space"])
        self.assertTrue(payload["ok"])
        self.assertEqual(delays, [1])

    def test_oauth_eof_is_retried(self):
        """Verify a temporary OAuth EOF is retried.

        Parameters:
            self: Current test case.
        """
        failure = subprocess.CompletedProcess(
            args = ["lark-cli"],
            returncode = 1,
            stdout = "",
            stderr = 'Post "https://accounts.feishu.cn/oauth/v3/token": EOF'
        )
        success = subprocess.CompletedProcess(
            args = ["lark-cli"],
            returncode = 0,
            stdout = json.dumps({"ok": True, "data": {"nodes": []}}),
            stderr = ""
        )
        delays = []
        executor = LarkCliExecutor(max_attempts = 2, sleep = delays.append)
        with patch("sync_feishu_wiki.subprocess.run", side_effect = [failure, success]):
            payload = executor.run(["wiki", "+node-list", "--space-id", "space"])
        self.assertTrue(payload["ok"])
        self.assertEqual(delays, [1])


class MutationExecutionTests(unittest.TestCase):
    """Verify serial create, recover, rename, update, and conflict execution."""

    def test_create_rename_and_content_update_use_fake_executor(self):
        """Verify a missing node is staged, renamed, and overwritten serially.

        Parameters:
            self: Current test case.
        """
        executor = FakeExecutor()
        executor.documents["doc-home-token"] = {"content": "", "revision_id": 1}
        home = remote_page("home", None, "home-token", HOME_TITLE, revision_id = -1)
        manifest = manifest_with_pages([home], status = "in_progress")
        specs = {
            "home": PageSpec("home", None, HOME_TITLE, "README.md", "home"),
            "child": PageSpec("child", None, "Child", "source.md", "child body"),
        }
        tree = {"home-token": tree_node(home)}
        home_revision = create_missing_pages(
            executor = executor,
            space_id = "space",
            specs = specs,
            tree = tree,
            manifest = manifest,
            home_revision = 1,
            commit = "def456"
        )
        self.assertEqual(len(executor.created), 1)
        self.assertEqual(executor.created[0][1], staging_title(specs["child"]))
        self.assertEqual(home_revision, 2)

        rename_pages(executor, specs, tree, manifest)
        self.assertIn(("created-1", "Child"), executor.renamed)
        synchronized = synchronize_contents(
            executor = executor,
            specs = specs,
            manifest = manifest,
            tree = tree,
            commit = "def456"
        )
        self.assertIn("doc-created-1", executor.updated)
        self.assertEqual(synchronized["child"].source_commit, "def456")
        self.assertGreater(synchronized["child"].revision_id, 1)

    def test_staging_node_is_adopted_after_create_crash(self):
        """Verify a deterministic staging title enables safe crash recovery.

        Parameters:
            self: Current test case.
        """
        executor = FakeExecutor()
        executor.documents["doc-home-token"] = {"content": "", "revision_id": 1}
        executor.documents["doc-orphan"] = {"content": "", "revision_id": 1}
        home = remote_page("home", None, "home-token", HOME_TITLE, revision_id = -1)
        manifest = manifest_with_pages([home], status = "in_progress")
        specs = {
            "home": PageSpec("home", None, HOME_TITLE, "README.md", "home"),
            "child": PageSpec("child", None, "Child", "source.md", "body"),
        }
        staged = TreeNode(
            node_token = "orphan",
            obj_token = "doc-orphan",
            parent_node_token = "",
            title = staging_title(specs["child"]),
            has_child = False
        )
        tree = {"home-token": tree_node(home), "orphan": staged}
        create_missing_pages(
            executor = executor,
            space_id = "space",
            specs = specs,
            tree = tree,
            manifest = manifest,
            home_revision = 1,
            commit = "def456"
        )
        self.assertEqual(executor.created, [])
        self.assertEqual(manifest.pages["child"].node_token, "orphan")

    def test_revision_conflict_aborts_before_manifest_completion(self):
        """Verify an optimistic-write conflict propagates as a failed apply.

        Parameters:
            self: Current test case.
        """
        executor = FakeExecutor()
        executor.force_conflict = True
        executor.documents["doc-child-token"] = {"content": "manual edit", "revision_id": 2}
        home = remote_page("home", None, "home-token", HOME_TITLE, revision_id = -1)
        child = remote_page("child", None, "child-token", "Child", revision_id = 1)
        manifest = manifest_with_pages([home, child], status = "in_progress")
        specs = {
            "home": PageSpec("home", None, HOME_TITLE, "README.md", "home"),
            "child": PageSpec("child", None, "Child", "source.md", "source body"),
        }
        tree = {
            "home-token": tree_node(home),
            "child-token": tree_node(child),
        }
        with self.assertRaises(LarkCliError):
            synchronize_contents(
                executor = executor,
                specs = specs,
                manifest = manifest,
                tree = tree,
                commit = "def456"
            )
        self.assertEqual(manifest.status, "in_progress")

    def test_post_write_revision_waits_for_async_conversion(self):
        """Verify the stored revision waits for asynchronous conversion changes.

        Parameters:
            self: Current test case.
        """
        revisions = iter([3, 4, 4])

        class DelayedRevisionExecutor:
            """Return a controlled asynchronous revision sequence."""

            sleep = staticmethod(lambda _: None)

            def fetch_document(self, obj_token):
                """Return the next controlled revision.

                Parameters:
                    obj_token: Backing Docx token.
                """
                return {
                    "ok": True,
                    "data": {"document": {"revision_id": next(revisions)}},
                }

        revision = stabilize_document_revision(
            executor = DelayedRevisionExecutor(),
            obj_token = "doc-token",
            starting_revision = 2,
            delay_seconds = 0
        )
        self.assertEqual(revision, 4)


if __name__ == "__main__":
    unittest.main()
