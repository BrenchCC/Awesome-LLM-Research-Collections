import os
import json
import shutil
import tempfile
import unittest
from pathlib import Path

# Add project root to Python path
os.sys.path.append(os.getcwd())
os.sys.path.append(str(Path.cwd() / "scripts"))

from sync_notes import Note  # noqa: E402
from feishu_wiki_sync.content import build_page_specs  # noqa: E402
from feishu_wiki_sync.content import convert_qmd_body  # noqa: E402
from feishu_wiki_sync.content import normalize_feishu_formulas  # noqa: E402
from feishu_wiki_sync.content import parse_note_downloads  # noqa: E402
from feishu_wiki_sync.content import render_managed_page  # noqa: E402
from feishu_wiki_sync.content import render_manifest_block  # noqa: E402
from feishu_wiki_sync.content import validate_bilingual_downloads  # noqa: E402
from feishu_wiki_sync.models import HOME_TITLE  # noqa: E402
from feishu_wiki_sync.models import MAX_MEDIA_BYTES  # noqa: E402
from feishu_wiki_sync.models import PageSpec  # noqa: E402
from feishu_wiki_sync.models import RemotePage  # noqa: E402
from feishu_wiki_sync.models import SafetyError  # noqa: E402
from feishu_wiki_sync.models import SyncManifest  # noqa: E402
from feishu_wiki_sync.models import parse_manifest, stable_hash  # noqa: E402


def remote_page(
    key,
    parent_key,
    node_token,
    title,
    content_hash = "hash",
    revision_id = 1,
    obj_edit_time = None
):
    """Build a concise managed page fixture.

    Parameters:
        key: Stable page key.
        parent_key: Stable parent key.
        node_token: Wiki node token.
        title: Wiki page title.
        content_hash: Stored source content hash.
        revision_id: Stored Docx revision.
        obj_edit_time: Stored remote object edit time.
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
        source_commit = "abc123",
        obj_edit_time = obj_edit_time
    )


def manifest_with_pages(
    pages,
    status = "complete",
    pending_create_key = None,
    schema_version = 2
):
    """Build a manifest fixture from managed pages.

    Parameters:
        pages: Managed page records.
        status: Synchronization status.
        pending_create_key: Optional interrupted create key.
        schema_version: Manifest schema version.
    """
    return SyncManifest(
        schema_version = schema_version,
        repository = "https://github.com/BrenchCC/Awesome-LLM-Research-Collections",
        space_id = "space",
        status = status,
        commit = "abc123",
        updated_at = "2026-08-13T04:00:00+00:00",
        pending_create_key = pending_create_key,
        pages = {page.key: page for page in pages}
    )


class LocalConversionTests(unittest.TestCase):
    """Verify repository parsing and QMD conversion behavior."""

    def test_generated_introductions_describe_page_purpose(self):
        """Keep reader-facing introductions focused on each page's purpose.

        Parameters:
            self: Current test case.
        """
        specs, _ = build_page_specs()
        self.assertIn("精选论文、研究笔记与技术博客", specs["home"].body)
        self.assertIn("按研究主题整理", specs["papers"].body)
        self.assertIn("论文解读与技术思考", specs["notes"].body)
        self.assertIn("值得阅读的技术文章", specs["blogs"].body)
        for spec in specs.values():
            self.assertNotIn("自动同步", spec.body)
            self.assertNotIn("飞书双语镜像", spec.body)

    def test_managed_page_starts_with_purpose_and_ends_with_provenance(self):
        """Move technical provenance after the reader-facing content.

        Parameters:
            self: Current test case.
        """
        spec = PageSpec(
            key = "example",
            parent_key = None,
            title = "Example",
            source_path = "docs/example.md",
            body = "# Example\n\nExplain the document's purpose.\n"
        )
        rendered = render_managed_page(
            spec = spec,
            body = spec.body,
            source_commit = "abc123"
        )
        self.assertTrue(rendered.startswith("# Example"))
        self.assertIn("Explain the document's purpose.", rendered)
        self.assertIn("## 文档信息 / Document information", rendered)
        self.assertIn("**Source:** `docs/example.md`", rendered)
        self.assertIn("**Git commit:** `abc123`", rendered)
        self.assertNotIn("自动同步", rendered)
        self.assertNotIn("Automatically synchronized", rendered)
        self.assertLess(
            rendered.index("Explain the document's purpose."),
            rendered.index("## 文档信息 / Document information")
        )

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
                        r"\operatorname{clip}(x) < y \tag{7}",
                        "$$",
                        "",
                        r"Inline $a \lt b$ math.",
                        "",
                        "```latex",
                        r"\operatorname{keep}(x) \tag{8}",
                        "```",
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
            self.assertIn("<latex> \\mathrm{clip}(x) < y </latex>", body)
            self.assertIn(r"Inline $a \lt b$ math.", body)
            self.assertIn(r"\operatorname{keep}(x) \tag{8}", body)
            self.assertIn("![Figure](@./", body)
            self.assertIn("feishu-wiki://notes/en/page/topic/target", body)
            self.assertIn("**Created:** 2026-08-13", body)
            self.assertIn("**Last modified:** 2026-08-15", body)
            self.assertEqual(media, [image_path.resolve()])
            self.assertNotIn("fig-alt", body)
        finally:
            shutil.rmtree(temporary_root)

    def test_display_formulas_preserve_latex_tokens_across_lines(self):
        """Avoid XML escaping and command concatenation in display math.

        Parameters:
            self: Current test case.
        """
        body = normalize_feishu_formulas(
            "\n".join(
                [
                    "$$",
                    r"\begin{aligned}",
                    r"Q_i^{(v)} &= \mathrm{Adapt}_v(M_i),\\",
                    r"a_i^{\mathrm{tok}} &\sim \pi_\theta(\cdot \mid",
                    r"p_i^{\mathrm{tok}}).",
                    r"\end{aligned}",
                    "$$",
                ]
            )
        )
        self.assertIn(r"Q_i^{(v)} &= \mathrm{Adapt}_v(M_i),\\", body)
        self.assertIn(r"\mid p_i^{\mathrm{tok}}", body)
        self.assertNotIn("&amp;", body)
        self.assertNotIn(r"\midp_i", body)

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

    def test_unsupported_local_image_format_fails_during_conversion(self):
        """Verify unsupported media fails before a remote document update.

        Parameters:
            self: Current test case.
        """
        temporary_root = Path(tempfile.mkdtemp(dir = Path.cwd()))
        try:
            source_dir = temporary_root / "notes" / "en" / "topic"
            source_dir.mkdir(parents = True)
            image_path = source_dir / "figure.avif"
            image_path.write_bytes(b"avif")
            source_path = source_dir / "source.qmd"
            source_path.write_text(
                "---\ntitle: Source\n---\n\n![Figure](figure.avif)\n",
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
            with self.assertRaisesRegex(
                ValueError,
                "Unsupported note image format for Feishu"
            ):
                convert_qmd_body(note = note, source_key_by_path = {})
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

    def test_explicit_note_downloads_become_feishu_resources(self):
        """Require resources and other-links before uploading note downloads.

        Parameters:
            self: Current test case.
        """
        temporary_root = Path(tempfile.mkdtemp(dir = Path.cwd()))
        try:
            source_dir = temporary_root / "notes" / "en" / "topic"
            asset_dir = temporary_root / "notes" / "assets" / "downloads"
            source_dir.mkdir(parents = True)
            asset_dir.mkdir(parents = True)
            pdf_path = asset_dir / "lecture.pdf"
            tex_path = asset_dir / "lecture.tex"
            pdf_path.write_bytes(b"pdf")
            tex_path.write_text("tex", encoding = "utf-8")
            source_path = source_dir / "source.qmd"
            source_path.write_text(
                "\n".join(
                    [
                        "---",
                        "title: Source",
                        "resources:",
                        "  - ../../assets/downloads/lecture.pdf",
                        "  - ../../assets/downloads/lecture.tex",
                        "other-links:",
                        "  - text: Download PDF",
                        "    href: ../../assets/downloads/lecture.pdf",
                        "    icon: file-earmark-pdf",
                        "  - text: Download TeX",
                        "    href: ../../assets/downloads/lecture.tex",
                        "    icon: file-earmark-code",
                        "---",
                        "",
                        "Body",
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
                date = "2026-08-27",
                date_modified = "2026-08-27",
                description = "Description",
                author = "Brench",
                order = 1,
                note_type = "technical-reflection",
                topic = "topic",
                tags = ["Tag"]
            )
            body, media = convert_qmd_body(note = note, source_key_by_path = {})
            self.assertIn("## Downloads", body)
            self.assertIn(
                '<source path="@./' + pdf_path.relative_to(Path.cwd()).as_posix(),
                body
            )
            self.assertIn('name="Download TeX"/>', body)
            self.assertEqual(media, [pdf_path.resolve(), tex_path.resolve()])
            first_hash = stable_hash(body, media)
            self.assertEqual(first_hash, stable_hash(body, media))
            pdf_path.write_bytes(b"updated pdf")
            self.assertNotEqual(first_hash, stable_hash(body, media))
        finally:
            shutil.rmtree(temporary_root)

    def test_note_downloads_are_opt_in_and_fail_closed(self):
        """Reject partial, unsafe, missing, duplicate, and oversized downloads.

        Parameters:
            self: Current test case.
        """
        temporary_root = Path(tempfile.mkdtemp(dir = Path.cwd()))
        try:
            source_dir = temporary_root / "notes" / "en" / "topic"
            asset_dir = temporary_root / "notes" / "assets"
            source_dir.mkdir(parents = True)
            asset_dir.mkdir(parents = True)
            pdf_path = asset_dir / "lecture.pdf"
            pdf_path.write_bytes(b"pdf")
            source_path = source_dir / "source.qmd"

            source_path.write_text(
                "---\ntitle: Source\nresources:\n  - ../../assets/lecture.pdf\n---\nBody\n",
                encoding = "utf-8"
            )
            self.assertEqual(
                parse_note_downloads(
                    source_path.read_text(encoding = "utf-8"),
                    source_path.resolve()
                ),
                []
            )

            cases = [
                (
                    "missing resources entry",
                    "---\ntitle: Source\nother-links:\n  - text: PDF\n    href: ../../assets/lecture.pdf\n---\nBody\n",
                    "must also be listed in resources"
                ),
                (
                    "missing file",
                    "---\ntitle: Source\nresources:\n  - ../../assets/missing.pdf\nother-links:\n  - text: PDF\n    href: ../../assets/missing.pdf\n---\nBody\n",
                    "Missing note download"
                ),
                (
                    "duplicate",
                    "---\ntitle: Source\nresources:\n  - ../../assets/lecture.pdf\nother-links:\n  - text: PDF\n    href: ../../assets/lecture.pdf\n  - text: PDF again\n    href: ../../assets/lecture.pdf\n---\nBody\n",
                    "Duplicate note download"
                ),
                (
                    "path escape",
                    "---\ntitle: Source\nresources:\n  - ../../../../../outside.pdf\nother-links:\n  - text: PDF\n    href: ../../../../../outside.pdf\n---\nBody\n",
                    "Path escapes the repository"
                ),
            ]
            for name, text_value, error_pattern in cases:
                with self.subTest(name = name):
                    source_path.write_text(text_value, encoding = "utf-8")
                    with self.assertRaisesRegex(ValueError, error_pattern):
                        parse_note_downloads(
                            source_path.read_text(encoding = "utf-8"),
                            source_path.resolve()
                        )

            large_path = asset_dir / "large.pdf"
            with large_path.open("wb") as output:
                output.truncate(MAX_MEDIA_BYTES + 1)
            source_path.write_text(
                "---\ntitle: Source\nresources:\n  - ../../assets/large.pdf\nother-links:\n  - text: PDF\n    href: ../../assets/large.pdf\n---\nBody\n",
                encoding = "utf-8"
            )
            with self.assertRaisesRegex(ValueError, "exceeds 20 MB"):
                parse_note_downloads(
                    source_path.read_text(encoding = "utf-8"),
                    source_path.resolve()
                )
        finally:
            shutil.rmtree(temporary_root)

    def test_bilingual_note_download_paths_must_match(self):
        """Reject attachment authorization drift between paired notes.

        Parameters:
            self: Current test case.
        """
        temporary_root = Path(tempfile.mkdtemp(dir = Path.cwd()))
        try:
            asset_dir = temporary_root / "notes" / "assets"
            asset_dir.mkdir(parents = True)
            (asset_dir / "zh.pdf").write_bytes(b"zh")
            (asset_dir / "en.pdf").write_bytes(b"en")
            notes = {}
            for language in ["zh", "en"]:
                source_dir = temporary_root / "notes" / language / "topic"
                source_dir.mkdir(parents = True)
                source_path = source_dir / "source.qmd"
                filename = f"{language}.pdf"
                source_path.write_text(
                    "\n".join(
                        [
                            "---",
                            "title: Source",
                            "resources:",
                            f"  - ../../assets/{filename}",
                            "other-links:",
                            "  - text: PDF",
                            f"    href: ../../assets/{filename}",
                            "---",
                            "Body",
                        ]
                    ),
                    encoding = "utf-8"
                )
                notes[language] = [
                    Note(
                        language = language,
                        source_path = source_path,
                        relative_path = Path("topic/source.qmd"),
                        title = "Source",
                        date = "2026-08-27",
                        date_modified = "2026-08-27",
                        description = "Description",
                        author = "Brench",
                        order = 1,
                        note_type = "technical-reflection",
                        topic = "topic",
                        tags = ["Tag"]
                    )
                ]
            with self.assertRaisesRegex(ValueError, "Bilingual note downloads differ"):
                validate_bilingual_downloads(notes)
        finally:
            shutil.rmtree(temporary_root)


class ManifestModelsContentTests(unittest.TestCase):
    """Verify manifest parsing, migration, and corruption handling."""

    def test_manifest_v2_round_trip_preserves_obj_edit_time(self):
        """Verify v2 manifests round-trip without losing obj_edit_time.

        Parameters:
            self: Current test case.
        """
        home = remote_page(
            "home",
            None,
            "home-token",
            HOME_TITLE,
            revision_id = -1,
            obj_edit_time = None
        )
        child = remote_page(
            "papers/en",
            "papers",
            "child-token",
            "English",
            obj_edit_time = "1723689600000"
        )
        manifest = manifest_with_pages([home, child], schema_version = 2)
        content = render_manifest_block(manifest)
        self.assertIn("## 内部维护清单 / Internal maintenance manifest", content)
        self.assertNotIn("## 同步清单 / Sync manifest", content)
        restored = parse_manifest(content)
        self.assertEqual(restored.schema_version, 2)
        self.assertIsNone(restored.pages["home"].obj_edit_time)
        self.assertEqual(restored.pages["papers/en"].obj_edit_time, "1723689600000")

    def test_manifest_v1_migrates_to_v2_with_none_obj_edit_time(self):
        """Verify strict v1 input migrates in memory to the v2 shape.

        Parameters:
            self: Current test case.
        """
        v1_manifest = {
            "schema_version": 1,
            "repository": "https://github.com/BrenchCC/Awesome-LLM-Research-Collections",
            "space_id": "space",
            "status": "complete",
            "commit": "abc123",
            "updated_at": "2026-08-13T04:00:00+00:00",
            "pending_create_key": None,
            "pages": {
                "home": {
                    "key": "home",
                    "parent_key": None,
                    "node_token": "home-token",
                    "obj_token": "doc-home-token",
                    "title": HOME_TITLE,
                    "content_hash": "hash",
                    "revision_id": -1,
                    "source_path": "README.md",
                    "source_commit": "abc123"
                }
            }
        }
        content = "\n".join(
            [
                "## Sync",
                "",
                "FEISHU_SYNC_MANIFEST_V1",
                "",
                "```json",
                json.dumps(v1_manifest, ensure_ascii = False, indent = 2, sort_keys = True),
                "```",
                "",
            ]
        )
        restored = parse_manifest(content)
        self.assertEqual(restored.schema_version, 2)
        self.assertIn("home", restored.pages)
        self.assertIsNone(restored.pages["home"].obj_edit_time)

    def test_manifest_corruption_and_unknown_fields_fail_closed(self):
        """Verify corrupted or non-strict manifests are rejected.

        Parameters:
            self: Current test case.
        """
        manifest = manifest_with_pages(
            [
                remote_page("home", None, "home-token", HOME_TITLE, revision_id = -1)
            ],
            schema_version = 2
        )
        content = render_manifest_block(manifest)

        with self.assertRaises(SafetyError):
            parse_manifest(content.replace('"schema_version": 2', '"schema_version": 3'))

        with self.assertRaises(SafetyError):
            parse_manifest(content.replace('"obj_edit_time": null,', ""))

        with self.assertRaises(SafetyError):
            parse_manifest(content.replace('"source_commit": "abc123"', '"source_commit": "abc123",\n      "extra": 1'))


if __name__ == "__main__":
    unittest.main()
