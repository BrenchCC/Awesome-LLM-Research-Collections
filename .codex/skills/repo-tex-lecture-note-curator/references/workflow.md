# TeX-to-Lecture Workflow

## 1. Establish the source contract

- Read the complete TeX and record counts for sections, subsections, displayed formulas, referenced equation labels, tables, figures, callouts or custom environments, and bibliography entries.
- Record SHA-256 for user-provided TeX/PDF files before moving them. Preserve those bytes unless the user asks for content changes or a rebuild.
- Classify the note by its main question using the repository taxonomy. Use matching `notes/zh/<topic>/<slug>.qmd` and `notes/en/<topic>/<slug>.qmd` paths.
- For a new pair, set `date` and `date-modified` to the current Asia/Shanghai date. Keep the bilingual structural metadata aligned.

## 2. Convert without changing claims

- Use Pandoc as a mechanical starting point when useful, then compare against the TeX rather than trusting its output. Pandoc can drop TikZ while retaining only captions and can leave unsupported equation environments.
- Convert TeX sections to QMD headings, intuition/quotation environments to localized Quarto callouts, LaTeX tables to Markdown tables, and bibliography items to a numbered list.
- Expand source-only macros. In QMD display math use `$$`; never use fenced `math` blocks. Replace `\operatorname` with `\mathrm` and raw mathematical `<`/`>` with `\lt`/`\gt` without changing meaning.
- Preserve formulas and cross-reference relationships. Unless the user asks for complete source numbering, retain the numbers of equations actually referenced by prose and do not manufacture visible numbers for every other display.
- Recreate unsupported TikZ as high-quality SVG under `notes/assets/<slug>/<lang>/`. Preserve layout and semantics; localize only textual labels in the paired figure.
- The Chinese page must faithfully preserve the source. The English page must keep the same section structure, formulas, tables, figure order, claims, and conclusions while translating reader-facing text.

## 3. Add downloads only with explicit authorization

Place shared user-provided downloads under `notes/assets/<slug>/`. Add both lists to each authorized QMD page:

```yaml
resources:
  - ../../assets/<slug>/<file>.pdf
  - ../../assets/<slug>/<file>.tex
other-links:
  - text: Download Original PDF
    href: ../../assets/<slug>/<file>.pdf
    icon: file-earmark-pdf
  - text: Download Original TeX
    href: ../../assets/<slug>/<file>.tex
    icon: file-earmark-code
```

Localize `text` in Chinese. The same normalized paths and order must appear in the bilingual pair. Do not use directory globs or global `_quarto.yml` resources. Do not add attachment links to README cards or Notes indexes.

The repository validator treats only local `.pdf`/`.tex` links present in both `resources` and `other-links` as authorized. It rejects missing labels, missing files, duplicates, repository escapes, bilingual path/order drift, and files over 20 MB. The Feishu converter emits the same files as `<source>` resources and includes their bytes in the page hash.

## 4. Synchronize and verify

Use the Conda environment from `.codex/project.local.json` for Python commands. Run writes first, then restart all shared README checks:

```bash
python scripts/sync_notes.py --write
python scripts/check_readme_qmd_sync.py --write
python scripts/sync_notes.py
python scripts/check_readme_qmd_sync.py
python scripts/sync_blog_shares.py
python scripts/check_note_attachments.py
python -m unittest discover -s tests -v
python scripts/sync_feishu_wiki.py --check
quarto render --no-execute
```

Verify source/QMD structure counts, bilingual heading and figure alignment, attachment hashes, absence of `\operatorname` and fenced QMD math, successful display-math rendering, and native `quarto-other-links` on only authorized pages. Confirm the attachment files exist in `_site` with unchanged hashes.

On GitHub Actions, `.github/workflows/quarto-gh-pages.yml` validates the same metadata and uploads a `note-downloads-<commit>` artifact containing one copy of each authorized file plus `manifest.json`. The Pages artifact carries the QMD `resources`; the Feishu workflow runs the same validator before synchronization.

Remote delivery remains a separate gate: after an authorized commit and push, run Feishu `plan`, inspect the exact page/resource diff, then run `apply` only with explicit authorization. Verify both language pages and finish with an unchanged no-op run.
