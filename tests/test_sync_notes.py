import os
import sys
import unittest
from pathlib import Path

# Add project root and scripts directory to Python path
sys.path.append(os.getcwd())
sys.path.append(str(Path.cwd() / "scripts"))

from sync_notes import Note, grouped_notes
from sync_notes import LANGUAGE_CONFIGS, render_note_card, render_readme_note


def build_note(title, created, modified):
    """Build a concise English note fixture.

    Parameters:
        title: Note title and stable fixture identifier.
        created: Note creation date.
        modified: Note last-modified date.
    """
    return Note(
        language = "en",
        source_path = Path(f"notes/en/agents/{title}.qmd"),
        relative_path = Path(f"agents/{title}.qmd"),
        title = title,
        date = created,
        date_modified = modified,
        description = "Description",
        author = "Brench",
        order = 1,
        note_type = "paper-reading",
        topic = "agents",
        tags = ["Agent"]
    )


class NoteDateTests(unittest.TestCase):
    """Verify creation and modification dates remain distinct."""

    def test_grouped_notes_sort_by_creation_date(self):
        """Sort notes by creation date even when an older note was updated later.

        Parameters:
            self: Current test case.
        """
        older_updated_later = build_note(
            title = "older",
            created = "2026-01-01",
            modified = "2026-12-31"
        )
        newer = build_note(
            title = "newer",
            created = "2026-02-01",
            modified = "2026-02-01"
        )

        grouped = grouped_notes([older_updated_later, newer])

        self.assertEqual(
            [note.title for note in grouped["paper-reading"]],
            ["newer", "older"]
        )

    def test_generated_surfaces_show_both_dates(self):
        """Show creation and update dates on index cards and README entries.

        Parameters:
            self: Current test case.
        """
        note = build_note(
            title = "sample",
            created = "2026-07-29",
            modified = "2026-08-15"
        )
        paired_note = build_note(
            title = "sample-zh",
            created = "2026-07-29",
            modified = "2026-08-15"
        )
        paired_note.language = "zh"
        config = LANGUAGE_CONFIGS["en"]

        card = render_note_card(note, config)
        readme_entry = render_readme_note(note, paired_note, config)

        self.assertIn("Created 2026-07-29", card)
        self.assertIn("Updated 2026-08-15", card)
        self.assertIn(
            "(Created: 2026-07-29; Updated: 2026-08-15)",
            readme_entry
        )


if __name__ == "__main__":
    unittest.main()
