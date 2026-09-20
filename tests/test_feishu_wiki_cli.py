"""Tests for Feishu Wiki command-line environment handling."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Add project root to Python path
sys.path.append(os.getcwd())
sys.path.append(str(Path.cwd() / "scripts"))

from feishu_wiki_sync.cli import load_local_environment  # noqa: E402


class CliEnvironmentTests(unittest.TestCase):
    """Verify local .env credential aliases."""

    def test_local_aliases_map_to_lark_cli_environment(self):
        """Load supported credentials without exposing unrelated values.

        Parameters:
            self: Current test case.
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            env_path = Path(temporary_directory) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "# Local Feishu credentials",
                        "FEISHU_APP_ID=cli_example",
                        "FEISHU_APP_SECRET='secret-example'",
                        "UNRELATED_VALUE=ignored",
                    ]
                ),
                encoding = "utf-8"
            )
            with patch.dict(os.environ, {}, clear = True):
                load_local_environment(env_path)
                self.assertEqual(os.environ["FEISHU_APP_ID"], "cli_example")
                self.assertEqual(os.environ["FEISHU_APP_SECRET"], "secret-example")
                self.assertEqual(os.environ["LARKSUITE_CLI_APP_ID"], "cli_example")
                self.assertEqual(
                    os.environ["LARKSUITE_CLI_APP_SECRET"],
                    "secret-example"
                )
                self.assertNotIn("UNRELATED_VALUE", os.environ)

    def test_explicit_environment_takes_precedence_over_dotenv(self):
        """Preserve explicit process credentials over local file values.

        Parameters:
            self: Current test case.
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            env_path = Path(temporary_directory) / ".env"
            env_path.write_text(
                "FEISHU_APP_ID=file-alias\nFEISHU_APP_SECRET=file-secret\n",
                encoding = "utf-8"
            )
            explicit = {
                "FEISHU_APP_ID": "process-alias",
                "LARKSUITE_CLI_APP_ID": "official-id",
                "LARKSUITE_CLI_APP_SECRET": "official-secret",
            }
            with patch.dict(os.environ, explicit, clear = True):
                load_local_environment(env_path)
                self.assertEqual(os.environ["FEISHU_APP_ID"], "process-alias")
                self.assertEqual(os.environ["LARKSUITE_CLI_APP_ID"], "official-id")
                self.assertEqual(
                    os.environ["LARKSUITE_CLI_APP_SECRET"],
                    "official-secret"
                )


if __name__ == "__main__":
    unittest.main()
