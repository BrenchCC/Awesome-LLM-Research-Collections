import os
import sys
import json
import unittest
import subprocess
from pathlib import Path
from unittest.mock import patch

# Add project root to Python path
sys.path.append(os.getcwd())
sys.path.append(str(Path.cwd() / "scripts"))

from feishu_wiki_sync.client import CREDENTIAL_ENV_KEYS  # noqa: E402
from feishu_wiki_sync.client import LarkCliExecutor  # noqa: E402


class LarkCliClientTests(unittest.TestCase):
    """Verify the extracted lark-cli adapter behavior."""

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
        with patch(
            "feishu_wiki_sync.client.subprocess.run",
            side_effect = [failures, success]
        ):
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
        with patch(
            "feishu_wiki_sync.client.subprocess.run",
            side_effect = [failure, success]
        ):
            payload = executor.run(["wiki", "+node-list", "--space-id", "space"])
        self.assertTrue(payload["ok"])
        self.assertEqual(delays, [1])

    def test_fetch_document_revision_uses_lightweight_raw_api(self):
        """Verify revision-only fetch uses the raw document info endpoint.

        Parameters:
            self: Current test case.
        """
        success = subprocess.CompletedProcess(
            args = ["lark-cli"],
            returncode = 0,
            stdout = json.dumps(
                {"ok": True, "data": {"document": {"revision_id": 7}}}
            ),
            stderr = ""
        )
        executor = LarkCliExecutor()
        with patch(
            "feishu_wiki_sync.client.subprocess.run",
            return_value = success
        ) as mocked_run:
            revision_id = executor.fetch_document_revision("doc-token")
        self.assertEqual(revision_id, 7)
        command = mocked_run.call_args.args[0]
        self.assertEqual(
            command,
            [
                "lark-cli",
                "api",
                "GET",
                "/open-apis/docx/v1/documents/doc-token",
                "--as",
                "bot",
                "--format",
                "json",
            ]
        )

    def test_get_node_parses_obj_edit_time(self):
        """Verify get_node returns a normalized TreeNode with edit time.

        Parameters:
            self: Current test case.
        """
        success = subprocess.CompletedProcess(
            args = ["lark-cli"],
            returncode = 0,
            stdout = json.dumps(
                {
                    "ok": True,
                    "data": {
                        "node": {
                            "node_token": "wiki-node",
                            "obj_token": "doc-node",
                            "parent_node_token": "parent-node",
                            "title": "Node Title",
                            "has_child": True,
                            "obj_type": "docx",
                            "obj_edit_time": "1723708800"
                        }
                    },
                }
            ),
            stderr = ""
        )
        executor = LarkCliExecutor()
        with patch(
            "feishu_wiki_sync.client.subprocess.run",
            return_value = success
        ) as mocked_run:
            node = executor.get_node("wiki-node")
        self.assertEqual(node.node_token, "wiki-node")
        self.assertEqual(node.obj_token, "doc-node")
        self.assertEqual(node.parent_node_token, "parent-node")
        self.assertEqual(node.title, "Node Title")
        self.assertTrue(node.has_child)
        self.assertEqual(node.obj_type, "docx")
        self.assertEqual(node.obj_edit_time, "1723708800")
        command = mocked_run.call_args.args[0]
        self.assertEqual(command[:4], ["lark-cli", "api", "GET", "/open-apis/wiki/v2/spaces/get_node"])
        self.assertIn("--params", command)
        self.assertEqual(
            json.loads(command[command.index("--params") + 1]),
            {"token": "wiki-node"}
        )

    def test_run_removes_credential_environment_variables(self):
        """Verify subprocess calls do not inherit sensitive credential env vars.

        Parameters:
            self: Current test case.
        """
        observed_env = {}
        success = subprocess.CompletedProcess(
            args = ["lark-cli"],
            returncode = 0,
            stdout = json.dumps({"ok": True, "data": {"nodes": []}}),
            stderr = ""
        )

        def fake_run(command, capture_output, text, env):
            """Capture the subprocess environment used by the executor.

            Parameters:
                command: Subprocess command.
                capture_output: Whether stdout and stderr are captured.
                text: Whether text mode is enabled.
                env: Environment mapping used for the subprocess.
            """
            observed_env.update(env)
            return success

        injected_env = {
            key: f"value-{index}"
            for index, key in enumerate(CREDENTIAL_ENV_KEYS, start = 1)
        }
        injected_env["KEEP_ME"] = "present"
        executor = LarkCliExecutor()
        with patch.dict(os.environ, injected_env, clear = False):
            with patch("feishu_wiki_sync.client.subprocess.run", side_effect = fake_run):
                executor.run(["wiki", "+node-list", "--space-id", "space"])
        self.assertEqual(observed_env["KEEP_ME"], "present")
        for key in CREDENTIAL_ENV_KEYS:
            self.assertNotIn(key, observed_env)


if __name__ == "__main__":
    unittest.main()
