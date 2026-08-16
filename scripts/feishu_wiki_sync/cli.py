"""Command-line orchestration for the Feishu Wiki synchronizer."""

import os
import sys
import time
import shutil
import logging
import argparse
import subprocess

# Add project root to Python path
sys.path.append(os.getcwd())

from feishu_wiki_sync.client import LarkCliExecutor
from feishu_wiki_sync.content import build_page_specs
from feishu_wiki_sync.engine import apply_sync_plan
from feishu_wiki_sync.models import SyncError
from feishu_wiki_sync.planner import build_sync_plan, load_remote_state

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments.

    No parameters.
    """
    parser = argparse.ArgumentParser(
        description = "Mirror papers, notes, and blogs into a Feishu Wiki space."
    )
    mode = parser.add_mutually_exclusive_group(required = True)
    mode.add_argument(
        "--check",
        action = "store_true",
        help = "Parse, convert, and validate local content without contacting Feishu."
    )
    mode.add_argument(
        "--plan",
        action = "store_true",
        help = "Read Feishu state and print the proposed changes without writing."
    )
    mode.add_argument(
        "--apply",
        action = "store_true",
        help = "Apply the synchronization plan with bounded content concurrency."
    )
    return parser.parse_args()


def get_git_commit():
    """Return GITHUB_SHA or the current Git HEAD.

    No parameters.
    """
    github_sha = os.environ.get("GITHUB_SHA", "").strip()
    if github_sha:
        return github_sha
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check = True,
        capture_output = True,
        text = True
    )
    return result.stdout.strip()


def require_remote_environment():
    """Validate bot credentials and return the target Wiki space ID.

    No parameters.
    """
    required = [
        "LARKSUITE_CLI_APP_ID",
        "LARKSUITE_CLI_APP_SECRET",
        "FEISHU_WIKI_SPACE_ID",
    ]
    missing = [name for name in required if not os.environ.get(name, "").strip()]
    if missing:
        raise SyncError("Missing required environment variables: " + ", ".join(missing))
    brand = os.environ.setdefault("LARKSUITE_CLI_BRAND", "feishu")
    if brand != "feishu":
        raise SyncError("LARKSUITE_CLI_BRAND must be feishu")
    strict_mode = os.environ.setdefault("LARKSUITE_CLI_STRICT_MODE", "bot")
    if strict_mode != "bot":
        raise SyncError("LARKSUITE_CLI_STRICT_MODE must be bot")
    if shutil.which("lark-cli") is None:
        raise SyncError("lark-cli is not installed or not available on PATH")
    return os.environ["FEISHU_WIKI_SPACE_ID"].strip()


def run_check(specs, statistics):
    """Report successful local conversion statistics.

    Parameters:
        specs: Validated desired page specifications.
        statistics: Parsed collection counts.
    """
    media_paths = {
        path.resolve()
        for spec in specs.values()
        for path in spec.media_paths
    }
    print("Feishu Wiki local validation passed")
    print(
        "Papers: "
        f"zh={statistics['paper_counts']['zh']}, "
        f"en={statistics['paper_counts']['en']}"
    )
    print(
        "Notes: "
        f"zh={statistics['note_counts']['zh']}, "
        f"en={statistics['note_counts']['en']}"
    )
    print(f"Blogs: zh={statistics['blog_count']}, en={statistics['blog_count']}")
    print(f"Wiki pages: {len(specs)}; local media: {len(media_paths)}")


def print_actions(actions):
    """Print a deterministic synchronization plan without secrets.

    Parameters:
        actions: Proposed synchronization actions.
    """
    for action in actions:
        print(f"{action.operation.upper():8} {action.key} — {action.detail}")
    counts = {}
    for action in actions:
        counts[action.operation] = counts.get(action.operation, 0) + 1
    summary = ", ".join(
        f"{key}={value}"
        for key, value in sorted(counts.items())
    )
    print(f"Summary: {summary or 'no changes'}")


def _load_and_plan(specs, executor, space_id, commit):
    """Load remote state and build a timed synchronization plan.

    Parameters:
        specs: Desired page specifications.
        executor: Lark CLI executor.
        space_id: Target Wiki space identifier.
        commit: Current source commit.
    """
    phase_start = time.monotonic()
    snapshot = load_remote_state(executor, space_id)
    logger.info(
        "Remote tree discovery and manifest load completed in %.2fs (%s nodes)",
        time.monotonic() - phase_start,
        len(snapshot.tree)
    )
    phase_start = time.monotonic()
    plan = build_sync_plan(
        specs = specs,
        snapshot = snapshot,
        executor = executor,
        commit = commit
    )
    logger.info(
        "Plan completed in %.2fs (%s lightweight revision read(s), %s action(s))",
        time.monotonic() - phase_start,
        plan.audited_revision_count,
        len(plan.actions)
    )
    return snapshot, plan


def run_plan(specs, executor, space_id, commit):
    """Read remote state and print a non-mutating synchronization diff.

    Parameters:
        specs: Desired page specifications.
        executor: Lark CLI executor.
        space_id: Target Wiki space identifier.
        commit: Current source commit.
    """
    _, plan = _load_and_plan(specs, executor, space_id, commit)
    print_actions(plan.actions)


def run_apply(specs, executor, space_id, commit):
    """Apply the audited plan and commit the homepage manifest last.

    Parameters:
        specs: Desired page specifications.
        executor: Lark CLI executor.
        space_id: Target Wiki space identifier.
        commit: Current source commit.
    """
    snapshot, plan = _load_and_plan(specs, executor, space_id, commit)
    print_actions(plan.actions)
    phase_start = time.monotonic()
    apply_sync_plan(
        executor = executor,
        space_id = space_id,
        specs = specs,
        snapshot = snapshot,
        plan = plan,
        commit = commit
    )
    logger.info("Apply phase completed in %.2fs", time.monotonic() - phase_start)
    print(f"Apply complete: {len(specs)} managed pages at commit {commit}")


def main():
    """Run the selected local check, remote plan, or remote apply mode.

    No parameters.
    """
    total_start = time.monotonic()
    args = parse_args()
    phase_start = time.monotonic()
    specs, statistics = build_page_specs()
    logger.info(
        "Local content build completed in %.2fs (%s pages)",
        time.monotonic() - phase_start,
        len(specs)
    )
    if args.check:
        run_check(specs, statistics)
        logger.info("Total runtime: %.2fs", time.monotonic() - total_start)
        return

    space_id = require_remote_environment()
    commit = get_git_commit()
    executor = LarkCliExecutor()
    if args.plan:
        run_plan(specs, executor, space_id, commit)
    else:
        run_apply(specs, executor, space_id, commit)
    logger.info("Total runtime: %.2fs", time.monotonic() - total_start)
