#!/usr/bin/env python3
"""Compatibility entrypoint for the modular Feishu Wiki synchronizer."""

import os
import sys
import logging
import subprocess

# Add project root to Python path
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(__file__))

from feishu_wiki_sync.cli import main
from feishu_wiki_sync.models import SyncError


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    try:
        main()
    except (OSError, ValueError, SyncError, subprocess.CalledProcessError) as error:
        logging.getLogger(__name__).error(
            "Feishu Wiki synchronization failed: %s",
            error
        )
        raise SystemExit(1) from error
