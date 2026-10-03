"""Step 1 - dataset acquisition.

Downloads the credit card fraud dataset from OpenML (or reuses the local
cache), verifies it can be parsed, and writes the normalised copy used by every
downstream task.

Usage:
    python -m scripts.download_data
    python -m scripts.download_data --force
"""

from __future__ import annotations

import argparse
import json

from src import config
from src.data_loader import describe_dataset, load_dataset, log_dataset_profile
from src.logging_utils import get_logger

LOGGER = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the credit card fraud dataset.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore the local cache and download again.",
    )
    args = parser.parse_args()

    config.ensure_directories()
    LOGGER.info("Loading dataset (force_download=%s)...", args.force)
    frame = load_dataset(force_download=args.force)

    profile = describe_dataset(frame)
    log_dataset_profile(profile)

    config.VALIDATION_REPORT_PATH.write_text(
        json.dumps(profile, indent=2, default=str), encoding="utf-8"
    )
    LOGGER.info("Wrote dataset profile to %s", config.VALIDATION_REPORT_PATH)
    LOGGER.info("Download step finished successfully.")


if __name__ == "__main__":
    main()
