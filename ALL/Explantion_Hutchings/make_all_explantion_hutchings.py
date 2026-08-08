#!/usr/bin/env python3
"""One-command entry point for the Hutchings explanation collection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from hutchings_explanation import build_collection  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the four-case Hutchings mixed-potential/EDL collection."
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=WORKSPACE_ROOT,
        help="Path to the 2026 workspace (default: inferred from this script).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Destination root (default: this Explantion_Hutchings package).",
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    result = build_collection(arguments.workspace_root, arguments.output_root)
    print(
        json.dumps(
            {
                "output_root": result["output_root"],
                "figure_count": len(result["figure_paths"]),
                "validation_passed": result["validation"]["passed"],
                "checksums_file": result["checksums_file"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
