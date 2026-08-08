from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SAME_LENGTH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SAME_LENGTH_DIR))

import same_length_common as common  # noqa: E402
from Figures.Figure_3 import make_figure_3_panels as base  # noqa: E402


def main() -> None:
    common.ensure_output_dirs()
    params, summary = common.load_inputs_for_scripts()
    common.assert_expected_values(summary)

    def load_params() -> dict[str, Any]:
        return copy.deepcopy(params)

    def load_summary() -> dict[str, str]:
        return dict(summary)

    base.RESULT_ID = common.OUTPUT_TAG
    base.OUT_DIR = common.FIGURE_3_DIR
    base.load_params = load_params
    base.load_summary = load_summary
    base.main()


if __name__ == "__main__":
    main()
