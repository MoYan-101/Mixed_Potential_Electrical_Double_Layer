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
from Figures.Figure_scheme import make_edl_vs_no_edl_bv_mixed_potential_with_pzc as base  # noqa: E402


def main() -> None:
    common.ensure_output_dirs()
    params, summary = common.load_inputs_for_scripts()
    common.assert_expected_values(summary)

    def load_inputs() -> tuple[dict[str, Any], dict[str, str]]:
        return copy.deepcopy(params), dict(summary)

    base.OUT_DIR = common.FIGURE_SCHEME_DIR
    base.RESULT_ID = common.OUTPUT_TAG
    base.load_inputs = load_inputs
    base.main()


if __name__ == "__main__":
    main()
