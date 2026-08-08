from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SAME_LENGTH_I0_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SAME_LENGTH_I0_DIR))

import same_length_i0_common as common  # noqa: E402


SCRIPTS = [
    common.FIGURE_3_DIR / "make_figure_3_panels_same_length_i0.py",
    common.FIGURE_SCHEME_DIR / "make_edl_vs_no_edl_bv_mixed_potential_with_pzc_same_length_i0.py",
    common.FIGRUE_RP_DIR / "make_phi_s_reactants_2d_same_length_i0.py",
]


def main() -> None:
    common.ensure_output_dirs()
    summary = common.load_summary_for_scripts()
    common.assert_expected_values(summary)
    common.save_traceability_inputs(summary=summary)
    common.print_summary(summary)

    for script in SCRIPTS:
        print(f"Running {script.relative_to(common.ROOT)}")
        subprocess.run([sys.executable, str(script)], cwd=common.ROOT, check=True)

    images = common.assert_output_counts()
    print(f"Verified {len(images)} PNG/SVG outputs and zero PDF outputs")
    for path in images:
        print(f"  {path}")


if __name__ == "__main__":
    main()
