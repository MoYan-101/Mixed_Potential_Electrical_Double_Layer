"""
Beginner-friendly batch runner (JSON-driven).

How to use:
    1) Edit params_template.json (numbers only).
    2) Run: python run_cases.py

This script will read the JSON and run two example cases:
    - case1: uses the JSON as-is
    - case2: overrides only L_gap = 100e-9 on top of the JSON

It runs the SAME workflow as Solve_Emix_updating.py main():
    baseline comparison of with EDL (FULL) vs w/o EDL
    + baseline profiles
    + OFAT comparison scans
    + optional heatmaps / sensitivities.
"""

import json
from datetime import datetime
from pathlib import Path

from Solve_Emix_updating import apply_param_overrides, default_params, load_overrides_json, run_full_workflow

SCRIPT_DIR = Path(__file__).resolve().parent


def run_one_case(case_name: str, base_params: dict, overrides: dict | None, template_path: Path) -> None:
    # Apply only the parameters you want to change.
    # This function also safely handles lambda_D for you.
    params = apply_param_overrides(base_params, overrides, reset_lambda_D=True)

    # Use a timestamp folder to avoid overwriting old results.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = SCRIPT_DIR / "results" / case_name / timestamp

    print(f"\n=== {case_name} ===")
    print(f"Output folder: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save both the raw JSON overrides and the actual merged parameters used in this run.
    template_copy = out_dir / "params_overrides_input.json"
    template_copy.write_text(template_path.read_text(encoding="utf-8"), encoding="utf-8")
    params_used = out_dir / "params_used.json"
    with params_used.open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, sort_keys=True)
    run_full_workflow(params, out_dir=out_dir, print_summary=True)


def main() -> None:
    # Load defaults, then apply JSON overrides on top.
    defaults = default_params()
    template_path = SCRIPT_DIR / "params_template.json"
    json_overrides = load_overrides_json(template_path)
    base_params = apply_param_overrides(defaults, json_overrides, reset_lambda_D=True)

    # Case 1: use the JSON as-is.
    run_one_case("case1", base_params, overrides=None, template_path=template_path)

    # Case 2: only change L_gap, everything else stays as in JSON.
    run_one_case("case2", base_params, overrides={"L_gap": 1e-07}, template_path=template_path)


if __name__ == "__main__":
    main()
