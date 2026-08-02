"""
Batch runner for the nonlinear-PB variant.

How to use:
    1) Edit params_template_nonlinearPB.json
    2) Run: python run_cases_nonlinearPB.py

This mirrors run_cases.py, but writes into results_nonlinearPB/ and calls the
nonlinear Poisson-Boltzmann solver variant.
"""

import json
from datetime import datetime
from pathlib import Path

from Solve_Emix_nonlinearPB import (
    RESULTS_ROOT_NAME,
    apply_param_overrides,
    default_params,
    load_overrides_json,
    run_full_workflow,
)

SCRIPT_DIR = Path(__file__).resolve().parent


def run_one_case(case_name: str, base_params: dict, overrides: dict | None, template_path: Path) -> None:
    params = apply_param_overrides(base_params, overrides, reset_lambda_D=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = SCRIPT_DIR / RESULTS_ROOT_NAME / case_name / timestamp

    print(f"\n=== {case_name} (nonlinearPB) ===")
    print(f"Output folder: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    template_copy = out_dir / "params_overrides_input.json"
    template_copy.write_text(template_path.read_text(encoding="utf-8"), encoding="utf-8")
    params_used = out_dir / "params_used.json"
    with params_used.open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, sort_keys=True)

    run_full_workflow(params, out_dir=out_dir, print_summary=True)


def main() -> None:
    defaults = default_params()
    template_path = SCRIPT_DIR / "params_template_nonlinearPB.json"
    json_overrides = load_overrides_json(template_path)
    base_params = apply_param_overrides(defaults, json_overrides, reset_lambda_D=True)

    run_one_case("case1", base_params, overrides=None, template_path=template_path)
    run_one_case("case2", base_params, overrides={"L_gap": 1e-07}, template_path=template_path)


if __name__ == "__main__":
    main()
