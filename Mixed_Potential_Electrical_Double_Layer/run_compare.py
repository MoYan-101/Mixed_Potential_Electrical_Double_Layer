from datetime import datetime
from pathlib import Path

from Solve_Emix_updating import apply_param_overrides, compare_edl_effects, default_params, load_overrides_json

SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    # Load defaults, then apply JSON overrides on top.
    defaults = default_params()
    json_overrides = load_overrides_json(SCRIPT_DIR / "params_template.json")
    params = apply_param_overrides(defaults, json_overrides, reset_lambda_D=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = compare_edl_effects(
        params,
        save_dir=SCRIPT_DIR / "results" / "compare_demo" / timestamp,
        save_data=True,
        save_fig=True,
    )
    print(out["comparison"])


if __name__ == "__main__":
    main()
