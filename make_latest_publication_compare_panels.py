import argparse
from pathlib import Path

from Solve_Emix_updating import (
    SCRIPT_DIR,
    apply_param_overrides,
    build_profiles_for_emix,
    default_params,
    ensure_dir,
    load_overrides_json,
    plot_publication_compare_panels,
    run_edl_comparison_pair,
)


def _latest_compare_params_json() -> Path | None:
    compare_root = SCRIPT_DIR / "results" / "compare_demo"
    if not compare_root.exists():
        return None
    candidates = sorted(
        (p for p in compare_root.iterdir() if p.is_dir() and (p / "params.json").exists()),
        key=lambda p: p.name,
        reverse=True,
    )
    if not candidates:
        return None
    return candidates[0] / "params.json"


def _resolve_params_path(source: str | None) -> Path | None:
    if source is None:
        return _latest_compare_params_json()
    path = Path(source).expanduser().resolve()
    if path.is_dir():
        candidate = path / "params.json"
        if not candidate.exists():
            raise FileNotFoundError(f"No params.json found under {path}")
        return candidate
    return path


def _load_params(params_path: Path | None) -> tuple[dict, Path, str]:
    out_dir = ensure_dir(SCRIPT_DIR / "results" / "publication_compare_latest" / "figures")
    if params_path is None:
        defaults = default_params()
        overrides = load_overrides_json(SCRIPT_DIR / "params_template.json")
        params = apply_param_overrides(defaults, overrides, reset_lambda_D=True)
        return params, out_dir, "params_template.json + default_params()"

    params_data = load_overrides_json(params_path)
    params = apply_param_overrides(default_params(), params_data, reset_lambda_D=False)
    return params, out_dir, str(params_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the publication-style 2x2 compare panels using the latest compare params by default."
    )
    parser.add_argument(
        "--params-json",
        type=str,
        default=None,
        help="Optional path to params.json, or a result directory containing params.json. "
        "If omitted, the newest results/compare_demo/*/params.json is used.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default="publication_compare_panels_latest",
        help="Base filename stem for the exported figure.",
    )
    args = parser.parse_args()

    params_path = _resolve_params_path(args.params_json)
    params, fig_dir, source_label = _load_params(params_path)

    pair = run_edl_comparison_pair(params, mode="FULL")
    res_edl = pair["with_edl"]
    res_no = pair["no_edl"]

    prof_edl, derived_edl = build_profiles_for_emix(params, float(res_edl["E_mix"]), use_edl=True)
    prof_no, derived_no = build_profiles_for_emix(params, float(res_no["E_mix"]), use_edl=False)

    out_base = fig_dir / args.output_stem
    paths = plot_publication_compare_panels(
        prof_edl=prof_edl,
        derived_edl=derived_edl,
        prof_no=prof_no,
        derived_no=derived_no,
        params=params,
        E_mix_edl=float(res_edl["E_mix"]),
        E_mix_no=float(res_no["E_mix"]),
        out_base=out_base,
        i_mix_abs_edl=float(res_edl["i_mix_abs_A"]),
        i_mix_abs_no=float(res_no["i_mix_abs_A"]),
    )

    print(f"Parameter source: {source_label}")
    print("Saved publication compare figure:")
    for kind, path in paths.items():
        print(f"  {kind}: {path}")


if __name__ == "__main__":
    main()
