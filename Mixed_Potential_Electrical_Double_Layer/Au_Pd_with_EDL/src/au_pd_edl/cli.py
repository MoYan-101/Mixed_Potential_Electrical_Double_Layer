"""Command-line entry points for single cases and the complete separation scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .io import make_run_tag, save_run, timestamped_run_directory
from .parameters import (
    ELECTROSTATIC_BACKEND,
    GEOMETRY_NAME,
    MAX_SEPARATION_M,
    MODEL_NAME,
    RESULT_SCHEMA_VERSION,
    apply_param_overrides,
    default_params,
    load_params,
)
from .scan import run_separation_scan


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_SOURCE_BASELINE = (
    WORKSPACE_ROOT
    / "Figures"
    / "Figure_same_length_i0_alpha"
    / "inputs"
    / "params_same_length_i0_alpha050_au25_pd25_20260528_111255.json"
)
DEFAULT_PARAMS = PACKAGE_ROOT / "params_template.json"
DEFAULT_RESULTS_ROOT = PACKAGE_ROOT / "results"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="au-pd-edl",
        description=f"Linear-PB {MODEL_NAME} mixed-potential model",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(target: argparse.ArgumentParser) -> None:
        target.add_argument(
            "--params",
            type=Path,
            default=DEFAULT_PARAMS,
            help="Au|Pd JSON parameter file (default: package params_template.json)",
        )
        target.add_argument(
            "--output",
            type=Path,
            default=None,
            help="New result directory (default: results/YYYYMMDD_HHMMSS)",
        )
        target.add_argument("--n-modes", type=int, default=None)
        target.add_argument("--surface-nx", type=int, default=None)
        target.add_argument(
            "--dh-action", choices=("ignore", "warn", "raise"), default=None
        )

    single = subparsers.add_parser("single", help="solve one separation")
    common(single)
    single.add_argument("--d-nm", type=float, required=True)
    single.add_argument("--display-nx", type=int, default=600)
    single.add_argument("--display-ny", type=int, default=320)

    scan = subparsers.add_parser(
        "scan", help="run the 0-100 nm scan, RP figures, and Figure-3-style panels"
    )
    common(scan)
    scan.add_argument("--display-nx", type=int, default=600)
    scan.add_argument("--display-ny", type=int, default=320)
    scan.add_argument(
        "--skip-validation",
        action="store_true",
        help="skip the convergence/analytic validation report",
    )
    return parser


def _load_cli_params(args: argparse.Namespace) -> dict[str, Any]:
    params = load_params(args.params) if args.params is not None else default_params()
    overrides: dict[str, Any] = {}
    if args.n_modes is not None:
        overrides["N_modes"] = args.n_modes
    if args.surface_nx is not None:
        overrides["Nx"] = args.surface_nx
    if args.dh_action is not None:
        overrides["dh_violation_action"] = args.dh_action
    return apply_param_overrides(params, overrides)


def _new_output_path(requested: Path | None) -> Path:
    """Resolve the requested path; ``save_run`` enforces no-overwrite."""

    return (
        requested.resolve()
        if requested is not None
        else timestamped_run_directory(DEFAULT_RESULTS_ROOT).resolve()
    )


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _result_contract_metadata() -> dict[str, Any]:
    """Return the required top-level identity for every run manifest."""

    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "electrostatic_backend": ELECTROSTATIC_BACKEND,
    }


def _run_single(args: argparse.Namespace) -> dict[str, Any]:
    maximum_nm = MAX_SEPARATION_M * 1.0e9
    if not 0.0 <= args.d_nm <= maximum_nm:
        raise ValueError(f"--d-nm must be within 0-{maximum_nm:g}")
    params = _load_cli_params(args)
    separation_m = float(args.d_nm) * 1.0e-9
    scan = run_separation_scan(
        params,
        [separation_m],
        retain_models_at_nm=[float(args.d_nm)],
    )
    output = _new_output_path(args.output)
    saved = save_run(
        output,
        params,
        scan,
        source_baseline=DEFAULT_SOURCE_BASELINE,
        package_root=PACKAGE_ROOT,
        display_nx=int(args.display_nx),
        display_ny=int(args.display_ny),
    )
    manifest = {
        **_result_contract_metadata(),
        "command": "single",
        "model": MODEL_NAME,
        "geometry": GEOMETRY_NAME,
        "d_Au_Pd_nm": float(args.d_nm),
        "tag": make_run_tag(params),
        "output_dir": saved["output_dir"],
        "summary": saved["summary"],
        "gouy_chapman_metadata": {
            "csv": saved["gouy_chapman"]["csv"],
            "analysis_json": saved["gouy_chapman"]["analysis_json"],
            "n_rows": saved["gouy_chapman"]["n_rows"],
        },
    }
    _write_manifest(output / "run_manifest.json", manifest)
    return manifest


def _run_scan(args: argparse.Namespace) -> dict[str, Any]:
    # Plotting and validation imports are intentionally local so ``single``
    # remains usable in minimal/headless environments.
    from .figure3 import generate_figure3_comparison_panels
    from .plotting import generate_rp_2d_figures, plot_separation_trends

    params = _load_cli_params(args)
    scan = run_separation_scan(params)
    output = _new_output_path(args.output)
    saved = save_run(
        output,
        params,
        scan,
        source_baseline=DEFAULT_SOURCE_BASELINE,
        package_root=PACKAGE_ROOT,
        display_nx=int(args.display_nx),
        display_ny=int(args.display_ny),
    )
    tag = make_run_tag(params)
    rp_dir = output / "figures" / "rp_2d"
    trend_dir = output / "figures" / "trends"
    rp_metadata = generate_rp_2d_figures(
        scan["cases"],
        rp_dir,
        tag,
        n_x=int(args.display_nx),
        n_y=int(args.display_ny),
        y_max_over_lambda=5.0,
        dpi=600,
    )
    trend_metadata = plot_separation_trends(
        scan,
        trend_dir,
        tag,
        dpi=600,
    )
    trend_zoom_metadata = plot_separation_trends(
        scan,
        trend_dir,
        tag,
        dpi=600,
        x_max_nm=20.0,
    )
    figure3_dir = output / "figures" / "Figure_3"
    figure3_metadata = generate_figure3_comparison_panels(
        output,
        figure3_dir,
        separation_nm=10.0,
        dpi=600,
    )
    validation: dict[str, Any] | None = None
    if not args.skip_validation:
        from .validation import run_numerical_validation

        validation = run_numerical_validation(params)
        _write_manifest(output / "validation.json", validation)

    png_count = len(list(rp_dir.glob("*.png")))
    svg_count = len(list(rp_dir.glob("*.svg")))
    pdf_count = len(list(rp_dir.glob("*.pdf")))
    if (png_count, svg_count, pdf_count) != (4, 4, 0):
        raise RuntimeError(
            "RP output count mismatch: "
            f"PNG={png_count}, SVG={svg_count}, PDF={pdf_count}"
        )
    trend_png_count = len(list(trend_dir.glob("*.png")))
    trend_svg_count = len(list(trend_dir.glob("*.svg")))
    trend_pdf_count = len(list(trend_dir.glob("*.pdf")))
    if (trend_png_count, trend_svg_count, trend_pdf_count) != (2, 2, 0):
        raise RuntimeError(
            "Trend output count mismatch: "
            f"PNG={trend_png_count}, SVG={trend_svg_count}, "
            f"PDF={trend_pdf_count}"
        )
    figure3_png_count = len(list(figure3_dir.glob("*.png")))
    figure3_svg_count = len(list(figure3_dir.glob("*.svg")))
    figure3_pdf_count = len(list(figure3_dir.glob("*.pdf")))
    if (figure3_png_count, figure3_svg_count, figure3_pdf_count) != (6, 6, 0):
        raise RuntimeError(
            "Figure 3 output count mismatch: "
            f"PNG={figure3_png_count}, SVG={figure3_svg_count}, "
            f"PDF={figure3_pdf_count}"
        )
    manifest = {
        **_result_contract_metadata(),
        "command": "scan",
        "model": MODEL_NAME,
        "geometry": GEOMETRY_NAME,
        "tag": tag,
        "output_dir": saved["output_dir"],
        "n_separations": len(scan["rows"]),
        "plateau_check": scan["plateau_check"],
        "rp_figure_count": {"png": png_count, "svg": svg_count, "pdf": pdf_count},
        "rp_metadata": rp_metadata,
        "trend_figure_count": {
            "png": trend_png_count,
            "svg": trend_svg_count,
            "pdf": trend_pdf_count,
        },
        "trend_metadata": trend_metadata,
        "trend_zoom_0_20nm_metadata": trend_zoom_metadata,
        "figure_3_count": {
            "png": figure3_png_count,
            "svg": figure3_svg_count,
            "pdf": figure3_pdf_count,
        },
        "figure_3_metadata": figure3_metadata,
        "gouy_chapman_metadata": {
            "csv": saved["gouy_chapman"]["csv"],
            "analysis_json": saved["gouy_chapman"]["analysis_json"],
            "n_rows": saved["gouy_chapman"]["n_rows"],
        },
        "validation": validation,
    }
    _write_manifest(output / "run_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = _run_single(args) if args.command == "single" else _run_scan(args)
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
