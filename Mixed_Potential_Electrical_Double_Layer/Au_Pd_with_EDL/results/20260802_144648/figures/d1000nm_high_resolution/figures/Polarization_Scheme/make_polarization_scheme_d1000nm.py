from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
OUTPUT_DIR = SCRIPT_PATH.parent
RUN_DIR = SCRIPT_PATH.parents[2]
PROJECT_DIR = SCRIPT_PATH.parents[4]
SOURCE_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SOURCE_DIR))

import au_pd_edl  # noqa: E402
from au_pd_edl.parameters import (  # noqa: E402
    ELECTROSTATIC_BACKEND,
    GEOMETRY_NAME,
    RESULT_SCHEMA_VERSION,
    apply_param_overrides,
)
from au_pd_edl.solver import (  # noqa: E402
    compute_polarization_curve,
    run_edl_comparison_pair,
    solve_case,
)


SEPARATION_NM = 1000.0
SEPARATION_M = SEPARATION_NM * 1.0e-9

# The 1050 nm-wide domain needs more modes and surface quadrature points than
# the source 0--100 nm scan.  The convergence table written beside the figure
# records the checks leading to these production values.
PRODUCTION_N_MODES = 4800
PRODUCTION_NX = 20000
CONVERGENCE_SPECS = (
    (960, 5000),
    (960, 10001),
    (1920, 10001),
    (2880, 10001),
    (3840, 10001),
    (3840, 20000),
    (4800, 20000),
)

POTENTIAL_MIN_V = -0.02
POTENTIAL_MAX_V = 1.02
N_POTENTIALS = 1201
CURRENT_TO_1E_MINUS_3_UA = 1.0e9

COLORS = {
    "dark": "#272727",
    "gray": "#767676",
    "green": "#3B7A57",
    "red": "#B64342",
    "accent": "#8A2387",
}

PUBLICATION_RCPARAMS: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Helvetica",
        "Nimbus Sans",
        "Arial",
        "DejaVu Sans",
        "sans-serif",
    ],
    "font.size": 8.6,
    "axes.linewidth": 0.9,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "mathtext.fontset": "dejavusans",
    "mathtext.default": "regular",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return loaded


def _load_and_validate_sources() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    params_path = RUN_DIR / "params.json"
    summary_path = RUN_DIR / "summary.json"
    manifest_path = RUN_DIR / "run_manifest.json"
    for path in (params_path, summary_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    params = _load_json_object(params_path)
    summary = _load_json_object(summary_path)
    manifest = _load_json_object(manifest_path)
    expected_identity = (RESULT_SCHEMA_VERSION, ELECTROSTATIC_BACKEND)
    for label, document in (("summary", summary), ("manifest", manifest)):
        identity = (
            document.get("result_schema_version"),
            document.get("electrostatic_backend"),
        )
        if identity != expected_identity:
            raise ValueError(
                f"Unsupported {label} result contract {identity!r}; "
                f"expected {expected_identity!r}"
            )

    tag = str(summary.get("tag", ""))
    if tag != str(manifest.get("tag", "")):
        raise ValueError("summary.json and run_manifest.json disagree on tag")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", tag):
        raise ValueError(f"Unsafe result tag: {tag!r}")
    return params, summary, manifest


def _assert_balanced_result(label: str, result: Mapping[str, Any]) -> None:
    i_au = float(result["I_Au_A"])
    i_pd = float(result["I_Pd_A"])
    i_mix = float(result["i_mix_abs_A"])
    denominator = abs(i_au) + abs(i_pd)
    relative_residual = abs(i_au + i_pd) / denominator
    if not math.isclose(abs(i_au), i_mix, rel_tol=2.0e-10, abs_tol=1.0e-20):
        raise ValueError(f"{label}: i_mix_abs_A does not match |I_Au|")
    if relative_residual > 1.0e-8:
        raise ValueError(
            f"{label}: mixed-current balance failed; relative residual="
            f"{relative_residual:.6g}"
        )


def _solve_production_case(
    source_params: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    overrides = {
        "d_Au_Pd": SEPARATION_M,
        "N_modes": PRODUCTION_N_MODES,
        "Nx": PRODUCTION_NX,
    }
    params = apply_param_overrides(source_params, overrides)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        pair = run_edl_comparison_pair(params, separation_m=SEPARATION_M)
    warning_messages = list(dict.fromkeys(str(item.message) for item in caught))

    with_result = dict(pair["with_edl"])
    without_result = dict(pair["without_edl"])
    _assert_balanced_result("with EDL", with_result)
    _assert_balanced_result("w/o EDL", without_result)
    return params, with_result, without_result, warning_messages


def _validate_saved_references(
    summary: Mapping[str, Any],
    with_result: Mapping[str, Any],
    without_result: Mapping[str, Any],
) -> dict[str, float]:
    saved_without = summary["without_edl_reference"]
    for name in ("E_mix_V", "i_mix_abs_A", "i_mix_avg_A_per_m2"):
        if not math.isclose(
            float(without_result[name]),
            float(saved_without[name]),
            rel_tol=2.0e-12,
            abs_tol=2.0e-15,
        ):
            raise ValueError(f"Computed w/o-EDL {name} disagrees with saved reference")

    saved_100 = summary["no_overlap_100nm_reference"]
    delta_e = float(with_result["E_mix_V"]) - float(saved_100["E_mix_V"])
    delta_i_relative = (
        float(with_result["i_mix_avg_A_per_m2"])
        / float(saved_100["i_mix_avg_A_per_m2"])
        - 1.0
    )
    plateau = summary["plateau_check"]
    if abs(delta_e) > float(plateau["E_tolerance_V"]):
        raise ValueError("1000 nm E_mix is inconsistent with the saved far-gap plateau")
    if abs(delta_i_relative) > float(plateau["i_relative_tolerance"]):
        raise ValueError("1000 nm i_mix is inconsistent with the saved far-gap plateau")
    return {
        "delta_E_1000nm_minus_saved_100nm_V": delta_e,
        "delta_i_1000nm_minus_saved_100nm_relative": delta_i_relative,
    }


def _compute_convergence_rows(
    source_params: Mapping[str, Any],
    production_result: Mapping[str, Any],
) -> list[dict[str, float | int | bool]]:
    raw_rows: list[dict[str, float | int | bool]] = []
    for n_modes, nx in CONVERGENCE_SPECS:
        is_production = n_modes == PRODUCTION_N_MODES and nx == PRODUCTION_NX
        if is_production:
            result = production_result
        else:
            params = apply_param_overrides(
                source_params,
                {
                    "d_Au_Pd": SEPARATION_M,
                    "N_modes": n_modes,
                    "Nx": nx,
                },
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result = solve_case(params, use_edl=True)
            gc.collect()

        total_length_nm = (
            float(result["derived"]["L_total"]) * 1.0e9
        )
        raw_rows.append(
            {
                "N_modes": n_modes,
                "Nx": nx,
                "is_production": is_production,
                "nominal_L_total_over_N_modes_nm": total_length_nm / n_modes,
                "E_mix_V": float(result["E_mix_V"]),
                "i_mix_abs_A": float(result["i_mix_abs_A"]),
                "i_mix_avg_A_per_m2": float(result["i_mix_avg_A_per_m2"]),
                "max_abs_phi_tilde": float(
                    result["debye_huckel_validity"]["max_abs_phi_tilde"]
                ),
                "relative_balance_residual": float(
                    result["relative_balance_residual"]
                ),
            }
        )

    final_e = float(production_result["E_mix_V"])
    final_i = float(production_result["i_mix_avg_A_per_m2"])
    for row in raw_rows:
        row["delta_E_vs_production_uV"] = (
            float(row["E_mix_V"]) - final_e
        ) * 1.0e6
        row["delta_i_vs_production_percent"] = (
            float(row["i_mix_avg_A_per_m2"]) / final_i - 1.0
        ) * 100.0

    penultimate = raw_rows[-2]
    if abs(float(penultimate["delta_E_vs_production_uV"])) > 10.0:
        raise ValueError("Final long-gap E_mix convergence exceeds 10 uV")
    if abs(float(penultimate["delta_i_vs_production_percent"])) > 0.05:
        raise ValueError("Final long-gap current convergence exceeds 0.05%")
    return raw_rows


def _compute_curves(
    params: Mapping[str, Any],
    with_result: Mapping[str, Any],
    without_result: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    potentials = np.linspace(POTENTIAL_MIN_V, POTENTIAL_MAX_V, N_POTENTIALS)
    with_curve = compute_polarization_curve(
        params,
        potentials,
        separation_m=SEPARATION_M,
        use_edl=True,
    )
    without_curve = compute_polarization_curve(
        params,
        potentials,
        separation_m=SEPARATION_M,
        use_edl=False,
    )

    for label, result, use_edl in (
        ("with EDL", with_result, True),
        ("w/o EDL", without_result, False),
    ):
        at_mix = compute_polarization_curve(
            params,
            [float(result["E_mix_V"])],
            separation_m=SEPARATION_M,
            use_edl=use_edl,
        )
        for field in ("I_Au_A", "I_Pd_A"):
            if not math.isclose(
                float(at_mix[field][0]),
                float(result[field]),
                rel_tol=2.0e-10,
                abs_tol=2.0e-20,
            ):
                raise ValueError(f"{label}: curve {field} disagrees with solved root")
    return potentials, with_curve, without_curve


def _add_mixed_current_annotations(
    ax: plt.Axes,
    result: Mapping[str, Any],
    *,
    reference: bool,
) -> None:
    e_mix = float(result["E_mix_V"])
    i_au = float(result["I_Au_A"]) * CURRENT_TO_1E_MINUS_3_UA
    i_pd = float(result["I_Pd_A"]) * CURRENT_TO_1E_MINUS_3_UA
    line_color = COLORS["gray"] if reference else COLORS["dark"]
    linestyle = (0, (2, 3)) if reference else (0, (3, 2))
    alpha = 0.68 if reference else 1.0
    zorder = 6 if reference else 7

    ax.axvline(
        e_mix,
        color=line_color,
        linestyle=linestyle,
        linewidth=1.0,
        alpha=alpha,
        zorder=zorder - 2,
    )
    for current, color in ((i_au, COLORS["green"]), (i_pd, COLORS["red"])):
        ax.annotate(
            "",
            xy=(e_mix, current),
            xytext=(e_mix, 0.0),
            arrowprops={
                "arrowstyle": "-|>",
                "mutation_scale": 9,
                "color": color,
                "linewidth": 1.05,
                "linestyle": linestyle,
                "alpha": alpha,
            },
            zorder=zorder,
        )
    ax.scatter(
        [e_mix, e_mix],
        [i_au, i_pd],
        s=46,
        color=[COLORS["green"], COLORS["red"]],
        edgecolor="white",
        linewidth=0.7,
        alpha=alpha,
        zorder=zorder + 1,
    )


def _plot_scheme(
    output_stem: Path,
    potentials: np.ndarray,
    with_curve: Mapping[str, np.ndarray],
    without_curve: Mapping[str, np.ndarray],
    with_result: Mapping[str, Any],
    without_result: Mapping[str, Any],
) -> list[Path]:
    plt.rcParams.update(PUBLICATION_RCPARAMS)
    fig, ax = plt.subplots(figsize=(6.25, 4.0))

    x = np.asarray(potentials, dtype=float)
    au_with = np.asarray(with_curve["I_Au_A"], dtype=float) * CURRENT_TO_1E_MINUS_3_UA
    pd_with = np.asarray(with_curve["I_Pd_A"], dtype=float) * CURRENT_TO_1E_MINUS_3_UA
    au_without = np.asarray(without_curve["I_Au_A"], dtype=float) * CURRENT_TO_1E_MINUS_3_UA
    pd_without = np.asarray(without_curve["I_Pd_A"], dtype=float) * CURRENT_TO_1E_MINUS_3_UA
    dash = (0, (4, 3))

    ax.plot(
        x,
        au_with,
        color=COLORS["green"],
        linewidth=2.2,
        label="Oxidation on Au, with EDL",
        zorder=4,
    )
    ax.plot(
        x,
        pd_with,
        color=COLORS["red"],
        linewidth=2.2,
        label="Reduction on Pd, with EDL",
        zorder=4,
    )
    ax.plot(
        x,
        au_without,
        color=COLORS["green"],
        linewidth=1.65,
        linestyle=dash,
        alpha=0.58,
        label="Oxidation on Au, w/o EDL",
        zorder=3,
    )
    ax.plot(
        x,
        pd_without,
        color=COLORS["red"],
        linewidth=1.65,
        linestyle=dash,
        alpha=0.58,
        label="Reduction on Pd, w/o EDL",
        zorder=3,
    )
    ax.axhline(0.0, color=COLORS["dark"], linewidth=0.9, zorder=2)

    _add_mixed_current_annotations(ax, without_result, reference=True)
    _add_mixed_current_annotations(ax, with_result, reference=False)

    e_with = float(with_result["E_mix_V"])
    e_without = float(without_result["E_mix_V"])
    i_with = float(with_result["i_mix_abs_A"]) * CURRENT_TO_1E_MINUS_3_UA
    i_without = float(without_result["i_mix_abs_A"]) * CURRENT_TO_1E_MINUS_3_UA
    delta_e = e_with - e_without
    current_drop_percent = (1.0 - i_with / i_without) * 100.0
    y_limit = max(0.13, 2.18 * max(i_with, i_without))

    shift_y = -0.735 * y_limit
    ax.annotate(
        "",
        xy=(e_with - 0.006, shift_y),
        xytext=(e_without + 0.006, shift_y),
        arrowprops={
            "arrowstyle": "->",
            "color": COLORS["dark"],
            "linewidth": 1.1,
        },
        zorder=8,
    )
    ax.text(
        0.5 * (e_with + e_without),
        shift_y - 0.055 * y_limit,
        rf"$E_{{\mathrm{{mix}}}}$ shifts up by {delta_e:.3f} V",
        ha="center",
        va="top",
        fontsize=8.4,
        color=COLORS["dark"],
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.90,
            "boxstyle": "round,pad=0.14",
        },
        zorder=9,
    )

    drop_x = 0.865
    ax.annotate(
        "",
        xy=(drop_x, i_with),
        xytext=(drop_x, i_without),
        arrowprops={
            "arrowstyle": "->",
            "color": COLORS["accent"],
            "linewidth": 1.2,
        },
        zorder=8,
    )
    ax.text(
        drop_x - 0.014,
        0.5 * (i_with + i_without),
        rf"$I_{{\mathrm{{mix}}}}$ drops by {current_drop_percent:.1f}%",
        ha="right",
        va="center",
        fontsize=8.1,
        color=COLORS["accent"],
        zorder=9,
    )

    ax.text(
        e_with + 0.018,
        0.60 * y_limit,
        "with EDL\n"
        rf"$E_{{\mathrm{{mix}}}}={e_with:.2f}$ V, "
        rf"$|I_{{\mathrm{{mix}}}}|={i_with:.3f}\times 10^{{-3}}$ uA",
        ha="left",
        va="center",
        fontsize=8.0,
        color=COLORS["dark"],
        linespacing=1.15,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.86,
            "boxstyle": "round,pad=0.12",
        },
        zorder=9,
    )
    ax.text(
        e_without - 0.018,
        -0.55 * y_limit,
        "w/o EDL\n"
        rf"$E_{{\mathrm{{mix}}}}={e_without:.2f}$ V, "
        rf"$|I_{{\mathrm{{mix}}}}|={i_without:.3f}\times 10^{{-3}}$ uA",
        ha="right",
        va="center",
        fontsize=8.0,
        color=COLORS["gray"],
        linespacing=1.15,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.86,
            "boxstyle": "round,pad=0.12",
        },
        zorder=9,
    )

    ax.set_xlim(POTENTIAL_MIN_V, POTENTIAL_MAX_V)
    ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax.set_ylim(-y_limit, y_limit)
    ax.set_xlabel("Potential (V vs. RHE)", fontsize=10.2)
    ax.set_ylabel(r"Current ($10^{-3}$ uA)", fontsize=10.2)
    ax.tick_params(length=3.4, width=0.9, labelsize=9.2)
    ax.set_title("Mixed-potential balance", loc="left", fontsize=11.3, pad=6)
    ax.text(
        0.995,
        0.995,
        r"$d_{\mathrm{Au-Pd}}=1000$ nm",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
        color=COLORS["gray"],
    )
    ax.legend(
        loc="upper left",
        fontsize=7.35,
        handlelength=2.8,
        ncols=2,
        columnspacing=0.95,
        handletextpad=0.55,
        borderaxespad=0.45,
    )
    for spine in (ax.spines["left"], ax.spines["bottom"]):
        spine.set_color(COLORS["dark"])
        spine.set_linewidth(0.9)

    fig.tight_layout(pad=0.85)
    saved: list[Path] = []
    figure_metadata = {
        "Creator": "au_pd_edl polarization schematic",
        "Description": "Signed absolute Au/Pd half-reaction currents at d_Au-Pd = 1000 nm",
    }
    for suffix in (".png", ".svg"):
        path = output_stem.with_suffix(suffix)
        fig.savefig(
            path,
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white",
            edgecolor="none",
            metadata=figure_metadata,
        )
        saved.append(path)
    plt.close(fig)
    return saved


def _write_curve_csv(
    path: Path,
    potentials: np.ndarray,
    with_curve: Mapping[str, np.ndarray],
    without_curve: Mapping[str, np.ndarray],
) -> None:
    fieldnames = [
        "potential_V",
        "I_Au_with_EDL_A",
        "I_Pd_with_EDL_A",
        "I_total_with_EDL_A",
        "I_Au_without_EDL_A",
        "I_Pd_without_EDL_A",
        "I_total_without_EDL_A",
        "I_Au_with_EDL_1e-3_uA",
        "I_Pd_with_EDL_1e-3_uA",
        "I_Au_without_EDL_1e-3_uA",
        "I_Pd_without_EDL_1e-3_uA",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, potential in enumerate(potentials):
            row = {
                "potential_V": float(potential),
                "I_Au_with_EDL_A": float(with_curve["I_Au_A"][index]),
                "I_Pd_with_EDL_A": float(with_curve["I_Pd_A"][index]),
                "I_total_with_EDL_A": float(with_curve["I_total_A"][index]),
                "I_Au_without_EDL_A": float(without_curve["I_Au_A"][index]),
                "I_Pd_without_EDL_A": float(without_curve["I_Pd_A"][index]),
                "I_total_without_EDL_A": float(without_curve["I_total_A"][index]),
                "I_Au_with_EDL_1e-3_uA": float(with_curve["I_Au_A"][index])
                * CURRENT_TO_1E_MINUS_3_UA,
                "I_Pd_with_EDL_1e-3_uA": float(with_curve["I_Pd_A"][index])
                * CURRENT_TO_1E_MINUS_3_UA,
                "I_Au_without_EDL_1e-3_uA": float(without_curve["I_Au_A"][index])
                * CURRENT_TO_1E_MINUS_3_UA,
                "I_Pd_without_EDL_1e-3_uA": float(without_curve["I_Pd_A"][index])
                * CURRENT_TO_1E_MINUS_3_UA,
            }
            writer.writerow(row)


def _write_convergence_csv(
    path: Path,
    rows: list[dict[str, float | int | bool]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_params, summary, manifest = _load_and_validate_sources()
    tag = str(summary["tag"])
    stem_name = f"polarization_scheme_d1000nm_{tag}"
    output_stem = OUTPUT_DIR / stem_name

    params, with_result, without_result, warning_messages = _solve_production_case(
        source_params
    )
    plateau_comparison = _validate_saved_references(
        summary, with_result, without_result
    )
    convergence_rows = _compute_convergence_rows(source_params, with_result)
    potentials, with_curve, without_curve = _compute_curves(
        params, with_result, without_result
    )

    curve_csv = OUTPUT_DIR / f"{stem_name}_curves.csv"
    convergence_csv = OUTPUT_DIR / f"{stem_name}_convergence.csv"
    effective_params_json = OUTPUT_DIR / f"{stem_name}_effective_params.json"
    metadata_json = OUTPUT_DIR / f"{stem_name}_metadata.json"
    _write_curve_csv(curve_csv, potentials, with_curve, without_curve)
    _write_convergence_csv(convergence_csv, convergence_rows)
    effective_params_json.write_text(
        json.dumps(params, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    figure_paths = _plot_scheme(
        output_stem,
        potentials,
        with_curve,
        without_curve,
        with_result,
        without_result,
    )

    e_with = float(with_result["E_mix_V"])
    e_without = float(without_result["E_mix_V"])
    i_with = float(with_result["i_mix_abs_A"])
    i_without = float(without_result["i_mix_abs_A"])
    metadata = {
        "figure_type": "signed_absolute_half_reaction_polarization_schematic",
        "result_id": RUN_DIR.name,
        "result_tag": tag,
        "source_result_contract": {
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "electrostatic_backend": ELECTROSTATIC_BACKEND,
            "geometry": GEOMETRY_NAME,
            "package_version": au_pd_edl.__version__,
        },
        "source_files": {
            "params_json": str((RUN_DIR / "params.json").resolve()),
            "params_sha256": _sha256(RUN_DIR / "params.json"),
            "summary_json": str((RUN_DIR / "summary.json").resolve()),
            "summary_sha256": _sha256(RUN_DIR / "summary.json"),
            "run_manifest_json": str((RUN_DIR / "run_manifest.json").resolve()),
            "run_manifest_sha256": _sha256(RUN_DIR / "run_manifest.json"),
        },
        "calculation_scope": {
            "separation_nm": SEPARATION_NM,
            "outside_saved_0_100nm_scan": True,
            "standard_scan_manifest_unchanged": True,
            "source_N_modes": int(source_params["N_modes"]),
            "source_Nx": int(source_params["Nx"]),
            "production_N_modes": PRODUCTION_N_MODES,
            "production_Nx": PRODUCTION_NX,
            "potential_min_V": POTENTIAL_MIN_V,
            "potential_max_V": POTENTIAL_MAX_V,
            "n_potentials": N_POTENTIALS,
        },
        "current_definition": (
            "Signed absolute half-reaction currents: I_Au > 0 and I_Pd < 0; "
            "the mixed-potential root is I_Au + I_Pd = 0."
        ),
        "display_current_conversion": (
            "I_display in units of 10^-3 uA equals I_A multiplied by 1e9."
        ),
        "with_EDL": {
            "E_mix_V": e_with,
            "I_Au_A": float(with_result["I_Au_A"]),
            "I_Pd_A": float(with_result["I_Pd_A"]),
            "i_mix_abs_A": i_with,
            "i_mix_display_1e-3_uA": i_with * CURRENT_TO_1E_MINUS_3_UA,
            "i_mix_avg_A_per_m2": float(with_result["i_mix_avg_A_per_m2"]),
            "relative_balance_residual": float(
                with_result["relative_balance_residual"]
            ),
            "max_abs_phi_tilde": float(
                with_result["debye_huckel_validity"]["max_abs_phi_tilde"]
            ),
        },
        "without_EDL": {
            "E_mix_V": e_without,
            "I_Au_A": float(without_result["I_Au_A"]),
            "I_Pd_A": float(without_result["I_Pd_A"]),
            "i_mix_abs_A": i_without,
            "i_mix_display_1e-3_uA": i_without * CURRENT_TO_1E_MINUS_3_UA,
            "i_mix_avg_A_per_m2": float(
                without_result["i_mix_avg_A_per_m2"]
            ),
            "relative_balance_residual": float(
                without_result["relative_balance_residual"]
            ),
        },
        "comparison": {
            "delta_E_with_minus_without_V": e_with - e_without,
            "I_mix_with_over_without": i_with / i_without,
            "I_mix_drop_percent": (1.0 - i_with / i_without) * 100.0,
            **plateau_comparison,
        },
        "linearized_PB_caveat": {
            "threshold": float(params["dh_warn_threshold"]),
            "threshold_exceeded": bool(
                with_result["debye_huckel_validity"]["threshold_exceeded"]
            ),
            "interpretation": with_result["debye_huckel_validity"][
                "interpretation"
            ],
            "warnings": warning_messages,
        },
        "supporting_outputs": {
            "figure_png": str(figure_paths[0].resolve()),
            "figure_svg": str(figure_paths[1].resolve()),
            "curve_csv": str(curve_csv.resolve()),
            "convergence_csv": str(convergence_csv.resolve()),
            "effective_params_json": str(effective_params_json.resolve()),
        },
        "source_manifest_tag": manifest["tag"],
        "pdf_generated": False,
    }
    metadata_json.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for path in (
        *figure_paths,
        curve_csv,
        convergence_csv,
        effective_params_json,
        metadata_json,
    ):
        print(f"Saved {path.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
