"""Load the four model cases used in the Hutchings explanation collection.

The public :func:`load_all_cases` function returns plotting-ready dictionaries
with two deliberately separate current bases.  Each case folder uses its
native computational cell, while the cross-case Summary uses a 1 cm
out-of-plane width and a common Au(4 nm)+Pd(4 nm) reactive area.  In both
contexts, ``I_mix`` means the magnitude of either balanced half-reaction
current, ``|I_Au| = |I_Pd|``; it is not the sum of both magnitudes.

The legacy Au|C|Pd cosine models are solved on Neumann half-cells.  Their
published display domains remain those computational half-cells:
``Au(2)|Pd(2)`` and ``Au(2)|C(10)|Pd(2)``.  The independent analytic model also
uses its original one-face Au(2)/Pd(2) areas.  These first three native currents
are multiplied by two only for the common-area Summary.  The C-supported Janus
source half-cell already contains Au(4)|Pd(4), so its Summary current needs no
area conversion and its spatial plots retain the actual 5|4|4|5 nm geometry.
"""

from __future__ import annotations

import csv
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


COMPARISON_AU_LENGTH_NM = 4.0
COMPARISON_PD_LENGTH_NM = 4.0
NATIVE_REFERENCE_LENGTH_NM = 2.0
COMPARISON_WIDTH_M = 0.01
COMPARISON_REACTIVE_AREA_M2 = (
    (COMPARISON_AU_LENGTH_NM + COMPARISON_PD_LENGTH_NM)
    * 1.0e-9
    * COMPARISON_WIDTH_M
)
JANUS_ON_C_RUN_ID = "20260808_163633"
BASE_RESULT_ID = "20260528_111255"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _csv_columns(path: Path) -> dict[str, np.ndarray]:
    """Read a modest trace CSV while preserving material/name columns."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        values: dict[str, list[Any]] = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                raw = row[name]
                if name in {"segment_id", "material"}:
                    values[name].append(str(raw))
                else:
                    values[name].append(float(raw))
    return {
        name: np.asarray(items, dtype=object if name in {"segment_id", "material"} else float)
        for name, items in values.items()
    }


def _source_path(path: Path) -> str:
    return str(path.resolve())


def _insert_import_path(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _legacy_engine(workspace_root: Path) -> Any:
    module_dir = workspace_root / "Figures" / "Figure_Au2nm_Pd2nm"
    module_path = module_dir / "legacy_au_c_pd_engine.py"
    if not module_path.is_file():
        raise FileNotFoundError(module_path)
    _insert_import_path(module_dir)
    module = importlib.import_module("legacy_au_c_pd_engine")
    loaded_path = Path(module.__file__).resolve()
    if loaded_path != module_path.resolve():
        raise ImportError(
            "legacy_au_c_pd_engine resolved to an unexpected source: "
            f"{loaded_path}"
        )
    return module


def _segments_with_edges(
    specifications: Sequence[tuple[str, str, float, float, bool]]
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for name, material, start_nm, end_nm, faradaic in specifications:
        segments.append(
            {
                "name": name,
                "material": material,
                "x_start_nm": float(start_nm),
                "x_end_nm": float(end_nm),
                "length_nm": float(end_nm - start_nm),
                "faradaic": bool(faradaic),
            }
        )
    return segments


def _segment_labels(
    x_nm: np.ndarray, segments: Sequence[Mapping[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    names = np.empty(x_nm.shape, dtype=object)
    materials = np.empty(x_nm.shape, dtype=object)
    names[:] = ""
    materials[:] = ""
    for index, segment in enumerate(segments):
        start = float(segment["x_start_nm"])
        end = float(segment["x_end_nm"])
        if index == len(segments) - 1:
            # Keep the plotted domain endpoint with the segment approaching it.
            mask = (x_nm >= start - 1.0e-10) & (x_nm <= end + 1.0e-10)
        else:
            mask = (x_nm >= start - 1.0e-10) & (x_nm < end - 1.0e-10)
        names[mask] = str(segment["name"])
        materials[mask] = str(segment["material"])
    if np.any(names == "") or np.any(materials == ""):
        missing = int(np.count_nonzero(names == ""))
        raise RuntimeError(f"Segment assignment left {missing} points unassigned")
    return names, materials

def _charge_segments_from_surface(
    surface: Mapping[str, np.ndarray],
    segments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    names = np.asarray(surface["segment_id"], dtype=object)
    result: list[dict[str, Any]] = []
    for segment in segments:
        mask = names == str(segment["name"])
        if not np.any(mask):
            raise RuntimeError(f"No charge samples for segment {segment['name']}")
        result.append(
            {
                "name": str(segment["name"]),
                "material": str(segment["material"]),
                "x_nm": np.asarray(surface["x_nm"], dtype=float)[mask].copy(),
                "phi_RP_V": np.asarray(surface["phi_RP_with_V"], dtype=float)[mask].copy(),
                "sigma_C_per_m2": np.asarray(
                    surface["sigma_with_C_per_m2"], dtype=float
                )[mask].copy(),
            }
        )
    return result


def _summary_block(
    E_with_V: float,
    E_no_V: float,
    I_with_A: float,
    I_no_A: float,
) -> dict[str, dict[str, float | str]]:
    common = {
        "current_definition": "I_mix = |I_Au| = |I_Pd|",
        "comparison_reactive_area_m2": COMPARISON_REACTIVE_AREA_M2,
    }
    return {
        "with_edl": {
            **common,
            "E_mix_V": float(E_with_V),
            "I_mix_A": float(I_with_A),
        },
        "without_edl": {
            **common,
            "E_mix_V": float(E_no_V),
            "I_mix_A": float(I_no_A),
        },
    }


def _legacy_case(
    workspace_root: Path,
    engine: Any,
    support_nm: float,
    *,
    key: str,
    directory_name: str,
    artifact_tag: str,
    display_label: str,
) -> dict[str, Any]:
    legacy = engine.build_case(float(support_nm), include_2d=True)
    half_length_nm = float(legacy.L_total_nm)
    period_nm = 2.0 * half_length_nm
    x_nm = np.asarray(legacy.x_nm, dtype=float).copy()

    if math.isclose(float(support_nm), 0.0, rel_tol=0.0, abs_tol=1.0e-12):
        segments = _segments_with_edges(
            (
                ("Au", "Au", 0.0, 2.0, True),
                ("Pd", "Pd", 2.0, 4.0, True),
            )
        )
        display_sequence = "Au(2 nm)|Pd(2 nm)"
        even_extension_sequence = "Au(2)|Pd(2)|Pd(2)|Au(2) nm"
    else:
        segments = _segments_with_edges(
            (
                ("Au", "Au", 0.0, 2.0, True),
                ("C", "C", 2.0, 2.0 + float(support_nm), False),
                (
                    "Pd",
                    "Pd",
                    2.0 + float(support_nm),
                    4.0 + float(support_nm),
                    True,
                ),
            )
        )
        display_sequence = (
            f"Au(2 nm)|C({float(support_nm):g} nm)|Pd(2 nm)"
        )
        even_extension_sequence = (
            f"Au(2)|C({float(support_nm):g})|Pd(2)|Pd(2)|"
            f"C({float(support_nm):g})|Au(2) nm"
        )

    segment_id, material = _segment_labels(x_nm, segments)
    thermal_voltage = (
        float(legacy.derived["R"])
        * float(legacy.derived["T"])
        / float(legacy.derived["F"])
    )
    phi_with = np.asarray(legacy.phi_rp_with_V, dtype=float).copy()
    phi_no = np.asarray(legacy.phi_rp_no_V, dtype=float).copy()
    E_with = float(legacy.res_edl["E_mix"])
    E_no = float(legacy.res_no["E_mix"])
    E1 = float(legacy.res_edl["E1_eq_eff"])
    E2 = float(legacy.res_edl["E2_eq_eff"])
    mask_au = material == "Au"
    mask_pd = material == "Pd"
    mask_c = material == "C"

    nan = np.full(x_nm.shape, np.nan, dtype=float)
    eta_au_with = nan.copy()
    eta_pd_with = nan.copy()
    eta_au_no = nan.copy()
    eta_pd_no = nan.copy()
    eta_au_with[mask_au] = E_with - E1 - phi_with[mask_au]
    eta_pd_with[mask_pd] = E_with - E2 - phi_with[mask_pd]
    eta_au_no[mask_au] = E_no - E1 - phi_no[mask_au]
    eta_pd_no[mask_pd] = E_no - E2 - phi_no[mask_pd]

    j_au_with = np.where(mask_au, np.asarray(legacy.i1_with, dtype=float), np.nan)
    j_pd_with = np.where(mask_pd, np.asarray(legacy.i2_with, dtype=float), np.nan)
    j_au_no = np.where(mask_au, np.asarray(legacy.i1_no, dtype=float), np.nan)
    j_pd_no = np.where(mask_pd, np.asarray(legacy.i2_no, dtype=float), np.nan)
    current_with = np.where(mask_au, j_au_with, np.where(mask_pd, j_pd_with, np.nan))
    current_no = np.where(mask_au, j_au_no, np.where(mask_pd, j_pd_no, np.nan))
    eta_with = np.where(mask_au, eta_au_with, np.where(mask_pd, eta_pd_with, np.nan))
    eta_no = np.where(mask_au, eta_au_no, np.where(mask_pd, eta_pd_no, np.nan))

    c_h = np.zeros(x_nm.shape, dtype=float)
    pzc = np.zeros(x_nm.shape, dtype=float)
    c_h[mask_au] = float(legacy.params["Cdl_Au"])
    pzc[mask_au] = float(legacy.params["pzc_Au"])
    c_h[mask_pd] = float(legacy.params["Cdl_Pd"])
    pzc[mask_pd] = float(legacy.params["pzc_Pd"])
    if np.any(mask_c):
        c_h[mask_c] = float(legacy.params["Cdl_C"])
        pzc[mask_c] = float(legacy.params["pzc_C"])
    sigma = c_h * (E_with - pzc - phi_with)

    surface = {
        "x_nm": x_nm,
        "segment_id": segment_id,
        "material": material,
        "phi_tilde_with": phi_with / thermal_voltage,
        "phi_tilde_no": phi_no / thermal_voltage,
        "phi_RP_with_V": phi_with,
        "phi_RP_no_V": phi_no,
        "c_R1_with": np.asarray(legacy.c_R1_norm, dtype=float).copy(),
        "c_O2_with": np.asarray(legacy.c_O2_norm, dtype=float).copy(),
        "c_R1_no": np.ones(x_nm.shape, dtype=float),
        "c_O2_no": np.ones(x_nm.shape, dtype=float),
        "eta_Au_with_V": eta_au_with,
        "eta_Pd_with_V": eta_pd_with,
        "eta_Au_no_V": eta_au_no,
        "eta_Pd_no_V": eta_pd_no,
        "eta_with_V": eta_with,
        "eta_no_V": eta_no,
        "j_Au_with_A_per_m2": j_au_with,
        "j_Pd_with_A_per_m2": j_pd_with,
        "j_Au_no_A_per_m2": j_au_no,
        "j_Pd_no_A_per_m2": j_pd_no,
        "current_density_with_A_per_m2": current_with,
        "current_density_no_A_per_m2": current_no,
        "sigma_with_C_per_m2": sigma,
    }

    x_2d_nm = np.asarray(legacy.x_2d_nm, dtype=float).copy()
    phi_s_mV = np.asarray(legacy.phi_s_mV, dtype=float).copy()
    phi_tilde_2d = phi_s_mV / (1000.0 * thermal_voltage)
    z_r1 = float(legacy.params["z_R1"])
    z_o2 = float(legacy.params["z_O2"])
    grid_2d = {
        "x_nm": x_2d_nm,
        "y_nm": np.asarray(legacy.y_2d_nm, dtype=float).copy(),
        "phi_s_with_mV": phi_s_mV,
        "phi_tilde_with": phi_tilde_2d,
        "c_R1_with": np.exp(np.clip(-z_r1 * phi_tilde_2d, -700.0, 700.0)),
        "c_O2_with": np.exp(np.clip(-z_o2 * phi_tilde_2d, -700.0, 700.0)),
        "phi_s_no_mV": np.zeros(phi_s_mV.shape, dtype=float),
        "phi_tilde_no": np.zeros(phi_s_mV.shape, dtype=float),
        "c_R1_no": np.ones(phi_s_mV.shape, dtype=float),
        "c_O2_no": np.ones(phi_s_mV.shape, dtype=float),
    }
    charge_segments = _charge_segments_from_surface(surface, segments)

    raw_with = float(legacy.res_edl["i_mix_abs_A"])
    raw_no = float(legacy.res_no["i_mix_abs_A"])
    scale_factor = 2.0
    I_with = scale_factor * raw_with
    I_no = scale_factor * raw_no
    input_dir = (
        workspace_root
        / "Figures"
        / "Figure_Au2nm_Pd2nm"
        / "Au_C_Pd"
        / "inputs"
    )
    suffix = f"{float(support_nm):g}nm_au2_pd2_{BASE_RESULT_ID}"
    saved_summary_path = input_dir / f"summary_compare_L_support_{suffix}.json"
    saved_params_path = input_dir / f"params_L_support_{suffix}.json"
    saved_overrides_path = input_dir / f"overrides_L_support_{suffix}.json"
    saved_summary = _read_json(saved_summary_path)
    source_agreement = {
        "delta_E_with_V": E_with - float(saved_summary["E_mix_with_V"]),
        "delta_E_no_V": E_no - float(saved_summary["E_mix_no_V"]),
        "delta_raw_I_with_A": raw_with - float(saved_summary["i_mix_abs_with_A"]),
        "delta_raw_I_no_A": raw_no - float(saved_summary["i_mix_abs_no_A"]),
    }

    diagnostics: dict[str, Any] = {
        "source_halfcell_length_nm": half_length_nm,
        "full_even_period_nm": period_nm,
        "display_domain": "Neumann computational half-cell",
        "display_sequence": display_sequence,
        "even_extension_sequence": even_extension_sequence,
        "display_Au_length_nm": 2.0,
        "display_Pd_length_nm": 2.0,
        "effective_Au_length_nm": COMPARISON_AU_LENGTH_NM,
        "effective_Pd_length_nm": COMPARISON_PD_LENGTH_NM,
        "relative_balance_with": float(legacy.res_edl["relative_balance_residual"]),
        "relative_balance_no": float(legacy.res_no["relative_balance_residual"]),
        "max_abs_phi_tilde_with": float(legacy.res_edl["max_abs_phi_tilde"]),
        "debye_huckel_threshold": float(legacy.params["dh_warn_threshold"]),
        "debye_huckel_threshold_exceeded": not bool(
            legacy.res_edl["debye_huckel_ok"]
        ),
        "phi_2d_surface_max_error_halfcell": float(
            legacy.phi_2d_surface_max_error
        ),
        "saved_source_agreement": source_agreement,
    }
    source_paths = {
        "engine": _source_path(
            workspace_root
            / "Figures"
            / "Figure_Au2nm_Pd2nm"
            / "legacy_au_c_pd_engine.py"
        ),
        "saved_summary": _source_path(saved_summary_path),
        "saved_params": _source_path(saved_params_path),
        "saved_overrides": _source_path(saved_overrides_path),
    }

    if math.isclose(float(support_nm), 10.0, rel_tol=0.0, abs_tol=1.0e-12):
        reference_path = (
            input_dir
            / f"summary_compare_L_support_1000nm_au2_pd2_{BASE_RESULT_ID}.json"
        )
        reference = _read_json(reference_path)
        source_paths["plateau_reference_1000nm_summary"] = _source_path(
            reference_path
        )
        reference_raw_I = float(reference["i_mix_abs_with_A"])
        reference_I = scale_factor * reference_raw_I
        delta_E_mV = 1000.0 * (
            E_with - float(reference["E_mix_with_V"])
        )
        delta_I_percent = 100.0 * (I_with / reference_I - 1.0)
        diagnostics["C10_vs_C1000_plateau"] = {
            "reference_source": _source_path(reference_path),
            "E_C10_V": E_with,
            "E_C1000_V": float(reference["E_mix_with_V"]),
            "delta_E_C10_minus_C1000_mV": delta_E_mV,
            "I_C10_A_on_comparison_area": I_with,
            "I_C1000_A_on_comparison_area": reference_I,
            "delta_I_C10_minus_C1000_A": I_with - reference_I,
            "delta_I_C10_minus_C1000_percent": delta_I_percent,
            "strictly_identical": False,
            "plateau_equivalent_at_0p3mV_0p1percent": bool(
                abs(delta_E_mV) < 0.3 and abs(delta_I_percent) < 0.1
            ),
        }

    return {
        "key": key,
        "directory_name": directory_name,
        "artifact_tag": artifact_tag,
        "display_label": display_label,
        "topology_kind": "continuous",
        "params": dict(legacy.params),
        "segments": segments,
        "E_with_V": E_with,
        "E_no_V": E_no,
        "I_with_A": I_with,
        "I_no_A": I_no,
        "summary": _summary_block(E_with, E_no, I_with, I_no),
        "comparison_reactive_area_m2": COMPARISON_REACTIVE_AREA_M2,
        "raw_current": {
            "with_A": raw_with,
            "no_A": raw_no,
            "source_reactive_area_m2": float(legacy.res_edl["reactive_area_m2"]),
            "source_cell": "Au(2)|C|Pd(2) Neumann computational half-cell",
            "scale_factor_to_comparison": scale_factor,
        },
        "raw_current_with_A": raw_with,
        "raw_current_no_A": raw_no,
        "scale_factor": scale_factor,
        "surface": surface,
        "grid_2d": grid_2d,
        "charge_segments": charge_segments,
        "independent_surfaces": None,
        "source_paths": source_paths,
        "diagnostics": diagnostics,
    }


def _independent_surface(
    material: str,
    *,
    params: Mapping[str, Any],
    derived: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    with_edl = summary["with_edl"]
    without_edl = summary["without_edl"]
    suffix = "Au" if material == "Au" else "Pd"
    x_nm = np.linspace(0.0, NATIVE_REFERENCE_LENGTH_NM, 801, dtype=float)
    lambda_nm = float(derived["lambda_D"]) * 1.0e9
    y_nm = np.linspace(0.0, 5.0 * lambda_nm, 320, dtype=float)
    thermal_voltage = float(derived["thermal_voltage_V"])
    phi_rp_V = float(with_edl[f"phi_RP_{suffix}_V"])
    phi_rp_tilde = float(with_edl[f"phi_RP_{suffix}_tilde"])
    phi_y_tilde = phi_rp_tilde * np.exp(-y_nm / lambda_nm)
    phi_tilde_2d = np.repeat(phi_y_tilde[:, None], x_nm.size, axis=1)
    z_r1 = float(params["z_R1"])
    z_o2 = float(params["z_O2"])
    E_with = float(with_edl["E_mix_V"])
    E_no = float(without_edl["E_mix_V"])
    if material == "Au":
        E_eq = float(summary["effective_reaction"]["E1_eq_eff"])
        pzc = float(params["pzc_Au"])
        c_h = float(params["C_H_Au"])
        j_with = float(with_edl["j_Au_A_per_m2"])
        j_no = float(without_edl["j_Au_A_per_m2"])
    else:
        E_eq = float(summary["effective_reaction"]["E2_eq_eff"])
        pzc = float(params["pzc_Pd"])
        c_h = float(params["C_H_Pd"])
        j_with = float(with_edl["j_Pd_A_per_m2"])
        j_no = float(without_edl["j_Pd_A_per_m2"])
    sigma = c_h * (E_with - pzc - phi_rp_V)
    surface = {
        "x_nm": x_nm,
        "material": np.full(x_nm.shape, material, dtype=object),
        "phi_tilde_with": np.full(x_nm.shape, phi_rp_tilde, dtype=float),
        "phi_tilde_no": np.zeros(x_nm.shape, dtype=float),
        "phi_RP_with_V": np.full(x_nm.shape, phi_rp_V, dtype=float),
        "phi_RP_no_V": np.zeros(x_nm.shape, dtype=float),
        "c_R1_with": np.full(
            x_nm.shape, math.exp(float(np.clip(-z_r1 * phi_rp_tilde, -700.0, 700.0)))
        ),
        "c_O2_with": np.full(
            x_nm.shape, math.exp(float(np.clip(-z_o2 * phi_rp_tilde, -700.0, 700.0)))
        ),
        "c_R1_no": np.ones(x_nm.shape, dtype=float),
        "c_O2_no": np.ones(x_nm.shape, dtype=float),
        "eta_with_V": np.full(x_nm.shape, E_with - E_eq - phi_rp_V),
        "eta_no_V": np.full(x_nm.shape, E_no - E_eq),
        "current_density_with_A_per_m2": np.full(x_nm.shape, j_with),
        "current_density_no_A_per_m2": np.full(x_nm.shape, j_no),
        "sigma_with_C_per_m2": np.full(x_nm.shape, sigma),
    }
    grid_2d = {
        "x_nm": x_nm,
        "y_nm": y_nm,
        "phi_s_with_mV": 1000.0 * thermal_voltage * phi_tilde_2d,
        "phi_tilde_with": phi_tilde_2d,
        "c_R1_with": np.exp(np.clip(-z_r1 * phi_tilde_2d, -700.0, 700.0)),
        "c_O2_with": np.exp(np.clip(-z_o2 * phi_tilde_2d, -700.0, 700.0)),
        "phi_s_no_mV": np.zeros(phi_tilde_2d.shape, dtype=float),
        "phi_tilde_no": np.zeros(phi_tilde_2d.shape, dtype=float),
        "c_R1_no": np.ones(phi_tilde_2d.shape, dtype=float),
        "c_O2_no": np.ones(phi_tilde_2d.shape, dtype=float),
    }
    return {
        "material": material,
        "coordinate_system": f"independent_{material}_half_space",
        "surface": surface,
        "grid_2d": grid_2d,
        "charge": {
            "x_nm": x_nm,
            "phi_RP_V": surface["phi_RP_with_V"],
            "sigma_C_per_m2": surface["sigma_with_C_per_m2"],
        },
    }


def _independent_case(workspace_root: Path) -> dict[str, Any]:
    base = (
        workspace_root
        / "Figures"
        / "Figure_Au2nm_Pd2nm"
        / "Au_Pd_independent"
    )
    summary_path = base / "summary.json"
    params_path = base / "params.json"
    derived_path = base / "derived.json"
    summary = _read_json(summary_path)
    params = _read_json(params_path)
    derived = _read_json(derived_path)
    surfaces = {
        material: _independent_surface(
            material,
            params=params,
            derived=derived,
            summary=summary,
        )
        for material in ("Au", "Pd")
    }
    raw_with = float(summary["with_edl"]["i_mix_abs_A"])
    raw_no = float(summary["without_edl"]["i_mix_abs_A"])
    scale_factor = 2.0
    I_with = scale_factor * raw_with
    I_no = scale_factor * raw_no
    E_with = float(summary["with_edl"]["E_mix_V"])
    E_no = float(summary["without_edl"]["E_mix_V"])
    segments = [
        {
            "name": "Au",
            "material": "Au",
            "x_start_nm": 0.0,
            "x_end_nm": NATIVE_REFERENCE_LENGTH_NM,
            "length_nm": NATIVE_REFERENCE_LENGTH_NM,
            "faradaic": True,
            "coordinate_system": "independent_Au_half_space",
        },
        {
            "name": "Pd",
            "material": "Pd",
            "x_start_nm": 0.0,
            "x_end_nm": NATIVE_REFERENCE_LENGTH_NM,
            "length_nm": NATIVE_REFERENCE_LENGTH_NM,
            "faradaic": True,
            "coordinate_system": "independent_Pd_half_space",
        },
    ]
    charge_segments = [
        {
            "name": material,
            "material": material,
            **surfaces[material]["charge"],
        }
        for material in ("Au", "Pd")
    ]
    return {
        "key": "au_pd_independent",
        "directory_name": "01_Au_Pd_independent",
        "artifact_tag": "au_pd_independent",
        "display_label": "Independent Au/Pd",
        "topology_kind": "independent",
        "params": params,
        "segments": segments,
        "E_with_V": E_with,
        "E_no_V": E_no,
        "I_with_A": I_with,
        "I_no_A": I_no,
        "summary": _summary_block(E_with, E_no, I_with, I_no),
        "comparison_reactive_area_m2": COMPARISON_REACTIVE_AREA_M2,
        "raw_current": {
            "with_A": raw_with,
            "no_A": raw_no,
            "source_reactive_area_m2": float(derived["reactive_area_m2"]),
            "source_cell": "two independent 2 nm one-face planar interfaces",
            "scale_factor_to_comparison": scale_factor,
        },
        "raw_current_with_A": raw_with,
        "raw_current_no_A": raw_no,
        "scale_factor": scale_factor,
        "surface": None,
        "grid_2d": None,
        "charge_segments": charge_segments,
        "independent_surfaces": surfaces,
        "source_paths": {
            "summary": _source_path(summary_path),
            "params": _source_path(params_path),
            "derived": _source_path(derived_path),
            "model": _source_path(
                workspace_root
                / "Mixed_Potential_Electrical_Double_Layer"
                / "Au_Pd_independent_EDLs"
                / "src"
                / "au_pd_independent_edls"
                / "model.py"
            ),
        },
        "diagnostics": {
            "area_rescale_only": True,
            "comparison_uses": (
                "native Au(2)+Pd(2) one-face currents multiplied by two for "
                "the common Au(4)+Pd(4) Summary area"
            ),
            "source_has_neumann_lateral_reflection": False,
            "display_Au_length_nm": 2.0,
            "display_Pd_length_nm": 2.0,
            "effective_Au_length_nm": COMPARISON_AU_LENGTH_NM,
            "effective_Pd_length_nm": COMPARISON_PD_LENGTH_NM,
            "relative_balance_with": float(
                summary["with_edl"]["relative_balance_residual"]
            ),
            "relative_balance_no": float(
                summary["without_edl"]["relative_balance_residual"]
            ),
            "max_abs_phi_tilde_with": float(
                summary["with_edl"]["debye_huckel_validity"]["max_abs_phi_tilde"]
            ),
            "debye_huckel_threshold": float(
                summary["with_edl"]["debye_huckel_validity"]["threshold"]
            ),
            "debye_huckel_threshold_exceeded": bool(
                summary["with_edl"]["debye_huckel_validity"]["threshold_exceeded"]
            ),
        },
    }


def _janus_on_c_case(workspace_root: Path) -> dict[str, Any]:
    run = (
        workspace_root
        / "Figures"
        / "Figure_Janus_on_C_support"
        / "results"
        / JANUS_ON_C_RUN_ID
    )
    summary_path = run / "summary.json"
    params_path = run / "params.json"
    derived_path = run / "derived.json"
    surface_path = run / "csv" / "surface_profiles.csv"
    charge_path = run / "csv" / "surface_charge_distribution.csv"
    grid_path = run / "npz" / "display_grid_2d.npz"
    validation_path = run / "validation.json"
    summary = _read_json(summary_path)
    params = _read_json(params_path)
    derived = _read_json(derived_path)
    validation = _read_json(validation_path)
    surface_csv = _csv_columns(surface_path)
    charge_csv = _csv_columns(charge_path)
    with np.load(grid_path, allow_pickle=False) as archive:
        grid_values = {name: np.asarray(archive[name]).copy() for name in archive.files}

    material = np.asarray(surface_csv["material"], dtype=object)
    eta_with = np.where(
        material == "Au",
        surface_csv["eta_au_with_V"],
        np.where(material == "Pd", surface_csv["eta_pd_with_V"], np.nan),
    )
    eta_no = np.where(
        material == "Au",
        surface_csv["eta_au_no_V"],
        np.where(material == "Pd", surface_csv["eta_pd_no_V"], np.nan),
    )
    current_with = np.where(
        material == "Au",
        surface_csv["j_au_with"],
        np.where(material == "Pd", surface_csv["j_pd_with"], np.nan),
    )
    current_no = np.where(
        material == "Au",
        surface_csv["j_au_no"],
        np.where(material == "Pd", surface_csv["j_pd_no"], np.nan),
    )
    sigma_lookup = {
        (str(segment), float(x)): float(sigma)
        for segment, x, sigma in zip(
            charge_csv["segment_id"],
            charge_csv["x_nm"],
            charge_csv["sigma_C_per_m2"],
            strict=True,
        )
    }
    sigma_surface = np.asarray(
        [
            sigma_lookup[(str(segment), float(x))]
            for segment, x in zip(
                surface_csv["segment_id"], surface_csv["x_nm"], strict=True
            )
        ],
        dtype=float,
    )
    surface = {
        "x_nm": surface_csv["x_nm"],
        "segment_id": surface_csv["segment_id"],
        "material": material,
        "phi_tilde_with": surface_csv["phi_tilde_with"],
        "phi_tilde_no": surface_csv["phi_tilde_no"],
        "phi_RP_with_V": surface_csv["phi_rp_with_V"],
        "phi_RP_no_V": surface_csv["phi_rp_no_V"],
        "c_R1_with": surface_csv["c_R1_with"],
        "c_O2_with": surface_csv["c_O2_with"],
        "c_R1_no": surface_csv["c_R1_no"],
        "c_O2_no": surface_csv["c_O2_no"],
        "eta_Au_with_V": surface_csv["eta_au_with_V"],
        "eta_Pd_with_V": surface_csv["eta_pd_with_V"],
        "eta_Au_no_V": surface_csv["eta_au_no_V"],
        "eta_Pd_no_V": surface_csv["eta_pd_no_V"],
        "eta_with_V": eta_with,
        "eta_no_V": eta_no,
        "j_Au_with_A_per_m2": surface_csv["j_au_with"],
        "j_Pd_with_A_per_m2": surface_csv["j_pd_with"],
        "j_Au_no_A_per_m2": surface_csv["j_au_no"],
        "j_Pd_no_A_per_m2": surface_csv["j_pd_no"],
        "current_density_with_A_per_m2": current_with,
        "current_density_no_A_per_m2": current_no,
        "sigma_with_C_per_m2": sigma_surface,
    }
    grid_2d = {
        "x_nm": grid_values["x_nm"],
        "y_nm": grid_values["y_nm"],
        "phi_s_with_mV": grid_values["phi_s_mV"],
        "phi_tilde_with": grid_values["phi_tilde"],
        "c_R1_with": grid_values["c_R1_norm"],
        "c_O2_with": grid_values["c_O2_norm"],
        "phi_s_no_mV": np.zeros(grid_values["phi_s_mV"].shape, dtype=float),
        "phi_tilde_no": np.zeros(grid_values["phi_tilde"].shape, dtype=float),
        "c_R1_no": np.ones(grid_values["phi_tilde"].shape, dtype=float),
        "c_O2_no": np.ones(grid_values["phi_tilde"].shape, dtype=float),
    }
    segments = [
        {
            "name": str(segment["name"]),
            "material": str(segment["material"]),
            "x_start_nm": 1.0e9 * float(segment["x_start_m"]),
            "x_end_nm": 1.0e9 * float(segment["x_end_m"]),
            "length_nm": 1.0e9 * float(segment["length_m"]),
            "faradaic": bool(segment["faradaic"]),
        }
        for segment in derived["segments"]
    ]
    charge_segments: list[dict[str, Any]] = []
    for segment in segments:
        mask = np.asarray(charge_csv["segment_id"], dtype=object) == segment["name"]
        charge_segments.append(
            {
                "name": segment["name"],
                "material": segment["material"],
                "x_nm": charge_csv["x_nm"][mask].copy(),
                "phi_RP_V": charge_csv["phi_rp_V"][mask].copy(),
                "sigma_C_per_m2": charge_csv["sigma_C_per_m2"][mask].copy(),
            }
        )

    with_edl = summary["condition_results"]["with_edl"]
    without_edl = summary["condition_results"]["without_edl"]
    E_with = float(with_edl["E_mix_V"])
    E_no = float(without_edl["E_mix_V"])
    raw_with = float(with_edl["i_mix_abs_halfcell_A"])
    raw_no = float(without_edl["i_mix_abs_halfcell_A"])
    scale_factor = 1.0
    I_with = scale_factor * raw_with
    I_no = scale_factor * raw_no
    return {
        "key": "janus_on_c_support",
        "directory_name": "04_Janus_on_C_support",
        "artifact_tag": "janus_on_c_support",
        "display_label": "C-supported Janus",
        "topology_kind": "continuous",
        "params": params,
        "segments": segments,
        "E_with_V": E_with,
        "E_no_V": E_no,
        "I_with_A": I_with,
        "I_no_A": I_no,
        "summary": _summary_block(E_with, E_no, I_with, I_no),
        "comparison_reactive_area_m2": COMPARISON_REACTIVE_AREA_M2,
        "raw_current": {
            "with_A": raw_with,
            "no_A": raw_no,
            "source_reactive_area_m2": float(
                derived["reactive_area_halfcell_m2"]
            ),
            "source_cell": "18 nm C5|Au4|Pd4|C5 Neumann half-cell",
            "scale_factor_to_comparison": scale_factor,
        },
        "raw_current_with_A": raw_with,
        "raw_current_no_A": raw_no,
        "scale_factor": scale_factor,
        "surface": surface,
        "grid_2d": grid_2d,
        "charge_segments": charge_segments,
        "independent_surfaces": None,
        "source_paths": {
            "summary": _source_path(summary_path),
            "params": _source_path(params_path),
            "derived": _source_path(derived_path),
            "surface_profiles": _source_path(surface_path),
            "surface_charge": _source_path(charge_path),
            "grid_2d": _source_path(grid_path),
            "validation": _source_path(validation_path),
        },
        "diagnostics": {
            "source_halfcell_length_nm": float(derived["L_halfcell_m"]) * 1.0e9,
            "source_full_period_nm": float(derived["L_full_period_nm"]),
            "comparison_uses": (
                "the native 18 nm half-cell current; it already contains the "
                "common Au(4)+Pd(4) Summary area"
            ),
            "display_Au_length_nm": 4.0,
            "display_Pd_length_nm": 4.0,
            "source_Au_length_nm": 4.0,
            "source_Pd_length_nm": 4.0,
            "effective_Au_length_nm": COMPARISON_AU_LENGTH_NM,
            "effective_Pd_length_nm": COMPARISON_PD_LENGTH_NM,
            "relative_balance_with": float(with_edl["relative_balance_residual"]),
            "relative_balance_no": float(
                without_edl["relative_balance_residual"]
            ),
            "max_abs_phi_tilde_with": float(
                np.max(np.abs(surface["phi_tilde_with"]))
            ),
            "debye_huckel_threshold": float(params["dh_warn_threshold"]),
            "debye_huckel_threshold_exceeded": bool(
                np.max(np.abs(surface["phi_tilde_with"]))
                > float(params["dh_warn_threshold"])
            ),
            "validation_passed": bool(validation.get("passed", False)),
        },
    }


def _validate_comparison_case(case: Mapping[str, Any]) -> None:
    required = {
        "key",
        "directory_name",
        "artifact_tag",
        "display_label",
        "topology_kind",
        "params",
        "segments",
        "E_with_V",
        "E_no_V",
        "I_with_A",
        "I_no_A",
        "raw_current",
        "scale_factor",
        "surface",
        "grid_2d",
        "charge_segments",
        "independent_surfaces",
        "source_paths",
        "diagnostics",
    }
    missing = sorted(required.difference(case))
    if missing:
        raise KeyError(f"Case {case.get('key', '<unknown>')} is missing {missing}")
    if case["topology_kind"] not in {"independent", "continuous"}:
        raise ValueError(f"Unknown topology kind: {case['topology_kind']}")
    if not math.isclose(
        float(case["comparison_reactive_area_m2"]),
        COMPARISON_REACTIVE_AREA_M2,
        rel_tol=0.0,
        abs_tol=1.0e-24,
    ):
        raise ValueError(f"Case {case['key']} uses the wrong comparison area")
    for name in ("E_with_V", "E_no_V", "I_with_A", "I_no_A"):
        value = float(case[name])
        if not math.isfinite(value) or (name.startswith("I_") and value <= 0.0):
            raise ValueError(f"Invalid {name} for {case['key']}: {value}")
    if case["topology_kind"] == "continuous":
        if case["surface"] is None or case["grid_2d"] is None:
            raise ValueError(f"Continuous case {case['key']} lacks field data")
    elif not case["independent_surfaces"]:
        raise ValueError("Independent case lacks its two analytic surfaces")


def load_all_cases(workspace_root: Path) -> list[dict[str, Any]]:
    """Return the four ordered Hutchings-comparison cases.

    Parameters
    ----------
    workspace_root:
        Path to the project ``2026`` directory.

    Returns
    -------
    list of dict
        Ordered as independent reference, touching Janus, C-separated physical
        mixture surrogate, and C-supported Janus.  ``I_with_A``/``I_no_A`` use
        the common Au4+Pd4 Summary area, while ``raw_current`` retains each
        case's native computational-cell current.
    """

    root = Path(workspace_root).expanduser().resolve()
    if not (root / "Figures").is_dir():
        raise FileNotFoundError(f"Not a 2026 workspace root: {root}")
    engine = _legacy_engine(root)
    cases = [
        _independent_case(root),
        _legacy_case(
            root,
            engine,
            0.0,
            key="janus_au_pd",
            directory_name="02_Janus_Au2_Pd2",
            artifact_tag="janus_au2_pd2",
            display_label="Janus Au|Pd",
        ),
        _legacy_case(
            root,
            engine,
            10.0,
            key="physical_mixture_c10",
            directory_name="03_Physical_mixture_C10",
            artifact_tag="physical_mixture_c10",
            display_label="Physical mixture (C = 10 nm)",
        ),
        _janus_on_c_case(root),
    ]
    keys = [str(case["key"]) for case in cases]
    directories = [str(case["directory_name"]) for case in cases]
    if len(keys) != len(set(keys)) or len(directories) != len(set(directories)):
        raise RuntimeError("Case keys and output directory names must be unique")
    for case in cases:
        _validate_comparison_case(case)
    return cases


__all__ = [
    "COMPARISON_AU_LENGTH_NM",
    "COMPARISON_PD_LENGTH_NM",
    "COMPARISON_REACTIVE_AREA_M2",
    "load_all_cases",
]
