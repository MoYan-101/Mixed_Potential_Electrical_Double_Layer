from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from hutchings_explanation import cases as case_loader  # noqa: E402
from hutchings_explanation import figures, pipeline, polarization  # noqa: E402
from hutchings_explanation.io import verify_checksums  # noqa: E402


def _surface(materials: list[str]) -> dict[str, np.ndarray]:
    size = len(materials)
    x = np.linspace(0.0, float(size - 1), size)
    material = np.asarray(materials, dtype=object)
    c_mask = material == "C"
    current = np.linspace(0.1, 0.3, size)
    current[c_mask] = np.nan
    return {
        "x_nm": x,
        "segment_id": np.asarray([f"segment_{index}" for index in range(size)], dtype=object),
        "material": material,
        "phi_tilde_with": np.linspace(-1.0, 1.0, size),
        "phi_tilde_no": np.zeros(size),
        "phi_RP_with_V": np.linspace(-0.02, 0.02, size),
        "phi_RP_no_V": np.zeros(size),
        "c_R1_with": np.linspace(0.8, 1.2, size),
        "c_O2_with": np.linspace(1.2, 0.8, size),
        "c_R1_no": np.ones(size),
        "c_O2_no": np.ones(size),
        "eta_Au_with_V": current,
        "eta_Pd_with_V": -current,
        "eta_Au_no_V": current,
        "eta_Pd_no_V": -current,
        "eta_with_V": current,
        "eta_no_V": current,
        "j_Au_with_A_per_m2": current,
        "j_Pd_with_A_per_m2": -current,
        "j_Au_no_A_per_m2": current,
        "j_Pd_no_A_per_m2": -current,
        "current_density_with_A_per_m2": current,
        "current_density_no_A_per_m2": current,
        "sigma_with_C_per_m2": np.linspace(-0.1, 0.1, size),
    }


def _grid() -> dict[str, np.ndarray]:
    x = np.linspace(0.0, 4.0, 4)
    y = np.linspace(0.0, 3.0, 3)
    phi = np.outer(np.exp(-y), np.linspace(-20.0, 20.0, x.size))
    return {
        "x_nm": x,
        "y_nm": y,
        "phi_s_with_mV": phi,
        "phi_tilde_with": phi / 25.7,
        "c_R1_with": np.exp(-phi / 25.7),
        "c_O2_with": np.exp(phi / 25.7),
        "phi_s_no_mV": np.zeros_like(phi),
        "phi_tilde_no": np.zeros_like(phi),
        "c_R1_no": np.ones_like(phi),
        "c_O2_no": np.ones_like(phi),
    }


def _charge(material: str) -> dict[str, Any]:
    return {
        "name": material,
        "material": material,
        "x_nm": np.asarray([0.0, 1.0]),
        "phi_RP_V": np.asarray([0.01, 0.01]),
        "sigma_C_per_m2": np.asarray([0.02, 0.02]),
    }


def _fake_cases(workspace: Path) -> list[dict[str, Any]]:
    source = workspace / "source.json"
    source.write_text('{"source": true}\n', encoding="utf-8")
    keys = (
        ("au_pd_independent", "01_Au_Pd_independent", "independent"),
        ("janus_au_pd", "02_Janus_Au2_Pd2", "continuous"),
        ("physical_mixture_c10", "03_Physical_mixture_C10", "continuous"),
        ("janus_on_c_support", "04_Janus_on_C_support", "continuous"),
    )
    cases: list[dict[str, Any]] = []
    for index, (key, directory, topology) in enumerate(keys, start=1):
        raw_with = (2.0 + index) * 1.0e-12
        scale = 2.0 if index < 4 else 1.0
        raw_no = 8.0e-12 / scale
        i_with = scale * raw_with
        i_no = scale * raw_no
        diagnostics: dict[str, Any] = {
            "effective_Au_length_nm": 4.0,
            "effective_Pd_length_nm": 4.0,
            "relative_balance_with": 1.0e-14,
            "relative_balance_no": 1.0e-14,
            "max_abs_phi_tilde_with": 4.0,
        }
        if key == "physical_mixture_c10":
            diagnostics["C10_vs_C1000_plateau"] = {
                "delta_E_C10_minus_C1000_mV": 0.239,
                "delta_I_C10_minus_C1000_percent": 0.026,
            }
        common_surface = _surface(["Au", "C", "Pd"])
        independent = {
            material: {
                "coordinate_system": f"independent_{material}_half_space",
                "surface": _surface([material, material]),
                "grid_2d": _grid(),
                "charge": _charge(material),
            }
            for material in ("Au", "Pd")
        }
        summary = {
            "with_edl": {
                "E_mix_V": 0.58 + 0.01 * index,
                "I_mix_A": i_with,
                "current_definition": pipeline.CURRENT_DEFINITION,
                "comparison_reactive_area_m2": pipeline.COMPARISON_REACTIVE_AREA_M2,
            },
            "without_edl": {
                "E_mix_V": 0.467,
                "I_mix_A": i_no,
                "current_definition": pipeline.CURRENT_DEFINITION,
                "comparison_reactive_area_m2": pipeline.COMPARISON_REACTIVE_AREA_M2,
            },
        }
        cases.append(
            {
                "key": key,
                "directory_name": directory,
                "artifact_tag": key,
                "display_label": key.replace("_", " "),
                "topology_kind": topology,
                "params": {"test_parameter": index},
                "segments": [
                    {
                        "name": name,
                        "material": material,
                        "x_start_nm": start,
                        "x_end_nm": end,
                        "length_nm": end - start,
                        "faradaic": material in {"Au", "Pd"},
                    }
                    for name, material, start, end in pipeline.EXPECTED_DISPLAY_SEGMENTS[
                        key
                    ]
                ],
                "E_with_V": summary["with_edl"]["E_mix_V"],
                "E_no_V": summary["without_edl"]["E_mix_V"],
                "I_with_A": i_with,
                "I_no_A": i_no,
                "summary": summary,
                "comparison_reactive_area_m2": pipeline.COMPARISON_REACTIVE_AREA_M2,
                "raw_current": {
                    "with_A": raw_with,
                    "no_A": raw_no,
                    "source_reactive_area_m2": pipeline.COMPARISON_REACTIVE_AREA_M2 / scale,
                    "source_cell": "test",
                    "scale_factor_to_comparison": scale,
                },
                "raw_current_with_A": raw_with,
                "raw_current_no_A": raw_no,
                "scale_factor": scale,
                "surface": None if topology == "independent" else common_surface,
                "grid_2d": None if topology == "independent" else _grid(),
                "charge_segments": [_charge("Au"), _charge("Pd")],
                "independent_surfaces": independent if topology == "independent" else None,
                "source_paths": {"test_source": str(source)},
                "diagnostics": diagnostics,
            }
        )
    return cases


def _write_pair(directory: Path, stem: str) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    png = directory / f"{stem}.png"
    svg = directory / f"{stem}.svg"
    png.write_bytes(b"test-png")
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><text>Potential</text></svg>\n',
        encoding="utf-8",
    )
    return [png, svg]


def _write_polarization_pair(directory: Path, stem: str) -> list[Path]:
    paths = _write_pair(directory, stem)
    paths[1].write_text(
        '<svg xmlns="http://www.w3.org/2000/svg">'
        '<text>Electrode potential (V vs. RHE)</text>'
        '<text>Oxidation on Au</text>'
        '<text>Reduction on Pd</text>'
        '<text><tspan style="font-style: normal">µ</tspan>A</text>'
        '</svg>\n',
        encoding="utf-8",
    )
    return paths


def _fake_summary_figures(cases: Any, output_root: str | Path) -> list[Path]:
    output = Path(output_root)
    paths: list[Path] = []
    for index in range(2):
        paths.extend(_write_pair(output, f"summary_{index + 1}"))
    return paths


def _fake_summary_polarization(
    cases: Any, workspace_root: str | Path, summary_root: str | Path
) -> dict[str, Any]:
    output = Path(summary_root)
    csv_path = output / "csv" / "four_case_polarization_curves.csv"
    json_path = output / "polarization_summary.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("condition,E_V,I_Au_A,I_Pd_A\nw/o EDL,0.467,1e-11,-1e-11\n", encoding="utf-8")
    json_path.write_text('{"validation": {"passed": true}}\n', encoding="utf-8")
    figure_paths = _write_polarization_pair(
        output / "figures", "summary_half_reaction_polarization_overlay"
    )
    figure_paths.extend(
        _write_polarization_pair(
            output / "figures",
            "summary_half_reaction_polarization_overlay_xmin045V",
        )
    )
    return {
        "data_paths": [csv_path, json_path],
        "figure_paths": figure_paths,
        "rows": [],
        "conditions": [],
        "validation": {"passed": True},
    }


def _fake_case_figures(case: Any, output_root: str | Path) -> list[Path]:
    output = Path(output_root)
    paths: list[Path] = []
    for index in range(6):
        paths.extend(_write_pair(output / "Figure_3", f"figure_3_{index + 1}"))
    for index in range(3):
        paths.extend(_write_pair(output / "Figure_RP", f"figure_rp_{index + 1}"))
    return paths


def test_build_collection_is_complete_and_non_overwriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    output = tmp_path / "published"
    (workspace / "Figures").mkdir(parents=True)
    cases = _fake_cases(workspace)
    monkeypatch.setattr(pipeline, "load_all_cases", lambda root: cases)
    monkeypatch.setattr(pipeline, "generate_summary_figures", _fake_summary_figures)
    monkeypatch.setattr(
        pipeline, "build_summary_polarization", _fake_summary_polarization
    )
    monkeypatch.setattr(pipeline, "generate_case_figures", _fake_case_figures)

    result = pipeline.build_collection(workspace, output)

    assert result["validation"]["passed"] is True
    assert len(result["figure_paths"]) == 80
    assert len(list(output.rglob("*.png"))) == 40
    assert len(list(output.rglob("*.svg"))) == 40
    assert not list(output.rglob("*.pdf"))
    assert (
        output
        / "Summary"
        / "figures"
        / "summary_half_reaction_polarization_overlay.png"
    ).is_file()
    assert (
        output
        / "Summary"
        / "figures"
        / "summary_half_reaction_polarization_overlay_xmin045V.png"
    ).is_file()
    assert (output / "Summary" / "csv" / "four_case_comparison.csv").is_file()
    assert (
        output / "Summary" / "csv" / "four_case_polarization_curves.csv"
    ).is_file()
    assert (output / "Summary" / "polarization_summary.json").is_file()
    for case in cases:
        case_root = output / case["directory_name"]
        assert len(list((case_root / "Figure_3").glob("*.png"))) == 6
        assert len(list((case_root / "Figure_RP").glob("*.svg"))) == 3
        assert (case_root / "data" / "surface_profiles.npz").is_file()
        assert (case_root / "data" / "display_grid_2d.npz").is_file()
    validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
    assert validation["figure_validation"]["visible_word_Local_absent"] is True
    assert validation["figure_validation"]["polarization_figure_layout"] == {
        "variant_count": 2,
        "variants": [
            {
                "variant_key": "xmin040V",
                "stem": "summary_half_reaction_polarization_overlay",
                "x_min_V": 0.4,
                "x_max_V": 0.64,
            },
            {
                "variant_key": "xmin045V",
                "stem": "summary_half_reaction_polarization_overlay_xmin045V",
                "x_min_V": 0.45,
                "x_max_V": 0.75,
            },
        ],
        "x_axis_label": "Electrode potential (V vs. RHE)",
        "y_axis_label": r"Current (10$^{-3}$ µA)",
        "reaction_legend_labels": ["Oxidation on Au", "Reduction on Pd"],
        "obsolete_half_reaction_title_absent": True,
        "current_unit_upright": True,
        "passed": True,
    }
    assert validation["polarization_validation"]["passed"] is True
    assert validation["case_validation"]["C10_vs_C1000_plateau"]["passed"] is True
    verified = verify_checksums(output, require_complete=True)
    assert "artifacts.json" in verified
    assert "manifest.json" in verified

    with pytest.raises(FileExistsError):
        pipeline.build_collection(workspace, output)


def test_incorrect_absolute_current_conversion_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    output = tmp_path / "published"
    (workspace / "Figures").mkdir(parents=True)
    cases = _fake_cases(workspace)
    cases[0]["I_with_A"] *= 1.01
    monkeypatch.setattr(pipeline, "load_all_cases", lambda root: cases)

    with pytest.raises(RuntimeError, match="area conversion"):
        pipeline.build_collection(workspace, output)
    assert not output.exists()


def test_C10_C1000_plateau_threshold_is_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    output = tmp_path / "published"
    (workspace / "Figures").mkdir(parents=True)
    cases = _fake_cases(workspace)
    cases[2]["diagnostics"]["C10_vs_C1000_plateau"][
        "delta_I_C10_minus_C1000_percent"
    ] = 0.05
    monkeypatch.setattr(pipeline, "load_all_cases", lambda root: cases)

    with pytest.raises(RuntimeError, match="plateau"):
        pipeline.build_collection(workspace, output)
    assert not output.exists()


def test_public_import() -> None:
    from hutchings_explanation import build_collection

    assert build_collection is pipeline.build_collection


def test_polarization_grid_covers_both_published_views() -> None:
    potentials = polarization.POTENTIAL_VALUES_V

    assert potentials.size == 1401
    assert potentials[0] == pytest.approx(0.40)
    assert potentials[-1] == pytest.approx(0.75)
    assert all(
        float(variant["x_min_V"]) >= float(potentials[0])
        and float(variant["x_max_V"]) <= float(potentials[-1])
        for variant in polarization.FIGURE_VARIANTS
    )
    assert polarization.AU_LEGEND_LABEL == "Oxidation on Au"
    assert polarization.PD_LEGEND_LABEL == "Reduction on Pd"
    assert polarization.Y_AXIS_LABEL == r"Current (10$^{-3}$ µA)"


def test_figure_parameter_lookup_falls_back_to_nested_params() -> None:
    case = {"params": {"E1_eq": 0.10, "pzc_Au": 0.93}}

    assert figures._parameter(case, "E1_eq") == pytest.approx(0.10)
    assert figures._parameter(case, "pzc_Au") == pytest.approx(0.93)


def test_profile_layout_omits_redundant_material_annotations() -> None:
    fig, ax = figures._profile_figure()
    segments = (
        figures._Segment("Au", "Au", 0.0, 2.0),
        figures._Segment("Pd", "Pd", 2.0, 4.0),
    )
    figures._finish_continuous_profile(fig, ax, segments)
    try:
        assert len(fig.axes) == 1
        assert ax.get_xlabel() == r"$x$ (nm)"
    finally:
        figures.plt.close(fig)

    fig, axes = figures._independent_profile_layout("Title", "Value")
    try:
        assert all(not axis.texts for axis in axes)
        assert [axis.get_title(loc="left") for axis in axes] == [
            "Au interface",
            "Pd interface",
        ]
    finally:
        figures.plt.close(fig)


def test_summary_plot_uses_one_common_without_edl_baseline() -> None:
    no_values = np.asarray([0.467, 0.467, 0.467, 0.467 + 2.0e-16])
    with_values = np.asarray([0.625, 0.625, 0.598, 0.611])

    values, conditions = figures._summary_plot_values(
        no_values, with_values, metric="E_mix"
    )

    assert values.shape == (5,)
    assert conditions.count("without") == 1
    assert conditions.count("with") == 4
    assert values[0] == pytest.approx(0.467)
    assert np.array_equal(values[1:], with_values)

    with pytest.raises(ValueError, match="one common w/o-EDL"):
        figures._summary_plot_values(
            np.asarray([0.467, 0.468]),
            np.asarray([0.625, 0.624]),
            metric="E_mix",
        )


def test_real_cases_keep_native_2nm_displays_and_4nm_summary_basis() -> None:
    workspace = PACKAGE_ROOT.parents[1]
    cases = case_loader.load_all_cases(workspace)

    assert [case["directory_name"] for case in cases] == [
        "01_Au_Pd_independent",
        "02_Janus_Au2_Pd2",
        "03_Physical_mixture_C10",
        "04_Janus_on_C_support",
    ]
    assert pipeline.COMPARISON_REACTIVE_AREA_M2 == pytest.approx(8.0e-11)
    assert [case["scale_factor"] for case in cases] == [2.0, 2.0, 2.0, 1.0]

    expected_edges = [
        [(0.0, 2.0), (0.0, 2.0)],
        [(0.0, 2.0), (2.0, 4.0)],
        [(0.0, 2.0), (2.0, 12.0), (12.0, 14.0)],
        [(0.0, 5.0), (5.0, 9.0), (9.0, 13.0), (13.0, 18.0)],
    ]
    for case, expected in zip(cases, expected_edges, strict=True):
        actual = [
            (float(segment["x_start_nm"]), float(segment["x_end_nm"]))
            for segment in case["segments"]
        ]
        assert np.asarray(actual) == pytest.approx(np.asarray(expected), abs=1.0e-12)
        assert case["I_with_A"] == pytest.approx(
            case["scale_factor"] * case["raw_current"]["with_A"], abs=1.0e-24
        )
        assert case["I_no_A"] == pytest.approx(
            case["scale_factor"] * case["raw_current"]["no_A"], abs=1.0e-24
        )

    assert cases[1]["surface"]["x_nm"][[0, -1]] == pytest.approx([0.0, 4.0])
    assert cases[2]["surface"]["x_nm"][[0, -1]] == pytest.approx([0.0, 14.0])
    assert cases[3]["surface"]["x_nm"][[0, -1]] == pytest.approx([0.0, 18.0])
