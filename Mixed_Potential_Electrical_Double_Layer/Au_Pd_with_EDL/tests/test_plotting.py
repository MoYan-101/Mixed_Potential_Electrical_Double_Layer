from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from au_pd_edl.figure3 import (
    generate_figure3_comparison_panels,
    load_figure3_saved_data,
)
from au_pd_edl.io import save_run
from au_pd_edl.plotting import generate_rp_2d_figures, plot_separation_trends
from au_pd_edl.scan import run_separation_scan


def test_save_run_rejects_nonempty_directory_before_other_work(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "existing"
    run_dir.mkdir()
    sentinel = run_dir / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_run(run_dir, {}, {})

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert list(run_dir.iterdir()) == [sentinel]


def test_save_run_does_not_relabel_legacy_scan_as_schema_v2(
    tmp_path: Path,
) -> None:
    output = tmp_path / "legacy-attempt"
    legacy_scan = {
        "rows": [{"d_Au_Pd_nm": 10.0, "n_nodes": 123}],
        "electrostatic_backend": "finite_height_q1",
    }

    with pytest.raises(ValueError, match="not a result-schema-v2"):
        save_run(output, {}, legacy_scan)

    assert output.is_dir()
    assert not list(output.iterdir())


@pytest.mark.parametrize(
    ("summary", "message"),
    [
        ({"tag": "legacy"}, "Unsupported summary.json result contract"),
        (
            {
                "result_schema_version": 2,
                "electrostatic_backend": "finite_height_q1",
                "tag": "wrong-backend",
            },
            "Unsupported summary.json result contract",
        ),
    ],
)
def test_figure3_loader_rejects_legacy_or_wrong_backend_summary(
    tmp_path: Path,
    summary: dict[str, object],
    message: str,
) -> None:
    (tmp_path / "summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_figure3_saved_data(tmp_path)


def test_figure3_loader_rejects_manifest_summary_contract_mismatch(
    tmp_path: Path,
) -> None:
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "result_schema_version": 2,
                "electrostatic_backend": "semi_infinite_cosine_fourier",
                "tag": "schema2",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "result_schema_version": 2,
                "electrostatic_backend": "finite_height_q1",
                "tag": "schema2",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="disagree on result version/backend"):
        load_figure3_saved_data(tmp_path)


@pytest.mark.integration
def test_rp_batch_shared_scales_and_output_contract(
    coarse_params: dict[str, object], tmp_path: Path
) -> None:
    scan = run_separation_scan(
        coarse_params,
        [0.0, 2.0e-9, 3.0e-9, 10.0e-9, 100.0e-9],
    )
    rp_dir = tmp_path / "rp"
    metadata = generate_rp_2d_figures(
        scan["cases"],
        rp_dir,
        "pytest",
        n_x=61,
        n_y=31,
        dpi=150,
    )

    assert len(list(rp_dir.glob("*.png"))) == 4
    assert len(list(rp_dir.glob("*.svg"))) == 4
    assert not list(rp_dir.glob("*.pdf"))
    assert metadata["grid"] == {
        "n_x": 61,
        "n_y": 31,
        "y_max_over_lambda": 5.0,
    }
    assert metadata["shared_color_limits"]["phi_s_mV"][0] == -metadata[
        "shared_color_limits"
    ]["phi_s_mV"][1]
    assert all("<text" in path.read_text(encoding="utf-8") for path in rp_dir.glob("*.svg"))

    trend_dir = tmp_path / "trends"
    trend = plot_separation_trends(scan, trend_dir, "pytest", dpi=150)
    assert len(trend["separations_nm"]) == 5
    assert "x_range_nm" not in trend
    assert Path(trend["saved_paths"][0]).name == (
        "emix_imix_vs_separation_pytest.png"
    )

    zoom = plot_separation_trends(
        scan,
        trend_dir,
        "pytest",
        dpi=150,
        x_max_nm=20.0,
    )
    assert zoom["separations_nm"] == [0.0, 2.0, 3.0, 10.0]
    assert zoom["x_range_nm"] == [0.0, 20.0]
    assert zoom["x_ticks_nm"] == [0.0, 5.0, 10.0, 15.0, 20.0]
    assert zoom["source_row_count"] == 5
    assert zoom["visible_row_count"] == 4
    assert zoom["n_visible_points"] == 4
    assert zoom["reference_100nm"] == trend["reference_100nm"]
    assert Path(zoom["saved_paths"][0]).name == (
        "emix_imix_vs_separation_0_20nm_pytest.png"
    )
    assert len(list(trend_dir.glob("*.png"))) == 2
    assert len(list(trend_dir.glob("*.svg"))) == 2
    assert not list(trend_dir.glob("*.pdf"))

    run_dir = tmp_path / "saved_run"
    saved_run = save_run(
        run_dir,
        coarse_params,
        scan,
        display_nx=61,
        display_ny=31,
    )
    for document_name in ("summary.json", "artifacts.json"):
        document = json.loads((run_dir / document_name).read_text(encoding="utf-8"))
        assert document["result_schema_version"] == 2
        assert document["electrostatic_backend"] == (
            "semi_infinite_cosine_fourier"
        )
    artifacts = json.loads((run_dir / "artifacts.json").read_text(encoding="utf-8"))
    assert set(artifacts["cases"]) == {"0.0", "2.0", "3.0", "10.0"}
    diagnostic_and_saved_counts_differ = False
    for artifact_paths in artifacts["cases"].values():
        metadata = json.loads(
            Path(artifact_paths["spectral_metadata_json"]).read_text(
                encoding="utf-8"
            )
        )["electrostatics"]
        with np.load(artifact_paths["full_spectral_npz"]) as spectral_field:
            saved_n_x = int(spectral_field["surface_x_tilde"].size)
            assert int(spectral_field["surface_phi_tilde"].size) == saved_n_x
        assert metadata["saved_surface_n_x_coordinates"] == saved_n_x
        assert metadata["n_x_coordinates"] == saved_n_x
        assert metadata["n_x_coordinates_role"] == (
            "saved spectral_field surface_x_tilde coordinate count"
        )
        diagnostic_n_x = metadata["diagnostic_surface_n_x_coordinates"]
        diagnostic_and_saved_counts_differ |= diagnostic_n_x != saved_n_x
    assert diagnostic_and_saved_counts_differ
    assert saved_run["result_schema_version"] == 2
    assert saved_run["electrostatic_backend"] == "semi_infinite_cosine_fourier"

    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "result_schema_version": 2,
                "electrostatic_backend": "semi_infinite_cosine_fourier",
                "tag": saved_run["tag"],
                "sentinel": "Figure 3 must not mutate this document",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_before = manifest_path.read_bytes()
    figure3 = generate_figure3_comparison_panels(run_dir, dpi=150)
    figure3_dir = run_dir / "figures" / "Figure_3"
    assert figure3["separation_nm"] == pytest.approx(10.0)
    assert figure3["geometry_nm"] == {
        "L_Au": pytest.approx(25.0),
        "insulating_substrate_gap": pytest.approx(10.0),
        "L_Pd": pytest.approx(25.0),
    }
    assert figure3["result_schema_version"] == 2
    assert figure3["electrostatic_backend"] == "semi_infinite_cosine_fourier"
    assert manifest_path.read_bytes() == manifest_before
    assert len(list(figure3_dir.glob("*.png"))) == 6
    assert len(list(figure3_dir.glob("*.svg"))) == 6
    assert not list(figure3_dir.glob("*.pdf"))
    assert all(
        "<text" in path.read_text(encoding="utf-8")
        for path in figure3_dir.glob("*.svg")
    )
    panel_f_svg = next(figure3_dir.glob("figure_3_panel_f_*.svg"))
    panel_f_text = panel_f_svg.read_text(encoding="utf-8")
    assert "PZC Au" in panel_f_text
    assert "PZC Pd" in panel_f_text
    assert "support" not in panel_f_text
    panel_b_svg = next(figure3_dir.glob("figure_3_panel_b_*.svg"))
    panel_b_text = panel_b_svg.read_text(encoding="utf-8")
    assert "insulating" in panel_b_text
    assert "substrate" in panel_b_text
    assert "electrolyte gap" not in panel_b_text

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_run(
            run_dir,
            coarse_params,
            scan,
            display_nx=61,
            display_ny=31,
        )


def test_separation_trend_rejects_invalid_x_max(tmp_path: Path) -> None:
    scan = {
        "rows": [
            {
                "d_Au_Pd_nm": 100.0,
                "E_mix_with_EDL_V": 0.6,
                "E_mix_without_EDL_V": 0.5,
                "i_mix_avg_with_EDL_A_per_m2": 0.08,
                "i_mix_avg_without_EDL_A_per_m2": 0.1,
            }
        ]
    }
    with pytest.raises(ValueError, match=r"x_max_nm must be within \(0, 100\]"):
        plot_separation_trends(scan, tmp_path, "pytest", x_max_nm=0.0)
