from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure

from janus_on_c_support import figures, pipeline
from janus_on_c_support.io import json_safe


@pytest.fixture
def fast_overrides() -> dict[str, int]:
    return {
        "N_modes": 16,
        "Nx": 81,
        "gl_order": 24,
        "Ny_2d": 17,
        "evaluation_block_size": 32,
        "mode_block_size": 16,
    }


def _fake_figures(bundle: dict[str, object], output_root: str | Path) -> list[Path]:
    root = Path(output_root)
    paths: list[Path] = []
    specifications = (("Figure_3", 6), ("Figure_RP", 3))
    for directory_name, count in specifications:
        directory = root / "figures" / directory_name
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            png = directory / f"test_{index + 1}.png"
            svg = directory / f"test_{index + 1}.svg"
            png.write_bytes(b"not-a-rendered-png")
            svg.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg"><text>test</text></svg>\n',
                encoding="utf-8",
            )
            paths.extend((png, svg))
    return paths


def test_build_result_bundle_without_publishing(
    monkeypatch: pytest.MonkeyPatch,
    fast_overrides: dict[str, int],
) -> None:
    monkeypatch.setattr(pipeline, "DISPLAY_NX", 65)
    bundle = pipeline.build_result_bundle(fast_overrides, publish=False)

    assert bundle["output"] is None
    assert bundle["validation"]["passed"] is True
    assert bundle["validation"]["geometry"]["boundaries_nm"] == pytest.approx(
        [0.0, 5.0, 9.0, 13.0, 18.0]
    )
    assert bundle["derived"]["L_full_period_nm"] == pytest.approx(36.0)
    assert bundle["surface"]["x_nm"].shape == (81,)
    assert bundle["surface"]["x_nm"][[0, -1]] == pytest.approx([0.0, 18.0])
    for boundary in (5.0, 9.0, 13.0):
        assert np.any(np.isclose(bundle["surface"]["x_nm"], boundary, atol=1.0e-12))
    assert bundle["grid_2d"]["phi_tilde"].shape == (17, 65)
    assert [item["segment_id"] for item in bundle["surface_charge_segments"]] == [
        "C_left",
        "Au",
        "Pd",
        "C_right",
    ]
    c_mask = bundle["surface"]["material"] == "C"
    assert np.all(np.isnan(bundle["surface"]["j_au_with"][c_mask]))
    assert np.all(np.isnan(bundle["surface"]["j_pd_with"][c_mask]))
    assert bundle["with_edl"]["relative_balance_residual"] < 1.0e-10
    assert bundle["without_edl"]["E_mix_V"] == pytest.approx(0.467, abs=1.0e-10)


def test_published_bundle_is_complete_strict_and_non_overwriting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fast_overrides: dict[str, int],
) -> None:
    monkeypatch.setattr(pipeline, "DISPLAY_NX", 41)
    monkeypatch.setattr(pipeline, "generate_all_figures", _fake_figures)
    bundle = pipeline.build_result_bundle(
        fast_overrides,
        output_root=tmp_path,
        run_id="test_run",
        publish=True,
    )
    output = tmp_path / "test_run"
    assert bundle["output"] == str(output.resolve())
    assert bundle["validation"]["convergence"]["performed"] is True
    assert bundle["validation"]["convergence"]["passed"] is True
    assert len(list(output.glob("figures/**/*.png"))) == 9
    assert len(list(output.glob("figures/**/*.svg"))) == 9
    assert not list(output.glob("**/*.pdf"))

    required = {
        "params.json",
        "derived.json",
        "summary.json",
        "validation.json",
        "run_manifest.json",
        "artifacts.json",
        "environment.json",
        "figure_metadata.json",
        "grid_2d_metadata.json",
        "checksums.sha256",
        "inputs/baseline_params.json",
        "inputs/overrides.json",
        "inputs/numerical_resolution.json",
        "csv/summary_compare.csv",
        "csv/convergence.csv",
        "csv/surface_profiles.csv",
        "csv/surface_charge_distribution.csv",
        "csv/gl_current_quadrature.csv",
        "csv/display_grid_2d_preview.csv",
        "npz/spectral_coefficients.npz",
        "npz/display_grid_2d.npz",
    }
    present = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert required <= present

    for path in output.glob("*.json"):
        decoded = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(decoded, dict)
        assert "NaN" not in path.read_text(encoding="utf-8")
        assert "Infinity" not in path.read_text(encoding="utf-8")

    checksum_lines = (output / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    assert checksum_lines
    for line in checksum_lines:
        expected, relative = line.split("  ", 1)
        actual = hashlib.sha256((output / relative).read_bytes()).hexdigest()
        assert actual == expected

    with pytest.raises(FileExistsError):
        pipeline.build_result_bundle(
            fast_overrides,
            output_root=tmp_path,
            run_id="test_run",
            publish=True,
        )


def test_json_safe_rejects_non_finite_values() -> None:
    assert json_safe({"value": np.float64(1.25)}) == {"value": 1.25}
    with pytest.raises(ValueError):
        json_safe({"value": float("nan")})
    with pytest.raises(ValueError):
        json_safe({"value": np.float64("inf")})


def test_noncanonical_geometry_cannot_be_published(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="canonical C\\(5\\)\\|Au\\(4\\)"):
        pipeline.build_result_bundle(
            {"L_C_left": 0.0, "L_Au": 2.0e-9, "L_Pd": 2.0e-9, "L_C_right": 0.0},
            output_root=tmp_path,
            run_id="wrong_geometry_label",
            publish=True,
        )
    assert not (tmp_path / "wrong_geometry_label").exists()


def test_explicit_g_override_keeps_exported_charge_consistent(
    monkeypatch: pytest.MonkeyPatch,
    fast_overrides: dict[str, int],
) -> None:
    monkeypatch.setattr(pipeline, "DISPLAY_NX", 41)
    overrides: dict[str, float | int] = {**fast_overrides, "g_C": 0.75}
    bundle = pipeline.build_result_bundle(overrides, publish=False)
    model = bundle["model"]
    E_mix = float(bundle["with_edl"]["E_mix_V"])
    for item in bundle["surface_charge_segments"]:
        expected = model.surface_charge_C_per_m2(
            E_mix,
            item["x_m"],
            material=item["material"],
        )
        assert item["sigma_C_per_m2"] == pytest.approx(expected, abs=2.0e-13)


def test_zero_width_C_segments_never_own_surface_endpoints(
    monkeypatch: pytest.MonkeyPatch,
    fast_overrides: dict[str, int],
) -> None:
    monkeypatch.setattr(pipeline, "DISPLAY_NX", 41)
    overrides: dict[str, float | int] = {
        **fast_overrides,
        "L_C_left": 0.0,
        "L_Au": 2.0e-9,
        "L_Pd": 2.0e-9,
        "L_C_right": 0.0,
    }
    bundle = pipeline.build_result_bundle(overrides, publish=False)
    assert bundle["surface"]["material"][[0, -1]].tolist() == ["Au", "Pd"]
    assert [item["segment_id"] for item in bundle["surface_charge_segments"]] == [
        "Au",
        "Pd",
    ]


@pytest.mark.parametrize("capacitance_name", ["C_H_C", "C_H_Au", "C_H_Pd"])
def test_zero_capacitance_is_a_valid_robin_limit(
    capacitance_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "DISPLAY_NX", 41)
    bundle = pipeline.build_result_bundle(
        {
            capacitance_name: 0.0,
            "N_modes": 960,
            "Nx": 81,
            "gl_order": 128,
            "Ny_2d": 17,
        },
        publish=False,
    )
    check = bundle["validation"]["numerical"][
        "stern_charge_vs_diffuse_normal_displacement"
    ]
    assert bundle["validation"]["passed"] is True
    assert check["passed"] is True


def test_real_figure_renderer_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fast_overrides: dict[str, int],
) -> None:
    """Exercise the real nine-figure renderer at a small test-only PNG DPI."""

    monkeypatch.setattr(pipeline, "DISPLAY_NX", 41)
    original_savefig = Figure.savefig

    def low_dpi_savefig(self: Figure, *args: object, **kwargs: object) -> object:
        kwargs["dpi"] = 120
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", low_dpi_savefig)
    bundle = pipeline.build_result_bundle(fast_overrides, publish=False)
    paths = figures.generate_all_figures(bundle, tmp_path)
    pngs = sorted(tmp_path.glob("figures/**/*.png"))
    svgs = sorted(tmp_path.glob("figures/**/*.svg"))
    assert len(paths) == 18
    assert len(pngs) == 9
    assert len(svgs) == 9
    assert len(list((tmp_path / "figures" / "Figure_3").glob("*.png"))) == 6
    assert len(list((tmp_path / "figures" / "Figure_RP").glob("*.png"))) == 3
    assert not list(tmp_path.glob("**/*.pdf"))
    assert all("<text" in path.read_text(encoding="utf-8") for path in svgs)

    svg_text = "\n".join(path.read_text(encoding="utf-8") for path in svgs)
    assert "Local " not in svg_text
    assert "Reactant concentration at RP" in svg_text
    assert "Overpotential at RP" in svg_text
    assert "Current density at RP" in svg_text
    assert r"$x$ (nm)" in svg_text
    assert r"$y$ (nm)" in svg_text
    assert r"$\mathit{\Phi}_{\mathrm{s}}$ (mV)" in svg_text
    assert r"$\mathrm{Red}_1^-$" in svg_text
    assert r"$\mathrm{Ox}_2^+$" in svg_text
    assert "(-)" not in svg_text
