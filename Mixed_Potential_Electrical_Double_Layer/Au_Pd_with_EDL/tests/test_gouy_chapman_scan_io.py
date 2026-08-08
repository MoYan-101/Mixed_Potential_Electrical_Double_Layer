from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import au_pd_edl.scan as scan_module
from au_pd_edl.io import save_gouy_chapman_outputs
from au_pd_edl.parameters import default_params
from au_pd_edl.scan import build_gouy_chapman_analysis, run_separation_scan


def _material_diagnostic(metal: str, mean_length_nm: float) -> dict[str, Any]:
    mean_sigma = 0.035 if metal == "Au" else 0.026
    return {
        "effective_C_H_F_per_m2": 0.2 if metal == "Au" else 0.4,
        "boundary_length_m": 25.0e-9,
        "boundary_area_m2": 25.0e-11,
        "n_boundary_quadrature_points": 8,
        "n_boundary_samples": 5,
        "mean_signed_sigma_C_per_m2": -mean_sigma,
        "mean_abs_sigma_C_per_m2": mean_sigma,
        "local_min_sigma_C_per_m2": -1.1 * mean_sigma,
        "local_max_sigma_C_per_m2": -0.9 * mean_sigma,
        "charge_sign": "negative",
        "has_sign_change": False,
        "has_zero_charge": False,
        "has_zero_crossing": False,
        "mean_gouy_chapman_length_m": mean_length_nm * 1.0e-9,
        "mean_gouy_chapman_length_nm": mean_length_nm,
        "mean_length_infinite": False,
        "local_min_gouy_chapman_length_m": 0.9 * mean_length_nm * 1.0e-9,
        "local_min_gouy_chapman_length_nm": 0.9 * mean_length_nm,
        "local_max_gouy_chapman_length_m": 1.1 * mean_length_nm * 1.0e-9,
        "local_max_gouy_chapman_length_nm": 1.1 * mean_length_nm,
        "local_max_length_infinite": False,
        "zero_tolerance_C_per_m2": 1.0e-15,
    }


def _diagnostic(d_nm: float) -> dict[str, Any]:
    return {
        "applicable": True,
        "status": "computed_from_stern_surface_charge",
        "formula": "l_GC = 2 epsilon_s R T / (F |sigma|)",
        "mean_definition": (
            "l_GC,mean = 2 epsilon_s R T / (F <|sigma|>_boundary)"
        ),
        "surface_charge_definition": (
            "sigma_M = C_H,M,eff (E_mix - PZC_M - phi_s)"
        ),
        "surface_charge_sign_convention": (
            "positive sigma denotes positive charge on the metal side"
        ),
        "interpretation": "charge-derived diagnostic",
        "gouy_chapman_numerator_C_per_m": 3.57e-11,
        "metals": {
            "Au": _material_diagnostic("Au", 1.0 + 0.001 * d_nm),
            "Pd": _material_diagnostic("Pd", 1.3 + 0.002 * d_nm),
        },
    }


def _synthetic_scan_result() -> dict[str, Any]:
    distances = np.asarray(
        [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 100.0]
    )
    E_reference = 0.598
    i_reference = 0.084
    E_overlap = 0.0025 * np.exp(-distances / 2.2)
    i_overlap = 0.0023 * np.exp(-distances / 2.7)
    rows = []
    gc_rows = []
    for d_nm, delta_E, delta_i in zip(
        distances, E_overlap, i_overlap, strict=True
    ):
        rows.append(
            {
                "d_Au_Pd_m": float(d_nm * 1.0e-9),
                "d_Au_Pd_nm": float(d_nm),
                "delta_E_overlap_vs_100nm_V": float(delta_E),
                "delta_i_overlap_vs_100nm_A_per_m2": float(-delta_i),
                "max_abs_phi_tilde": 6.0,
            }
        )
        diagnostic = _diagnostic(float(d_nm))
        for metal in ("Au", "Pd"):
            gc_rows.append(
                {
                    "d_Au_Pd_m": float(d_nm * 1.0e-9),
                    "d_Au_Pd_nm": float(d_nm),
                    "d_over_lambda_D": float(d_nm / 3.041217889),
                    "applicable": True,
                    "diagnostic_status": diagnostic["status"],
                    "formula": diagnostic["formula"],
                    "mean_definition": diagnostic["mean_definition"],
                    "surface_charge_definition": diagnostic[
                        "surface_charge_definition"
                    ],
                    "surface_charge_sign_convention": diagnostic[
                        "surface_charge_sign_convention"
                    ],
                    "interpretation": diagnostic["interpretation"],
                    "gouy_chapman_numerator_C_per_m": diagnostic[
                        "gouy_chapman_numerator_C_per_m"
                    ],
                    "metal": metal,
                    **diagnostic["metals"][metal],
                }
            )
    derived = {
        "epsilon_s": 6.950537433048001e-10,
        "R": 8.314,
        "T": 298.0,
        "F": 96485.0,
        "lambda_D": 3.0412178890177212e-9,
    }
    return {
        "rows": rows,
        "gouy_chapman_rows": gc_rows,
        "no_overlap_100nm_reference": {
            "E_mix_V": E_reference,
            "i_mix_avg_A_per_m2": i_reference,
            "derived": derived,
            "params": {"dh_warn_threshold": 1.0},
        },
    }


def test_analysis_recovers_zero_offset_decay_lengths_and_paper_benchmark() -> None:
    analysis = build_gouy_chapman_analysis(_synthetic_scan_result())
    fits = analysis["overlap_exponential_fits"]

    assert fits["potential"]["n_points"] == 9
    assert fits["potential"]["decay_length_nm"] == pytest.approx(2.2)
    assert fits["potential"]["r_squared"] == pytest.approx(1.0)
    assert fits["current_density"]["decay_length_nm"] == pytest.approx(2.7)
    assert fits["current_density"]["r_squared"] == pytest.approx(1.0)
    assert analysis["paper_benchmark"]["gouy_chapman_length_nm"] == pytest.approx(
        0.35696, rel=2.0e-4
    )
    assert analysis["model_length_scales"]["debye_length_nm"] == pytest.approx(
        3.041217889
    )


def test_gc_writer_is_strict_json_and_does_not_rewrite_separation_csv(
    tmp_path: Path,
) -> None:
    separation_path = tmp_path / "csv" / "separation_scan.csv"
    separation_path.parent.mkdir(parents=True)
    separation_path.write_text("sentinel\n", encoding="utf-8")
    digest_before = hashlib.sha256(separation_path.read_bytes()).hexdigest()

    saved = save_gouy_chapman_outputs(tmp_path, _synthetic_scan_result())

    assert hashlib.sha256(separation_path.read_bytes()).hexdigest() == digest_before
    json_text = Path(saved["analysis_json"]).read_text(encoding="utf-8")
    assert "NaN" not in json_text
    assert "Infinity" not in json_text
    parsed = json.loads(
        json_text,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    assert parsed["schema_version"] == 1
    assert parsed["result_schema_version"] == 2
    assert parsed["electrostatic_backend"] == "semi_infinite_cosine_fourier"
    with Path(saved["csv"]).open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 20
    assert {row["metal"] for row in csv_rows} == {"Au", "Pd"}
    assert "n_boundary_samples" in csv_rows[0]
    assert "n_boundary_nodes" not in csv_rows[0]


def test_run_separation_scan_collects_two_gc_rows_per_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_solve_case(
        params: dict[str, Any],
        separation_m: float,
        *,
        use_edl: bool,
        return_model: bool = False,
    ) -> Any:
        d_nm = float(separation_m * 1.0e9)
        result = {
            "d_Au_Pd_m": float(separation_m),
            "d_Au_Pd_nm": d_nm,
            "E_mix_V": 0.598
            + (0.0025 * np.exp(-d_nm / 2.2) if use_edl else 0.0),
            "i_mix_avg_A_per_m2": (
                0.084 - 0.0023 * np.exp(-d_nm / 2.7) if use_edl else 0.1
            ),
            "I_Au_A": 1.0e-11,
            "I_Pd_A": -1.0e-11,
            "relative_balance_residual": 0.0,
            "debye_huckel_validity": {"max_abs_phi_tilde": 0.5},
            "electrostatics": (
                {
                    "n_modes": 9,
                    "n_coefficients": 10,
                    "surface_quadrature_target_points": 101,
                }
                if use_edl
                else None
            ),
            "derived": {"lambda_D": 3.0e-9},
            "gouy_chapman": _diagnostic(d_nm) if use_edl else {
                "applicable": False,
                "metals": None,
            },
        }
        return (result, object()) if return_model else result

    monkeypatch.setattr(scan_module, "solve_case", fake_solve_case)
    scan = run_separation_scan(
        default_params(),
        [0.0, 100.0e-9],
        retain_models_at_nm=(),
    )

    assert len(scan["gouy_chapman_rows"]) == 4
    assert {
        (row["d_Au_Pd_nm"], row["metal"])
        for row in scan["gouy_chapman_rows"]
    } == {(0.0, "Au"), (0.0, "Pd"), (100.0, "Au"), (100.0, "Pd")}
