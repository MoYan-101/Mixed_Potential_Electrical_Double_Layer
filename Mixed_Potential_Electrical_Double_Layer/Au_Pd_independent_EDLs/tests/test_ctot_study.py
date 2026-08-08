import csv
import json

import pytest

from au_pd_independent_edls.ctot_study import (
    build_ctot_results,
    compute_scan_rows,
    concentration_scan_values_m,
    row_at,
)
from au_pd_independent_edls.model import default_params


@pytest.fixture(scope="module")
def scan_rows():
    return compute_scan_rows(default_params())


def test_scan_grid_contains_exact_reference_cases() -> None:
    values = concentration_scan_values_m()
    assert len(values) == 36
    for value in (0.01, 1.0, 10.0, 1000.0):
        assert value in values


def test_representative_independent_edl_regression(scan_rows) -> None:
    low = row_at(scan_rows, 0.01)
    middle = row_at(scan_rows, 1.0)
    high = row_at(scan_rows, 1000.0)

    assert low["lambda_D_nm"] == pytest.approx(3.0412178890177213, rel=2e-13)
    assert low["E_mix_with_EDL_V"] == pytest.approx(0.6008249905513605, abs=2e-12)
    assert low["i_mix_avg_with_EDL_A_per_m2"] == pytest.approx(
        0.07995142584528597, rel=2e-11
    )
    assert middle["E_mix_with_EDL_V"] == pytest.approx(0.5046235116332828, abs=2e-12)
    assert middle["i_mix_avg_with_EDL_A_per_m2"] == pytest.approx(
        0.1255923041387018, rel=2e-11
    )
    assert high["E_mix_with_EDL_V"] == pytest.approx(0.46849409592372837, abs=2e-12)
    assert high["i_mix_avg_ratio_with_over_without"] == pytest.approx(
        1.0043024743342899, rel=2e-11
    )


def test_mechanism_reconstructs_charge_and_kinetics(scan_rows) -> None:
    for concentration_m in (0.01, 1.0, 1000.0):
        row = row_at(scan_rows, concentration_m)
        for material in ("Au", "Pd"):
            assert abs(row[f"{material}_charge_relation_residual_C_per_m2"]) < 1e-13
            assert abs(row[f"{material}_j_reconstruction_residual_A_per_m2"]) < 1e-13
        assert row["relative_current_balance_residual_with_EDL"] < 1e-12


def test_without_edl_is_concentration_invariant(scan_rows) -> None:
    e_values = {round(row["E_mix_without_EDL_V"], 14) for row in scan_rows}
    i_values = {
        round(row["i_mix_avg_without_EDL_A_per_m2"], 14) for row in scan_rows
    }
    assert len(e_values) == 1
    assert len(i_values) == 1


def test_complete_result_build_is_traceable_and_atomic(tmp_path) -> None:
    output = tmp_path / "ctot_result"
    result = build_ctot_results(default_params(), output)
    assert result["output"] == str(output)
    assert output.is_dir()
    assert len(list(output.glob("figures/**/*.png"))) == 5
    assert len(list(output.glob("figures/**/*.svg"))) == 5
    assert not list(output.glob("**/*.pdf"))

    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    artifacts = json.loads((output / "artifacts.json").read_text(encoding="utf-8"))
    assert manifest["figure_counts"] == {"pdf": 0, "png": 5, "svg": 5}
    assert manifest["execution"]["atomic_output_staging"] is True
    assert manifest["formal_high_salt_extension_starts_M"] == 10.0
    assert "run_manifest.json" in artifacts["metadata"]
    assert "artifacts.json" in artifacts["metadata"]
    assert set(artifacts["sha256"]) == set(artifacts["size_bytes"])

    with (output / "csv" / "ctot_scan.csv").open(encoding="utf-8") as handle:
        scan = list(csv.DictReader(handle))
    with (
        output / "csv" / "ctot_phi_bar_profiles_0p01M_1M_with_without_edl.csv"
    ).open(encoding="utf-8") as handle:
        profiles = list(csv.DictReader(handle))
    au_low = next(row for row in scan if float(row["C_tot_M"]) == 0.01)
    au_surface = next(
        row
        for row in profiles
        if row["material"] == "Au"
        and row["condition"] == "with EDL"
        and float(row["C_tot_M"]) == 0.01
        and float(row["distance_from_RP_nm"]) == 0.0
    )
    assert float(au_surface["phi_bar_tilde"]) == pytest.approx(
        float(au_low["Au_phi_RP_tilde"]), abs=1e-14
    )
