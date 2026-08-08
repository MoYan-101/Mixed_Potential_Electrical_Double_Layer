from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import au_pd_edl.cli as cli_module
from au_pd_edl.parameters import default_params


def test_single_run_manifest_has_schema_v2_and_spectral_backend(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    params = default_params()
    output = tmp_path / "single"

    monkeypatch.setattr(cli_module, "_load_cli_params", lambda args: params)
    monkeypatch.setattr(
        cli_module,
        "run_separation_scan",
        lambda *args, **kwargs: {"synthetic": True},
    )

    def fake_save_run(output_dir: str | Path, *args: Any, **kwargs: Any) -> dict[str, Any]:
        destination = Path(output_dir).resolve()
        destination.mkdir(parents=True)
        return {
            "result_schema_version": 2,
            "electrostatic_backend": "semi_infinite_cosine_fourier",
            "output_dir": str(destination),
            "summary": {
                "result_schema_version": 2,
                "electrostatic_backend": "semi_infinite_cosine_fourier",
            },
            "gouy_chapman": {
                "csv": str(destination / "csv" / "gc.csv"),
                "analysis_json": str(destination / "gc.json"),
                "n_rows": 2,
            },
        }

    monkeypatch.setattr(cli_module, "save_run", fake_save_run)
    args = SimpleNamespace(
        d_nm=3.0,
        output=output,
        display_nx=31,
        display_ny=21,
    )

    returned = cli_module._run_single(args)
    written = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))

    for manifest in (returned, written):
        assert manifest["result_schema_version"] == 2
        assert manifest["electrostatic_backend"] == (
            "semi_infinite_cosine_fourier"
        )


def test_manifest_contract_helper_is_shared_by_cli_commands() -> None:
    assert cli_module._result_contract_metadata() == {
        "result_schema_version": 2,
        "electrostatic_backend": "semi_infinite_cosine_fourier",
    }
