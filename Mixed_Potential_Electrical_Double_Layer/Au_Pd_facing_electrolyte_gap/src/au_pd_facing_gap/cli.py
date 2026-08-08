"""Command-line entry point for the facing Au/electrolyte/Pd slit model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .figures import build_results, default_output
from .model import load_params


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--params",
        type=Path,
        default=PACKAGE_ROOT / "params_template.json",
    )
    result.add_argument(
        "--backend",
        choices=("linear", "nonlinear"),
        default="linear",
        help="Electrostatic solver; the existing analytic linear model remains the default.",
    )
    result.add_argument("--output", type=Path, default=None)
    return result


def main() -> int:
    args = parser().parse_args()
    params = load_params(args.params.resolve())
    output = (
        args.output.resolve()
        if args.output is not None
        else default_output(PACKAGE_ROOT).resolve()
    )
    if args.backend == "nonlinear":
        from .nonlinear_model import (
            NonlinearFacingElectrolyteSlitModel,
            solve_cases_nonlinear,
        )

        result = build_results(
            params,
            output,
            model_class=NonlinearFacingElectrolyteSlitModel,
            solve_function=solve_cases_nonlinear,
            solution_label="1D nonlinear-PB solution",
        )
    else:
        result = build_results(params, output)
    print(json.dumps(result["manifest"]["figure_counts"], sort_keys=True))
    print(result["output"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
