"""Command-line entry point for the C-supported Janus result pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .pipeline import DEFAULT_OUTPUT_ROOT, build_result_bundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Solve C(5 nm)|Au(4 nm)|Pd(4 nm)|C(5 nm), validate the mirror "
            "half-cell, and export Figure 3/Figure RP artifacts."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Parent directory for the new timestamped run (default: %(default)s)",
    )
    parser.add_argument(
        "--run-id",
        help="Explicit non-existing result directory name (default: local timestamp)",
    )
    parser.add_argument(
        "--n-modes",
        type=int,
        help="Override the default 960 cosine modes",
    )
    parser.add_argument(
        "--gl-order",
        type=int,
        help="Override the default 128-point Gauss-Legendre integration",
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Compute and validate without writing a result directory or figures",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    overrides: dict[str, int] = {}
    if args.n_modes is not None:
        overrides["N_modes"] = args.n_modes
    if args.gl_order is not None:
        overrides["gl_order"] = args.gl_order

    bundle = build_result_bundle(
        params=overrides,
        output_root=args.output_root,
        run_id=args.run_id,
        publish=not args.no_publish,
    )
    summary = bundle["summary"]
    for condition in ("with_edl", "without_edl"):
        result = summary["condition_results"][condition]
        print(
            f"{condition}: E_mix={float(result['E_mix_V']):.12f} V, "
            f"i_mix_avg={float(result['i_mix_avg_A_per_m2']):.12g} A/m^2, "
            f"I_halfcell={float(result['i_mix_abs_halfcell_A']):.12g} A, "
            f"I_full_period={float(result['i_mix_abs_full_period_A']):.12g} A"
        )
    if args.no_publish:
        print("validated only; no files published")
    else:
        print(f"output = {bundle['output']}")
        print("figures = 9 PNG + 9 SVG; PDF = 0")
        print(f"checksums = {len(bundle['checksums'])} files")
    return 0


__all__ = ["main"]

