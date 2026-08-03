"""Command line entry point.

    spikenaut-etl validate --input <dir>     gates only, writes nothing
    spikenaut-etl clean    --input <dir> --output <dir>
    spikenaut-etl report   --input <dir>

Exit status is 1 if any source fails a gate, so CI fails on corrupt data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import SOURCES, run_all

DEFAULT_REPORTS = Path("reports")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spikenaut-etl",
        description="Clean and validate Spikenaut telemetry before publication.",
    )
    parser.add_argument(
        "command", choices=("validate", "clean", "report"), help="action to perform"
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="directory holding the source JSONL"
    )
    parser.add_argument(
        "--output", type=Path, help="dataset repo root (required for 'clean')"
    )
    parser.add_argument(
        "--reports", type=Path, default=DEFAULT_REPORTS, help="report output directory"
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=[s.key for s in SOURCES],
        help="restrict to specific sources",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "clean" and args.output is None:
        print("error: --output is required for 'clean'", file=sys.stderr)
        return 2

    write_output = args.command == "clean"
    outcomes = run_all(
        input_root=args.input,
        output_root=args.output or Path("."),
        report_dir=args.reports,
        only=args.only,
        write_output=write_output,
    )

    for outcome in outcomes:
        print(outcome.rendered)
        if outcome.output_path:
            print(f"      wrote     {outcome.output_path}")
        for sample in outcome.sample_paths:
            print(f"      sample    {sample}")

    failed = [o for o in outcomes if not o.ok]
    print()
    if failed:
        print(f"{len(failed)}/{len(outcomes)} source(s) failed validation: "
              f"{', '.join(o.key for o in failed)}")
        return 1
    print(f"All {len(outcomes)} source(s) passed.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
