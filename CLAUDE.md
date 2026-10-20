@AGENTS.md

# CLAUDE.md

Claude-specific notes only. Layout, toolchain, the full CI command list (including the corrupt-fixture loop), and
conventions are in `AGENTS.md` (imported above); do not duplicate them here.

## Commands

Run from the repo root. "Verified" means it ran successfully in a fresh checkout on Python 3.14 (Linux, CPU only).
The system `python3` may be older than 3.14, so create a 3.14 venv first (for example `uv venv --python 3.14 .venv`).

| Command | Status |
|---|---|
| `pip install -e ".[dev]"` | verified (inside a Python 3.14 venv) |
| `ruff check src tests tools` | verified |
| `mypy` | verified (18 source files) |
| `pytest -q` | verified (517 passed, about 70 s) |
| `spikenaut-etl validate --input tests/fixtures --reports /tmp/reports` | verified (5 sources passed) |
| Corrupt-fixture rejection loop from `AGENTS.md` | verified (all three fixtures rejected) |
| `spikenaut-etl clean`, `report`, `build-v3` | not run (need real telemetry or a dataset checkout; none is in the repo) |

Run ruff and mypy before committing; CI fails on either.

## Architecture map

- `src/spikenaut_etl/pipeline.py`: ingest, clean, validate, write. Output is written only after `validate.assert_publishable` returns.
- `src/spikenaut_etl/cli.py`: `spikenaut-etl` entry point (`validate`, `clean`, `report`, `build-v3`, `audit-v3`, `prepare-anticipation`).
- `ingest.py`, `v1.py`, `clean.py`, `validate.py`, `timestamps.py`, `report.py`: the pipeline stages.
- `schemas.py`, `contracts.py`: declared record schemas and named producer/consumer contracts.
- `v2_parquet.py`, `v3_build.py`, `audit_v3.py`, `teacher_policy.py`, `anticipation.py`, `system_telemetry.py`, `artifacts.py`: dataset builders, audits, and gaming-telemetry ingest.
- `tests/` holds the pytest suite; `tests/fixtures/` has valid inputs and `tests/fixtures/corrupt/` has inputs the gates must reject.

## Invariants and non-goals

- Fail loud, never fabricate. Do not fill missing values with `0.0` or `""`, and do not guess timestamps. Schemas set `extra="forbid"` so drift raises instead of dropping columns.
- Do not weaken a gate to make a test or fixture pass. A gate regression means corrupt data can ship again.
- A change to a validator or schema should come with a regression fixture (valid or `corrupt/`).
- No datasets or full telemetry captures in the repo. `*.jsonl` is gitignored except under `tests/fixtures/`.
- Publishing to the Hugging Face dataset is a git push against its remote, not an SDK call. Do not add an SDK dependency or publish from here.
- Base install depends only on `pydantic`; `pyarrow` and `datasets` stay in the `v3`/`dev` extras.
- `pyproject.toml` has no `exclude`/`include` list (the wheel target is `src/spikenaut_etl`), so adding `CLAUDE.md` needs no manifest change.
- Source files do not carry SPDX headers; do not add them.

## Gotchas

- `requires-python >=3.14`; installing under 3.13 fails.
- `.agents/` and `.claude/` are gitignored, so keep shared agent guidance in this file or `AGENTS.md`.
- Issue tracking is GitHub issues.
- Edit only what the task needs.
