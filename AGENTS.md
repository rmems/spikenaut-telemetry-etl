# AGENTS.md

Guidance for coding agents (Amp, Codex, Cursor, Claude Code, and others) working in this repository.

## Purpose

`spikenaut-telemetry-etl` is the cleaning and validation pipeline for Spikenaut SNN telemetry
(see `README.md`). It sits between the Rust collectors (`rmems/Theseus-Quarry`, raw JSONL schema v1)
and the published Hugging Face dataset (`rmems/Spikenaut-SNN-Telemetry`): ingest → clean →
validate → publish (`src/spikenaut_etl/pipeline.py`). Output is written only if every gate passes on
the cleaned data. The repository holds code and small test fixtures only: no datasets, no LFS.

## Layout

| Path | Contents |
|------|----------|
| `src/spikenaut_etl/` | Package: `cli.py` (`spikenaut-etl`), `ingest`, `validate`, `clean`, `report`, schemas (`v1`, `v2_parquet`, `v3_build`), `system_telemetry`, `anticipation`, `audit_v3`, ... |
| `tests/` (+ `tests/fixtures/`, `tests/fixtures/corrupt/`) | pytest suite and valid/corrupt fixtures |
| `tools/make_fixtures.py` | Fixture generator |
| `docs/` | Pilot/research notes |

## Toolchain

- Python **3.14** (`requires-python >=3.14`; `validate.yml`). Install with `pip install -e ".[dev]"`.
- ruff (`line-length = 90`, `target-version = "py314"`, rules `E, F, I, UP, B`) and mypy
  (`files = ["src/spikenaut_etl"]`) are configured in `pyproject.toml`.
- No GPU needed.

## Commands (from `.github/workflows/validate.yml`)

```bash
pip install -e ".[dev]"
ruff check src tests tools
mypy
pytest -q

# The pipeline must accept the real fixtures...
spikenaut-etl validate --input tests/fixtures --reports /tmp/reports

# ...and must reject each corrupt fixture. Each one is copied into its own directory under the
# file name its source expects, then validated with --only. The loop exits non-zero if any corrupt
# fixture is accepted, as in CI.
failures=0
for fixture in all_empty_telemetry all_zero_telemetry fabricated_timestamps; do
  case "$fixture" in
    all_empty_telemetry) key=neuromorphic_data; name=neuromorphic_data.jsonl ;;
    *)                   key=node_sync_harvest; name=node_sync_harvest.jsonl ;;
  esac
  dir="/tmp/corrupt-$fixture"; mkdir -p "$dir"
  cp "tests/fixtures/corrupt/$fixture.jsonl" "$dir/$name"
  if spikenaut-etl validate --input "$dir" --reports "/tmp/reports-$fixture" --only "$key"; then
    echo "REGRESSION: $fixture was accepted"
    failures=$((failures + 1))
  fi
done
exit $failures
```

CLI usage (`validate`, `clean`, `report`, `--only`, `--profile`) is documented in the README.

## Conventions visible in the repo

- The gates exist to stop uninformative or fabricated records from shipping. Don't invent zeros
  or fill missing values (see "Why this exists" and the `--profile` table in the README).
- Issue tracking for this repo is **GitHub issues** (README "Issue tracking"), unlike the Vault's
  beads setup.
- Don't commit datasets or full telemetry captures. The small fixtures under `tests/fixtures/`
  (valid and `corrupt/`) are committed on purpose; add one when a validator or schema change needs a
  regression input. `tools/make_fixtures.py` regenerates the seeded samples from the recovered
  originals.
