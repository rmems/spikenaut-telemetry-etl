# spikenaut-telemetry-etl

Cleaning and validation pipeline for Spikenaut SNN telemetry, sitting between the
Rust collectors that produce it and the Hugging Face dataset that publishes it.

```
rmems/Theseus-Quarry            Rust collectors -> raw JSONL (schema v1)
        |
        v
rmems/spikenaut-telemetry-etl   ingest -> validate -> clean -> publish   <- this repo
        |
        v
rmems/Spikenaut-SNN-Telemetry   published dataset artifact
        |
        v
rmems/Spikenaut-SNN             model training
```

This repository holds **code only**. No data, no LFS.

---

## Why this exists

On 2026-08-03 an audit of `rmems/Spikenaut-SNN-Telemetry` found that **934,287 of
its 953,290 advertised records carried no information**:

| File | Rows | Usable | Condition |
|------|-----:|-------:|-----------|
| `neuromorphic_data.jsonl` | 813,973 | 0 | every line the identical empty record `{"telemetry":{}}` |
| `node_sync_harvest.jsonl` | 120,314 | 0 | every numeric field `0.0`, `blockchain` `""`, in 100% of rows |

The collection was fine. The **cleaning step** destroyed the data, and reported
success while doing it.

### The bug

The previous pipeline (`clean_datasets.jl`) read JSON into a dictionary with
**Symbol** keys, then filtered and looked up fields with **String** keys:

```julia
tel_dict = Dict(tel)                    # Dict{Symbol, Any}

clean_tel = Dict(k => v for (k, v) in tel_dict
                 if k in CANONICAL_NEURO_FIELDS)   # Set{String} -> never matches

hashrate = Float64(get(clean_tel, "hashrate_mh", 0.0))   # String key -> always 0.0
```

Neither construct errors. `in` simply returned false for every field, so every
record collapsed to `{}`. `get` simply returned its default, so every numeric
field became `0.0`. Nothing downstream checked whether the output still carried
information, so 813,973 identical empty records were written and published.

Two further defects came from the same script:

- Its "is this already ISO-8601?" check tested for a literal `T`. The real
  timestamps used a space (`2026-03-19 11:55:13.132`), so **114,250 real
  timestamps were overwritten** with a generated `base + 10s × index` sequence.
  That fabricated sequence then became the dataset's advertised collection window.
- Chain attribution was read from a `blockchain` field that does not exist in the
  source; the coin is encoded in the timestamp (`dynex:919876`). All 120,314 rows
  were written with `blockchain: ""`, and the dataset card then advertised a
  per-coin breakdown that the data could not support.

Its "trim synthetic tail rows" heuristic fired on the all-zero output of its own
bug and removed 20 good rows, taking 120,334 to exactly the 120,314 that shipped.

### The design response

Language choice was not the fix. **Absent assertions** were the fix. Every gate in
`validate.py` maps to a defect above, and the pipeline writes output only after
all of them pass:

| Gate | Catches |
|------|---------|
| `payload_columns_remain` | every payload field dropped as dead, leaving only an index |
| `distinct_rows` | 813,973 rows carrying one repeated payload |
| `no_constant_columns` | fields that collapsed to a single value during cleaning |
| `no_all_null_columns` | fields that vanished entirely |
| `row_count_delta` | silent row loss |
| `timestamps_not_fabricated` | perfectly uniform spacing — the signature of generated time |
| `schema_matches` | undeclared columns appearing, declared columns disappearing |

Plus, structurally:

- **Schemas are declared** (`pydantic`, `extra="forbid"`). A missing required
  field raises; an unexpected field raises. No `get(..., default)` anywhere in the
  read path.
- **Nothing is ever fabricated.** An unparseable timestamp is quarantined and
  reported, never replaced with a synthesized value.
- **Dead columns are measured, not hardcoded.** The prior script's `DEAD_COLS_*`
  constants would drift silently as the source changed; `find_dead_columns`
  recomputes per run and `report_dead_column_drift` flags divergence from what was
  expected.
- **Missing attribution is `None`, never `""`.** An empty string reads as a value.

---

## Usage

```bash
pip install -e ".[dev]"

# Gates only; writes nothing. Exit 1 if any source fails.
spikenaut-etl validate --input ~/spikenaut-data-backup/2026-08-03

# Clean and write into a dataset repo checkout.
spikenaut-etl clean \
    --input  ~/spikenaut-data-backup/2026-08-03 \
    --output ../Spikenaut-SNN-Telemetry

# Per-file data-quality profiles, written to reports/
spikenaut-etl report --input ~/spikenaut-data-backup/2026-08-03
```

Restrict to one source with `--only node_sync_harvest`.

---

## Sources

| Key | Input | Output | Notes |
|-----|-------|--------|-------|
| `neuromorphic_data` | `neuromorphic_data.jsonl` | `full_data/neuromorphic_data.jsonl` | GPU telemetry. **The source carries no timestamp**, so output is positional (`row_index`). |
| `node_sync_harvest` | `node_sync_harvest.jsonl` | `full_data/node_sync_harvest.jsonl` | Mining telemetry. Three timestamp forms; real ones preserved, coin tags decomposed. |
| `qubic_ticks_snn` | `qubic_ticks.jsonl` | `full_data/qubic_ticks_snn.jsonl` | Derived columns suffixed `_derived`. |
| `ghost_market_log` | `ghost_market_log.jsonl` | `full_data/ghost_market_log.jsonl` | Already healthy; validated passthrough. |

### Timestamp forms in `node_sync_harvest.jsonl`

| Count | Form | Example | Handling |
|------:|------|---------|----------|
| 114,238 | datetime, space separator | `2026-03-19 11:55:13.132` | parsed, preserved |
| 12 | datetime + UTC offset | `2026-03-19 17:00:05.551-05:00` | parsed, preserved |
| 5,001 | `coin:height` | `dynex:919876` | decomposed to `blockchain` + `block_height` |
| 1,083 | `coin:epoch:tick` | `qubic:204:46075040` | decomposed to `blockchain` + `chain_epoch` + `block_height` |

A coin tag is an identifier, not a clock. Those rows get `timestamp: null` rather
than an invented datetime.

### Derived columns on Qubic ticks

`qubic_ticks_snn.jsonl` previously presented `hashrate_mh`, `power_w` and
`gpu_temp_c` as GPU telemetry. They are a fixed function of `tick_rate` — 16
distinct values across 27,430 rows, with `power_w / hashrate_mh` pinned at 210.9.
They are preserved for continuity but every one now carries a `_derived` suffix.
The independent signals are `tick_rate` and `qubic_tick_trace`.

---

## Tests

```bash
pytest
```

`tests/test_regression_shipped_defects.py` reproduces each defect that reached
production and asserts this pipeline rejects it. The corrupt fixtures under
`tests/fixtures/corrupt/` are byte-faithful reproductions of what shipped.

Real fixtures are **seeded random samples**, regenerable with:

```bash
python tools/make_fixtures.py --source ~/spikenaut-data-backup/2026-08-03
```

They are deliberately not prefixes. The dataset's published `*_SAMPLE_*` files
were byte-exact prefixes of their parent files, which made them unrepresentative
(`ch2_mode` is constant for the first 200 rows of `ghost_market_log.jsonl` but
varies across the file) and double-counted rows when loaded alongside the full
file. `node_sync_harvest.jsonl` is stratified across all three timestamp forms,
since a uniform sample would be ~95% datetime and would never exercise coin parsing.

---

## Issue tracking

**GitHub issues, for this repository only.**

`Spikenaut-Vault/CLAUDE.md` directs all task tracking to `bd` (beads), and vault-
and model-scoped work still lives there. This repo is the exception: the vault's
git remote currently 404s, and beads syncs through `refs/dolt/data` on that remote,
so every issue in it sits in one local Dolt database with no off-machine copy.
Issues filed here survive a disk failure, link from commits and PRs, and are
visible to anyone who reaches the repo through the dataset's provenance chain.

Fold this back into beads once the vault remote is fixed, if one tracker is wanted.

## License

MIT OR Apache-2.0.
