"""Builder for the v3 action-proposal trajectory configs.

Reads the **published v2 JSONL** in a checkout of the dataset repo
(``rmems/Spikenaut-SNN-Telemetry``) and writes the additive ``v3/`` Parquet
tree next to it. v2 files are never touched; deleting ``v3/`` and re-running
this builder is a full, deterministic rebuild.

Outputs (one directory per Hugging Face config):

    v3/gpu_telemetry/       gpu_telemetry_v3 -- v2 GPU sensors minus the
                            interleaved qubic_* columns and the duplicate
                            ``clock_mhz``
    v3/qubic_signals/       the qubic_* columns split into their own config
    v3/state_telemetry/     the full v3 state schema, backfilled where v2
                            carries the signal, null elsewhere
    v3/action_proposals/    teacher labels (see ``teacher_policy``)
    v3/safety_filter_log/   filter verdicts -- no historical filter ran, so
                            this ships as an empty, typed table
    v3/outcomes/            within-episode horizon deltas + RLDS step flags
    v3/encoding_params/     per-feature encoder sidecar (axon-encoder params)

plus ``v2_parquet/<config>/`` next to ``v3/`` — Parquet conversions that serve
the four v2 configs (see ``v2_parquet.py`` for why mixed JSONL/Parquet configs
cannot work on the Hub). The v2 JSONL files themselves are never touched.

Honesty rules inherited from the rest of this repo:

- **Never synthesize a timestamp.** The v2 GPU source carries none and no
  collection cadence is documented anywhere (the Theseus-Quarry
  telemetry-collector default of 5 s belongs to a different source schema), so
  ``ts_utc`` *and* ``ts_synthetic_offset_ms`` are published as typed nulls.
- **Missing means null**, never ``0.0``, ``""``, or a guessed enum value.
  ``regime`` stays null for the backfill: classifying idle/steady/overload
  from values would be inference, not provenance.
- **Assert non-degenerate before writing.** The builder refuses to write if
  the backfilled columns are all-null or the split layout is inconsistent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.json as pa_json
import pyarrow.parquet as pq

from .teacher_policy import TEACHER_POLICY_VERSION, TeacherState, propose

SCHEMA_VERSION = "3.0.0"

# Episode and split geometry. The GPU capture has no wall clock, so episodes
# are fixed rolling windows over the (verified contiguous) row_index order and
# the split is chronological by episode. One whole episode is left out between
# blocks as the purge/embargo gap; it exceeds OUTCOME_HORIZON_STEPS by design.
EPISODE_LEN = 4096
OUTCOME_HORIZON_STEPS = 64
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
EMBARGO_EPISODES = 1

GPU_SOURCE = "full_data/neuromorphic_data.jsonl"
EPISODE_PREFIX = "gpu"

QUBIC_COLUMNS = ("qubic_tick_trace", "qubic_tick_rate", "qubic_epoch_progress")

# v2 column -> v3 state_telemetry column. ``sm_clock_mhz`` is backfilled from
# v2 ``gpu_clock_mhz``: NVML reports SM and graphics clocks separately but the
# v2 collector sampled a single graphics clock; the card documents the mapping.
STATE_BACKFILL = {
    "mem_util_pct": "mem_util_pct",
    "power_w": "power_w",
    "gpu_temp_c": "gpu_temp_c",
    "vram_temp_c": "vram_temp_c",
    "gpu_clock_mhz": "sm_clock_mhz",
    "mem_clock_mhz": "mem_clock_mhz",
    "fan_speed_pct": "fan_speed_pct",
    "vddcr_gfx_v": "vddcr_gfx_v",
}

PARAMS_STRUCT = pa.struct(
    [
        pa.field("target", pa.string()),
        pa.field("magnitude", pa.float32()),
        pa.field("duration_ms", pa.int32()),
        pa.field("paired_action", pa.string()),
    ]
)

STATE_SCHEMA = pa.schema(
    [
        # Alignment keys
        pa.field("ts_utc", pa.int64()),
        pa.field("episode_id", pa.string()),
        pa.field("step_idx", pa.int32()),
        pa.field("schema_version", pa.string()),
        pa.field("window_ms", pa.int32()),
        # GPU / NVML block
        pa.field("gpu_util_pct", pa.float32()),
        pa.field("mem_util_pct", pa.float32()),
        pa.field("fb_used_mib", pa.float32()),
        pa.field("fb_total_mib", pa.float32()),
        pa.field("power_w", pa.float32()),
        pa.field("power_limit_w", pa.float32()),
        pa.field("gpu_temp_c", pa.float32()),
        pa.field("vram_temp_c", pa.float32()),
        pa.field("sm_clock_mhz", pa.float32()),
        pa.field("mem_clock_mhz", pa.float32()),
        pa.field("fan_speed_pct", pa.float32()),
        pa.field("vddcr_gfx_v", pa.float32()),
        pa.field("pcie_replay_counter", pa.int64()),
        pa.field("ecc_sbe_vol", pa.int64()),
        pa.field("ecc_dbe_vol", pa.int64()),
        pa.field("throttle_reasons", pa.int64()),
        pa.field("thr_sw_power_cap", pa.bool_()),
        pa.field("thr_hw_slowdown", pa.bool_()),
        pa.field("thr_sw_thermal", pa.bool_()),
        pa.field("thr_hw_thermal", pa.bool_()),
        pa.field("thr_hw_power_brake", pa.bool_()),
        # Inference-server block (vLLM metric names)
        pa.field("num_requests_running", pa.int32()),
        pa.field("num_requests_waiting", pa.int32()),
        pa.field("gpu_cache_usage_perc", pa.float32()),
        pa.field("ttft_p50_s", pa.float32()),
        pa.field("ttft_p99_s", pa.float32()),
        pa.field("itl_p50_s", pa.float32()),
        pa.field("tokens_per_s", pa.float32()),
        pa.field("prompt_tokens_total", pa.int64()),
        pa.field("generation_tokens_total", pa.int64()),
        pa.field("num_preemptions_total", pa.int64()),
        pa.field("e2e_latency_p99_s", pa.float32()),
        pa.field("request_timeouts", pa.int64()),
        # CPU / board block
        pa.field("cpu_util_pct", pa.float32()),
        pa.field("cpu_temp_c", pa.float32()),
        pa.field("ram_used_pct", pa.float32()),
        pa.field("board_power_w", pa.float32()),
        pa.field("chassis_fan_rpm", pa.int32()),
        # FPGA block
        pa.field("fpga_spike_rate_hz", pa.float32()),
        pa.field("fpga_membrane_q88", pa.int16()),
        pa.field("fpga_clock_gated", pa.bool_()),
        pa.field("fpga_temp_c", pa.float32()),
        pa.field("fpga_uart_frame_seq", pa.int64()),
        # Provenance
        pa.field("source_host", pa.string()),
        pa.field("gpu_uuid", pa.string()),
        pa.field("regime", pa.string()),
        pa.field("synthetic", pa.bool_()),
    ]
)

PROPOSALS_SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.string()),
        pa.field("step_idx", pa.int32()),
        pa.field("ts_utc", pa.int64()),
        pa.field("schema_version", pa.string()),
        pa.field("proposed_action", pa.string()),
        pa.field("proposed_action_id", pa.int16()),
        pa.field("proposed_params", PARAMS_STRUCT),
        pa.field("proposal_logits", pa.list_(pa.float32())),
        pa.field("teacher_action", pa.string()),
        pa.field("teacher_action_id", pa.int16()),
        # Traceability contract: every published label names the exact teacher
        # clause (TR-xxx) that produced it.
        pa.field("teacher_rule_id", pa.string()),
        pa.field("label_source", pa.string()),
        pa.field("label_confidence", pa.float32()),
    ]
)

SAFETY_SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.string()),
        pa.field("step_idx", pa.int32()),
        pa.field("ts_utc", pa.int64()),
        pa.field("schema_version", pa.string()),
        pa.field("filter_verdict", pa.string()),
        pa.field("filter_rule_id", pa.string()),
        pa.field("filter_reason", pa.string()),
        pa.field("executed_action", pa.string()),
        pa.field("executed_action_id", pa.int16()),
        pa.field("executed_params", PARAMS_STRUCT),
        pa.field("override", pa.bool_()),
        pa.field("override_reason", pa.string()),
        pa.field("mode", pa.string()),
    ]
)

OUTCOMES_SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.string()),
        pa.field("step_idx", pa.int32()),
        pa.field("ts_utc", pa.int64()),
        pa.field("schema_version", pa.string()),
        pa.field("reward", pa.float32()),
        pa.field("d_gpu_temp_c", pa.float32()),
        pa.field("d_ttft_p99_s", pa.float32()),
        pa.field("d_tokens_per_s", pa.float32()),
        pa.field("d_kv_cache_usage", pa.float32()),
        pa.field("thermal_violation", pa.bool_()),
        pa.field("oom", pa.bool_()),
        pa.field("timeout_burst", pa.bool_()),
        pa.field("sla_met", pa.bool_()),
        pa.field("discount", pa.float32()),
        pa.field("is_terminal", pa.bool_()),
        pa.field("is_first", pa.bool_()),
        pa.field("is_last", pa.bool_()),
    ]
)

ENCODING_SCHEMA = pa.schema(
    [
        pa.field("feature_name", pa.string()),
        pa.field("encoder_type", pa.string()),
        pa.field("min", pa.float64()),
        pa.field("max", pa.float64()),
        pa.field("base_rate_hz", pa.float32()),
        pa.field("max_rate_hz", pa.float32()),
        pa.field("dt_seconds", pa.float32()),
        pa.field("q8_8_scale", pa.int32()),
        pa.field("threshold", pa.float32()),
        pa.field("schema_version", pa.string()),
    ]
)

# Default axon-encoder parameters. These are *prescriptive encoder settings*
# (RateEncoder::try_new(base_rate_hz, max_rate_hz, (min, max), dt_seconds)),
# not measurements: dt_seconds is the replay integration step, chosen, and
# independent of the (undocumented) collection cadence. q8_8_scale is the Q8.8
# fixed-point scale of the FPGA backend (1 LSB = 1/256).
BASE_RATE_HZ = 5.0
MAX_RATE_HZ = 100.0
DT_SECONDS = 0.010
Q8_8_SCALE = 256

# encoder_type per state feature: "rate" for continuous magnitudes, "delta"
# (spike on change >= threshold, threshold = 1 count) for monotone counters.
RATE_FEATURES = (
    "gpu_util_pct",
    "mem_util_pct",
    "fb_used_mib",
    "fb_total_mib",
    "power_w",
    "power_limit_w",
    "gpu_temp_c",
    "vram_temp_c",
    "sm_clock_mhz",
    "mem_clock_mhz",
    "fan_speed_pct",
    "vddcr_gfx_v",
    "num_requests_running",
    "num_requests_waiting",
    "gpu_cache_usage_perc",
    "ttft_p50_s",
    "ttft_p99_s",
    "itl_p50_s",
    "tokens_per_s",
    "e2e_latency_p99_s",
    "cpu_util_pct",
    "cpu_temp_c",
    "ram_used_pct",
    "board_power_w",
    "chassis_fan_rpm",
    "fpga_spike_rate_hz",
    "fpga_membrane_q88",
    "fpga_temp_c",
)
DELTA_FEATURES = (
    "pcie_replay_counter",
    "ecc_sbe_vol",
    "ecc_dbe_vol",
    "prompt_tokens_total",
    "generation_tokens_total",
    "num_preemptions_total",
    "request_timeouts",
    "fpga_uart_frame_seq",
)


class BuildError(RuntimeError):
    """A precondition, fidelity, or non-degeneracy check failed.

    All v3 tables are computed and validated before the first write, but the
    build has several write stages (v2_parquet conversions, then the v3
    configs), so a failure can leave a partial tree. Completion is marked
    by ``v3/build_report.json``, written last: a tree without it is
    incomplete, and re-running the builder first removes every artifact it
    owns, so no mixed-generation tree can survive a retry.
    """


@dataclass(frozen=True)
class BuildReport:
    """What a build produced; also serialized to ``v3/build_report.json``."""

    schema_version: str
    teacher_policy_version: str
    episode_len: int
    outcome_horizon_steps: int
    n_episodes: int
    embargo_episodes: tuple[str, ...]
    configs: dict[str, dict[str, int]]

    def to_json(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "teacher_policy_version": self.teacher_policy_version,
            "episode_len": self.episode_len,
            "outcome_horizon_steps": self.outcome_horizon_steps,
            "n_episodes": self.n_episodes,
            "embargo_episodes": list(self.embargo_episodes),
            "configs": self.configs,
        }
        return json.dumps(payload, indent=2) + "\n"


@dataclass(frozen=True)
class SplitLayout:
    """Episode index ranges per split, embargo episodes excluded everywhere."""

    train: range
    validation: range
    test: range
    embargo: tuple[int, ...]
    n_episodes: int

    def split_of(self, episode: int) -> str | None:
        if episode in self.embargo:
            return None
        if episode in self.train:
            return "train"
        if episode in self.validation:
            return "validation"
        if episode in self.test:
            return "test"
        raise BuildError(f"episode {episode} outside every split range")


def episode_of(row_index: int, episode_len: int = EPISODE_LEN) -> int:
    return row_index // episode_len


def episode_id_str(episode: int) -> str:
    return f"{EPISODE_PREFIX}-{episode:06d}"


def split_episodes(
    n_episodes: int,
    train_fraction: float = TRAIN_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
    embargo_episodes: int = EMBARGO_EPISODES,
) -> SplitLayout:
    """Chronological blocked split with an embargo gap between blocks.

    The gap is whole episodes (>= OUTCOME_HORIZON_STEPS rows each), so no
    outcome window computed inside one block can reach into the next, and no
    two split blocks are temporally adjacent.
    """
    if n_episodes < 2 * embargo_episodes + 3:
        raise BuildError(
            f"{n_episodes} episodes cannot host three blocks and two embargo gaps"
        )
    n_train = int(n_episodes * train_fraction)
    n_validation = int(n_episodes * validation_fraction)
    val_start = n_train + embargo_episodes
    test_start = val_start + n_validation + embargo_episodes
    if test_start >= n_episodes:
        raise BuildError("split fractions leave no test episodes")
    embargo = tuple(
        list(range(n_train, val_start))
        + list(range(val_start + n_validation, test_start))
    )
    return SplitLayout(
        train=range(0, n_train),
        validation=range(val_start, val_start + n_validation),
        test=range(test_start, n_episodes),
        embargo=embargo,
        n_episodes=n_episodes,
    )


def _nulls(n: int, typ: pa.DataType) -> pa.Array:
    return pa.nulls(n, type=typ)


def _const_str(n: int, value: str) -> pa.Array:
    return pa.array([value] * n, type=pa.string())


def _read_gpu_v2(dataset_root: Path) -> pa.Table:
    source = dataset_root / GPU_SOURCE
    if not source.exists():
        raise BuildError(f"missing v2 source {source}")
    try:
        table = pa_json.read_json(source)
    except (pa.ArrowInvalid, OSError) as exc:
        raise BuildError(f"cannot read {source}: {exc}") from exc
    n = table.num_rows
    if n == 0:
        raise BuildError(f"{source} is empty")
    # fill_null(False): pc.all skips nulls by default, and a null in either
    # guard column must fail the gate, not slip past it.
    row_index = table.column("row_index")
    expected = pa.chunked_array([pa.array(range(n), type=pa.int64())])
    if not pc.all(pc.fill_null(pc.equal(row_index, expected), False)).as_py():
        raise BuildError("row_index is not contiguous 0..N-1; refusing to episode it")
    if not pc.all(
        pc.fill_null(
            pc.equal(table.column("clock_mhz"), table.column("gpu_clock_mhz")), False
        )
    ).as_py():
        raise BuildError(
            "clock_mhz != gpu_clock_mhz; the duplicate-column assumption broke"
        )
    return table


def build_gpu_telemetry_v3(gpu: pa.Table) -> pa.Table:
    """v2 GPU sensors, minus qubic_* and the duplicate clock_mhz column."""
    n = gpu.num_rows
    keep = [
        name
        for name in gpu.column_names
        if name not in (*QUBIC_COLUMNS, "clock_mhz", "row_index")
    ]
    cols: dict[str, pa.Array | pa.ChunkedArray] = {
        "row_index": gpu.column("row_index"),
        "ts_utc": _nulls(n, pa.int64()),
        "ts_synthetic_offset_ms": _nulls(n, pa.int64()),
        "schema_version": _const_str(n, SCHEMA_VERSION),
    }
    for name in keep:
        cols[name] = gpu.column(name)
    return pa.table(cols)


def build_qubic_signals(gpu: pa.Table) -> pa.Table:
    n = gpu.num_rows
    cols: dict[str, pa.Array | pa.ChunkedArray] = {
        "row_index": gpu.column("row_index"),
        "ts_utc": _nulls(n, pa.int64()),
        "ts_synthetic_offset_ms": _nulls(n, pa.int64()),
        "schema_version": _const_str(n, SCHEMA_VERSION),
    }
    for name in QUBIC_COLUMNS:
        cols[name] = gpu.column(name)
    return pa.table(cols)


def build_state_telemetry(
    gpu: pa.Table, episode_len: int = EPISODE_LEN
) -> pa.Table:
    """One state row per v2 GPU row, in the full v3 schema.

    Signals v2 never carried stay typed nulls. ``synthetic`` is False: these
    are real measurements, only re-shaped.
    """
    n = gpu.num_rows
    episodes = [episode_of(i, episode_len) for i in range(n)]
    cols: dict[str, pa.Array | pa.ChunkedArray] = {}
    backfilled = set(STATE_BACKFILL.values())
    for fld in STATE_SCHEMA:
        if fld.name == "episode_id":
            cols[fld.name] = pa.array(
                [episode_id_str(e) for e in episodes], type=pa.string()
            )
        elif fld.name == "step_idx":
            cols[fld.name] = pa.array(
                [i - e * episode_len for i, e in enumerate(episodes)],
                type=pa.int32(),
            )
        elif fld.name == "schema_version":
            cols[fld.name] = _const_str(n, SCHEMA_VERSION)
        elif fld.name == "synthetic":
            cols[fld.name] = pa.array([False] * n, type=pa.bool_())
        elif fld.name in backfilled:
            v2_name = next(k for k, v in STATE_BACKFILL.items() if v == fld.name)
            cols[fld.name] = pc.cast(gpu.column(v2_name), fld.type)
        else:
            cols[fld.name] = _nulls(n, fld.type)
    return pa.table(cols, schema=STATE_SCHEMA)


def build_outcomes(
    state: pa.Table,
    episode_len: int = EPISODE_LEN,
    horizon: int = OUTCOME_HORIZON_STEPS,
) -> pa.Table:
    """RLDS step structure plus the one horizon delta v2 can support.

    ``d_gpu_temp_c[t] = gpu_temp_c[t + horizon] - gpu_temp_c[t]`` strictly
    within an episode; the last ``horizon`` steps of each episode are null.
    Episodes end by windowing, not by environment termination, so
    ``is_last`` marks truncation and ``is_terminal`` stays False (RLDS
    convention). Reward and the vLLM-derived deltas have no v2 source and
    stay null.
    """
    n = state.num_rows
    episode_ids = state.column("episode_id").to_pylist()
    step_idx = state.column("step_idx").to_pylist()
    temps = state.column("gpu_temp_c").to_pylist()

    d_temp: list[float | None] = [None] * n
    is_first = [False] * n
    is_last = [False] * n
    for i in range(n):
        if step_idx[i] == 0:
            is_first[i] = True
        last_of_episode = i + 1 == n or episode_ids[i + 1] != episode_ids[i]
        if last_of_episode:
            is_last[i] = True
        j = i + horizon
        if (
            j < n
            and episode_ids[j] == episode_ids[i]
            and temps[i] is not None
            and temps[j] is not None
        ):
            d_temp[i] = temps[j] - temps[i]

    cols: dict[str, pa.Array | pa.ChunkedArray] = {
        "episode_id": state.column("episode_id"),
        "step_idx": state.column("step_idx"),
        "ts_utc": state.column("ts_utc"),
        "schema_version": _const_str(n, SCHEMA_VERSION),
        "reward": _nulls(n, pa.float32()),
        "d_gpu_temp_c": pa.array(d_temp, type=pa.float32()),
        "d_ttft_p99_s": _nulls(n, pa.float32()),
        "d_tokens_per_s": _nulls(n, pa.float32()),
        "d_kv_cache_usage": _nulls(n, pa.float32()),
        "thermal_violation": _nulls(n, pa.bool_()),
        "oom": _nulls(n, pa.bool_()),
        "timeout_burst": _nulls(n, pa.bool_()),
        "sla_met": _nulls(n, pa.bool_()),
        "discount": pa.array([1.0] * n, type=pa.float32()),
        "is_terminal": pa.array([False] * n, type=pa.bool_()),
        "is_first": pa.array(is_first, type=pa.bool_()),
        "is_last": pa.array(is_last, type=pa.bool_()),
    }
    return pa.table(cols, schema=OUTCOMES_SCHEMA)


def build_action_proposals(state: pa.Table) -> pa.Table:
    """Run the deterministic teacher over every state row.

    The teacher refuses to label rows whose core signals are missing, so a
    pure-v2 backfill legitimately yields an empty table -- the schema still
    ships so collectors can start filling it without a schema change.
    """
    episode_ids = state.column("episode_id").to_pylist()
    step_idx = state.column("step_idx").to_pylist()
    throttle = state.column("throttle_reasons").to_pylist()
    ecc_dbe = state.column("ecc_dbe_vol").to_pylist()
    kv = state.column("gpu_cache_usage_perc").to_pylist()
    waiting = state.column("num_requests_waiting").to_pylist()
    tps = state.column("tokens_per_s").to_pylist()
    preempt = state.column("num_preemptions_total").to_pylist()

    rows: list[dict[str, object]] = []
    history: list[TeacherState] = []
    prev_episode: str | None = None
    for i in range(state.num_rows):
        if episode_ids[i] != prev_episode:
            history = []
            prev_episode = episode_ids[i]
        history.append(
            TeacherState(
                throttle_reasons=throttle[i],
                ecc_dbe_vol=ecc_dbe[i],
                gpu_cache_usage_perc=kv[i],
                num_requests_waiting=waiting[i],
                tokens_per_s=tps[i],
                num_preemptions_total=preempt[i],
            )
        )
        decision = propose(history)
        if decision is None:
            continue
        paired = (
            decision.paired_action.action_name if decision.paired_action else None
        )
        rows.append(
            {
                "episode_id": episode_ids[i],
                "step_idx": step_idx[i],
                "ts_utc": None,
                "schema_version": SCHEMA_VERSION,
                "proposed_action": None,
                "proposed_action_id": None,
                "proposed_params": {
                    "target": None,
                    "magnitude": None,
                    "duration_ms": None,
                    "paired_action": paired,
                },
                "proposal_logits": None,
                "teacher_action": decision.action.action_name,
                "teacher_action_id": int(decision.action),
                "teacher_rule_id": decision.rule_id,
                "label_source": "teacher_rule",
                "label_confidence": decision.confidence,
            }
        )
    if not rows:
        return PROPOSALS_SCHEMA.empty_table()
    return pa.Table.from_pylist(rows, schema=PROPOSALS_SCHEMA)


def build_encoding_params(train_state: pa.Table) -> pa.Table:
    """Per-feature encoder sidecar; min/max fitted on the train split only."""
    rows: list[dict[str, object]] = []
    for name, encoder in [(f, "rate") for f in RATE_FEATURES] + [
        (f, "delta") for f in DELTA_FEATURES
    ]:
        column = train_state.column(name)
        lo = pc.min(column).as_py()
        hi = pc.max(column).as_py()
        rows.append(
            {
                "feature_name": name,
                "encoder_type": encoder,
                "min": float(lo) if lo is not None else None,
                "max": float(hi) if hi is not None else None,
                "base_rate_hz": BASE_RATE_HZ if encoder == "rate" else None,
                "max_rate_hz": MAX_RATE_HZ if encoder == "rate" else None,
                "dt_seconds": DT_SECONDS,
                "q8_8_scale": Q8_8_SCALE,
                "threshold": 1.0 if encoder == "delta" else None,
                "schema_version": SCHEMA_VERSION,
            }
        )
    return pa.Table.from_pylist(rows, schema=ENCODING_SCHEMA)


def _split_table(table: pa.Table, layout: SplitLayout) -> dict[str, pa.Table]:
    """Slice an (episode-ordered, contiguous) table into split blocks."""
    out: dict[str, pa.Table] = {}
    episode_ids = table.column("episode_id")
    for split_name, episodes in (
        ("train", layout.train),
        ("validation", layout.validation),
        ("test", layout.test),
    ):
        wanted = pa.array([episode_id_str(e) for e in episodes], type=pa.string())
        mask = pc.is_in(episode_ids, value_set=wanted)
        out[split_name] = table.filter(mask)
    return out


def _assert_non_degenerate(splits: dict[str, pa.Table]) -> None:
    for split_name, table in splits.items():
        if table.num_rows == 0:
            raise BuildError(f"state_telemetry {split_name} split is empty")
        for name in STATE_BACKFILL.values():
            column = table.column(name)
            if column.null_count == table.num_rows:
                raise BuildError(
                    f"backfilled column {name} is all-null in {split_name}; "
                    "refusing to publish a degenerate build"
                )
    total = sum(t.num_rows for t in splits.values())
    if total == 0:
        raise BuildError("no rows survived the split")


# Every artifact the builder owns under v3/. Anything else in v3/ is foreign
# and is left alone.
OWNED_CONFIG_DIRS = (
    "gpu_telemetry",
    "qubic_signals",
    "state_telemetry",
    "action_proposals",
    "safety_filter_log",
    "outcomes",
    "encoding_params",
)


def _clean_owned_outputs(out_root: Path) -> None:
    """Remove builder-owned artifacts from a previous run.

    A rebuild that only overwrites matching filenames would leave stale shards
    (say, an old ``train-00001.parquet``) mixed into the new tree. Deleting
    exactly what this builder owns keeps rebuilds deterministic without
    touching anything a human may have parked under ``v3/``.
    """
    for name in OWNED_CONFIG_DIRS:
        directory = out_root / name
        if not directory.is_dir():
            continue
        for stale in directory.glob("*.parquet"):
            stale.unlink()
    report = out_root / "build_report.json"
    if report.exists():
        report.unlink()


def _write(table: pa.Table, directory: Path, split: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{split}-00000.parquet"
    pq.write_table(table, path)
    return path


def build_v3(
    dataset_root: Path,
    output_root: Path | None = None,
    episode_len: int = EPISODE_LEN,
    horizon: int = OUTCOME_HORIZON_STEPS,
) -> BuildReport:
    """Build every v3 config and write the tree plus a build report."""
    # Deferred import: v2_parquet imports this module for BuildError, and it
    # pulls in the (heavy) datasets dependency only when actually building.
    # Import and convert before cleaning v3/ so a missing extra or a bad aux
    # JSONL cannot leave a half-replaced tree; build_v2_parquet itself loads
    # every source fully before replacing any shard.
    from .v2_parquet import build_v2_parquet

    out_root = (output_root or dataset_root) / "v3"
    gpu = _read_gpu_v2(dataset_root)

    gpu_v3 = build_gpu_telemetry_v3(gpu)
    qubic = build_qubic_signals(gpu)
    state = build_state_telemetry(gpu, episode_len=episode_len)
    n_episodes = episode_of(gpu.num_rows - 1, episode_len) + 1
    layout = split_episodes(n_episodes)

    state_splits = _split_table(state, layout)
    _assert_non_degenerate(state_splits)

    outcomes = build_outcomes(state, episode_len=episode_len, horizon=horizon)
    outcome_splits = _split_table(outcomes, layout)
    proposal_splits = {
        name: build_action_proposals(split_state)
        for name, split_state in state_splits.items()
    }
    safety_splits = {name: SAFETY_SCHEMA.empty_table() for name in state_splits}
    encoding = build_encoding_params(state_splits["train"])

    v2_counts = build_v2_parquet(dataset_root, output_root or dataset_root)

    _clean_owned_outputs(out_root)

    # The two flat corpus derivations are complete captures, not training
    # sets: they cover validation, test, and embargo episodes too. Publishing
    # them under a "train" split would mislabel those rows, so they ship as a
    # single "full" split; episode membership is recoverable from row_index.
    configs: dict[str, dict[str, int]] = {}
    _write(gpu_v3, out_root / "gpu_telemetry", "full")
    configs["gpu_telemetry_v3"] = {"full": gpu_v3.num_rows}
    _write(qubic, out_root / "qubic_signals", "full")
    configs["qubic_signals"] = {"full": qubic.num_rows}
    for name, tables in (
        ("state_telemetry", state_splits),
        ("action_proposals", proposal_splits),
        ("safety_filter_log", safety_splits),
        ("outcomes", outcome_splits),
    ):
        configs[name] = {}
        for split_name, table in tables.items():
            _write(table, out_root / name, split_name)
            configs[name][split_name] = table.num_rows
    _write(encoding, out_root / "encoding_params", "train")
    configs["encoding_params"] = {"train": encoding.num_rows}
    for config, rows in v2_counts.items():
        configs[config] = {"train": rows}

    report = BuildReport(
        schema_version=SCHEMA_VERSION,
        teacher_policy_version=TEACHER_POLICY_VERSION,
        episode_len=episode_len,
        outcome_horizon_steps=horizon,
        n_episodes=n_episodes,
        embargo_episodes=tuple(episode_id_str(e) for e in layout.embargo),
        configs=configs,
    )
    (out_root / "build_report.json").write_text(report.to_json())
    return report
