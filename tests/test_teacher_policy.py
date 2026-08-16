"""Unit tests for the deterministic teacher policy.

Each test pins one clause of the labeling contract. If a change here is ever
needed to make a test pass, that change is a behaviour change and
``TEACHER_POLICY_VERSION`` must be bumped in the same commit.
"""

from __future__ import annotations

import pytest

from spikenaut_etl.teacher_policy import (
    KV_PRESSURE,
    SUSTAIN_STEPS,
    THR_GPU_IDLE,
    THR_HW_POWER_BRAKE_SLOWDOWN,
    THR_HW_SLOWDOWN,
    THR_HW_THERMAL_SLOWDOWN,
    THR_SW_POWER_CAP,
    THR_SW_THERMAL_SLOWDOWN,
    Action,
    TeacherState,
    propose,
)

# A GPU with core signals present and nothing wrong.
NOMINAL = TeacherState(throttle_reasons=0, ecc_dbe_vol=0)


def state(**kwargs: object) -> TeacherState:
    """Nominal core signals unless overridden."""
    defaults: dict[str, object] = {"throttle_reasons": 0, "ecc_dbe_vol": 0}
    defaults.update(kwargs)
    return TeacherState(**defaults)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# No guessed labels
# --------------------------------------------------------------------------- #


def test_missing_throttle_mask_emits_no_label():
    assert propose([TeacherState(ecc_dbe_vol=0)]) is None


def test_missing_ecc_counter_emits_no_label():
    assert propose([TeacherState(throttle_reasons=0)]) is None


def test_v2_backfill_state_emits_no_label():
    """The historical GPU capture carries neither core signal: zero labels."""
    assert propose([TeacherState()]) is None


def test_nominal_state_is_labeled_no_op():
    decision = propose([NOMINAL])
    assert decision is not None
    assert decision.action is Action.NO_OP
    assert decision.rule_id == "TR-000-NOMINAL"
    assert decision.confidence == 1.0


# --------------------------------------------------------------------------- #
# ECC double-bit errors dominate everything
# --------------------------------------------------------------------------- #


def test_ecc_dbe_escalates_and_migrates():
    decision = propose([state(ecc_dbe_vol=1)])
    assert decision is not None
    assert decision.action is Action.ESCALATE_TO_HUMAN
    assert decision.paired_action is Action.MIGRATE_WORKLOAD
    assert decision.rule_id == "TR-001-ECC-DBE"


def test_ecc_dbe_beats_thermal_throttling():
    decision = propose(
        [state(ecc_dbe_vol=2, throttle_reasons=THR_HW_THERMAL_SLOWDOWN)]
    )
    assert decision is not None
    assert decision.action is Action.ESCALATE_TO_HUMAN


# --------------------------------------------------------------------------- #
# Throttle-mask rules: keyed off NVML bits, never off a temperature
# --------------------------------------------------------------------------- #


def test_hw_power_brake_caps_power():
    decision = propose([state(throttle_reasons=THR_HW_POWER_BRAKE_SLOWDOWN)])
    assert decision is not None
    assert decision.action is Action.CAP_POWER
    assert decision.rule_id == "TR-002-HW-POWER-BRAKE"


def test_hw_thermal_throttles_clocks():
    decision = propose([state(throttle_reasons=THR_HW_THERMAL_SLOWDOWN)])
    assert decision is not None
    assert decision.action is Action.THROTTLE_CLOCKS
    assert decision.rule_id == "TR-003-HW-THERMAL"


def test_hw_slowdown_throttles_clocks():
    decision = propose([state(throttle_reasons=THR_HW_SLOWDOWN)])
    assert decision is not None
    assert decision.action is Action.THROTTLE_CLOCKS
    assert decision.rule_id == "TR-004-HW-SLOWDOWN"


def test_sw_thermal_raises_fan():
    decision = propose([state(throttle_reasons=THR_SW_THERMAL_SLOWDOWN)])
    assert decision is not None
    assert decision.action is Action.RAISE_FAN
    assert decision.rule_id == "TR-005-SW-THERMAL"


def test_power_brake_outranks_thermal_bits():
    mask = THR_HW_POWER_BRAKE_SLOWDOWN | THR_HW_THERMAL_SLOWDOWN
    decision = propose([state(throttle_reasons=mask)])
    assert decision is not None
    assert decision.action is Action.CAP_POWER


def test_benign_bits_are_nominal():
    """Idle and sw-power-cap bits describe normal operation, not incidents."""
    for mask in (THR_GPU_IDLE, THR_SW_POWER_CAP, THR_GPU_IDLE | THR_SW_POWER_CAP):
        decision = propose([state(throttle_reasons=mask)])
        assert decision is not None
        assert decision.action is Action.NO_OP


# --------------------------------------------------------------------------- #
# KV-cache pressure: sustained, then escalating
# --------------------------------------------------------------------------- #


def _kv_history(n: int, usage: float) -> list[TeacherState]:
    return [state(gpu_cache_usage_perc=usage) for _ in range(n)]


def test_kv_pressure_needs_sustain():
    history = _kv_history(SUSTAIN_STEPS - 1, KV_PRESSURE)
    decision = propose(history)
    assert decision is not None
    assert decision.action is Action.NO_OP


def test_sustained_kv_pressure_evicts_cache():
    decision = propose(_kv_history(SUSTAIN_STEPS, KV_PRESSURE))
    assert decision is not None
    assert decision.action is Action.EVICT_KV_CACHE
    assert decision.rule_id == "TR-006-KV-PRESSURE-EVICT"


def test_persistent_kv_pressure_reduces_batch():
    decision = propose(_kv_history(2 * SUSTAIN_STEPS, KV_PRESSURE))
    assert decision is not None
    assert decision.action is Action.REDUCE_BATCH_SIZE
    assert decision.rule_id == "TR-007-KV-PRESSURE-BATCH"


def test_kv_gap_breaks_the_sustain_run():
    """A window with a missing gauge is not evidence of pressure."""
    history = _kv_history(SUSTAIN_STEPS, KV_PRESSURE)
    history[-2] = state(gpu_cache_usage_perc=None)
    decision = propose(history)
    assert decision is not None
    assert decision.action is Action.NO_OP


# --------------------------------------------------------------------------- #
# Queue growth with flat throughput
# --------------------------------------------------------------------------- #


def _queue_history(
    queues: list[int], rates: list[float], preempts: list[int] | None = None
) -> list[TeacherState]:
    preempt_values: list[int | None]
    if preempts is None:
        preempt_values = [None] * len(queues)
    else:
        preempt_values = list(preempts)
    return [
        state(num_requests_waiting=q, tokens_per_s=r, num_preemptions_total=p)
        for q, r, p in zip(queues, rates, preempt_values, strict=True)
    ]


def test_queue_growth_flat_throughput_sheds_load():
    decision = propose(_queue_history([1, 5, 9], [100.0, 101.0, 99.0]))
    assert decision is not None
    assert decision.action is Action.SHED_LOAD
    assert decision.rule_id == "TR-008-QUEUE-GROWTH"


def test_queue_growth_with_rising_throughput_is_nominal():
    """A queue that grows while throughput scales up is a ramp, not a stall."""
    decision = propose(_queue_history([1, 5, 9], [100.0, 200.0, 400.0]))
    assert decision is not None
    assert decision.action is Action.NO_OP


def test_queue_growth_with_preemptions_reroutes():
    decision = propose(
        _queue_history([1, 5, 9], [100.0, 101.0, 99.0], preempts=[0, 3, 7])
    )
    assert decision is not None
    assert decision.action is Action.REROUTE_REQUEST
    assert decision.rule_id == "TR-009-QUEUE-PREEMPT"


def test_non_monotonic_queue_is_nominal():
    decision = propose(_queue_history([5, 9, 9], [100.0, 100.0, 100.0]))
    assert decision is not None
    assert decision.action is Action.NO_OP


# --------------------------------------------------------------------------- #
# Structural properties
# --------------------------------------------------------------------------- #


def test_determinism_same_history_same_decision():
    history = _queue_history([1, 5, 9], [100.0, 101.0, 99.0])
    assert propose(history) == propose(history)


def test_action_ids_are_frozen():
    """Published *_action_id values; renumbering breaks every consumer."""
    assert [(a.action_name, int(a)) for a in Action] == [
        ("no_op", 0),
        ("throttle_clocks", 1),
        ("power_gate_fpga", 2),
        ("clock_gate_fpga", 3),
        ("reduce_batch_size", 4),
        ("evict_kv_cache", 5),
        ("downshift_precision", 6),
        ("shed_load", 7),
        ("reroute_request", 8),
        ("migrate_workload", 9),
        ("cap_power", 10),
        ("raise_fan", 11),
        ("pause_workload", 12),
        ("escalate_to_human", 13),
    ]


def test_empty_history_raises():
    with pytest.raises(ValueError):
        propose([])
