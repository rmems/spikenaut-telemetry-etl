"""Deterministic teacher policy for the v3 action-proposal dataset.

The Spikenaut control hierarchy is *learned policy -> action proposal ->
deterministic safety filter -> execution*. This module is the **teacher** that
labels historical state windows with the action the deterministic layer would
have wanted proposed. It is not the safety filter and it never executes
anything; it exists so ``action_proposals.teacher_action`` in the published
dataset has a reproducible, versioned origin.

Design rules, in priority order:

1. **Pure and deterministic.** :func:`propose` is a pure function of the state
   window it is given. Same input, same label, forever. Any behaviour change
   bumps :data:`TEACHER_POLICY_VERSION`.
2. **No guessed labels.** If the signals a rule needs are absent, the rule is
   skipped; if the *core* signals (``throttle_reasons``, ``ecc_dbe_vol``) are
   absent, no label is emitted at all -- ``None``, never a default. This is the
   same contract as the cleaning pipeline: missing means missing, not ``0``.
   In particular the v2 backfill (GPU sensors only, no NVML throttle mask, no
   ECC counters, no vLLM metrics) receives **zero** labels by construction.
3. **Key off the NVML throttle mask, not hard-coded temperatures.** A GPU at
   85 C with no slowdown bit set is a GPU doing its job. The bits below follow
   ``nvmlClocksThrottleReasons`` in the NVML API.

Every decision carries the ``rule_id`` that produced it, so a published label
can always be traced to the exact clause of the exact teacher version.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

# Semantic version of the labeling behaviour, recorded in the dataset changelog
# alongside the git SHA of the merged teacher. PATCH = docs only, MINOR =
# additive rules that cannot change existing labels, MAJOR = any label changes.
TEACHER_POLICY_VERSION = "1.0.0"

# --------------------------------------------------------------------------- #
# NVML throttle_reasons bitmask (nvmlClocksThrottleReasons)
# --------------------------------------------------------------------------- #

THR_GPU_IDLE = 0x0000000000000001
THR_APPLICATIONS_CLOCKS_SETTING = 0x0000000000000002
THR_SW_POWER_CAP = 0x0000000000000004
THR_HW_SLOWDOWN = 0x0000000000000008
THR_SYNC_BOOST = 0x0000000000000010
THR_SW_THERMAL_SLOWDOWN = 0x0000000000000020
THR_HW_THERMAL_SLOWDOWN = 0x0000000000000040
THR_HW_POWER_BRAKE_SLOWDOWN = 0x0000000000000080
THR_DISPLAY_CLOCK_SETTING = 0x0000000000000100


class Action(IntEnum):
    """Action taxonomy. Integer values are the published ``*_action_id``
    columns and are frozen -- append new actions, never renumber."""

    NO_OP = 0
    THROTTLE_CLOCKS = 1
    POWER_GATE_FPGA = 2
    CLOCK_GATE_FPGA = 3
    REDUCE_BATCH_SIZE = 4
    EVICT_KV_CACHE = 5
    DOWNSHIFT_PRECISION = 6
    SHED_LOAD = 7
    REROUTE_REQUEST = 8
    MIGRATE_WORKLOAD = 9
    CAP_POWER = 10
    RAISE_FAN = 11
    PAUSE_WORKLOAD = 12
    ESCALATE_TO_HUMAN = 13

    @property
    def action_name(self) -> str:
        """The string published in ``*_action`` columns, e.g. ``"no_op"``."""
        return self.name.lower()


# --------------------------------------------------------------------------- #
# Inputs and outputs
# --------------------------------------------------------------------------- #

# "Sustained" means this many consecutive windows, current one included.
SUSTAIN_STEPS = 3
# KV-cache pressure threshold on vLLM's gpu_cache_usage_perc gauge (0..1).
KV_PRESSURE = 0.95
# Throughput is "flat" when its relative range over the queue-growth lookback
# stays within this fraction of its maximum.
FLAT_THROUGHPUT_REL_RANGE = 0.05


@dataclass(frozen=True)
class TeacherState:
    """The slice of one ``state_telemetry`` row the teacher reads.

    Every field is optional because historical captures genuinely lack most of
    them. ``None`` means the collector did not report the signal -- it is never
    treated as zero.
    """

    throttle_reasons: int | None = None
    ecc_dbe_vol: int | None = None
    gpu_cache_usage_perc: float | None = None
    num_requests_waiting: int | None = None
    tokens_per_s: float | None = None
    num_preemptions_total: int | None = None


@dataclass(frozen=True)
class Decision:
    """One emitted label. ``paired_action`` is a secondary action the rule
    demands alongside the primary (e.g. migrate after escalate); it is
    published inside ``proposed_params``, not as a second row, so the
    (episode_id, step_idx) join stays unique."""

    action: Action
    rule_id: str
    paired_action: Action | None = None
    # Deterministic rules are certain by definition. A future shadow-policy or
    # human label source will carry real uncertainty here.
    confidence: float = 1.0


def _bit(mask: int | None, bit: int) -> bool:
    return mask is not None and bool(mask & bit)


def _sustained_kv_pressure(history: Sequence[TeacherState], steps: int) -> bool:
    """True when the last ``steps`` windows all report kv usage >= threshold.

    A window with a missing gauge breaks the run -- absence of evidence is not
    sustained pressure.
    """
    if len(history) < steps:
        return False
    recent = history[-steps:]
    return all(
        s.gpu_cache_usage_perc is not None and s.gpu_cache_usage_perc >= KV_PRESSURE
        for s in recent
    )


def _queue_growing_throughput_flat(
    history: Sequence[TeacherState], steps: int
) -> bool:
    """Strictly growing request queue while token throughput stays flat."""
    if len(history) < steps:
        return False
    recent = history[-steps:]
    queues = [s.num_requests_waiting for s in recent]
    rates = [s.tokens_per_s for s in recent]
    known_queues = [q for q in queues if q is not None]
    known_rates = [r for r in rates if r is not None]
    if len(known_queues) != steps or len(known_rates) != steps:
        return False
    if not all(b > a for a, b in zip(known_queues, known_queues[1:], strict=False)):
        return False
    top = max(known_rates)
    if top <= 0.0:
        # No throughput at all while the queue grows is the degenerate flat case.
        return True
    return (top - min(known_rates)) / top <= FLAT_THROUGHPUT_REL_RANGE


def _preemptions_rising(history: Sequence[TeacherState], steps: int) -> bool:
    if len(history) < steps:
        return False
    first = history[-steps].num_preemptions_total
    last = history[-1].num_preemptions_total
    return first is not None and last is not None and last > first


def propose(history: Sequence[TeacherState]) -> Decision | None:
    """Label the newest state window given its recent history.

    ``history`` is chronological and must contain at least the current window
    (``history[-1]``). Rules are evaluated in a fixed priority order; the first
    match wins. Returns ``None`` -- no label, rather than a guessed one -- when
    the core signals are absent from the current window.
    """
    if not history:
        raise ValueError("propose() needs at least the current state window")
    cur = history[-1]

    # Core signal gate: without the throttle mask and the double-bit ECC
    # counter, even "no_op" would be a guess.
    if cur.throttle_reasons is None or cur.ecc_dbe_vol is None:
        return None

    # TR-001: uncorrectable memory errors. Nothing else matters; get the
    # workload off this device and a human on it.
    if cur.ecc_dbe_vol > 0:
        return Decision(
            Action.ESCALATE_TO_HUMAN,
            rule_id="TR-001-ECC-DBE",
            paired_action=Action.MIGRATE_WORKLOAD,
        )

    # TR-002..005: hardware-reported slowdowns, most severe first. These key
    # off the NVML mask, never off a temperature threshold.
    if _bit(cur.throttle_reasons, THR_HW_POWER_BRAKE_SLOWDOWN):
        return Decision(Action.CAP_POWER, rule_id="TR-002-HW-POWER-BRAKE")
    if _bit(cur.throttle_reasons, THR_HW_THERMAL_SLOWDOWN):
        return Decision(Action.THROTTLE_CLOCKS, rule_id="TR-003-HW-THERMAL")
    if _bit(cur.throttle_reasons, THR_HW_SLOWDOWN):
        return Decision(Action.THROTTLE_CLOCKS, rule_id="TR-004-HW-SLOWDOWN")
    if _bit(cur.throttle_reasons, THR_SW_THERMAL_SLOWDOWN):
        return Decision(Action.RAISE_FAN, rule_id="TR-005-SW-THERMAL")

    # TR-006/007: sustained KV-cache pressure. Evict first; if pressure has
    # already persisted twice the sustain horizon, eviction was not enough --
    # shrink the batch.
    if _sustained_kv_pressure(history, 2 * SUSTAIN_STEPS):
        return Decision(Action.REDUCE_BATCH_SIZE, rule_id="TR-007-KV-PRESSURE-BATCH")
    if _sustained_kv_pressure(history, SUSTAIN_STEPS):
        return Decision(Action.EVICT_KV_CACHE, rule_id="TR-006-KV-PRESSURE-EVICT")

    # TR-008/009: queue growing while throughput is flat. If the scheduler is
    # also preempting, this replica is thrashing -- send work elsewhere.
    if _queue_growing_throughput_flat(history, SUSTAIN_STEPS):
        if _preemptions_rising(history, SUSTAIN_STEPS):
            return Decision(Action.REROUTE_REQUEST, rule_id="TR-009-QUEUE-PREEMPT")
        return Decision(Action.SHED_LOAD, rule_id="TR-008-QUEUE-GROWTH")

    # Core signals present and nominal. THR_SW_POWER_CAP alone is a GPU
    # operating at its configured limit, which is normal, not actionable.
    return Decision(Action.NO_OP, rule_id="TR-000-NOMINAL")
