"""Timestamp parsing that never fabricates.

The defect this module exists to prevent:

    if occursin("T", ts) && occursin("-", ts)   # "already ISO?"
        return ts
    end
    return string(base_ts + Second(idx * 10))   # otherwise, invent one

The real timestamps in ``node_sync_harvest.jsonl`` look like
``2026-03-19 11:55:13.132`` -- a space separator, not ``T``. All 114,250 of them
failed that check and were overwritten with an index-derived sequence. The
fabricated values then became the dataset's advertised collection window
(2026-03-20 to 2026-04-02), which was never a real observation period.

Rules here:

* Parsing is total: every input either yields a real value or is reported.
* An unparseable timestamp yields ``None`` and a quarantine entry. It never
  yields a generated substitute.
* ``coin:height`` tags are recognized as *identifiers*, not clocks, and are
  decomposed rather than coerced into a datetime.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Real datetime forms observed across the sources, by frequency in the recovered
# 120,334-row node_sync_harvest original:
#
#   114,238  2026-03-19 11:55:13.132            space separator
#        12  2026-03-19 17:00:05.551-05:00      space separator + UTC offset
#
# Both use a space, not "T". The prior pipeline tested `occursin("T", ts)` and
# discarded all 114,250 of them. Ordered most specific first.
_DATETIME_PATTERNS = (
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f%z",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
)

# Chain-tagged identifiers, in two real shapes:
#
#    5,001  dynex:919876              coin:height
#    1,083  qubic:204:46075040        coin:epoch:tick
#
# These are identifiers, not clocks. Decomposed rather than coerced.
_COIN_TAG = re.compile(
    r"^(?P<coin>[a-z][a-z0-9_-]*):(?P<first>\d+)(?::(?P<second>\d+))?$",
    re.IGNORECASE,
)

# Unix epoch seconds as a bare string, e.g. "1773996924".
_EPOCH_SECONDS = re.compile(r"^\d{9,11}$")

# Sub-second precision finer than microseconds. ghost_market_log.jsonl is written
# by a Rust producer whose timestamps carry *nanoseconds*
# ("2026-03-11T18:22:37.433458521+00:00" -- 9 digits). Python's %f accepts at most
# 6, and datetime cannot represent finer than microseconds at all, so the tail is
# truncated. Caught by CI on 3.10; newer interpreters are more permissive here,
# which is precisely why the pinned matrix matters.
_SUBSECOND = re.compile(r"(?<=\.)(\d{7,})")


def _truncate_subseconds(text: str) -> str:
    """Trim fractional seconds to the 6 digits datetime can represent."""
    return _SUBSECOND.sub(lambda m: m.group(1)[:6], text)


@dataclass(frozen=True)
class ParsedTimestamp:
    """Outcome of parsing one timestamp field.

    Exactly one of ``moment`` / ``coin`` is populated for a well-formed input;
    both are ``None`` when the value could not be interpreted.
    """

    raw: str
    moment: datetime | None = None
    coin: str | None = None
    height: int | None = None
    epoch_number: int | None = None

    @property
    def ok(self) -> bool:
        return self.moment is not None or self.coin is not None

    @property
    def iso(self) -> str | None:
        return self.moment.isoformat() if self.moment else None

    @property
    def epoch(self) -> float | None:
        return self.moment.timestamp() if self.moment else None


def parse(raw: object) -> ParsedTimestamp:
    """Interpret a legacy timestamp field. Never invents a value."""
    if not isinstance(raw, str):
        # Numeric epoch seconds appear in qubic_ticks.jsonl as a real int.
        if isinstance(raw, (int, float)):
            return ParsedTimestamp(
                raw=str(raw),
                moment=datetime.fromtimestamp(float(raw), tz=timezone.utc),
            )
        return ParsedTimestamp(raw=repr(raw))

    text = raw.strip()
    if not text:
        return ParsedTimestamp(raw=raw)

    tag = _COIN_TAG.match(text)
    if tag:
        first, second = tag.group("first"), tag.group("second")
        # Two parts -> coin:height. Three parts -> coin:epoch:tick.
        epoch_number = int(first) if second is not None else None
        height = int(second if second is not None else first)
        return ParsedTimestamp(
            raw=raw,
            coin=tag.group("coin").lower(),
            height=height,
            epoch_number=epoch_number,
        )

    if _EPOCH_SECONDS.match(text):
        return ParsedTimestamp(
            raw=raw, moment=datetime.fromtimestamp(int(text), tz=timezone.utc)
        )

    # Theseus-Quarry chrono DateTime<Utc> serializes with a trailing Z.
    # Python 3.10's fromisoformat rejects Z; the %z patterns want ±HH:MM.
    # Some producers emit lowercase z; treat it the same as Z.
    if text.endswith(("Z", "z")) and "T" in text:
        text = text[:-1] + "+00:00"

    normalized = _truncate_subseconds(text)

    for pattern in _DATETIME_PATTERNS:
        try:
            return ParsedTimestamp(
                raw=raw, moment=datetime.strptime(normalized, pattern)
            )
        except ValueError:
            continue

    # Last resort: stdlib ISO parser handles offsets this module hasn't enumerated.
    try:
        return ParsedTimestamp(raw=raw, moment=datetime.fromisoformat(normalized))
    except ValueError:
        return ParsedTimestamp(raw=raw)


UNPARSEABLE = "unparseable"
OUT_OF_ORDER = "out-of-order"


@dataclass
class QuarantineLog:
    """Rows excluded from output, with the reason, for the run report.

    Quarantined rows are counted and surfaced rather than absorbed, and are never
    replaced with a synthesized value. Reasons are tracked separately because they
    mean different things: an unparseable timestamp is a malformed record, while
    an out-of-order trailing run is a different writer's data appended to the file.
    """

    source: str
    entries: list[tuple[int, str, str]] = field(default_factory=list)

    def record(self, row_index: int, raw: str, reason: str = UNPARSEABLE) -> None:
        self.entries.append((row_index, raw, reason))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.entries)

    def counts(self) -> dict[str, int]:
        tally: dict[str, int] = {}
        for _, _, reason in self.entries:
            tally[reason] = tally.get(reason, 0) + 1
        return tally

    def summary(self, limit: int = 10) -> str:
        if not self.entries:
            return f"{self.source}: 0 quarantined rows"
        n = len(self.entries)
        head = ", ".join(
            f"row {i} ({reason}): {v!r}" for i, v, reason in self.entries[:limit]
        )
        more = "" if n <= limit else f" (+{n - limit} more)"
        tally = ", ".join(f"{k}={v}" for k, v in sorted(self.counts().items()))
        return f"{self.source}: {n} quarantined [{tally}] -- {head}{more}"
