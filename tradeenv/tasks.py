"""Task families and a deterministic, time-split task sampler.

A task is a (family, symbol, window) triple plus family parameters. Tasks are generated from
history the way Spreadsheet-RL generated tasks from forum posts: many cheap, checkable
instances, each with a reward computed from the same recorded data.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass

import numpy as np

from .data import Bars

HISTORY_BARS = 7 * 1440  # a task needs 7 days of past bars (for the volume-profile tool)

# Time split of the pilot data (UTC). Train: July and August 2026. Validation: Sep 1 to 15.
# Test: Sep 16 to 30.
SPLITS = {
    "train": (dt.datetime(2026, 7, 1), dt.datetime(2026, 9, 1)),
    "val": (dt.datetime(2026, 9, 1), dt.datetime(2026, 9, 16)),
    "test": (dt.datetime(2026, 9, 16), dt.datetime(2026, 10, 1)),
    "all": (dt.datetime(2026, 7, 1), dt.datetime(2026, 10, 1)),
}

FAMILIES = {
    # Buy or sell a parent order over 60 minutes, one decision every 5 minutes.
    "execution": dict(n_steps=12, bars_per_step=5),
    # Hold -1, 0 or +1 unit of one asset over 4 hours, one decision every 15 minutes.
    "pnl": dict(n_steps=16, bars_per_step=15),
}

EXEC_PARTICIPATION = 0.05   # parent order = 5% of the trailing average step volume x n_steps
PNL_UNIT_FRACTION = 0.001   # position unit = 0.1% of the trailing average step volume


@dataclass(frozen=True)
class Task:
    task_id: str
    family: str
    symbol: str
    start: int            # index of the first bar of the window
    n_steps: int
    bars_per_step: int
    side: int = 0         # execution: +1 buy, -1 sell
    qty: float = 0.0      # execution: parent order size in base units
    unit: float = 0.0     # pnl: position unit in base units

    def to_dict(self) -> dict:
        return asdict(self)


def _epoch(d: dt.datetime) -> int:
    return int(d.replace(tzinfo=dt.timezone.utc).timestamp())


def candidate_starts(bars: Bars, family: str, split: str) -> np.ndarray:
    """Non-overlapping window starts inside the split, with full history and no empty step."""
    p = FAMILIES[family]
    length = p["n_steps"] * p["bars_per_step"]
    lo, hi = (_epoch(x) for x in SPLITS[split])
    starts = np.arange(HISTORY_BARS, len(bars) - length, length)
    ok = []
    for s in starts:
        if not (lo <= bars.ts[s] and bars.ts[s + length - 1] < hi):
            continue
        if np.any(np.diff(bars.ts[s - 1:s + length]) != 60):
            continue  # gap in the data
        vol = bars.volume[s:s + length].reshape(p["n_steps"], p["bars_per_step"]).sum(axis=1)
        if np.any(vol <= 0):
            continue
        ok.append(s)
    return np.asarray(ok, dtype=np.int64)


def make_task(bars: Bars, family: str, start: int, rng: np.random.Generator) -> Task:
    p = FAMILIES[family]
    n, b = p["n_steps"], p["bars_per_step"]
    past = bars.volume[start - n * b:start].reshape(n, b).sum(axis=1).mean()
    tid = f"{family}:{bars.symbol}:{int(bars.ts[start])}"
    if family == "execution":
        side = int(rng.choice([1, -1]))
        return Task(tid, family, bars.symbol, int(start), n, b, side=side, qty=float(EXEC_PARTICIPATION * past * n))
    return Task(tid, family, bars.symbol, int(start), n, b, unit=float(PNL_UNIT_FRACTION * past))


def sample_tasks(data: dict[str, Bars], family: str, split: str, n: int, seed: int) -> list[Task]:
    """Deterministic sample of n tasks across symbols (without replacement)."""
    rng = np.random.default_rng(seed)
    pool = [(sym, int(s)) for sym in sorted(data) for s in candidate_starts(data[sym], family, split)]
    if n > len(pool):
        raise ValueError(f"only {len(pool)} {family} windows in split {split}")
    idx = rng.choice(len(pool), size=n, replace=False)
    idx.sort()
    return [make_task(data[pool[i][0]], family, pool[i][1], rng) for i in idx]


@dataclass(frozen=True)
class Hindsight:
    """Future information for scripted calibration policies and oracles only. Never shown to agents."""

    step_volume: np.ndarray
    step_vwap: np.ndarray
    step_close: np.ndarray


def hindsight(task: Task, bars: Bars) -> Hindsight:
    n, b = task.n_steps, task.bars_per_step
    sl = slice(task.start, task.start + n * b)
    vol = bars.volume[sl].reshape(n, b).sum(axis=1)
    quote = bars.quote[sl].reshape(n, b).sum(axis=1)
    return Hindsight(step_volume=vol, step_vwap=quote / vol, step_close=bars.close[sl].reshape(n, b)[:, -1])
