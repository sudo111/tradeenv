"""Bar-replay execution runtime: the part that plays the role Excel played in Spreadsheet-RL.

The caller owns the clock. Each call to `process_step` executes working orders against one
decision step (a block of 1-minute bars) and advances time by one step. Fills are a pure
function of the recorded bars, the orders, and the fill model, so replay is deterministic.

Fill model (declared assumptions, not estimates of any real venue):
  * Market orders fill at the step's exact VWAP (sum of quote volume / sum of base volume),
    plus a half-spread, plus temporary impact that is linear in participation
    (order quantity / step volume). Impact does not persist: later prices never react.
  * One step can absorb at most `max_participation` of its volume across all orders;
    the rest keeps working in later steps.
  * Limit orders fill at the limit price, up to the same cap, only if the step trades
    strictly through the limit (low < limit for buys, high > limit for sells). No queue model.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from .data import Bars


@dataclass(frozen=True)
class FillModel:
    half_spread_bps: float = 1.0
    impact_bps: float = 50.0          # cost in bps if an order took 100% of a step's volume
    max_participation: float = 0.25


@dataclass
class Order:
    order_id: int
    side: int                  # +1 buy, -1 sell
    remaining: float
    kind: str                  # "market" or "limit"
    limit: float | None = None  # raw (unmasked) price


@dataclass
class BookState:
    """Everything that changes during an episode. Copying it is a full snapshot."""

    step: int = 0
    cash: float = 0.0
    position: float = 0.0
    next_id: int = 1
    orders: list[Order] = field(default_factory=list)
    fills: list[tuple[int, int, int, float, float]] = field(default_factory=list)  # step, id, side, qty, price


class BarReplay:
    def __init__(self, bars: Bars, start: int, n_steps: int, bars_per_step: int, fill: FillModel):
        end = start + n_steps * bars_per_step
        if start < 1 or end > len(bars):
            raise ValueError("window out of range")
        self.bars, self.start, self.n_steps, self.bps = bars, start, n_steps, bars_per_step
        self.fill = fill
        sl = slice(start, end)
        shape = (n_steps, bars_per_step)
        self.step_volume = bars.volume[sl].reshape(shape).sum(axis=1)
        self.step_quote = bars.quote[sl].reshape(shape).sum(axis=1)
        self.step_vwap = self.step_quote / self.step_volume
        self.step_high = bars.high[sl].reshape(shape).max(axis=1)
        self.step_low = bars.low[sl].reshape(shape).min(axis=1)
        self.step_close = bars.close[sl].reshape(shape)[:, -1]
        self.window_vwap = float(self.step_quote.sum() / self.step_volume.sum())
        self.ref_price = float(bars.close[start - 1])  # last close the agent can see at reset

    def visible_end(self, step: int) -> int:
        """Index one past the last bar visible at the start of `step` (no lookahead)."""
        return self.start + step * self.bps

    def _execute(self, st: BookState, order: Order, qty: float, price: float) -> None:
        st.cash -= order.side * qty * price
        st.position += order.side * qty
        order.remaining -= qty
        st.fills.append((st.step, order.order_id, order.side, qty, price))

    def process_step(self, st: BookState, force: bool = False) -> list[tuple]:
        """Fill working orders against the current step, then advance the clock.

        With force=True (used at an execution deadline) market orders ignore the
        participation cap.
        """
        s = st.step
        vol, vwap = self.step_volume[s], self.step_vwap[s]
        cap_left = self.fill.max_participation * vol
        n_before = len(st.fills)
        for order in st.orders:
            room = order.remaining if (force and order.kind == "market") else min(order.remaining, cap_left)
            if room <= 0:
                continue
            if order.kind == "market":
                part = room / vol
                cost_bps = self.fill.half_spread_bps + self.fill.impact_bps * part
                price = vwap * (1.0 + order.side * cost_bps / 1e4)
            else:
                crossed = (self.step_low[s] < order.limit) if order.side > 0 else (self.step_high[s] > order.limit)
                if not crossed:
                    continue
                price = order.limit
            self._execute(st, order, room, price)
            cap_left = max(0.0, cap_left - room)
        st.orders = [o for o in st.orders if o.remaining > 1e-12]
        st.step += 1
        return st.fills[n_before:]

    def force_trade(self, st: BookState, signed_qty: float) -> None:
        """Trade a signed quantity in the step just processed, uncapped (deadline completion)."""
        if abs(signed_qty) <= 1e-12:
            return
        s = st.step - 1
        side = 1 if signed_qty > 0 else -1
        qty = abs(signed_qty)
        part = qty / self.step_volume[s]
        cost_bps = self.fill.half_spread_bps + self.fill.impact_bps * part
        price = self.step_vwap[s] * (1.0 + side * cost_bps / 1e4)
        order = Order(order_id=-1, side=side, remaining=qty, kind="market")
        st.cash -= side * qty * price
        st.position += side * qty
        st.fills.append((s, order.order_id, side, qty, price))

    @staticmethod
    def snapshot(st: BookState) -> BookState:
        return copy.deepcopy(st)
