"""Multi-turn trading environment exposed as a tool API.

An agent (LLM or scripted) interacts only through `call(tool_name, **args)`. Every call and
its result is appended to a running SHA-256 hash, so two runs are identical exactly when
their trajectory hashes match.

Tools: get_task, get_bars, get_volume_profile, get_portfolio, place_order, cancel, end_turn.
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np

from .data import Bars
from .engine import BarReplay, BookState, FillModel, Order
from .tasks import HISTORY_BARS, Task

MAX_BARS_PER_CALL = 1440


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=float)


TOOL_SCHEMAS = [
    {"name": "get_task", "description": "Describe the task, the deadline, and the reward.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "get_bars", "description": "Last n closed 1-minute bars [t, open, high, low, close, volume]; "
     "t is minutes relative to the window start (negative means before it).",
     "parameters": {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1, "maximum": MAX_BARS_PER_CALL}}}},
    {"name": "get_volume_profile", "description": "Expected volume per decision step of this window, "
     "averaged over the same time of day on the previous `days` days (past data only).",
     "parameters": {"type": "object", "properties": {"days": {"type": "integer", "minimum": 1, "maximum": 7}}}},
    {"name": "get_portfolio", "description": "Cash, position, progress, open orders, and steps left.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "place_order", "description": "Submit a market or limit order. Orders work until filled or cancelled.",
     "parameters": {"type": "object", "properties": {
         "side": {"type": "string", "enum": ["buy", "sell"]}, "qty": {"type": "number", "exclusiveMinimum": 0},
         "order_type": {"type": "string", "enum": ["market", "limit"]}, "limit_price": {"type": "number"}},
         "required": ["side", "qty"]}},
    {"name": "cancel", "description": "Cancel an open order.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "integer"}}, "required": ["order_id"]}},
    {"name": "end_turn", "description": "Let one decision step of market time pass; working orders fill.",
     "parameters": {"type": "object", "properties": {}}},
]


class TradingEnv:
    def __init__(self, task: Task, bars: Bars, fill: FillModel = FillModel(), mask: bool = True):
        if task.start < HISTORY_BARS:
            raise ValueError("task window needs history")
        self.task, self.bars, self.mask = task, bars, mask
        self.rt = BarReplay(bars, task.start, task.n_steps, task.bars_per_step, fill)
        self.scale = 100.0 / self.rt.ref_price if mask else 1.0
        self.state = BookState()
        self.done = False
        self.reward: float | None = None
        self.n_calls = 0
        self.count_chars = False  # when True, obs_chars counts characters of tool results (token-size proxy)
        self.obs_chars = 0
        self._hash = hashlib.sha256()
        self._hash.update(_canon({"task": task.to_dict(), "fill": vars(fill), "mask": mask}).encode())
        self._tools = {
            "get_task": self._get_task, "get_bars": self._get_bars, "get_volume_profile": self._get_volume_profile,
            "get_portfolio": self._get_portfolio, "place_order": self._place_order, "cancel": self._cancel,
            "end_turn": self._end_turn,
        }

    # ----- public interface -------------------------------------------------------------
    def call(self, tool: str, **args) -> dict:
        if tool not in self._tools:
            result = {"error": f"unknown tool {tool}"}
        elif self.done and tool not in ("get_task", "get_portfolio"):
            result = {"error": "episode finished"}
        else:
            try:
                result = self._tools[tool](**args)
            except TypeError as e:  # bad arguments from the agent
                result = {"error": str(e)}
        self.n_calls += 1
        self._hash.update(_canon({"tool": tool, "args": args, "result": result}).encode())
        if self.count_chars:
            self.obs_chars += len(_canon(result))
        return result

    def trajectory_hash(self) -> str:
        return self._hash.hexdigest()

    def snapshot(self) -> dict:
        return {"state": copy.deepcopy(self.state), "hash": self._hash.copy(), "n_calls": self.n_calls,
                "done": self.done, "reward": self.reward}

    def restore(self, snap: dict) -> None:
        self.state = copy.deepcopy(snap["state"])
        self._hash = snap["hash"].copy()
        self.n_calls, self.done, self.reward = snap["n_calls"], snap["done"], snap["reward"]

    # ----- helpers ----------------------------------------------------------------------
    def _px(self, p: float) -> float:
        return float(p) * self.scale

    def _progress(self) -> float:
        """Execution: quantity done in the parent order's direction."""
        return self.task.side * self.state.position

    # ----- tools ------------------------------------------------------------------------
    def _get_task(self) -> dict:
        t = self.task
        out = {"family": t.family, "asset": "ASSET" if self.mask else t.symbol, "n_steps": t.n_steps,
               "minutes_per_step": t.bars_per_step, "steps_left": t.n_steps - self.state.step}
        if t.family == "execution":
            out.update(side="buy" if t.side > 0 else "sell", qty=t.qty, instructions=(
                f"{'Buy' if t.side > 0 else 'Sell'} {t.qty:.6g} units within {t.n_steps * t.bars_per_step} minutes. "
                "Reward = minus your shortfall versus the window VWAP, in basis points. Anything unfilled at the "
                "deadline is completed at the last step without the participation cap."))
        else:
            out.update(unit=t.unit, instructions=(
                f"Hold a position between -{t.unit:.6g} and +{t.unit:.6g} units for {t.n_steps} steps. "
                "Reward = final profit and loss after costs, in basis points of one unit's starting value."))
        return out

    def _get_bars(self, n: int = 10) -> dict:
        n = int(max(1, min(n, MAX_BARS_PER_CALL)))
        end = self.rt.visible_end(self.state.step)
        lo = end - n
        b = self.bars
        rows = [[int(i - self.task.start), self._px(b.open[i]), self._px(b.high[i]), self._px(b.low[i]),
                 self._px(b.close[i]), float(b.volume[i])] for i in range(lo, end)]
        return {"bars": rows}

    def _get_volume_profile(self, days: int = 7) -> dict:
        days = int(max(1, min(days, 7)))
        t = self.task
        length = t.n_steps * t.bars_per_step
        hist = np.stack([self.bars.volume[t.start - 1440 * d:t.start - 1440 * d + length] for d in range(1, days + 1)])
        per_step = hist.mean(axis=0).reshape(t.n_steps, t.bars_per_step).sum(axis=1)
        return {"expected_step_volume": [float(v) for v in per_step]}

    def _get_portfolio(self) -> dict:
        st, t = self.state, self.task
        out = {"cash": self._px(st.cash), "position": st.position, "step": st.step,
               "steps_left": t.n_steps - st.step,
               "open_orders": [{"order_id": o.order_id, "side": "buy" if o.side > 0 else "sell",
                                "remaining": o.remaining, "order_type": o.kind,
                                "limit_price": None if o.limit is None else self._px(o.limit)} for o in st.orders]}
        if t.family == "execution":
            out.update(filled_qty=self._progress(), remaining_qty=t.qty - self._progress())
        return out

    def _place_order(self, side: str, qty: float, order_type: str = "market", limit_price: float | None = None) -> dict:
        if side not in ("buy", "sell") or not (qty > 0) or order_type not in ("market", "limit"):
            return {"error": "invalid order"}
        if order_type == "limit" and (limit_price is None or not limit_price > 0):
            return {"error": "limit order needs a positive limit_price"}
        sgn = 1 if side == "buy" else -1
        st = self.state
        if self.task.family == "pnl":
            pending = sum(o.side * o.remaining for o in st.orders)
            if abs(st.position + pending + sgn * qty) > self.task.unit * (1 + 1e-9):
                return {"error": "position limit exceeded"}
        limit = None if order_type == "market" else float(limit_price) / self.scale
        order = Order(order_id=st.next_id, side=sgn, remaining=float(qty), kind=order_type, limit=limit)
        st.next_id += 1
        st.orders.append(order)
        return {"order_id": order.order_id}

    def _cancel(self, order_id: int) -> dict:
        before = len(self.state.orders)
        self.state.orders = [o for o in self.state.orders if o.order_id != order_id]
        return {"ok": len(self.state.orders) < before}

    def _end_turn(self) -> dict:
        t, st = self.task, self.state
        last = st.step == t.n_steps - 1
        fills = self.rt.process_step(st, force=(last and t.family == "execution"))
        out = {"fills": [{"order_id": f[1], "side": "buy" if f[2] > 0 else "sell", "qty": f[3],
                          "price": self._px(f[4])} for f in fills], "step": st.step}
        if last:
            if t.family == "execution":
                self.rt.force_trade(st, t.side * t.qty - st.position)  # finish exactly the parent order
            st.orders = []
            self.done = True
            self.reward = self._final_reward()
            out.update(done=True, reward=self.reward)
        else:
            out.update(done=False)
        return out

    def _final_reward(self) -> float:
        t, st = self.task, self.state
        if t.family == "execution":
            cost = -st.cash  # money paid (negative when selling)
            bench = t.qty * self.rt.window_vwap
            shortfall_bps = (cost - t.side * bench) / bench * 1e4  # > 0 means worse than VWAP
            return float(-shortfall_bps)
        equity = st.cash + st.position * self.rt.step_close[-1]
        return float(equity / (t.unit * self.rt.ref_price) * 1e4)
