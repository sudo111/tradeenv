"""Scripted policies that act only through the tool API.

Two kinds:
  * Honest policies use only what the tools return (TWAP, front-load, volume-profile,
    random schedule, buy-and-hold, momentum).
  * Calibration policies are given hindsight (future step volumes or price direction) with a
    controlled amount of noise. They exist only to create policies of KNOWN relative quality,
    so the reward's ability to rank them can be measured. An agent never gets hindsight.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .data import Bars
from .engine import FillModel
from .env import TradingEnv
from .tasks import Hindsight, Task, hindsight


# ----- execution ------------------------------------------------------------------------
def run_schedule(env: TradingEnv, weights: np.ndarray) -> float:
    """Each step: look at recent bars and the portfolio, cancel leftovers, and order enough to
    reach the cumulative target implied by `weights`."""
    task = env.call("get_task")
    q, side = task["qty"], task["side"]
    w = np.asarray(weights, dtype=float)
    cum = np.cumsum(w) / w.sum()
    for s in range(task["n_steps"]):
        env.call("get_bars", n=5)
        pf = env.call("get_portfolio")
        for o in pf["open_orders"]:
            env.call("cancel", order_id=o["order_id"])
        need = q * cum[s] - pf["filled_qty"]
        if need > 1e-12 * q:
            env.call("place_order", side=side, qty=float(need))
        env.call("end_turn")
    return env.reward


def twap(env, rng, hs):
    return run_schedule(env, np.ones(env.task.n_steps))


def front_load(env, rng, hs):
    w = np.zeros(env.task.n_steps)
    w[0] = 1.0
    return run_schedule(env, w)


def volume_profile(env, rng, hs):
    prof = env.call("get_volume_profile", days=7)["expected_step_volume"]
    return run_schedule(env, np.asarray(prof))


def random_schedule(env, rng, hs):
    return run_schedule(env, rng.dirichlet(np.ones(env.task.n_steps)))


def noisy_vwap(sigma: float):
    """Calibration: schedule proportional to the realized step volumes times lognormal noise.
    sigma = 0 is the hindsight-optimal schedule under this fill model."""
    def policy(env, rng, hs: Hindsight):
        noise = np.exp(sigma * rng.standard_normal(env.task.n_steps)) if sigma > 0 else 1.0
        return run_schedule(env, hs.step_volume * noise)
    return policy


# ----- directional PnL ------------------------------------------------------------------
def run_positions(env: TradingEnv, decide: Callable[[int, dict], int]) -> float:
    task = env.call("get_task")
    unit = task["unit"]
    for s in range(task["n_steps"]):
        bars = env.call("get_bars", n=15)
        pf = env.call("get_portfolio")
        for o in pf["open_orders"]:
            env.call("cancel", order_id=o["order_id"])
        delta = decide(s, bars) * unit - pf["position"]
        if abs(delta) > 1e-12 * unit:
            env.call("place_order", side="buy" if delta > 0 else "sell", qty=float(abs(delta)))
        env.call("end_turn")
    return env.reward


def _truth(hs: Hindsight, s: int) -> int:
    """Sign of the move a position opened at step s earns (fill at VWAP_s, exit at VWAP_{s+1},
    or at the final close for the last step)."""
    nxt = hs.step_vwap[s + 1] if s + 1 < len(hs.step_vwap) else hs.step_close[-1]
    return int(np.sign(nxt - hs.step_vwap[s]))


def noisy_direction(q: float):
    """Calibration: knows the sign of the next move and follows it with probability q."""
    def policy(env, rng, hs: Hindsight):
        def decide(s, bars):
            t = _truth(hs, s)
            if t == 0:
                t = 1 if rng.random() < 0.5 else -1
            return t if rng.random() < q else -t
        return run_positions(env, decide)
    return policy


def buy_hold(env, rng, hs):
    return run_positions(env, lambda s, bars: 1)


def momentum(env, rng, hs):
    def decide(s, bars):
        b = bars["bars"]
        return 1 if b[-1][4] >= b[0][1] else -1
    return run_positions(env, decide)


POLICIES = {
    "execution": {
        "vwap_oracle": (noisy_vwap(0.0), False), "vwap_noise_0.25": (noisy_vwap(0.25), True),
        "vwap_noise_0.5": (noisy_vwap(0.5), True), "vwap_noise_1.0": (noisy_vwap(1.0), True),
        "volume_profile": (volume_profile, False), "twap": (twap, False), "front_load": (front_load, False),
        "random_schedule": (random_schedule, True),
    },
    "pnl": {
        "direction_oracle": (noisy_direction(1.0), False), "direction_q0.60": (noisy_direction(0.60), True),
        "direction_q0.55": (noisy_direction(0.55), True), "direction_q0.53": (noisy_direction(0.53), True),
        "direction_q0.51": (noisy_direction(0.51), True), "random_q0.50": (noisy_direction(0.50), True),
        "buy_hold": (buy_hold, False), "momentum": (momentum, False),
    },
}


def run_episode(task: Task, bars: Bars, policy: str, seed: int, fill: FillModel = FillModel(),
                mask: bool = True) -> tuple[float, str, int]:
    """Run one episode. Returns (reward, trajectory hash, number of tool calls)."""
    fn, _ = POLICIES[task.family][policy]
    env = TradingEnv(task, bars, fill=fill, mask=mask)
    rng = np.random.default_rng(seed)
    fn(env, rng, hindsight(task, bars))
    return env.reward, env.trajectory_hash(), env.n_calls
