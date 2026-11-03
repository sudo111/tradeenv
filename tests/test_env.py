"""Determinism, reward correctness, no-lookahead, and GRPO invariance tests."""
import pathlib

import numpy as np
import pytest

from tradeenv.data import Bars, synthetic_bars
from tradeenv.engine import FillModel
from tradeenv.env import TradingEnv
from tradeenv.grpo import group_advantage
from tradeenv.policies import POLICIES, run_episode
from tradeenv.tasks import HISTORY_BARS, Task, make_task, sample_tasks

ZERO_COST = FillModel(half_spread_bps=0.0, impact_bps=0.0, max_participation=1.0)
DATA = pathlib.Path(__file__).resolve().parents[1] / "data"


def hand_bars(window_vol, window_px, closes=None, ref_close=10.0) -> Bars:
    """History of flat bars (price ref_close, volume 1), then a hand-made window."""
    n_hist = HISTORY_BARS
    vol = np.concatenate([np.ones(n_hist), np.asarray(window_vol, float)])
    px = np.concatenate([np.full(n_hist, ref_close), np.asarray(window_px, float)])
    close = px.copy() if closes is None else np.concatenate([np.full(n_hist, ref_close), np.asarray(closes, float)])
    n = len(vol)
    return Bars("HAND", (1_700_000_000 + 60 * np.arange(n)).astype(np.int64), px, np.maximum(px, close) + 1,
                np.minimum(px, close) - 1, close, vol, px * vol)


def synth_tasks(family: str, k: int, seed: int = 0):
    bars = synthetic_bars(n=HISTORY_BARS + 4000, seed=seed)
    rng = np.random.default_rng(seed)
    length = 60 if family == "execution" else 240
    starts = HISTORY_BARS + length * np.arange(k)
    return bars, [make_task(bars, family, int(s), rng) for s in starts]


# ----- reward correctness ---------------------------------------------------------------
def test_execution_reward_by_hand():
    bars = hand_bars(window_vol=[100, 300], window_px=[10.0, 11.0])
    task = Task("t", "execution", "HAND", HISTORY_BARS, n_steps=2, bars_per_step=1, side=1, qty=20.0)
    fill = FillModel(half_spread_bps=1.0, impact_bps=50.0, max_participation=0.25)
    env = TradingEnv(task, bars, fill=fill, mask=False)
    env.call("place_order", side="buy", qty=20.0)
    env.call("end_turn")
    env.call("end_turn")
    price = 10.0 * (1 + (1.0 + 50.0 * 20 / 100) / 1e4)   # participation 0.2
    vwap = (100 * 10 + 300 * 11) / 400
    expected = -((20 * price - 20 * vwap) / (20 * vwap) * 1e4)
    assert env.done and env.reward == pytest.approx(expected, abs=1e-9)


def test_unfilled_parent_is_completed_at_deadline():
    bars = hand_bars(window_vol=[100, 300], window_px=[10.0, 11.0])
    task = Task("t", "execution", "HAND", HISTORY_BARS, n_steps=2, bars_per_step=1, side=-1, qty=50.0)
    env = TradingEnv(task, bars, fill=ZERO_COST, mask=False)
    env.call("end_turn")
    out = env.call("end_turn")  # nothing ordered: all 50 sold at the last step's VWAP
    vwap = (100 * 10 + 300 * 11) / 400
    assert out["done"] and env.state.position == pytest.approx(-50.0)
    assert env.reward == pytest.approx(-((-50 * 11.0 + 50 * vwap) / (50 * vwap) * 1e4))


def test_pnl_reward_by_hand():
    bars = hand_bars(window_vol=[100, 100], window_px=[10.0, 11.0], closes=[10.0, 12.0])
    task = Task("t", "pnl", "HAND", HISTORY_BARS, n_steps=2, bars_per_step=1, unit=1.0)
    env = TradingEnv(task, bars, fill=FillModel(half_spread_bps=1.0, impact_bps=0.0), mask=False)
    env.call("place_order", side="buy", qty=1.0)
    env.call("end_turn")
    env.call("end_turn")
    equity = -10.0 * (1 + 1e-4) + 12.0
    assert env.reward == pytest.approx(equity / 10.0 * 1e4, abs=1e-9)


def test_vwap_oracle_has_zero_shortfall_without_costs():
    bars, tasks = synth_tasks("execution", 20)
    for t in tasks:
        r, _, _ = run_episode(t, bars, "vwap_oracle", seed=0, fill=ZERO_COST)
        assert abs(r) < 1e-8


@pytest.mark.skipif(not (DATA / "BTCUSDT.npz").exists(), reason="real data not downloaded")
def test_vwap_oracle_zero_shortfall_on_real_data():
    from tradeenv.data import load_all
    data = load_all(["BTCUSDT", "ETHUSDT"])
    for t in sample_tasks(data, "execution", "test", 20, seed=1):
        r, _, _ = run_episode(t, data[t.symbol], "vwap_oracle", seed=0, fill=ZERO_COST)
        assert abs(r) < 1e-8


# ----- determinism and isolation --------------------------------------------------------
@pytest.mark.parametrize("family,policy", [("execution", "random_schedule"), ("pnl", "direction_q0.55")])
def test_same_seed_same_hash(family, policy):
    bars, tasks = synth_tasks(family, 5)
    for t in tasks:
        a = run_episode(t, bars, policy, seed=7)
        b = run_episode(t, bars, policy, seed=7)
        c = run_episode(t, bars, policy, seed=8)
        assert a == b
        assert a[1] != c[1]


def _scripted_actions(env, steps):
    for s in steps:
        env.call("get_bars", n=3)
        env.call("place_order", side="buy", qty=1.0 + s)
        env.call("end_turn")


def test_snapshot_restore_replays_identically():
    bars, tasks = synth_tasks("execution", 3)
    for t in tasks:
        env = TradingEnv(t, bars)
        _scripted_actions(env, range(4))
        snap = env.snapshot()
        _scripted_actions(env, range(4, t.n_steps))
        first = (env.trajectory_hash(), env.reward)
        env.restore(snap)
        _scripted_actions(env, range(4, t.n_steps))
        assert (env.trajectory_hash(), env.reward) == first


def test_rollouts_are_isolated():
    bars, tasks = synth_tasks("execution", 1)
    a, b = TradingEnv(tasks[0], bars), TradingEnv(tasks[0], bars)
    a.call("place_order", side="buy", qty=5.0)
    a.call("end_turn")
    assert b.call("get_portfolio")["position"] == 0.0 and b.state.step == 0


def test_sampler_is_deterministic(monkeypatch):
    import datetime as dt
    from tradeenv import tasks as T
    monkeypatch.setitem(T.SPLITS, "all", (dt.datetime(2023, 1, 1), dt.datetime(2030, 1, 1)))
    data = {"SYN": synthetic_bars(n=HISTORY_BARS + 4000, seed=3)}
    a = sample_tasks(data, "execution", "all", 10, seed=5)
    b = sample_tasks(data, "execution", "all", 10, seed=5)
    assert a == b and len(set(t.task_id for t in a)) == 10


# ----- no lookahead -----------------------------------------------------------------------
def test_get_bars_never_returns_future_bars():
    bars, tasks = synth_tasks("pnl", 1)
    t = tasks[0]
    env = TradingEnv(t, bars, mask=False)
    for s in range(t.n_steps):
        rows = env.call("get_bars", n=5)["bars"]
        last = rows[-1]
        assert last[0] == s * t.bars_per_step - 1
        assert last[4] == bars.close[t.start + s * t.bars_per_step - 1]
        env.call("end_turn")


def test_volume_profile_uses_only_past_days():
    bars, tasks = synth_tasks("execution", 1)
    t = tasks[0]
    env = TradingEnv(t, bars)
    prof = env.call("get_volume_profile", days=1)["expected_step_volume"]
    past = bars.volume[t.start - 1440:t.start - 1440 + 60].reshape(12, 5).sum(axis=1)
    assert np.allclose(prof, past)


def test_pnl_position_limit_and_invalid_orders():
    bars, tasks = synth_tasks("pnl", 1)
    t = tasks[0]
    env = TradingEnv(t, bars)
    assert "error" in env.call("place_order", side="buy", qty=2 * t.unit)
    assert "error" in env.call("place_order", side="hold", qty=1.0)
    assert "order_id" in env.call("place_order", side="sell", qty=t.unit)


# ----- GRPO facts ---------------------------------------------------------------------------
def test_group_advantage_invariant_to_hindsight_regret():
    rng = np.random.default_rng(0)
    r = rng.normal(size=(100, 8))
    best = rng.normal(size=(100, 1)) + 5
    scale = rng.uniform(0.1, 10, size=(100, 1))
    assert np.allclose(group_advantage(r), group_advantage((r - best) / scale), atol=1e-9)


def test_mean_advantage_order_equals_mean_reward_order():
    rng = np.random.default_rng(1)
    r = rng.normal(size=(1000, 8))
    adv = group_advantage(r)
    assert np.array_equal(adv[:, :4].mean(1) > adv[:, 4:].mean(1), r[:, :4].mean(1) > r[:, 4:].mean(1))


def test_policy_registry_complete():
    assert set(POLICIES) == {"execution", "pnl"}
