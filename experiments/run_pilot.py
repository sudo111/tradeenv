"""Reproduce every number and figure in the pilot write-up.

Usage:  python experiments/run_pilot.py            (full run, a few minutes on 14 cores)
        python experiments/run_pilot.py --quick    (small K, for a smoke test)

Writes results/pilot_results.json, results/summary.md, results/fig1_rank_vs_group.png,
results/fig2_groups_needed.png. All randomness is seeded from SEED.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pathlib
import platform
import time
import zlib

import numpy as np

from tradeenv import grpo
from tradeenv.data import load_all
from tradeenv.engine import FillModel
from tradeenv.env import TradingEnv
from tradeenv.policies import POLICIES, run_episode
from tradeenv.tasks import FAMILIES, candidate_starts, hindsight, sample_tasks

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
SEED = 20261102
GROUP_SIZES = [2, 4, 8, 16, 32, 64]
FILLS = {"base": FillModel(), "impact0": FillModel(impact_bps=0.0), "impact10": FillModel(impact_bps=10.0)}
SENSITIVITY_POLICIES = ["vwap_oracle", "vwap_noise_0.25", "vwap_noise_1.0", "twap", "front_load"]

EXEC_PAIRS = [("vwap_oracle", "twap"), ("vwap_noise_0.25", "vwap_noise_1.0"), ("twap", "front_load"),
              ("volume_profile", "twap"), ("twap", "random_schedule")]
PNL_PAIRS = [("direction_oracle", "buy_hold"), ("direction_q0.60", "random_q0.50"),
             ("direction_q0.55", "random_q0.50"), ("direction_q0.53", "random_q0.50"),
             ("direction_q0.51", "random_q0.50"), ("momentum", "random_q0.50")]
FIG1_PAIRS = {"execution": [("vwap_noise_0.25", "vwap_noise_1.0"), ("twap", "random_schedule"), ("twap", "front_load")],
              "pnl": [("direction_q0.60", "random_q0.50"), ("direction_q0.55", "random_q0.50"),
                      ("direction_q0.53", "random_q0.50")]}

_G: dict = {}


def seed_for(*parts) -> int:
    return (SEED + zlib.crc32("|".join(map(str, parts)).encode())) % (2**32)


def _init(tasks):
    _G["data"] = load_all(SYMBOLS)
    _G["tasks"] = tasks


def _job(args):
    fam, i, pol, j, fkey = args
    t = _G["tasks"][fam][i]
    r, h, n = run_episode(t, _G["data"][t.symbol], pol, seed_for(fam, i, pol, j), FILLS[fkey])
    return fam, i, pol, j, fkey, r, h, n


def collect(pool, tasks, k, r_stoch):
    jobs = []
    for fam in ("execution", "pnl"):
        for pol, (_, stochastic) in POLICIES[fam].items():
            for i in range(k):
                for j in range(r_stoch if stochastic else 1):
                    jobs.append((fam, i, pol, j, "base"))
    for fkey in ("impact0", "impact10"):
        for pol in SENSITIVITY_POLICIES:
            stochastic = POLICIES["execution"][pol][1]
            for i in range(k):
                for j in range(r_stoch if stochastic else 1):
                    jobs.append(("execution", i, pol, j, fkey))
    rewards: dict = {}
    t0 = time.perf_counter()
    for fam, i, pol, j, fkey, r, _, _ in pool.imap_unordered(_job, jobs, chunksize=256):
        key = (fam, fkey, pol)
        if key not in rewards:
            width = r_stoch if POLICIES[fam][pol][1] else 1
            rewards[key] = np.full((k, width), np.nan)
        rewards[key][i, j] = r
    elapsed = time.perf_counter() - t0
    assert all(not np.isnan(v).any() for v in rewards.values())
    return rewards, len(jobs), elapsed


def pair_stats(ra, rb, rng, quick):
    d = ra.mean(1) - rb.mean(1)
    unpaired_sd = float(np.sqrt(ra.mean(1).var(ddof=1) + rb.mean(1).var(ddof=1)))
    out = {
        "mean_gap_bps": float(d.mean()),
        "paired_sd_bps": float(d.std(ddof=1)),
        "unpaired_sd_bps": unpaired_sd,
        "path_ceiling": grpo.path_ceiling(ra, rb),
        "rank_prob_by_G": {g: grpo.rank_probability(ra, rb, g, 10 if quick else 50, rng) for g in GROUP_SIZES},
        "k90_paired_normal": grpo.normal_k90(float(d.mean()), float(d.std(ddof=1))),
        "k90_unpaired_normal": grpo.normal_k90(float(d.mean()), unpaired_sd),
    }
    gaps = grpo.advantage_gaps(ra, rb, 8, 4 if quick else 16, rng)
    out["k90_groups_bootstrap_G8"] = grpo.groups_for_90(gaps, rng, n_boot=500 if quick else 2000)
    return out


def determinism_checks(pool, tasks, n):
    data = _G["data"]
    eps = [("execution", i, "random_schedule", 0, "base") for i in range(n)] + \
          [("pnl", i, "direction_q0.55", 0, "base") for i in range(n)]
    first = [_job(e)[6] for e in eps]
    second = [_job(e)[6] for e in eps]
    pooled = [x[6] for x in pool.map(_job, eps, chunksize=16)]
    other_seed = [_job((f, i, p, 1, k))[6] for (f, i, p, _, k) in eps]
    # Snapshot and restore mid-episode with a fixed action script.
    snap_ok = 0
    for t in tasks["execution"][:n]:
        env = TradingEnv(t, data[t.symbol])
        q = env.call("get_task")["qty"] / t.n_steps
        side = env.call("get_task")["side"]

        def act(steps):
            for _ in steps:
                env.call("get_bars", n=5)
                env.call("place_order", side=side, qty=q)
                env.call("end_turn")
        act(range(5))
        snap = env.snapshot()
        act(range(5, t.n_steps))
        h1 = (env.trajectory_hash(), env.reward)
        env.restore(snap)
        act(range(5, t.n_steps))
        snap_ok += h1 == (env.trajectory_hash(), env.reward)
    return {
        "episodes": len(eps),
        "rerun_identical": sum(a == b for a, b in zip(first, second)),
        "pool_vs_single_identical": sum(a == b for a, b in zip(first, pooled)),
        "different_seed_differs": sum(a != b for a, b in zip(first, other_seed)),
        "snapshot_restore_identical": snap_ok, "snapshot_restore_episodes": n,
    }


def throughput(pool, tasks, quick):
    out = {}
    n1 = 100 if quick else 1000
    for fam, pol in (("execution", "twap"), ("pnl", "momentum")):
        eps = [(fam, i % len(tasks[fam]), pol, 0, "base") for i in range(n1)]
        t0 = time.perf_counter()
        res = [_job(e) for e in eps]
        dt = time.perf_counter() - t0
        calls = sum(r[7] for r in res)
        steps = n1 * FAMILIES[fam]["n_steps"]
        out[f"{fam}_single"] = {"episodes_per_s": n1 / dt, "tool_calls_per_s": calls / dt, "env_steps_per_s": steps / dt,
                                "tool_calls_per_episode": calls / n1}
        n_par = n1 * 20
        eps = [(fam, i % len(tasks[fam]), pol, 0, "base") for i in range(n_par)]
        t0 = time.perf_counter()
        res = pool.map(_job, eps, chunksize=256)
        dt = time.perf_counter() - t0
        calls = sum(r[7] for r in res)
        out[f"{fam}_parallel"] = {"workers": pool._processes, "episodes_per_s": n_par / dt,
                                  "tool_calls_per_s": calls / dt,
                                  "env_steps_per_s": n_par * FAMILIES[fam]["n_steps"] / dt}
    # Observation size per episode (characters of tool results; tokens ~ chars / 4 is an estimate).
    data = _G["data"]
    for fam, pol in (("execution", "volume_profile"), ("pnl", "momentum")):
        chars = []
        for t in tasks[fam][:200]:
            env = TradingEnv(t, data[t.symbol])
            env.count_chars = True
            POLICIES[fam][pol][0](env, np.random.default_rng(0), None)
            chars.append(env.obs_chars)
        out[f"{fam}_obs_chars_per_episode"] = float(np.mean(chars))
    return out


def analytic_pnl(task, bars, q: float, fill: FillModel) -> float:
    """Exact expected reward (bps) of the noisy-direction policy with accuracy q on one window.

    Equity = sum_s pos_s * (p_{s+1} - p_s) - trading costs, with p_s the step VWAP and p_n the
    final close. E[pos_s] = (2q - 1) * sign(move_s) * unit. The position flips between steps with
    probability 2q(1 - q) when the true direction repeats, q^2 + (1 - q)^2 when it changes, and
    1/2 when either direction is a tie (then the policy's position is a fair coin).
    """
    hs = hindsight(task, bars)
    unit, vw = task.unit, hs.step_vwap
    move = np.append(vw[1:], hs.step_close[-1]) - vw
    truth = np.sign(move)
    gain = (2 * q - 1) * np.abs(move).sum() * unit

    def cost(k, s):
        qty = k * unit
        return qty * vw[s] * (fill.half_spread_bps + fill.impact_bps * qty / hs.step_volume[s]) / 1e4
    total_cost = cost(1, 0)
    for s in range(1, task.n_steps):
        if truth[s] == 0 or truth[s - 1] == 0:
            p_flip = 0.5
        elif truth[s] == truth[s - 1]:
            p_flip = 2 * q * (1 - q)
        else:
            p_flip = q * q + (1 - q) * (1 - q)
        total_cost += p_flip * cost(2, s)
    return float((gain - total_cost) / (unit * bars.close[task.start - 1]) * 1e4)


def analytic_block(rewards, tasks):
    data = _G["data"]
    qs = {"direction_oracle": 1.0, "direction_q0.60": 0.60, "direction_q0.55": 0.55, "direction_q0.53": 0.53,
          "direction_q0.51": 0.51, "random_q0.50": 0.50}
    exp = {pol: np.array([analytic_pnl(t, data[t.symbol], q, FILLS["base"]) for t in tasks["pnl"]])
           for pol, q in qs.items()}
    out = {"agreement": {}, "ceiling": {}}
    for pol in qs:
        emp = rewards[("pnl", "base", pol)]
        se = emp.std(axis=1, ddof=1) / np.sqrt(emp.shape[1]) if emp.shape[1] > 1 else np.zeros(len(emp))
        diff = emp.mean(axis=1) - exp[pol]
        z = np.where(se > 0, diff / np.where(se > 0, se, 1), 0.0)
        out["agreement"][pol] = {"mean_abs_diff_bps": float(np.abs(diff).mean()),
                                 "corr": float(np.corrcoef(emp.mean(axis=1), exp[pol])[0, 1]),
                                 "frac_abs_z_below_3": float((np.abs(z) < 3).mean())}
    for a in ("direction_q0.60", "direction_q0.55", "direction_q0.53", "direction_q0.51"):
        d = exp[a] - exp["random_q0.50"]
        out["ceiling"][f"{a} > random_q0.50"] = float(((d > 0).sum() + 0.5 * (d == 0).sum()) / len(d))
    return out


def regret_check(rewards, rng):
    out = {}
    for fam, oracle, a, b, ref in (("execution", "vwap_oracle", "vwap_noise_0.25", "vwap_noise_1.0", "twap"),
                                   ("pnl", "direction_oracle", "direction_q0.55", "random_q0.50", "random_q0.50")):
        rstar = rewards[(fam, "base", oracle)][:, :1]
        grp = np.concatenate([grpo._draw(rewards[(fam, "base", a)], 4, rng),
                              grpo._draw(rewards[(fam, "base", b)], 4, rng)], axis=1)
        diff = np.abs(grpo.group_advantage(grp) - grpo.group_advantage((grp - rstar) / (np.abs(rstar) + 1.0)))
        raw = rewards[(fam, "base", ref)].mean(1)
        out[fam] = {"max_abs_advantage_change": float(diff.max()),
                    "cross_task_sd_raw_bps": float(raw.std(ddof=1)),
                    "cross_task_sd_regret_bps": float((raw - rstar[:, 0]).std(ddof=1)),
                    "reference_policy": ref}
    return out


def make_figures(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    markers = ["o", "s", "^"]
    label = {"vwap_noise_0.25": "VWAP noise 0.25", "vwap_noise_1.0": "noise 1.0", "twap": "TWAP",
             "front_load": "front-load", "volume_profile": "volume profile", "random_schedule": "random schedule", "direction_q0.60": "60% oracle",
             "direction_q0.55": "55% oracle", "direction_q0.53": "53% oracle", "random_q0.50": "random"}
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b"})
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7), sharey=True)
    for ax, fam, title in zip(axes, ("execution", "pnl"), ("Execution vs VWAP", "Directional P&L")):
        for c, m, (a, b) in zip(colors, markers, FIG1_PAIRS[fam]):
            st = res["pairs"][fam][f"{a} > {b}"]
            ys = [st["rank_prob_by_G"][g] for g in GROUP_SIZES]
            ax.plot(GROUP_SIZES, ys, color=c, marker=m, markersize=4, linewidth=2,
                    label=f"{label[a]} vs {label[b]}")
            ceil = res["pnl_analytic"]["ceiling"][f"{a} > {b}"] if fam == "pnl" else st["path_ceiling"]
            ax.axhline(ceil, color=c, linewidth=1, linestyle=":")
        ax.axhline(0.9, color="#8a8984", linewidth=1, linestyle="--")
        ax.set_xscale("log", base=2)
        ax.set_xticks(GROUP_SIZES, [str(g) for g in GROUP_SIZES])
        ax.set_xlabel("GRPO group size G (half from each policy)")
        ax.set_title(title, fontsize=9, loc="left")
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    axes[0].set_ylabel("P(group ranks better policy higher)")
    axes[0].set_ylim(0.45, 1.01)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_rank_vs_group.png", dpi=200)
    plt.close(fig)

    names, vals, fams = [], [], []
    for fam in ("execution", "pnl"):
        for a, b in FIG1_PAIRS[fam]:
            names.append(f"{label[a]} vs {label[b]}")
            vals.append(res["pairs"][fam][f"{a} > {b}"]["k90_groups_bootstrap_G8"])
            fams.append(fam)
    names.append("51% oracle vs random")
    vals.append(res["pairs"]["pnl"]["direction_q0.51 > random_q0.50"]["k90_groups_bootstrap_G8"])
    fams.append("pnl")
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.5, 2.5))
    for fam, c, lab in (("execution", colors[0], "Execution vs VWAP"), ("pnl", colors[1], "Directional P&L")):
        idx = [i for i, f in enumerate(fams) if f == fam]
        ax.barh(y[idx], [vals[i] for i in idx], height=0.6, color=c, label=lab)
    for yi, v in zip(y, vals):
        ax.text(v * 1.08, yi, f"{v:,}", va="center", fontsize=7, color="#0b0b0b")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.8, max(vals) * 3)
    ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])
    ax.set_xlabel("GRPO groups needed (G = 8) for the summed advantage to point the right way 90% of the time")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_groups_needed.png", dpi=200)
    plt.close(fig)


def write_summary(res):
    L = ["# Pilot results (generated by experiments/run_pilot.py)", "", f"Config: {json.dumps(res['config'])}", ""]
    L += ["## Task counts per split", "", "| family | train | val | test | all |", "|---|---|---|---|---|"]
    for fam, c in res["task_counts"].items():
        L.append(f"| {fam} | {c['train']} | {c['val']} | {c['test']} | {c['all']} |")
    L += ["", "## Determinism", "", json.dumps(res["determinism"]), "", "## Throughput", ""]
    for k, v in res["throughput"].items():
        L.append(f"- {k}: {json.dumps(v) if isinstance(v, dict) else round(v, 1)}")
    L += ["", "## Policy rewards (bps; mean and sd across tasks of the per-task mean)", "",
          "| family | fill | policy | mean | sd |", "|---|---|---|---|---|"]
    for k, v in res["policy_rewards"].items():
        L.append(f"| {k} | {v['mean']:.3f} | {v['sd']:.3f} |")
    L += ["", "## Pair statistics", ""]
    for fam, pairs in res["pairs"].items():
        L += [f"### {fam}", "", "| pair | gap bps | paired sd | unpaired sd | ceiling | "
              + " | ".join(f"P(G={g})" for g in GROUP_SIZES) + " | K90 boot G8 | K90 paired normal | K90 unpaired normal |",
              "|---" * (8 + len(GROUP_SIZES)) + "|"]
        for name, st in pairs.items():
            L.append(f"| {name} | {st['mean_gap_bps']:.3f} | {st['paired_sd_bps']:.3f} | {st['unpaired_sd_bps']:.3f} | "
                     f"{st['path_ceiling']:.3f} | " + " | ".join(f"{st['rank_prob_by_G'][g]:.3f}" for g in GROUP_SIZES)
                     + f" | {st['k90_groups_bootstrap_G8']} | {st['k90_paired_normal']:.1f} | {st['k90_unpaired_normal']:.1f} |")
        L.append("")
    L += ["## P&L analytic expectations (exact per-window expected reward of noisy-direction policies)", "",
          json.dumps(res["pnl_analytic"], indent=1), ""]
    L += ["## Hindsight-regret check", "", json.dumps(res["regret"], indent=1), "",
          f"Rollout collection: {res['rollouts']['episodes']} episodes in {res['rollouts']['seconds']:.1f} s"]
    (OUT / "summary.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    k, r_stoch = (100, 8) if args.quick else (1000, 32)
    OUT.mkdir(exist_ok=True)
    data = load_all(SYMBOLS)
    tasks = {fam: sample_tasks(data, fam, "all", k, seed=SEED) for fam in ("execution", "pnl")}
    _init(tasks)  # main process also runs single-process checks
    res = {"config": {"seed": SEED, "K": k, "R_stochastic": r_stoch, "symbols": SYMBOLS, "split": "all",
                      "fills": {n: vars(f) for n, f in FILLS.items()}, "families": FAMILIES,
                      "python": platform.python_version(), "numpy": np.__version__,
                      "machine": platform.machine(), "cpu_count": os.cpu_count(), "workers": args.workers}}
    res["task_counts"] = {fam: {sp: int(sum(len(candidate_starts(data[s], fam, sp)) for s in SYMBOLS))
                                for sp in ("train", "val", "test", "all")} for fam in FAMILIES}
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init, initargs=(tasks,)) as pool:
        res["determinism"] = determinism_checks(pool, tasks, 20 if args.quick else 150)
        res["throughput"] = throughput(pool, tasks, args.quick)
        rewards, n_eps, secs = collect(pool, tasks, k, r_stoch)
    res["rollouts"] = {"episodes": n_eps, "seconds": secs}
    res["policy_rewards"] = {f"{fam} | {fkey} | {pol}": {"mean": float(v.mean()), "sd": float(v.mean(1).std(ddof=1))}
                             for (fam, fkey, pol), v in sorted(rewards.items())}
    rng = np.random.default_rng(SEED)
    res["pairs"] = {"execution": {}, "pnl": {}, "execution_impact0": {}, "execution_impact10": {}}
    for a, b in EXEC_PAIRS:
        res["pairs"]["execution"][f"{a} > {b}"] = pair_stats(rewards[("execution", "base", a)],
                                                              rewards[("execution", "base", b)], rng, args.quick)
    for a, b in PNL_PAIRS:
        res["pairs"]["pnl"][f"{a} > {b}"] = pair_stats(rewards[("pnl", "base", a)], rewards[("pnl", "base", b)],
                                                        rng, args.quick)
    for fkey in ("impact0", "impact10"):
        for a, b in (("vwap_noise_0.25", "vwap_noise_1.0"), ("twap", "front_load"), ("vwap_oracle", "twap")):
            res["pairs"][f"execution_{fkey}"][f"{a} > {b}"] = pair_stats(
                rewards[("execution", fkey, a)], rewards[("execution", fkey, b)], rng, args.quick)
    res["regret"] = regret_check(rewards, rng)
    res["pnl_analytic"] = analytic_block(rewards, tasks)
    (OUT / "pilot_results.json").write_text(json.dumps(res, indent=1, default=float))
    write_summary(res)
    make_figures(res)
    print(f"done: {n_eps} episodes in {secs:.1f} s; results in {OUT}")


if __name__ == "__main__":
    main()
