# tradeenv: a multi-turn trading environment for RL training of LLM agents (pilot)

A small, deterministic environment in which an agent trades over historical market data through a tool API,
in the spirit of Spreadsheet-RL's spreadsheet gym (arXiv 2605.22642). The agent calls tools (look at bars,
check its portfolio, place and cancel orders, let time pass) over many turns, holds inventory across decisions,
and gets a reward that is computed exactly from the recorded data.

This repository is a pilot. It contains the environment, scripted baseline policies, tests, and one experiment:
how well the reward ranks two policies of known relative quality inside a GRPO group. No LLM is trained here.

## One command

```bash
./run_all.sh
```

This creates `.venv`, installs the package, downloads about 30 MB of Binance 1-minute bars (checksums verified),
runs the tests, and writes every pilot number and figure to `results/`. Python 3.11 or newer.
The full experiment takes under a minute on a 14-core laptop. Timing numbers (throughput) vary by machine and run;
every other number is exactly reproducible from the seed in `experiments/run_pilot.py`.

## What is in it

```
tradeenv/
  data.py       load bars; synthetic bars for tests
  engine.py     BarReplay: the bar-replay runtime and its fill model (the "Excel" of this gym)
  tasks.py      task families, time split, deterministic task sampler, hindsight (for calibration only)
  env.py        TradingEnv: the tool API, trajectory hashing, snapshot and restore
  policies.py   scripted policies that act only through the tool API
  grpo.py       group-relative advantage and ranking measurements
scripts/download_data.py    fetch and verify Binance Vision klines
experiments/run_pilot.py    reproduce every number and figure
tests/test_env.py           determinism, reward correctness, no lookahead, GRPO invariance
```

### Tool API (`TradingEnv.call(name, **args)`, schemas in `env.TOOL_SCHEMAS`)

| tool | arguments | returns |
|---|---|---|
| get_task | none | family, instructions, size, steps left |
| get_bars | n (1 to 1440) | last n closed 1-minute bars, time relative to the window start |
| get_volume_profile | days (1 to 7) | expected volume per decision step from past days at the same time of day |
| get_portfolio | none | cash, position, progress, open orders, steps left |
| place_order | side, qty, order_type (market or limit), limit_price | order id or an error |
| cancel | order_id | ok |
| end_turn | none | fills during the next decision step; reward when the episode ends |

By default prices are rescaled to 100 at the window start and the asset is named ASSET (masking against
memorized price levels and dates).

### Task families

- **execution**: buy or sell a parent order (5% of recent volume) within 60 minutes, one decision every 5 minutes
  (12 steps). Reward = minus the shortfall versus the window's VWAP, in basis points. Unfilled quantity is
  completed at the deadline without the participation cap.
- **pnl**: hold between -1 and +1 unit for 4 hours, one decision every 15 minutes (16 steps). Reward = final
  profit and loss after costs, in basis points of one unit's starting value.

Windows are non-overlapping and split by time: train July to August 2026, validation September 1 to 15,
test September 16 to 30 (execution: 6,600 / 1,800 / 1,795 windows; pnl: 1,650 / 450 / 445).

### Fill model (assumptions, not estimates of a real venue)

Market orders fill at the decision step's exact VWAP plus a 1 bp half-spread plus temporary impact of 50 bp times
participation (order size / step volume). At most 25% of a step's volume can be taken; the rest keeps working.
Limit orders fill at the limit price only if the step trades through it. Prices never react to the agent:
replay has no permanent market impact. The experiment also reports results with impact 0 and 10 bp.

## Determinism and systems properties (from `results/pilot_results.json`)

- Same task, seed and actions give an identical SHA-256 trajectory hash: 300 of 300 episodes on rerun and
  300 of 300 when run in a 14-process pool versus a single process; a different seed changed the hash in 300 of 300.
- Snapshot and restore mid-episode, then replaying the same actions, reproduced the hash and reward in 150 of 150.
- Throughput with scripted policies on an Apple M3 Max: about 2,900 execution episodes per second on one core and
  about 30,600 with 14 processes (about 49 tool calls per episode).

## Pilot result in one paragraph

Within a GRPO group all rollouts replay the same window, so shared market moves cancel. What remains decides
whether the reward ranks a better policy above a worse one. With the fill model above, execution tasks ranked a
schedule with a better volume forecast above a worse one in 83% of groups of size 8, and 2 groups sufficed for the
summed signal to point the right way 90% of the time. For directional P&L, a policy that knows the next 15-minute
direction 55% of the time beat a random policy in only 67% of groups of size 8 (9 groups needed; 168 for a 51%
policy), even though it has the higher expected reward on every one of the 1,000 windows (computed exactly). The
execution signal comes from the impact model: with impact set to zero, the same pairs are ranked at chance.
Details: `results/summary.md`.

## Tests

```bash
.venv/bin/python -m pytest -q
```

## Data and license

Code: MIT (see LICENSE). Data: Binance public data, CC BY-NC-SA 4.0, not redistributed here; see DATA.md.
