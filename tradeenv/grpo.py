"""Measure how well a reward ranks two policies under GRPO's group-relative advantage.

Setting: G rollouts per task form one GRPO group, and all of them replay the same market
window. Half come from policy A (better in expectation), half from policy B.

Facts used here (provable, also checked in tests):
  * Within a group, advantage = (r - mean) / std is a strictly increasing function of r, so
    "mean advantage of A > mean advantage of B" is the same event as "mean reward of A >
    mean reward of B".
  * Advantages are unchanged by any per-group transform r -> a*r + b with a > 0. Subtracting
    the hindsight-optimal reward of the task (hindsight regret) is such a transform.
"""
from __future__ import annotations

import numpy as np

Z90 = 1.2815515655446004  # standard normal 0.9 quantile


def group_advantage(r: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    return (r - r.mean(axis=-1, keepdims=True)) / (r.std(axis=-1, keepdims=True) + eps)


def _draw(r: np.ndarray, h: int, rng: np.random.Generator) -> np.ndarray:
    """Draw h rollouts per task (rows). Without replacement when the pool is large enough."""
    k, n = r.shape
    if n == 1:
        return np.repeat(r, h, axis=1)
    if n >= h:
        idx = rng.random((k, n)).argsort(axis=1)[:, :h]
    else:
        idx = rng.integers(0, n, size=(k, h))
    return np.take_along_axis(r, idx, axis=1)


def rank_probability(ra: np.ndarray, rb: np.ndarray, g: int, n_resample: int, rng: np.random.Generator) -> float:
    """P(group ranks A above B) for groups of size g (g/2 rollouts each). Ties count half."""
    h = g // 2
    wins = 0.0
    for _ in range(n_resample):
        ma, mb = _draw(ra, h, rng).mean(axis=1), _draw(rb, h, rng).mean(axis=1)
        wins += (ma > mb).sum() + 0.5 * (ma == mb).sum()
    return float(wins / (n_resample * ra.shape[0]))


def path_ceiling(ra: np.ndarray, rb: np.ndarray) -> float:
    """Fraction of tasks on which A beats B in expectation over policy randomness: the limit of
    rank_probability as g grows, because a group never sees more than one market path."""
    ma, mb = ra.mean(axis=1), rb.mean(axis=1)
    return float(((ma > mb).sum() + 0.5 * (ma == mb).sum()) / len(ma))


def advantage_gaps(ra: np.ndarray, rb: np.ndarray, g: int, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """(tasks, n_draws) array of mean-advantage(A) minus mean-advantage(B) in random groups."""
    h = g // 2
    out = np.empty((ra.shape[0], n_draws))
    for m in range(n_draws):
        grp = np.concatenate([_draw(ra, h, rng), _draw(rb, h, rng)], axis=1)
        adv = group_advantage(grp)
        out[:, m] = adv[:, :h].mean(axis=1) - adv[:, h:].mean(axis=1)
    return out


def groups_for_90(gaps: np.ndarray, rng: np.random.Generator, n_boot: int = 2000, k_max: int = 100_000) -> int:
    """Smallest number of groups K such that the summed advantage gap over K random groups
    (tasks drawn with replacement) is positive with probability >= 0.9. Bootstrap search on a
    doubling grid, then refined by bisection."""
    flat_t, flat_m = gaps.shape

    def p_pos(k: int) -> float:
        ti = rng.integers(0, flat_t, size=(n_boot, k))
        mi = rng.integers(0, flat_m, size=(n_boot, k))
        return float((gaps[ti, mi].sum(axis=1) > 0).mean())

    if p_pos(1) >= 0.9:
        return 1
    lo, hi = 1, 2
    while p_pos(hi) < 0.9:
        lo, hi = hi, hi * 2
        if hi > k_max:
            return k_max + 1  # effectively never at this budget
    while hi - lo > max(1, lo // 20):
        mid = (lo + hi) // 2
        if p_pos(mid) >= 0.9:
            hi = mid
        else:
            lo = mid
    return hi


def normal_k90(mean: float, sd: float) -> float:
    """Normal approximation: K with P(sum of K i.i.d. draws > 0) = 0.9."""
    if mean <= 0:
        return float("inf")
    return float((Z90 * sd / mean) ** 2)
