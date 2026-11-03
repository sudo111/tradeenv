"""Load 1-minute bars produced by scripts/download_data.py."""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data"


@dataclass(frozen=True)
class Bars:
    """Contiguous 1-minute bars for one symbol. All arrays have the same length."""

    symbol: str
    ts: np.ndarray       # unix seconds, bar open time
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray   # base-asset volume
    quote: np.ndarray    # quote-asset volume (sum of price x size), so quote/volume is the bar VWAP

    def __len__(self) -> int:
        return len(self.ts)


def load_symbol(symbol: str, data_dir: pathlib.Path | str = DATA_DIR) -> Bars:
    arr = np.load(pathlib.Path(data_dir) / f"{symbol}.npz")["bars"]
    for col in arr.T:
        col.setflags(write=False)
    return Bars(symbol=symbol, ts=arr[:, 0].astype(np.int64), open=arr[:, 1], high=arr[:, 2],
                low=arr[:, 3], close=arr[:, 4], volume=arr[:, 5], quote=arr[:, 6])


def load_all(symbols: list[str], data_dir: pathlib.Path | str = DATA_DIR) -> dict[str, Bars]:
    return {s: load_symbol(s, data_dir) for s in symbols}


def synthetic_bars(n: int = 3000, seed: int = 0, symbol: str = "SYN") -> Bars:
    """Deterministic random-walk bars for tests that must not depend on downloaded data."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
    open_ = np.concatenate([[100.0], close[:-1]])
    spread = np.abs(rng.normal(0, 5e-4, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.lognormal(3.0, 0.5, n)
    vwap = np.clip((open_ + close + high + low) / 4, low, high)
    return Bars(symbol=symbol, ts=(1_700_000_000 + 60 * np.arange(n)).astype(np.int64), open=open_,
                high=high, low=low, close=close, volume=volume, quote=vwap * volume)
