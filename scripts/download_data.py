"""Download Binance Vision spot 1-minute klines and convert them to compact .npz files.

Data source: https://data.binance.vision (Binance public data).
Terms: CC BY-NC-SA 4.0; permitted non-commercial uses include academic research
(binance/binance-public-data TERMS_AND_CONDITIONS.md, Sec 3.1 and 4.1).
Raw and derived files are kept out of git; see DATA.md.

Usage: python scripts/download_data.py [--symbols BTCUSDT ...] [--months 2026-07 ...]
"""
import argparse
import csv
import hashlib
import io
import pathlib
import sys
import urllib.request
import zipfile

import numpy as np

BASE = "https://data.binance.vision/data/spot/monthly/klines/{sym}/1m/{sym}-1m-{month}.zip"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
DEFAULT_MONTHS = ["2026-07", "2026-08", "2026-09"]
ROOT = pathlib.Path(__file__).resolve().parents[1]


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "tradeenv-research/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def download(sym: str, month: str, raw_dir: pathlib.Path) -> pathlib.Path:
    url = BASE.format(sym=sym, month=month)
    path = raw_dir / f"{sym}-1m-{month}.zip"
    if not path.exists():
        path.write_bytes(fetch(url))
    expected = fetch(url + ".CHECKSUM").decode().split()[0]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        path.unlink()
        raise RuntimeError(f"checksum mismatch for {path.name}")
    return path


def parse(zpath: pathlib.Path) -> np.ndarray:
    """Return an (n, 8) float64 array: ts_sec, open, high, low, close, volume, quote_volume, trades."""
    with zipfile.ZipFile(zpath) as zf:
        name = zf.namelist()[0]
        text = io.TextIOWrapper(zf.open(name), encoding="ascii")
        rows = []
        for rec in csv.reader(text):
            if not rec or not rec[0].isdigit():
                continue  # skip a header line if present
            t = int(rec[0])
            # Spot files from 2025 on use microseconds; older files use milliseconds.
            ts = t // 1_000_000 if t > 10**14 else t // 1000
            rows.append([ts, float(rec[1]), float(rec[2]), float(rec[3]), float(rec[4]),
                         float(rec[5]), float(rec[7]), float(rec[8])])
    return np.asarray(rows, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    ap.add_argument("--months", nargs="+", default=DEFAULT_MONTHS)
    ap.add_argument("--out", default=str(ROOT / "data"))
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for sym in args.symbols:
        parts = [parse(download(sym, m, raw_dir)) for m in args.months]
        arr = np.concatenate(parts)
        arr = arr[np.argsort(arr[:, 0], kind="stable")]
        np.savez_compressed(out / f"{sym}.npz", bars=arr, months=np.array(args.months))
        print(f"{sym}: {len(arr)} bars, {args.months[0]}..{args.months[-1]}", file=sys.stderr)


if __name__ == "__main__":
    main()
