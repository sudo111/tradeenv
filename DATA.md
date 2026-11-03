# Data

The pilot uses Binance spot 1-minute klines from Binance's public data site, data.binance.vision.

- Symbols: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT.
- Months: 2026-07, 2026-08, 2026-09 (UTC). 132,480 bars per symbol, no gaps.
- Each monthly zip is checked against Binance's published SHA-256 checksum by `scripts/download_data.py`.
- Fields used: open time, open, high, low, close, base volume, quote volume. Bar VWAP = quote volume / base volume.

Terms. The binance/binance-public-data TERMS_AND_CONDITIONS.md (version 1.0, 26 Aug 2026, read on 2026-10-27) says
the datasets are provided under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license
(Sec 3.1) and that "Permitted non-commercial uses are limited to academic research, ..." (Sec 4.1).

What this repo does: it does not redistribute raw or derived data files. `data/` is git-ignored; run the download
script to fetch the data yourself. Aggregate statistics computed from the data appear in `results/`.

Attribution: market data from Binance (data.binance.vision), CC BY-NC-SA 4.0. Binance does not endorse this project.
