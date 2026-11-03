#!/usr/bin/env bash
# One command: create a venv, download data, run tests, reproduce every pilot number and figure.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/pip install -q -e ".[dev]"
.venv/bin/python scripts/download_data.py
.venv/bin/python -m pytest -q
.venv/bin/python experiments/run_pilot.py
