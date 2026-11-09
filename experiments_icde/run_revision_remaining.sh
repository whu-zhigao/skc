#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python3}"
cd "$ROOT"

"$PY" experiments_icde/exp_f4_integrated.py \
  --data "$ROOT/data/diginetica.pkl" \
  --n-query 3000 --drift-items 80 --evict-after-read --force
"$PY" experiments_icde/exp_f4_integrated.py \
  --data "$ROOT/data/gowalla.pkl" \
  --n-query 3000 --drift-items 80 --evict-after-read --force

"$PY" experiments_icde/exp_f5_contract_power.py \
  --data "$ROOT/data/diginetica.pkl" --n-query 0 --batch-size 512 --force
"$PY" experiments_icde/exp_f5_contract_power.py \
  --data "$ROOT/data/gowalla.pkl" --n-query 10000 --batch-size 512 --force

"$PY" experiments_icde/exp_f6_sla_repeated.py \
  --data "$ROOT/data/diginetica.pkl" --n-query 1000 --requests 2000 \
  --traces 30 --target-qps 100 --batch-size 32 \
  --maintenance-interval-ms 500 --dirty-window-ms 2 \
  --drift-trace "$ROOT/results/systems/drift_trace_diginetica.npz" --force
"$PY" experiments_icde/exp_f6_sla_repeated.py \
  --data "$ROOT/data/gowalla.pkl" --n-query 1000 --requests 2000 \
  --traces 30 --target-qps 100 --batch-size 32 \
  --maintenance-interval-ms 500 --dirty-window-ms 2 \
  --drift-trace "$ROOT/results/systems/drift_trace_gowalla.npz" --force
