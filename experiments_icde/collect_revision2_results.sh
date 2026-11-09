#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-/root/skc_icde/icde_revision2_results_$(date +%Y%m%d_%H%M%S).tar.gz}"
cd "$ROOT"

mkdir -p results/systems/a3 results/systems/f4 results/systems/f5 \
  results/systems/f6 results/real
tar -czf "$OUT" \
  results/systems/a3 \
  results/systems/f4 \
  results/systems/f5 \
  results/systems/f6 \
  results/systems/dai2_pageledger_diginetica.json \
  results/systems/dai2_pageledger_gowalla.json \
  results/real/diginetica_a3_cotrain_seed0.pt \
  results/real/gowalla_a3_cotrain_seed0.pt \
  revision2.log \
  2>/dev/null
sha256sum "$OUT" > "${OUT}.sha256"
echo "$OUT"
