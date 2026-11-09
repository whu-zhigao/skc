#!/usr/bin/env bash
# Thin wrapper: full-matrix parallel run via run_parallel.py.
#
# Usage:
#   bash run_all_parallel.sh --smoke        # 1 dataset (diginetica),
#                                          # 1 seed, tiny caps
#                                          # (SMOKE_FLAGS in
#                                          # run_parallel.py); minutes
#   bash run_all_parallel.sh                # full paper matrix:
#                                          # 14 stages x {diginetica,
#                                          # gowalla} x seeds {0,1,2}
#   bash run_all_parallel.sh --jobs a1,b1 --datasets gowalla
#   OLD_CKPT=results/real/abl_no_anchor_diginetica.pt \
#     bash run_all_parallel.sh --jobs b7,b4 --datasets diginetica
#   bash run_all_parallel.sh --force        # rerun existing configs
#
# Notes:
# - Results land in <repo>/code/results/systems/<exp>/<tag>.json;
#   existing configs are skipped automatically (idempotent resume).
# - Logs per job in <repo>/code/logs/<tag>.log; manifest JSON in
#   results/systems/parallel_manifest_<ts>.json.
# - b4/b7 need OLD_CKPT (earlier-generation checkpoint); without it
#   those jobs are skipped with a log entry, not an error.
# - Path tip: this repo's root contains a space
#   ("Session Recommendation-WWW"); ALWAYS quote paths in shell.
set -euo pipefail
SUITE="$(cd "$(dirname "$0")" && pwd)"
if [ "${1:-}" = "--smoke" ]; then
  shift
  exec python3 "$SUITE/run_parallel.py" --smoke "$@"
fi
exec python3 "$SUITE/run_parallel.py" "$@"
