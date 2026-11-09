#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python3}"
cd "$ROOT"

"$PY" experiments_icde/run_revision_experiments.py --profile revision2 \
  2>&1 | tee revision2.log

bash experiments_icde/collect_revision2_results.sh \
  /root/skc_icde/icde_revision2_results.tar.gz
