#!/usr/bin/env bash
# GPU-scale versions of the Round-13 experiments (run on the cloud node).
# Full-protocol consumer replication + large-window DAI envelope.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python3}
echo "[cloud-r13] full-protocol consumer replication (12 epochs co-training protocol)"
for R in 1 inf; do
  $PY experiments_icde/exp_b8_consumer_replication.py --epochs 12 --steps 300 --batch-size 1024 --refresh $R --seeds 0,1,2 2>&1 | tail -3
done
echo "[cloud-r13] DAI large-window envelope"
$PY experiments_icde/exp_b9_dai2_dispersion.py --windows 64,256,1024 2>&1 | tail -3
echo "[cloud-r13] done -> results/systems/"
