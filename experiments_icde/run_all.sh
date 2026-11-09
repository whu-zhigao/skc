#!/usr/bin/env bash
# Orchestrates the ICDE experiment suite. Auto-locates the repo code/
# root; override with SKC_CODE_ROOT=/path/to/repo/code.
# Usage: DATA=data/diginetica.pkl bash run_all.sh
#        STAGES="a1 a2 b1" DATA=data/gowalla.pkl bash run_all.sh
set -euo pipefail
ROOT="${SKC_CODE_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)/code}"
cd "$ROOT"
PY=${PY:-python3}
# Dataset dir: default $ROOT/data; override with SKC_DATA_ROOT when the
# .pkl files live elsewhere (GPU_GUIDE section (c)).
if [ -z "${DATA:-}" ]; then
  if [ -n "${SKC_DATA_ROOT:-}" ]; then DATA="$SKC_DATA_ROOT/diginetica.pkl"
  else DATA="data/diginetica.pkl"; fi
fi
STEM=$(basename "$DATA" .pkl)
STAGES=${STAGES:-"a1 a2 b1 b2 b3 b5 b6 c1 c2 d1 d3 d4 f4 f5 f6"}
# d2 (Yoochoose-1/4 scaling) resolves its own ckpt; data defaults next
# to the main DATA pkl.
D2_DATA=${D2_DATA:-"$(dirname "$DATA")/yoochoose_quarter.pkl"}
SUITE="$(dirname "$0")"
mkdir -p results/systems logs
if [ -z "${CKPT:-}" ]; then
  case "$STEM" in
    diginetica) CKPT=results/real/abl_no_residual_diginetica.pt ;;
    gowalla) CKPT=results/real/mplignn_gowalla.pt ;;
    yoochoose_quarter) CKPT=results/real/mplignn_yoochoose_quarter.pt ;;
    *) CKPT=results/real/abl_no_residual_diginetica.pt ;;
  esac
fi

for s in $STAGES; do
  echo "=== stage $s ($(date +%H:%M:%S)) ==="
  case $s in
    a1) $PY "$SUITE/exp_a1_workload.py" --data "$DATA" ;;
    a2) $PY "$SUITE/exp_a2_drift_trace.py" --data "$DATA" ;;
    b1) $PY "$SUITE/exp_b1_cache_pareto.py" --data "$DATA" ;;
    b2) $PY "$SUITE/exp_b2_e2e_latency.py" --data "$DATA" --ckpt "$CKPT" ;;
    b3) $PY "$SUITE/exp_b3_dai_replay.py" --data "$DATA" --live ;;
    b4) $PY "$SUITE/exp_b4_mixed_generation.py" --data "$DATA" \
          --ckpt "$CKPT" --old-ckpt "${OLD_CKPT:?b4 needs OLD_CKPT}" ;;
    b5) $PY "$SUITE/exp_b5_budget_split.py" --data "$DATA" ;;
    b6) $PY "$SUITE/exp_b6_precision.py" --data "$DATA" --ckpt "$CKPT" ;;
    b7) $PY "$SUITE/exp_b7_staleness.py" --data "$DATA" \
          --new-ckpt "$CKPT" \
          --old-ckpt "${OLD_CKPT:?b7 needs OLD_CKPT (see README)}" ;;
    c1) $PY "$SUITE/exp_c1_tier2.py" --data "$DATA" ;;
    c2) $PY "$SUITE/exp_c2_gate_spectrum.py" --data "$DATA" --ckpt "$CKPT" ;;
    d1) $PY "$SUITE/exp_d1_ann_baselines.py" --data "$DATA" --ckpt "$CKPT" ;;
    d2) $PY "$SUITE/exp_d2_yoochoose.py" --data "$D2_DATA" ;;
    d3) $PY "$SUITE/exp_d3_concurrency.py" --data "$DATA" --ckpt "$CKPT" ;;
    d4) $PY "$SUITE/exp_d4_igms_layout.py" --data "$DATA" ;;
    f4) $PY "$SUITE/exp_f4_integrated.py" --data "$DATA" --ckpt "$CKPT" ${F4_ARGS:-} ;;
    f5) $PY "$SUITE/exp_f5_contract_power.py" --data "$DATA" --ckpt "$CKPT" ${F5_ARGS:-} ;;
    f6) $PY "$SUITE/exp_f6_sla_repeated.py" --data "$DATA" --ckpt "$CKPT" ${F6_ARGS:-} ;;
    *)  echo "unknown stage $s"; exit 1 ;;
  esac 2>&1 | tee "logs/${s}_${STEM}.log"
done
$PY "$SUITE/summarize.py"
echo "all done -> $ROOT/results/systems/"
