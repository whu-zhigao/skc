#!/usr/bin/env python3
"""Temporally split DAI threshold calibration and hold-out replay."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mplignn.data import load_sessions  # noqa: E402


def skip_rate(row, threshold, item_to_pages, n_pages):
    moved = np.flatnonzero(row > threshold)
    dirty = set().union(*(item_to_pages[int(i)] for i in moved)) \
        if len(moved) else set()
    return 1.0 - len(dirty) / n_pages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--drift", type=Path, required=True)
    ap.add_argument("--train-fraction", type=float, default=0.7)
    ap.add_argument("--page", type=int, default=128)
    args = ap.parse_args()
    data = load_sessions(args.data)
    drift = np.load(args.drift)["drift"]
    split = max(1, int(args.train_fraction * len(drift)))
    threshold = float(np.percentile(drift[:split].ravel(), 90))
    item_to_pages = defaultdict(set)
    for sid, session in enumerate(data.train):
        page = sid // args.page
        for item in set(session):
            item_to_pages[item].add(page)
    n_pages = (len(data.train) + args.page - 1) // args.page
    fixed_holdout = [skip_rate(row, threshold, item_to_pages, n_pages)
                     for row in drift[split:]]
    rolling_holdout = []
    for pos in range(split, len(drift)):
        rolling_threshold = float(
            np.percentile(drift[max(0, pos - 10):pos].ravel(), 90))
        rolling_holdout.append(
            skip_rate(row := drift[pos], rolling_threshold,
                      item_to_pages, n_pages))
    out = {
        "dataset": args.data.stem,
        "n_intervals": len(drift),
        "calibration_intervals": split,
        "holdout_intervals": len(drift) - split,
        "calibration_q90_theta": threshold,
        "holdout_skip_mean": float(np.mean(fixed_holdout))
            if fixed_holdout else None,
        "holdout_skip_per_interval": fixed_holdout,
        "rolling_q90_holdout_skip_mean": float(np.mean(rolling_holdout))
            if rolling_holdout else None,
        "rolling_q90_holdout_skip_per_interval": rolling_holdout,
    }
    dest = Path("results/systems") / f"dai_temporal_{args.data.stem}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("wrote", dest)


if __name__ == "__main__":
    main()
