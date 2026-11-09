#!/usr/bin/env python3
"""Object-level bounded-dispersion contract matrix.

The replay uses real session item sets and controlled directional updates to
separate the conservative ledger bound from realized profile/projection error.
It is a correctness and tightness audit; it does not claim end-to-end serving
performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from mplignn.data import load_sessions


def quantize_rows(values: np.ndarray, bits: int) -> Tuple[np.ndarray, np.ndarray, float]:
    levels = float(2 ** (bits - 1) - 1)
    scale = np.maximum(np.max(np.abs(values), axis=1), 1e-12) / levels
    codes = np.rint(values / scale[:, None]).astype(np.int8)
    restored = codes.astype(np.float32) * scale[:, None]
    actual = np.linalg.norm(restored - values, axis=1)
    bound = np.sqrt(values.shape[1]) * np.max(np.abs(values), axis=1) / (2.0 * levels)
    return codes.astype(np.float32), scale.astype(np.float32), float(np.max(bound - actual))


def build_pages(sessions: Sequence[Sequence[int]], page_size: int) -> List[np.ndarray]:
    return [
        np.asarray(sorted({int(i) for s in sessions[start:start + page_size] for i in s}), dtype=np.int32)
        for start in range(0, len(sessions), page_size)
    ]


def profile_rows(embeddings: np.ndarray, sessions: Sequence[Sequence[int]], rows: Iterable[int]) -> np.ndarray:
    ids = list(rows)
    return np.stack([embeddings[sorted(set(map(int, sessions[sid])))].mean(axis=0) for sid in ids])


def summarize_points(points: List[Dict]) -> Dict:
    keys = [
        "profile_checks", "profile_violations", "projection_checks",
        "projection_violations", "dirty_decisions", "false_dirty",
        "false_clean", "max_actual_profile_dispersion",
        "max_reported_profile_bound", "max_actual_projection_dispersion",
        "max_reported_projection_bound", "quant_bound_margin",
    ]
    out: Dict[str, float] = {k: 0 for k in keys}
    for p in points:
        for k in keys:
            out[k] += float(p[k])
    for k in keys:
        if k.startswith("max_") or k == "quant_bound_margin":
            out[k] /= max(1, len(points))
        else:
            out[k] = int(round(out[k]))
    out["configurations"] = len(points)
    return out


def run_case(
    sessions: Sequence[Sequence[int]], dimension: int, bits: int,
    page_size: int, pattern: str, theta: float, intervals: int,
    seed: int,
) -> Dict:
    rng = np.random.default_rng(seed)
    max_item = max(int(i) for s in sessions for i in s) + 1
    embeddings = rng.normal(0, 0.08, size=(max_item, dimension)).astype(np.float32)
    rows = np.arange(len(sessions), dtype=np.int64)
    source = profile_rows(embeddings, sessions, rows)
    cached, scales, quant_margin = quantize_rows(source, bits)
    dequant = cached * scales[:, None]
    initial = dequant.copy()
    pages = build_pages(sessions, page_size)
    page_rows = [rows[i:i + page_size] for i in range(0, len(rows), page_size)]
    ledgers = [np.zeros(items.size, dtype=np.float64) for items in pages]
    last_refresh = [source[rows].copy() for rows in page_rows]
    projection = rng.normal(0, 0.08, size=(dimension, dimension)).astype(np.float32)
    op_norm = float(np.linalg.norm(projection, ord=2))

    result = {
        "dimension": dimension, "bits": bits, "page_size": page_size,
        "pattern": pattern, "theta": theta, "intervals": intervals,
        "profile_checks": 0, "profile_violations": 0,
        "projection_checks": 0, "projection_violations": 0,
        "dirty_decisions": 0, "false_dirty": 0, "false_clean": 0,
        "max_actual_profile_dispersion": 0.0,
        "max_reported_profile_bound": 0.0,
        "max_actual_projection_dispersion": 0.0,
        "max_reported_projection_bound": 0.0,
        "quant_bound_margin": quant_margin,
    }

    for _ in range(intervals):
        if pattern == "diffuse":
            item_drift = np.full(max_item, 0.02, dtype=np.float64)
        elif pattern == "concentrated":
            item_drift = np.zeros(max_item, dtype=np.float64)
            hot = rng.choice(max_item, size=min(80, max_item), replace=False)
            item_drift[hot] = 0.04
        else:
            item_drift = np.zeros(max_item, dtype=np.float64)
            item_drift[np.arange(max_item) % 31 == 0] = 0.025

        directions = rng.normal(size=(max_item, dimension)).astype(np.float32)
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
        embeddings += (directions * item_drift[:, None]).astype(np.float32)
        current = profile_rows(embeddings, sessions, rows)

        for pid, items in enumerate(pages):
            ledgers[pid] += item_drift[items]
            reported_drift = float(ledgers[pid].max())
            dirty = reported_drift > theta
            row_ids = page_rows[pid]
            if dirty:
                result["dirty_decisions"] += 1
                actual_drift = float(np.max(np.linalg.norm(current[row_ids] - last_refresh[pid], axis=1)))
                if actual_drift <= theta:
                    result["false_dirty"] += 1
                continue

            actual = np.linalg.norm(dequant[row_ids] - current[row_ids], axis=1)
            initial_quant = np.linalg.norm(initial[row_ids] - source[row_ids], axis=1)
            bound = reported_drift + initial_quant
            result["profile_checks"] += int(row_ids.size)
            result["profile_violations"] += int(np.count_nonzero(actual > bound + 2e-5))
            result["max_actual_profile_dispersion"] = max(result["max_actual_profile_dispersion"], float(actual.max()))
            result["max_reported_profile_bound"] = max(result["max_reported_profile_bound"], float(np.max(bound)))

            delta = dequant[row_ids] - current[row_ids]
            actual_proj = np.linalg.norm(projection @ delta.T, axis=0)
            proj_bound = op_norm * bound
            result["projection_checks"] += int(row_ids.size)
            result["projection_violations"] += int(np.count_nonzero(actual_proj > proj_bound + 3e-5))
            result["max_actual_projection_dispersion"] = max(result["max_actual_projection_dispersion"], float(actual_proj.max()))
            result["max_reported_projection_bound"] = max(result["max_reported_projection_bound"], float(np.max(proj_bound)))

        # Refresh every dirty page from the same current-source snapshot.
        for pid in range(len(pages)):
            if float(ledgers[pid].max()) > theta:
                row_ids = page_rows[pid]
                values = current[row_ids]
                q, scale, _ = quantize_rows(values, bits)
                dequant[row_ids] = q * scale[:, None]
                last_refresh[pid] = values.copy()
                ledgers[pid].fill(0.0)

    result["pages"] = len(pages)
    result["max_operator_norm"] = op_norm
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--sessions", type=int, default=8192)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    data = load_sessions(args.data)
    sessions = data.train[:args.sessions]
    points = []
    for dimension in (64, 128, 256):
        for bits in (8, 4):
            for page_size in (64, 128, 256):
                for pattern in ("diffuse", "concentrated", "real_sparse"):
                    points.append(run_case(
                        sessions, dimension, bits, page_size, pattern,
                        theta=0.10, intervals=10, seed=17011,
                    ))
    result = {
        "dataset": args.data.stem,
        "protocol": "real session item sets; directional synthetic drift; profile and frozen-linear projection checks; dirty pages refreshed from one current-source snapshot",
        "summary": summarize_points(points),
        "points": points,
    }
    out = args.out or (Path(__file__).resolve().parents[1] / "results" / "systems" / "contract_matrix" / f"{args.data.stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(out)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
