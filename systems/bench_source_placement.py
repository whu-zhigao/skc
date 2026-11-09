#!/usr/bin/env python3
"""Source-placement sensitivity for cached versus fresh profile materialization.

This microbenchmark keeps the real A3 embedding table, real session item sets,
and exact top-10 neighbor ids.  Dimensions above the trained 64-D table are
obtained by deterministic repetition, so those cells isolate source-plane byte
movement rather than claim a new trained model.  macOS F_NOCACHE is used for
storage-backed reads; no result is extrapolated to an unmeasured network.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from mplignn.data import load_sessions


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def stats(values: Sequence[float]) -> Dict[str, float]:
    return {
        "p50_ms": percentile(values, 50) * 1e3,
        "p95_ms": percentile(values, 95) * 1e3,
        "p99_ms": percentile(values, 99) * 1e3,
        "mean_ms": float(np.mean(values)) * 1e3,
    }


def quantize(values: np.ndarray, bits: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    levels = float(2 ** (bits - 1) - 1)
    scale = np.maximum(np.max(np.abs(values), axis=1), 1e-12) / levels
    codes = np.rint(values / scale[:, None]).astype(np.int8)
    return codes, scale.astype(np.float32)


def materialize(source: np.ndarray, sessions: Sequence[Sequence[int]], ids: np.ndarray) -> np.ndarray:
    unique, inverse = np.unique(ids, return_inverse=True)
    values = np.stack([source[sorted(set(map(int, sessions[sid])))].mean(axis=0) for sid in unique])
    return values[inverse]


def disk_materialize(
    fd: int, row_bytes: int, dimension: int,
    sessions: Sequence[Sequence[int]], ids: np.ndarray,
) -> np.ndarray:
    unique, inverse = np.unique(ids, return_inverse=True)
    values = np.empty((len(unique), dimension), dtype=np.float32)
    for out_row, sid in enumerate(unique):
        needed = sorted(set(map(int, sessions[int(sid)])))
        chunks = []
        for item in needed:
            raw = os.pread(fd, row_bytes, int(item) * row_bytes)
            if len(raw) != row_bytes:
                raise IOError(f"short source read for item {item}")
            chunks.append(np.frombuffer(raw, dtype=np.float32))
        values[out_row] = np.mean(np.stack(chunks), axis=0)
    return values[inverse]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--neighbors", type=Path, required=True)
    ap.add_argument("--batches", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    data = load_sessions(args.data)
    neighbors = np.load(args.neighbors)[:args.batches * 128, :10]
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base = state["embedding.weight"].numpy().astype(np.float32, copy=False)
    result = {
        "dataset": args.data.stem,
        "protocol": {
            "neighbor_source": str(args.neighbors),
            "checkpoint": str(args.checkpoint),
            "batch_sizes": [1, 8, 32, 128],
            "batches": args.batches,
            "storage_cache": "macOS F_NOCACHE per-read",
            "dimension_note": "64-D uses the trained table; larger dimensions deterministically repeat its coordinates for source-byte sensitivity only",
        },
        "dimensions": {},
    }

    for dimension in (64, 128, 256, 512):
        source = np.repeat(base, dimension // base.shape[1], axis=1)
        row_bytes = dimension * 4
        source_path = Path("/tmp") / f"skc_source_{args.data.stem}_{dimension}.bin"
        source_path.write_bytes(source.astype(np.float32, copy=False).tobytes())
        fd = os.open(source_path, os.O_RDONLY)
        try:
            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
            batch_results = {}
            # Offline-profiled active-set ceiling: build the cache once from
            # every neighbor id in the measured workload.  This is not an
            # online admission claim.
            cached_ids = np.unique(neighbors.reshape(-1))
            cached_values = materialize(source, data.train, cached_ids)
            codes, scales = quantize(cached_values)
            sid_to_local = {int(sid): i for i, sid in enumerate(cached_ids)}
            cache_resident_mb = float(
                codes.nbytes + scales.nbytes +
                np.asarray(cached_ids, dtype=np.int64).nbytes
            ) / 1e6
            for batch_size in (1, 8, 32, 128):
                timings = {"resident_fresh": [], "storage_fresh": [], "int8_cache": []}
                usable_batches = min(args.batches, len(neighbors) // batch_size)
                for b in range(usable_batches):
                    chunk = neighbors[b * batch_size:(b + 1) * batch_size].reshape(-1)

                    start = time.perf_counter_ns()
                    fresh = materialize(source, data.train, chunk)
                    timings["resident_fresh"].append((time.perf_counter_ns() - start) / 1e9)

                    start = time.perf_counter_ns()
                    cold = disk_materialize(fd, row_bytes, dimension, data.train, chunk)
                    timings["storage_fresh"].append((time.perf_counter_ns() - start) / 1e9)
                    if cold.shape != fresh.shape:
                        raise AssertionError("source-placement result shape mismatch")

                    local = np.asarray([sid_to_local[int(sid)] for sid in chunk], dtype=np.int64)
                    start = time.perf_counter_ns()
                    _ = codes[local].astype(np.float32) * scales[local, None]
                    timings["int8_cache"].append((time.perf_counter_ns() - start) / 1e9)
                batch_results[str(batch_size)] = {k: stats(v) for k, v in timings.items()}
            batch_results["offline_cache_resident_mb"] = cache_resident_mb
            result["dimensions"][str(dimension)] = batch_results
        finally:
            os.close(fd)
            source_path.unlink(missing_ok=True)

    out = args.out or (ROOT / "results" / "systems" / "source_placement" / f"{args.data.stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(out)
    print(json.dumps(result["dimensions"], indent=2))


if __name__ == "__main__":
    main()
