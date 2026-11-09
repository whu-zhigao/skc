#!/usr/bin/env python3
"""CUDA stress test for snapshot-pinned SKC page publication.

The benchmark exercises the implementation invariant missing from the CPU
test: optimizer updates, immutable GPU source snapshots, reader streams, and
atomic metadata pinning execute concurrently on the A800.  Each replacement
page carries a model-epoch marker; a torn or mixed-version copy is detected
when rows in one immutable page view do not share that marker.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Dict, List, Tuple

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover - cloud prerequisite
    raise SystemExit("PyTorch with CUDA is required") from exc


@dataclass
class GpuPageView:
    page_id: int
    rows: torch.Tensor
    model_epoch: int
    dirty: bool


class GpuSnapshotPageStore:
    def __init__(
        self, source: torch.Tensor, page_rows: List[torch.Tensor], theta: float
    ) -> None:
        self.source = source
        self.page_rows = page_rows
        self.theta = theta
        self.pages: List[GpuPageView] = []
        self.epoch = 0
        self.lock = threading.RLock()
        for pid, rows in enumerate(page_rows):
            self.pages.append(GpuPageView(pid, source[rows].clone(), 0, False))

    def snapshot_versioned(self, page_ids: List[int]) -> Tuple[List[GpuPageView], int]:
        with self.lock:
            return [self.pages[int(pid)] for pid in page_ids], int(self.epoch)

    def mark_dirty_and_advance(self, page_ids: List[int]) -> int:
        with self.lock:
            self.epoch += 1
            for pid in page_ids:
                old = self.pages[pid]
                self.pages[pid] = GpuPageView(
                    old.page_id, old.rows, self.epoch, True
                )
            return int(self.epoch)

    def publish_snapshot(self, page_ids: List[int]) -> int:
        # Clone a complete immutable source snapshot while the metadata lock is
        # held.  Trainer updates use the same default CUDA stream, so the copy
        # has a well-defined pre-publication ordering.  Readers use separate
        # streams but touch only the cloned rows.
        with self.lock:
            snapshot = self.source.clone()
            published = 0
            for pid in page_ids:
                old = self.pages[pid]
                rows = self.page_rows[pid].to(self.source.device)
                replacement = snapshot[rows].clone()
                self.pages[pid] = GpuPageView(
                    old.page_id, replacement, self.epoch, False
                )
                published += 1
            return published


def percentile(values: List[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q)) if values else 0.0


def run_case(
    readers: int, dirty_items: int, cycles: int, n_items: int = 4096,
    dimension: int = 64, page_size: int = 128, seed: int = 1907,
    dirty_window_ms: float = 2.0,
) -> Dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda")
    n_sessions = page_size * 64
    source = torch.normal(0, 0.05, (n_sessions, dimension), device=device)
    source[:, -1] = 0
    optimizer = torch.optim.SGD([source], lr=0.01)

    # Synthetic sessions and physical pages. Item reuse makes drift update a
    # page-coherence problem rather than a trivial one-page test.
    sessions = [
        np.unique(rng.integers(0, n_items, size=int(rng.integers(2, 9))))
        for _ in range(n_sessions)
    ]
    page_items = [
        np.asarray(sorted({int(i) for s in sessions[start:start + page_size] for i in s}), dtype=np.int64)
        for start in range(0, len(sessions), page_size)
    ]
    item_to_pages: Dict[int, List[int]] = {}
    for pid, items in enumerate(page_items):
        for item in items.tolist():
            item_to_pages.setdefault(item, []).append(pid)
    page_rows = [
        torch.arange(start, min(start + page_size, len(sessions)), device=device)
        for start in range(0, len(sessions), page_size)
    ]
    store = GpuSnapshotPageStore(source, page_rows, theta=0.1)
    stop = threading.Event()
    counters = {
        "clean_reads": 0, "soft_misses": 0, "torn_pages": 0,
        "version_mismatches": 0, "publications": 0,
    }
    acq_ms: List[float] = []
    pub_ms: List[float] = []
    lock = threading.Lock()

    def reader(reader_id: int) -> None:
        stream = torch.cuda.Stream(device=device)
        local_rng = np.random.default_rng(seed + 1000 + reader_id)
        while not stop.is_set():
            pids = local_rng.choice(len(page_rows), size=8, replace=False).tolist()
            start_time = time.perf_counter_ns()
            views, epoch = store.snapshot_versioned(pids)
            elapsed = (time.perf_counter_ns() - start_time) / 1e6
            with lock:
                acq_ms.append(elapsed)
            if any(v.dirty for v in views):
                with lock:
                    counters["soft_misses"] += 1
                time.sleep(1e-5)
                continue
            with torch.cuda.stream(stream):
                rows = torch.cat([v.rows for v in views], dim=0)
                markers = rows[:, -1]
                expected = torch.as_tensor(
                    [float(v.model_epoch) for v in views], device=device
                )
                per_view_uniform = torch.stack([
                    markers[j * page_size:(j + 1) * page_size].std()
                    for j in range(len(views))
                ])
                page_epochs = markers.view(len(views), -1)[:, 0]
                stream.synchronize()
            uniform = bool(per_view_uniform.max().item() < 1e-5)
            matches = bool(torch.equal(page_epochs, expected))
            with lock:
                counters["clean_reads"] += 1
                counters["torn_pages"] += int(not uniform)
                counters["version_mismatches"] += int(not matches)
            # Yield the Python scheduler so reader spinning cannot starve the
            # publisher under the GIL.
            time.sleep(1e-5)

    threads = [threading.Thread(target=reader, args=(i,), daemon=True) for i in range(readers)]
    for thread in threads:
        thread.start()

    for cycle in range(1, cycles + 1):
        touched = rng.choice(n_items, size=dirty_items, replace=False)
        affected = sorted({pid for item in touched.tolist() for pid in item_to_pages.get(item, [])})
        store.mark_dirty_and_advance(affected)
        if dirty_window_ms:
            time.sleep(dirty_window_ms / 1e3)

        # Real optimizer update on the CUDA default stream.  The marker is a
        # deterministic coordinate used to detect mixed-version rows.
        optimizer.zero_grad(set_to_none=True)
        affected_rows = torch.cat([page_rows[pid] for pid in affected])
        grad = torch.zeros_like(source)
        grad[affected_rows, :] = torch.normal(
            0, 0.01, (len(affected_rows), dimension), device=device
        )
        source.grad = grad
        optimizer.step()
        with torch.no_grad():
            source[:, -1] = float(cycle)

        start_time = time.perf_counter_ns()
        published = store.publish_snapshot(affected)
        torch.cuda.synchronize(device)
        elapsed = (time.perf_counter_ns() - start_time) / 1e6
        with lock:
            pub_ms.append(elapsed)
            counters["publications"] += published

    stop.set()
    for thread in threads:
        thread.join(timeout=10)
        if thread.is_alive():
            raise RuntimeError("GPU reader thread did not stop")

    return {
        "readers": readers,
        "dirty_items": dirty_items,
        "cycles": cycles,
        "dirty_window_ms": dirty_window_ms,
        "device": torch.cuda.get_device_name(device),
        **counters,
        "acquisition_ms": {"p50": percentile(acq_ms, 50), "p99": percentile(acq_ms, 99)},
        "publication_ms": {"p50": percentile(pub_ms, 50), "p99": percentile(pub_ms, 99)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=120)
    ap.add_argument("--dirty-window-ms", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    cases = []
    for readers in (1, 2, 4, 8):
        cases.append(run_case(readers, 80, args.cycles, dirty_window_ms=args.dirty_window_ms))
    for dirty_items in (20, 80, 320):
        cases.append(run_case(4, dirty_items, args.cycles, dirty_window_ms=args.dirty_window_ms))
    summary = {
        "protocol": "A800 CUDA default-stream optimizer updates; immutable source snapshot; reader CUDA streams; atomic CPU metadata pinning",
        "cases": cases,
        "summary": {
            key: sum(case[key] for case in cases)
            for key in (
                "clean_reads", "soft_misses", "torn_pages",
                "version_mismatches", "publications",
            )
        },
    }
    out = args.out or Path(__file__).resolve().parents[1] / "results" / "systems" / "gpu_publication_matrix.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(out)
    print(json.dumps(summary["summary"], indent=2))


if __name__ == "__main__":
    main()
