#!/usr/bin/env python3
"""F2: request-level SLA under bounded-page maintenance.

The benchmark batches prefix-only requests, runs the real recommendation
forward pass, and concurrently refreshes packed profile pages. It reports
arrival-to-response latency, soft misses, and throughput for no maintenance,
full invalidation, and page-local DAI-2.
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (RESULTS, load_data, load_model, neighbor_ids,  # noqa: E402
                    pool_profiles, run_forward, serving_prefixes,
                    write_json)
from mplignn.data import pad_batch  # noqa: E402
from systems.hot_pages import (BoundedPackedPageStore,  # noqa: E402
                               HotPageDirtyError)


def pct(values, q):
    return float(np.percentile(np.asarray(values), q)) if values else 0.0


def run_arm(model, data, neighbors, profiles, mode: str,
            hot_budget_mb: float, target_qps: int, batch_size: int,
            n_requests: int, maintenance_interval_ms: float,
            theta: float) -> dict:
    device = next(model.parameters()).device
    freq = np.bincount(neighbors.reshape(-1), minlength=len(data.train))
    order = np.argsort(-freq)
    store = BoundedPackedPageStore(
        int(hot_budget_mb * 1e6), dimension=profiles.shape[1],
        page_size=128, theta_max=theta,
    )
    store.admit(order.tolist(), profiles, data.train)

    jobs = queue.Queue()
    stop = threading.Event()
    records = []
    maintenance_cycles = 0
    refresh_ms = []

    def maintainer():
        nonlocal maintenance_cycles
        rng = np.random.default_rng(917)
        while not stop.is_set():
            t0 = time.perf_counter()
            if mode == "none":
                pass
            elif mode == "full":
                store.mark_all_dirty()
            elif mode == "dai2":
                items = rng.choice(len(model.embedding.weight), size=80,
                                   replace=False)
                store.update_drift({int(i): theta * 1.5 for i in items})
            refreshed = store.refresh(profiles)
            refresh_ms.append((time.perf_counter() - t0) * 1000.0)
            maintenance_cycles += 1
            stop.wait(maintenance_interval_ms / 1000.0)

    worker = threading.Thread(target=maintainer, daemon=True)
    worker.start()

    rng_requests = np.random.default_rng(41)
    # Sample consecutive inter-arrival periods. Sorting i.i.d. exponentials
    # would condition on exactly n arrivals and compress the workload duration.
    arrivals = np.cumsum(rng_requests.exponential(
        1.0 / target_qps, size=n_requests))
    start = time.perf_counter()
    request_index = 0
    completed = 0
    soft_misses = 0
    hit_rows = 0
    total_rows = 0

    while completed < n_requests:
        next_arrival = (start + arrivals[request_index]
                        if request_index < n_requests else float("inf"))
        now = time.perf_counter()
        if now < next_arrival and jobs.empty():
            time.sleep(min(0.002, max(0.0, next_arrival - now)))
        while (request_index < n_requests and
               start + arrivals[request_index] <= time.perf_counter()):
            jobs.put((request_index, start + arrivals[request_index]))
            request_index += 1
        if jobs.empty():
            continue
        batch = []
        while len(batch) < batch_size and not jobs.empty():
            batch.append(jobs.get())
        gather_start = time.perf_counter()
        profile_rows = []
        gather_times = []
        for idx, _arrival in batch:
            ids = neighbors[idx]
            total_rows += len(ids)
            try:
                values, hit, _bound = store.service(ids, profiles)
                hit_rows += int(hit * len(ids))
            except HotPageDirtyError:
                values = profiles[ids]
                soft_misses += 1
            profile_rows.append(values)
        profile_batch = torch.from_numpy(
            np.asarray(profile_rows, dtype=np.float32)).to(device)
        gather_done = time.perf_counter()
        sessions = [data.test[idx] for idx, _ in batch]
        seqs, lens, _ = pad_batch(sessions, str(device))
        _ = run_forward(model, seqs, profile_batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        done = time.perf_counter()
        records.extend((done - arrival) * 1000.0 for _, arrival in batch)
        completed += len(batch)

    stop.set()
    worker.join()
    wall = time.perf_counter() - start
    latencies = [x for x in records if np.isfinite(x)]
    return {
        "mode": mode,
        "hot_budget_mb": hot_budget_mb,
        "target_qps": target_qps,
        "batch_size": batch_size,
        "requests": len(latencies),
        "achieved_qps": len(latencies) / wall,
        "resident_mb": store.resident_bytes() / 1e6,
        "cap_respected": store.resident_bytes() <= store.memory_budget_bytes,
        "hot_row_hit_rate": hit_rows / max(1, total_rows),
        "soft_miss_request_fraction": soft_misses / max(1, len(latencies)),
        "request_latency_ms": {
            "p50": pct(latencies, 50),
            "p90": pct(latencies, 90),
            "p99": pct(latencies, 99),
            "p999": pct(latencies, 99.9),
            "max": max(latencies, default=0.0),
        },
        "refresh_p50_ms": pct(refresh_ms, 50),
        "refresh_p99_ms": pct(refresh_ms, 99),
        "maintenance_cycles": maintenance_cycles,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--n-query", type=int, default=1000)
    parser.add_argument("--n-request", type=int, default=500)
    parser.add_argument("--target-qps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hot-budget-mb", type=float, default=5.0)
    parser.add_argument("--maintenance-interval-ms", type=float, default=100.0)
    parser.add_argument("--theta", type=float, default=0.1)
    parser.add_argument("--modes", default="none,dai2,full")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(args.data)
    model = load_model(data, args.ckpt, device=str(device))
    sessions = data.test[:min(args.n_query, len(data.test))]
    neighbors = np.asarray(
        neighbor_ids(serving_prefixes(sessions), data.train, 10),
        dtype=np.int64)
    profiles = pool_profiles(
        model.embedding.weight.detach().cpu().numpy(), data.train)

    arms = []
    for mode in args.modes.split(","):
        result = run_arm(
            model, data, neighbors, profiles, mode,
            args.hot_budget_mb, args.target_qps, args.batch_size,
            args.n_request, args.maintenance_interval_ms, args.theta,
        )
        arms.append(result)
        print(json.dumps(result, indent=2), flush=True)

    output = {
        "dataset": args.data.stem,
        "device": str(device),
        "gpu": (torch.cuda.get_device_name(0)
                if device.type == "cuda" else None),
        "n_query_pool": len(sessions),
        "protocol": ("closed-loop batching with real forward pass; Poisson "
                     "arrivals; concurrent packed-page maintenance"),
        "arms": arms,
    }
    out = args.out or (RESULTS / "f2" /
                       f"serving_sla_{args.data.stem}.json")
    write_json(out, output)


if __name__ == "__main__":
    main()
