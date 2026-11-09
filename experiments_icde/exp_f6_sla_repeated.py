#!/usr/bin/env python3
"""F6: repeated full-request SLA experiment with bounded packed pages.

F6 replaces the one-trace F2 diagnostic with repeated Poisson traces, paired
invalidation arms, a configurable dirty window, optional real drift traces,
and profile recomputation from the embedding table.  The default path still
uses precomputed neighbor ids to isolate maintenance/forward latency; pass
--use-posting to include the cold/hard-cap posting tier in the request path.
"""
from __future__ import annotations

import argparse
import gc
import queue
import resource
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (REAL, config_out, load_data, load_model,  # noqa: E402
                    neighbor_ids, pool_profiles, run_forward,
                    serving_prefixes, write_json)
from mplignn.data import pad_batch  # noqa: E402
from systems.bounded_tier import build_bounded_tier  # noqa: E402
from systems.bounded_tier import BoundedTieredPostingStore  # noqa: E402
from systems.hot_pages import BoundedPackedPageStore  # noqa: E402


def default_ckpt(dataset: str) -> Path:
    for name in (f"abl_no_residual_{dataset}.pt", f"mplignn_{dataset}.pt"):
        path = REAL / name
        if path.exists():
            return path
    raise FileNotFoundError(f"no checkpoint for {dataset}")


def pct(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def maxrss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return value / (1024.0 ** 2)
    return value / 1024.0


def pad_neighbor_ids(ids: Sequence[int], k: int) -> np.ndarray:
    """Pad a variable-length posting result to the model's fixed K slots."""
    row = [int(x) for x in ids]
    if not row:
        return np.zeros(k, dtype=np.int64)
    reps = (k + len(row) - 1) // max(1, len(row))
    return np.asarray((row * reps)[:k], dtype=np.int64)


def make_source_fn(embeddings: np.ndarray, sessions):
    stats = {"calls": 0, "rows": 0, "seconds": 0.0}

    def recompute(source_ids: Sequence[int]) -> np.ndarray:
        t0 = time.perf_counter()
        ids = np.asarray(source_ids, dtype=np.int64)
        out = np.empty((len(ids), embeddings.shape[1]), dtype=np.float32)
        for row, sid in enumerate(ids.tolist()):
            items = np.asarray(sorted(set(sessions[int(sid)])), dtype=np.int64)
            out[row] = embeddings[items].mean(axis=0, dtype=np.float32)
        stats["calls"] += 1
        stats["rows"] += int(len(ids))
        stats["seconds"] += time.perf_counter() - t0
        return out

    recompute.stats = stats
    return recompute


def summarize(values: Sequence[float]) -> dict:
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "p50": pct(values, 50),
        "p90": pct(values, 90),
        "p99": pct(values, 99),
        "max": float(np.max(values)),
    }


def bootstrap_ci(values: Sequence[float], iterations: int, seed: int = 0) -> List[float]:
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        samples[i] = arr[rng.integers(0, len(arr), size=len(arr))].mean()
    return [float(np.percentile(samples, 2.5)),
            float(np.percentile(samples, 97.5))]


def run_arm(
    model,
    data,
    neighbors: np.ndarray,
    embeddings: np.ndarray,
    packed_template: BoundedPackedPageStore,
    mode: str,
    trace_seed: int,
    args,
    posting=None,
    drift_trace: Optional[np.ndarray] = None,
) -> dict:
    import torch

    device = next(model.parameters()).device
    if posting is not None:
        # Clone a fresh reader from the same immutable block files so every
        # arm and trace starts from the same hot-tier state.
        posting = BoundedTieredPostingStore(
            posting.root,
            memory_budget_bytes=posting.memory_budget_bytes,
            bands=posting.bands,
            rows=posting.rows,
            seed=posting.meta.get("seed", 0),
            k=posting.k,
            max_candidates=posting.max_candidates,
            direct_io=posting.direct_io,
            evict_after_read=posting.evict_after_read,
        )
    # Every arm/trace starts from an identical encoded page set without a new
    # corpus-wide profile materialization pass.
    store = packed_template.clone()

    source_fn = make_source_fn(embeddings, data.train)
    stats = source_fn.stats

    jobs = queue.Queue()
    stop = threading.Event()
    records: List[float] = []
    refresh_ms: List[float] = []
    maintenance_cycles = 0
    profile_lookup_rows = 0
    profile_miss_rows = 0
    posting_lookup_ms: List[float] = []

    rng_requests = np.random.default_rng(trace_seed)
    arrivals = np.cumsum(
        rng_requests.exponential(1.0 / args.target_qps, size=args.requests)
    )
    query_pool = len(neighbors)

    def maintainer() -> None:
        nonlocal maintenance_cycles
        rng = np.random.default_rng(917 + trace_seed)
        interval = 0
        while not stop.is_set():
            t0 = time.perf_counter()
            if mode in ("dai2", "adaptive"):
                if drift_trace is not None:
                    if args.drift_scenario == "real":
                        drift = drift_trace[interval % len(drift_trace)]
                        ids = np.flatnonzero(drift > 0).astype(np.int64)
                        values = drift[ids]
                    else:
                        rng = np.random.default_rng(
                            917 + trace_seed * 1000 + interval
                        )
                        n_items = min(
                            args.drift_items, len(drift_trace[0])
                        )
                        ids = rng.choice(
                            len(drift_trace[0]), size=n_items,
                            replace=False,
                        ).astype(np.int64)
                        values = np.full(n_items, args.drift, dtype=np.float64)
                    store.update_drift({
                        int(item): float(value)
                        for item, value in zip(ids, values)
                    })
                    interval += 1
                else:
                    items = rng.choice(
                        embeddings.shape[0], size=args.drift_items,
                        replace=False,
                    )
                    store.update_drift({
                        int(i): args.drift for i in items
                    })
                if (
                    mode == "adaptive"
                    and len(store.dirty_pages()) / max(1, len(store.pages))
                    >= args.adaptive_full_fraction
                ):
                    store.mark_all_dirty()
            elif mode == "full":
                store.mark_all_dirty()
            if args.dirty_window_ms > 0:
                stop.wait(args.dirty_window_ms / 1000.0)
            refreshed = store.refresh(source_fn=source_fn)
            refresh_ms.append((time.perf_counter() - t0) * 1000.0)
            maintenance_cycles += 1
            if refreshed:
                pass
            stop.wait(args.maintenance_interval_ms / 1000.0)

    worker = threading.Thread(target=maintainer, daemon=True)
    worker.start()

    start = time.perf_counter()
    request_index = 0
    completed = 0
    while completed < args.requests:
        next_arrival = (
            start + arrivals[request_index]
            if request_index < len(arrivals) else float("inf")
        )
        now = time.perf_counter()
        if now < next_arrival and jobs.empty():
            time.sleep(min(0.002, max(0.0, next_arrival - now)))
        while (
            request_index < len(arrivals)
            and start + arrivals[request_index] <= time.perf_counter()
        ):
            jobs.put((request_index, start + arrivals[request_index]))
            request_index += 1
        if jobs.empty():
            continue
        batch = []
        while len(batch) < args.batch_size and not jobs.empty():
            batch.append(jobs.get())
        if not batch:
            continue

        profile_rows = []
        gather_start = time.perf_counter()
        for idx, _arrival in batch:
            if posting is None:
                ids = neighbors[idx % query_pool]
            else:
                t0 = time.perf_counter()
                ids = posting.search(
                    serving_prefixes([data.test[idx % query_pool]])[0],
                    return_profiles=False,
                )
                ids = pad_neighbor_ids(ids, neighbors.shape[1])
                posting_lookup_ms.append((time.perf_counter() - t0) * 1000.0)
            if mode == "fresh":
                values = source_fn(ids)
                hit = 0.0
            else:
                values, hit, _bound = store.service(
                    ids, fallback_fn=source_fn, strict_dirty=False
                )
            profile_lookup_rows += len(ids)
            profile_miss_rows += int(round((1.0 - hit) * len(ids)))
            profile_rows.append(values)
        profile_batch = torch.from_numpy(
            np.asarray(profile_rows, dtype=np.float32)
        ).to(device)
        gather_done = time.perf_counter()

        sessions = [data.test[idx % query_pool] for idx, _ in batch]
        seqs, _lens, _targets = pad_batch(sessions, str(device))
        _ = run_forward(model, seqs, profile_batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        done = time.perf_counter()
        records.extend((done - arrival) * 1000.0 for _, arrival in batch)
        completed += len(batch)

    stop.set()
    worker.join()
    wall = time.perf_counter() - start
    finite = [x for x in records if np.isfinite(x)]
    return {
        "mode": mode,
        "trace_seed": trace_seed,
        "requests": len(finite),
        "achieved_qps": len(finite) / wall,
        "packed_resident_mb": store.resident_bytes() / 1e6,
        "packed_components_mb": {
            k: v / 1e6 for k, v in store.resident_components().items()
        },
        "source_embedding_copy_mb": embeddings.nbytes / 1e6,
        "cap_respected": store.resident_bytes() <= store.memory_budget_bytes,
        "profile_lookup_rows": profile_lookup_rows,
        "profile_miss_rows": profile_miss_rows,
        "profile_miss_rate": profile_miss_rows / max(1, profile_lookup_rows),
        "request_latency_ms": {
            "p50": pct(finite, 50),
            "p90": pct(finite, 90),
            "p99": pct(finite, 99),
            "p999": pct(finite, 99.9),
            "max": float(np.max(finite)),
        },
        "posting_lookup_ms": (
            {
                "p50": pct(posting_lookup_ms, 50),
                "p90": pct(posting_lookup_ms, 90),
                "p99": pct(posting_lookup_ms, 99),
            }
            if posting_lookup_ms else None
        ),
        "refresh_ms": summarize(refresh_ms),
        "maintenance_cycles": maintenance_cycles,
        "source_recompute": dict(stats),
        "gather_excluding_forward_ms": (gather_done - gather_start) * 1000.0,
        "process_maxrss_mb": maxrss_mb(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--n-query", type=int, default=1000)
    ap.add_argument("--requests", type=int, default=2000)
    ap.add_argument("--traces", type=int, default=30)
    ap.add_argument("--first-trace-seed", type=int, default=1000)
    ap.add_argument("--target-qps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--packed-budget-mb", type=float, default=5.0)
    ap.add_argument("--theta", type=float, default=0.1)
    ap.add_argument("--maintenance-interval-ms", type=float, default=500.0)
    ap.add_argument("--dirty-window-ms", type=float, default=2.0)
    ap.add_argument("--drift-trace", type=Path, default=None)
    ap.add_argument("--drift-items", type=int, default=80)
    ap.add_argument("--drift", type=float, default=0.15)
    ap.add_argument("--modes", default="none,dai2,full")
    ap.add_argument("--adaptive-full-fraction", type=float, default=0.25,
                    help="adaptive arm switches to full invalidation at this dirty-page fraction")
    ap.add_argument("--drift-scenario", choices=["real", "concentrated"],
                    default="real")
    ap.add_argument("--use-posting", action="store_true")
    ap.add_argument("--posting-budget-mb", type=float, default=8.0)
    ap.add_argument("--block-sessions", type=int, default=8192)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import torch

    stem = args.data.stem
    ckpt = args.ckpt or default_ckpt(stem)
    tag = (
        f"{stem}_sla_repeated_{args.drift_scenario}_"
        f"{'posting_' if args.use_posting else ''}"
        f"t{args.traces}_q{args.target_qps}_"
        f"r{args.requests}_theta{args.theta:g}"
    )
    out_path, skip = config_out("f6", tag, args.out, args.force)
    if skip:
        print(f"[f6] exists, skipping: {out_path} (--force to rerun)")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(args.data)
    n_query = min(args.n_query, len(data.test))
    sessions = data.test[:n_query]
    neighbors = np.asarray(
        neighbor_ids(serving_prefixes(sessions), data.train, 10),
        dtype=np.int64,
    )
    # Pad variable-length heuristic neighbor rows to exactly k.
    if neighbors.ndim != 2 or neighbors.shape[1] != 10:
        fixed = np.zeros((len(sessions), 10), dtype=np.int64)
        for i, row in enumerate(neighbors):
            reps = (10 + len(row) - 1) // max(1, len(row))
            fixed[i] = np.tile(row, reps)[:10]
        neighbors = fixed

    model = load_model(data, ckpt, device=str(device))
    embeddings = model.embedding.weight.detach().cpu().numpy().copy()
    frequencies = np.bincount(neighbors.reshape(-1), minlength=len(data.train))
    packed_order = np.argsort(-frequencies, kind="stable")
    build_profiles = pool_profiles(embeddings, data.train)
    packed_template = BoundedPackedPageStore(
        int(args.packed_budget_mb * 1e6),
        dimension=embeddings.shape[1],
        page_size=128,
        theta_max=args.theta,
        quantize=True,
        quant_bits=8,
    )
    packed_template.admit(packed_order.tolist(), build_profiles, data.train)
    del build_profiles
    gc.collect()

    posting = None
    posting_root = None
    if args.use_posting:
        posting_root = Path("/tmp") / f"skc_f6_{stem}_{args.first_trace_seed}"
        if posting_root.exists():
            shutil.rmtree(posting_root)
        posting = build_bounded_tier(
            data.train, None, posting_root,
            block_sessions=args.block_sessions, hot_blocks=1,
        )
        posting.memory_budget_bytes = int(args.posting_budget_mb * 1e6)
        posting.evict_after_read = True
        posting.enforce_budget()

    drift_trace = None
    if args.drift_trace and args.drift_trace.exists():
        with np.load(args.drift_trace) as npz:
            drift_trace = npz["drift"].astype(np.float64)

    modes = [x.strip() for x in args.modes.split(",")]
    traces = []
    for t in range(args.traces):
        trace_seed = args.first_trace_seed + t
        print(f"trace {t + 1}/{args.traces} seed={trace_seed}", flush=True)
        for mode in modes:
            result = run_arm(
                model, data, neighbors, embeddings, packed_template,
                mode, trace_seed, args, posting=posting,
                drift_trace=drift_trace,
            )
            traces.append(result)
            print(
                f"  {mode}: p99={result['request_latency_ms']['p99']:.2f}ms "
                f"p50={result['request_latency_ms']['p50']:.2f}ms",
                flush=True,
            )

    summary = {}
    for mode in modes:
        rows = [r for r in traces if r["mode"] == mode]
        p99 = [r["request_latency_ms"]["p99"] for r in rows]
        p50 = [r["request_latency_ms"]["p50"] for r in rows]
        achieved = [r["achieved_qps"] for r in rows]
        refresh = [r["refresh_ms"]["mean"] for r in rows]
        summary[mode] = {
            "n_traces": len(rows),
            "p99_ms": summarize(p99),
            "p50_ms": summarize(p50),
            "achieved_qps": summarize(achieved),
            "refresh_ms_mean": summarize(refresh),
            "p99_bootstrap_ci95": bootstrap_ci(
                p99, args.bootstrap, seed=77
            ),
        }
    if "dai2" in modes and "full" in modes:
        n = min(summary["dai2"]["n_traces"], summary["full"]["n_traces"])
        deltas = [
            summary["dai2"]["p99_ms"]["mean"] - summary["full"]["p99_ms"]["mean"]
        ]
        # Pair by trace seed order; rows were appended mode-major within trace.
        dai = [r for r in traces if r["mode"] == "dai2"]
        full = [r for r in traces if r["mode"] == "full"]
        deltas = [
            a["request_latency_ms"]["p99"] - b["request_latency_ms"]["p99"]
            for a, b in zip(dai[:n], full[:n])
        ]
        summary["paired_dai2_minus_full_p99"] = {
            **summarize(deltas),
            "bootstrap_ci95": bootstrap_ci(deltas, args.bootstrap, seed=91),
        }

    if posting_root is not None:
        shutil.rmtree(posting_root, ignore_errors=True)
    result = {
        "dataset": stem,
        "checkpoint": str(ckpt),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0)
        if device.type == "cuda" else None,
        "protocol": (
            "repeated Poisson traces; paired arms; real forward pass; "
            "int8 packed pages; source profiles recomputed on miss/refresh"
        ),
        "n_query_pool": n_query,
        "target_qps": args.target_qps,
        "batch_size": args.batch_size,
        "maintenance_interval_ms": args.maintenance_interval_ms,
        "dirty_window_ms": args.dirty_window_ms,
        "drift_trace": str(args.drift_trace) if args.drift_trace else None,
        "drift_scenario": args.drift_scenario,
        "adaptive_full_fraction": args.adaptive_full_fraction,
        "drift_semantics": (
            "trace norms drive ledger dirtiness and refresh work; source "
            "profile values remain the checkpoint snapshot in this latency test"
        ),
        "drift_items": args.drift_items,
        "drift_l2": args.drift,
        "posting_in_path": bool(args.use_posting),
        "source_embedding_copy_mb": embeddings.nbytes / 1e6,
        "source_mode": "CPU copy of live embedding table; model weights excluded",
        "traces": traces,
        "summary": summary,
    }
    write_json(out_path, result)


if __name__ == "__main__":
    main()
