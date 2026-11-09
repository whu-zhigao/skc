#!/usr/bin/env python3
"""A800 source-placement serving benchmark with the real MPLI forward path.

Routes compare GPU-resident, CPU-pinned, storage-backed, and clean int8 cached
profile materialization followed by the same recommendation forward pass.  The
cached arm is an offline active-set ceiling built from all measured exact
neighbor ids; it does not include maintenance or online admission.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments_icde.common import load_data, load_model, run_forward  # noqa: E402
from mplignn.data import pad_batch  # noqa: E402


def pct(values: List[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q)) if values else 0.0


def summarize(values: List[float]) -> Dict[str, float]:
    return {
        "p50_ms": pct(values, 50),
        "p95_ms": pct(values, 95),
        "p99_ms": pct(values, 99),
        "mean_ms": float(np.mean(values)) if values else 0.0,
    }


def build_active_csr(data, active_ids):
    local = {int(sid): i for i, sid in enumerate(active_ids)}
    lists = [sorted(set(data.train[int(sid)])) for sid in active_ids]
    width = max(max(map(len, lists)), 1)
    ids = np.zeros((len(active_ids), width), dtype=np.int64)
    mask = np.zeros_like(ids, dtype=np.float32)
    for row, items in enumerate(lists):
        if items:
            ids[row, :len(items)] = items
            mask[row, :len(items)] = 1.0
    return local, ids, mask


def quantize(values: np.ndarray):
    levels = float(2 ** 7 - 1)
    scale = np.maximum(np.max(np.abs(values), axis=1), 1e-12) / levels
    codes = np.rint(values / scale[:, None]).astype(np.int8)
    return codes, scale.astype(np.float32)


def profiles_from_matrix(source, item_matrix, mask, local_ids, device):
    """Materialize profiles from a source matrix already on the chosen device."""
    if torch.is_tensor(source):
        local_tensor = torch.as_tensor(local_ids, device=source.device)
        if not torch.is_tensor(item_matrix):
            item_matrix = torch.as_tensor(item_matrix, device=source.device)
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask, device=source.device)
        ids = item_matrix[local_tensor]
        m = mask[local_tensor]
        vals = source[ids] * m.unsqueeze(-1)
        out = vals.sum(dim=1) / m.sum(dim=1, keepdim=True).clamp_min(1)
        return out.to(device, non_blocking=True)
    ids = item_matrix[local_ids]
    m = mask[local_ids]
    vals = source[ids] * m[:, :, None]
    out = vals.sum(axis=1) / np.maximum(m.sum(axis=1, keepdims=True), 1.0)
    return torch.as_tensor(out, device=device, non_blocking=True)


def disk_profiles(fd, row_bytes, source_shape, item_matrix, mask, local_ids, device):
    unique_rows = np.unique(local_ids)
    row_to_local = {int(row): i for i, row in enumerate(unique_rows)}
    values = np.empty((len(unique_rows), source_shape[1]), dtype=np.float32)
    for row in unique_rows:
        needed = item_matrix[row][mask[row] > 0]
        chunks = []
        for item in needed:
            raw = os.pread(fd, row_bytes, int(item) * row_bytes)
            if len(raw) != row_bytes:
                raise IOError(f"short disk source read for item {item}")
            chunks.append(np.frombuffer(raw, dtype=np.float32))
        values[row_to_local[int(row)]] = np.mean(np.stack(chunks), axis=0)
    profiles = values[np.asarray([row_to_local[int(x)] for x in local_ids])]
    return torch.from_numpy(profiles).to(device, non_blocking=True)


def route_logits(route_state, route, local_ids, device):
    state = route_state
    profiles = None
    if route == "gpu_resident":
        profiles = profiles_from_matrix(
            state["source_gpu"], state["item_gpu"], state["mask_gpu"], local_ids, device
        )
    if route == "cpu_pinned":
        profiles = profiles_from_matrix(
            state["source_cpu"], state["item_cpu"], state["mask_cpu"], local_ids, device
        )
    if route == "storage":
        profiles = disk_profiles(
            state["fd"], state["row_bytes"], state["source_shape"],
            state["item_cpu"], state["mask_cpu"], local_ids, device,
        )
    if route == "int8_cache":
        local_tensor = torch.as_tensor(local_ids, device=device)
        profiles = (
            state["codes_gpu"][local_tensor].float()
            * state["scales_gpu"][local_tensor][:, None]
        )
    if profiles is None:
        raise KeyError(route)
    return profiles.view(-1, 10, profiles.shape[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--neighbors", type=Path, required=True)
    ap.add_argument("--queries", type=int, default=2048)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    data = load_data(args.data)
    model = load_model(data, args.checkpoint, device=str(device))
    model.eval()
    neighbors = np.load(args.neighbors)[:args.queries, :10]
    queries = data.test[:len(neighbors)]
    active_ids = np.unique(neighbors.reshape(-1))
    local_map, item_matrix, mask_matrix = build_active_csr(data, active_ids)

    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    source_np = state_dict["embedding.weight"].numpy().astype(np.float32, copy=False)
    active_profiles = np.stack([
        source_np[sorted(set(data.train[int(sid)]))].mean(axis=0)
        for sid in active_ids
    ])
    codes, scales = quantize(active_profiles)
    source_cpu = torch.as_tensor(source_np).pin_memory()
    source_gpu = source_cpu.to(device, non_blocking=True)
    codes_gpu = torch.as_tensor(codes, device=device)
    scales_gpu = torch.as_tensor(scales, device=device)
    item_gpu = torch.as_tensor(item_matrix, device=device)
    mask_gpu = torch.as_tensor(mask_matrix, device=device)

    source_file = Path(tempfile.gettempdir()) / f"skc_gpu_source_{args.data.stem}.bin"
    source_file.write_bytes(source_np.tobytes())
    fd = os.open(source_file, os.O_RDONLY)
    row_bytes = source_np.shape[1] * 4
    route_state = {
        "source_gpu": source_gpu,
        "source_cpu": source_cpu,
        "fd": fd,
        "row_bytes": row_bytes,
        "source_shape": source_np.shape,
        "item_cpu": item_matrix,
        "mask_cpu": mask_matrix,
        "item_gpu": item_gpu,
        "mask_gpu": mask_gpu,
        "codes_gpu": codes_gpu,
        "scales_gpu": scales_gpu,
    }

    routes = ("gpu_resident", "cpu_pinned", "storage", "int8_cache")
    batch_configs = (1, 8, 32, 128)
    results = {str(batch): {route: {"materialize": [], "end_to_end": []} for route in routes} for batch in batch_configs}
    quality_reference = {}
    try:
        for batch in batch_configs:
            usable = min(64 if batch == 1 else 32 if batch == 8 else 16 if batch == 32 else 8, len(neighbors) // batch)
            batches = []
            for b in range(usable):
                qids = range(b * batch, (b + 1) * batch)
                seqs, _lens, targets = pad_batch([queries[i] for i in qids], str(device))
                nbs = neighbors[b * batch:(b + 1) * batch].reshape(-1)
                local_ids = np.asarray([local_map[int(x)] for x in nbs], dtype=np.int64)
                batches.append((seqs, targets, local_ids))

            # Warm CUDA kernels outside timed measurements.
            for route in routes:
                with torch.inference_mode():
                    probe = route_logits(route_state, route, batches[0][2], device)
                    _ = run_forward(model, batches[0][0], probe)
                torch.cuda.synchronize(device)

            for route in routes:
                for seqs, targets, local_ids in batches:
                    if route == "storage":
                        try:
                            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                        except Exception:
                            pass
                    torch.cuda.synchronize(device)
                    start = time.perf_counter_ns()
                    with torch.inference_mode():
                        profiles = route_logits(route_state, route, local_ids, device)
                        mat_done = time.perf_counter_ns()
                        logits = run_forward(model, seqs, profiles)
                        torch.cuda.synchronize(device)
                    end = time.perf_counter_ns()
                    results[str(batch)][route]["materialize"].append((mat_done - start) / 1e6)
                    results[str(batch)][route]["end_to_end"].append((end - start) / 1e6)
                    if batch == 32:
                        quality_reference.setdefault(route, []).append(logits.float().cpu())
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                except Exception:
                    pass
    finally:
        os.close(fd)
        source_file.unlink(missing_ok=True)

    output = {batch: {} for batch in results}
    for batch, route_map in results.items():
        for route, series in route_map.items():
            output[batch][route] = {
                "materialize_ms": summarize(series["materialize"]),
                "end_to_end_ms": summarize(series["end_to_end"]),
            }

    # Route fidelity at batch 32: compare top-20 logits to GPU-resident source.
    ref = torch.cat(quality_reference["gpu_resident"], dim=0)
    fidelity = {}
    for route in routes:
        if route == "gpu_resident":
            continue
        other = torch.cat(quality_reference[route], dim=0)
        ref_top = ref.topk(20, dim=1).indices
        other_top = other.topk(20, dim=1).indices
        overlap = torch.tensor([
            len(set(ref_top[i].tolist()) & set(other_top[i].tolist())) / 20.0
            for i in range(ref_top.shape[0])
        ])
        fidelity[route] = {
            "top20_overlap_mean": float(overlap.mean()),
            "logit_l2_mean": float((ref - other).norm(dim=1).mean()),
        }

    result = {
        "dataset": args.data.stem,
        "device": torch.cuda.get_device_name(device),
        "protocol": "real A3 checkpoint, exact top-10 neighbor ids, real test sequences; profiles followed by MPLI forward; storage route uses pread and fadvise(DONTNEED); int8 active-set cache is an offline ceiling",
        "active_sessions": int(len(active_ids)),
        "offline_cache_bytes": int(codes.nbytes + scales.nbytes + active_ids.nbytes),
        "batch_results": output,
        "fidelity_batch32": fidelity,
    }
    out = args.out or (
        ROOT / "results" / "systems" / "gpu_source_serving" / f"{args.data.stem}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(out)
    for batch in ("32", "128"):
        print(batch, {
            route: output[batch][route]["end_to_end_ms"]["p50_ms"]
            for route in routes
        })


if __name__ == "__main__":
    main()
